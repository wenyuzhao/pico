from dataclasses import dataclass
from pathlib import Path
import time
from typing import Literal
import warnings
from simple_parsing import ArgumentParser
from slugify import slugify
import torch
from torch import optim
from torch.utils.data import DataLoader
from contextlib import nullcontext
from model.pixie import Pixie as Model, Config
from scripts.dataset import DuckDBDataset
from torch.utils.data import Dataset
import wandb
import dotenv
import torch.nn.functional as F
from simple_parsing.helpers import flag, field
import pytorch_warmup as warmup
import git

dotenv.load_dotenv()

warnings.filterwarnings("ignore")


@dataclass
class TrainDataset:
    path: str
    limit: int | None = None


DATASET = {
    "pretrain": TrainDataset(path="datasets/pile", limit=1048576),  # 1B tokens
    "sft": TrainDataset(path="datasets/magpielm-sft-data-v0.1", limit=1000),
}


@dataclass
class TrainingConfig:
    """Training parameters."""

    out_dir: str = "./out"
    epochs: int = 1
    batch_size: int = 4
    learning_rate: float = 5e-4
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16" if torch.cuda.is_available() else "float32"
    wandb: bool = flag(default=False)
    wandb_project_prefix: str = "pixie"
    accumulation_steps: int = 8
    grad_clip: float = 1.0
    log_interval: int = 100
    warmup_period: int | None = 400
    max_seq_len: int | None = None
    """Maximum sequence length for training. If None, uses the model's context length."""
    checkpoint: str | None = field(alias="ckpt", default=None)
    """Path to the checkpoint file to continue training from."""
    max_steps: int | None = None
    name: str | None = None
    type: Literal["base", "sft"] = "base"

    def __post_init__(self):
        self.max_seq_len = self.max_seq_len or Config().training_context_length
        if self.checkpoint is not None:
            ckpt_path = Path(self.checkpoint)
            segments = ckpt_path.name.split("-")
            if len(segments) > 1 and segments[-1].isdigit():
                self.checkpoint_epoch = int(segments[-1])
            else:
                self.checkpoint_epoch = None
            self.checkpoint_runid = ckpt_path.parent.name
        else:
            self.checkpoint_epoch = None
            self.checkpoint_runid = None
        if self.type != "base":
            assert self.checkpoint is not None


class Trainer:
    def __init__(self, args: TrainingConfig):
        self.runid = Model.NAME
        # get git branch name
        try:
            repo = git.Repo(search_parent_directories=True)
            branch = repo.active_branch.name
            if branch != "main":
                self.runid += f"-{slugify(branch)}"
        except Exception as e:
            ...
        if args.name is not None:
            self.runid += "-" + slugify(args.name)
        self.runid += "-" + time.strftime("%Y%m%d-%H%M%S")
        self.args = args
        match args.type:
            case "base":
                self.type = "pretrain"
            case "sft":
                self.type = "sft"
            case _:
                raise ValueError(f"Unknown model type: {args.type}")
        self.save_dir = Path(args.out_dir) / self.type / self.runid
        self.save_dir.mkdir(parents=True, exist_ok=True)
        # Get model configs
        if args.checkpoint is not None:
            prev_cfg = Path(args.checkpoint).parent / "config.json"
            assert prev_cfg.exists(), "Previous config file not found."
            self.config = Config.from_pretrained(prev_cfg)
            print(f"Loaded config from {prev_cfg}")
        else:
            self.config = Config()
            self.config.save_pretrained(self.save_dir)
        # Setup device and context
        self.device_type = "cuda" if "cuda" in args.device else "cpu"
        self.ctx = (
            nullcontext()
            if self.device_type == "cpu"
            else torch.amp.autocast_mode.autocast("cuda")
        )
        # Initialize WandB if enabled
        if args.wandb:
            config = {
                **args.__dict__,
                **self.config.to_dict(),
                "runid": self.runid,
                "dataset": DATASET[self.type].path,
                "dataset_limit": DATASET[self.type].limit,
            }
            wandb.init(
                project=args.wandb_project_prefix + "-" + self.type,
                name=self.runid,
                config=config,
            )
        # Initialize model, tokenizer, and data loader
        self.model, self.tokenizer = self.init_model()
        self.data_loader = self.init_data_loader()
        self.iter_per_epoch = len(self.data_loader)
        self.scaler = torch.amp.grad_scaler.GradScaler(
            "cuda", enabled=(self.args.dtype in ["float16", "bfloat16"])
        )
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
        )
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.epochs * self.iter_per_epoch,
            eta_min=self.args.learning_rate / 10,
        )
        self.warmup_scheduler = (
            warmup.ExponentialWarmup(
                self.optimizer, warmup_period=self.args.warmup_period
            )
            if self.args.warmup_period
            else None
        )

    def init_model(self):
        tokenizer = Model.tokenizer(self.config)
        model = Model(self.config).to(self.args.device)  # type: ignore
        print(
            f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} M"
        )
        if self.args.checkpoint is not None:
            model.load_state_dict(
                torch.load(self.args.checkpoint, map_location=self.args.device),
                strict=True,
            )
            if self.args.checkpoint_epoch is not None:
                print(
                    f"Loaded checkpoint from {self.args.checkpoint} (epoch: {self.args.checkpoint_epoch})"
                )
            else:
                print(f"Loaded checkpoint from {self.args.checkpoint}")
        return model, tokenizer

    def init_data_loader(self):
        ds_config = DATASET[self.type]
        train_ds: Dataset
        train_ds = DuckDBDataset(
            ds_config.path,
            self.tokenizer,
            max_length=self.args.max_seq_len or self.config.training_context_length,
            limit=ds_config.limit,
        )
        return DataLoader(
            train_ds,
            batch_size=self.args.batch_size,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
            num_workers=1,
            sampler=None,
        )

    def log(self, epoch: int, step: int, loss: float, lr: float, epoch_time: float):
        print(
            "Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_time:{}min".format(
                epoch + 1,
                self.args.epochs,
                step,
                self.iter_per_epoch,
                loss,
                lr,
                epoch_time,
            )
        )
        if self.args.wandb:
            wandb.log(
                {
                    "loss": loss,
                    "learning_rate": lr,
                    "epoch_time": epoch_time,
                    "epoch": epoch + 1,
                }
            )

    def train_epoch(self, epoch: int):
        start_time = time.time()
        for step, (X, Y, loss_mask) in enumerate(self.data_loader):
            # Move data to target device
            X = X.to(self.args.device)
            Y = Y.to(self.args.device)
            loss_mask = loss_mask.to(self.args.device)

            with self.ctx:
                out = self.model(X)
                pad = self.tokenizer.pad_token_id
                assert isinstance(pad, int)
                loss = F.cross_entropy(
                    out.logits.view(-1, out.logits.size(-1)),
                    Y.view(-1),
                    ignore_index=pad,
                )
                # loss += res.aux_loss
                loss = loss / self.args.accumulation_steps

            self.scaler.scale(loss).backward()
            if self.warmup_scheduler is not None:
                with self.warmup_scheduler.dampening():
                    self.lr_scheduler.step()

            if (step + 1) % self.args.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.grad_clip
                )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.optimizer.zero_grad(set_to_none=True)

            if step % self.args.log_interval == 0:
                spend_time = time.time() - start_time
                _loss = loss.item() * self.args.accumulation_steps
                _lr = self.optimizer.param_groups[-1]["lr"]
                _epoch_time = (
                    spend_time / (step + 1) * self.iter_per_epoch // 60
                    - spend_time // 60
                )
                self.log(epoch, step, _loss, _lr, _epoch_time)
                if (
                    self.args.max_steps
                    and step / self.args.log_interval >= self.args.max_steps
                ):
                    break
        self.save_model(epoch=epoch)

    def save_model(self, epoch: int | None = None, final: bool = False):
        self.model.eval()
        if epoch is not None:
            ckp = self.save_dir / f"{Model.NAME}-{self.args.type}-{epoch}.pth"
        elif final:
            ckp = self.save_dir / f"{Model.NAME}-{self.args.type}.pth"
        else:
            ckp = self.save_dir / f"{Model.NAME}-{self.args.type}.pth"

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
        torch.save(state_dict, ckp)
        self.model.train()

    def train(self):
        for epoch in range(self.args.epochs):
            if self.args.checkpoint_epoch is not None:
                epoch = self.args.checkpoint_epoch + 1 + epoch
            self.train_epoch(epoch)
        self.save_model(final=True)


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available. Please check your setup."
    parser = ArgumentParser()
    parser.add_arguments(TrainingConfig, dest="training_config")
    ns = parser.parse_args()  # parse the given `args`
    training_config = ns.training_config
    assert isinstance(training_config, TrainingConfig)
    trainer = Trainer(training_config)
    trainer.train()
