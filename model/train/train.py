from dataclasses import dataclass
from pathlib import Path
import time
from typing import Literal
import warnings
from slugify import slugify
import torch
from torch import optim
from torch.utils.data import DataLoader
from contextlib import nullcontext
from model.config import DatasetConfig, Config
from model.train.dataset import DuckDBDataset
from torch.utils.data import Dataset
import wandb
import dotenv
import torch.nn.functional as F
import pytorch_warmup as warmup
import git

dotenv.load_dotenv()

warnings.filterwarnings("ignore")


@dataclass
class TrainingConfig:
    """Training parameters."""

    # Fields from model config
    dataset: DatasetConfig
    context_length: int
    batch_size: int
    epochs: int
    learning_rate: float
    grad_clip: float
    warmup_steps: int | None
    type: Literal["pretrain", "sft"]

    # Additional fields
    out_dir: str = "./out"
    wandb: bool = False
    accumulation_steps: int = 8
    log_interval: int = 100
    checkpoint: str | None = None
    max_steps: int | None = None
    name: str | None = None
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16" if torch.cuda.is_available() else "float32"

    @staticmethod
    def pretrain(
        config: Config, checkpoint: str | None, wandb: bool
    ) -> "TrainingConfig":
        assert config.pretrain is not None
        dataset = config.pretrain.dataset
        if isinstance(dataset, str):
            dataset = DatasetConfig(path=dataset)
        return TrainingConfig(
            epochs=config.pretrain.epochs,
            batch_size=config.pretrain.batch_size,
            learning_rate=config.pretrain.learning_rate,
            warmup_steps=config.pretrain.warmup_steps,
            context_length=config.pretrain.context_length,
            grad_clip=config.pretrain.grad_clip,
            checkpoint=checkpoint,
            wandb=wandb,
            dataset=dataset,
            type="pretrain",
        )

    @staticmethod
    def sft(config: Config, checkpoint: str | None, wandb: bool) -> "TrainingConfig":
        raise NotImplementedError("SFT training config not implemented yet.")

    def __post_init__(self):
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
        if self.type != "pretrain":
            assert self.checkpoint is not None


class Trainer:
    def __init__(
        self,
        config: Config,
        project: str | None,
        checkpoint: str | None,
        use_wandb: bool,
        train_type: Literal["pretrain", "sft"],
    ):
        assert (
            torch.cuda.is_available()
        ), "CUDA is not available. Please check your setup."
        assert config.name
        self.runid = config.name
        if project is not None:
            self.runid += "-" + slugify(project)
        # get git branch name
        try:
            repo = git.Repo(search_parent_directories=True)
            branch = repo.active_branch.name
            if branch != "main":
                self.runid += f"-{slugify(branch)}"
        except Exception as e:
            ...
        self.runid += "-" + time.strftime("%Y%m%d-%H%M%S")

        self.config = config
        self.args = (
            TrainingConfig.pretrain(config, checkpoint=checkpoint, wandb=use_wandb)
            if train_type == "pretrain"
            else TrainingConfig.sft(config, checkpoint=checkpoint, wandb=use_wandb)
        )

        assert self.config.name
        self.save_dir = (
            Path(self.args.out_dir) / self.config.name / self.args.type / self.runid
        )
        self.save_dir.mkdir(parents=True, exist_ok=True)
        # symlink to latest
        latest = Path(self.args.out_dir) / self.config.name / self.args.type / "latest"
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(self.save_dir.resolve(), target_is_directory=True)

        # Get model configs
        if self.args.checkpoint is not None:
            prev_config = Config.load(Path(self.args.checkpoint).parent / "config.yaml")
            assert prev_config == self.config, "Config mismatch with checkpoint."
        else:
            self.config.save(self.save_dir / "config.yaml")

        # Setup device and context
        self.ctx = (
            nullcontext()
            if self.args.device == "cpu"
            else torch.amp.autocast_mode.autocast("cuda")
        )
        # Initialize WandB if enabled
        if self.args.wandb:
            wandb_config = {
                "runid": self.runid,
                "config": self.config.model_dump(),
                "train_config": self.args.__dict__,
            }
            assert self.config.name
            wandb.init(
                project=self.config.name + "-" + self.args.type,
                name=self.runid,
                config=wandb_config,
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
                self.optimizer, warmup_period=self.args.warmup_steps
            )
            if self.args.warmup_steps
            else None
        )

    def init_model(self):
        tokenizer = self.config.load_tokenizer()
        model = self.config.load_model(compile=True).to(self.args.device)  # type: ignore
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
        train_ds: Dataset
        train_ds = DuckDBDataset(
            self.args.dataset.path,
            self.tokenizer,
            max_length=self.args.context_length,
            limit=self.args.dataset.limit,
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

    def save_model(self, epoch: int | None = None):
        self.model.eval()
        if epoch is not None:
            ckp = self.save_dir / f"model-{epoch}.pth"
        else:
            ckp = self.save_dir / f"model.pth"

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
        self.save_model()
