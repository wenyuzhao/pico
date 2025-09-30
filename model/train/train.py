from pathlib import Path
import time
from typing import Any, Literal
from pydantic import BaseModel
from slugify import slugify
import torch
from torch import optim
from torch.utils.data import DataLoader
from contextlib import nullcontext
import yaml
from model.config import (
    DatasetConfig,
    Config,
    AdamWOptimizerConfig,
    LionOptimizerConfig,
)
from lion_pytorch import Lion
from model.train.dataset import DuckDBDataset
from torch.utils.data import Dataset
import wandb
import torch.nn.functional as F
import pytorch_warmup as warmup
import git


class TrainingArgs(BaseModel):
    """Training parameters."""

    # Fields from model config
    context_length: int
    batch_size: int
    epochs: int
    grad_clip: float
    warmup_steps: int | None
    accumulation_steps: int
    gradient_checkpointing: bool
    optimizer: AdamWOptimizerConfig | LionOptimizerConfig
    dataset: DatasetConfig
    type: Literal["pretrain", "sft"]

    # Additional fields
    config: Config
    runid: str
    out_dir: str = "./out"
    wandb: bool = False
    log_interval: int = 100
    checkpoint: Path | None = None
    max_steps: int | None = None
    name: str | None = None
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16" if torch.cuda.is_available() else "float32"
    checkpoint_epoch: int | None = None
    checkpoint_runid: str | None = None

    @staticmethod
    def pretrain(runid: str, config: Config, wandb: bool) -> "TrainingArgs":
        assert config.pretrain is not None, "Pretrain configuration is not set."
        cfg = config.pretrain
        dataset = cfg.dataset
        if isinstance(dataset, str):
            dataset = DatasetConfig(path=dataset)
        optimizer = cfg.optimizer
        if isinstance(optimizer, str):
            if optimizer == "adamw":
                optimizer = AdamWOptimizerConfig()
            elif optimizer == "lion":
                optimizer = LionOptimizerConfig()
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer}")
        return TrainingArgs(
            context_length=cfg.context_length,
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            grad_clip=cfg.grad_clip,
            warmup_steps=cfg.warmup_steps,
            accumulation_steps=cfg.accumulation_steps,
            gradient_checkpointing=cfg.gradient_checkpointing,
            optimizer=optimizer,
            dataset=dataset,
            checkpoint=None,
            wandb=wandb,
            type="pretrain",
            config=config,
            runid=runid,
        )

    @staticmethod
    def sft(
        runid: str, config: Config, checkpoint: Path | None, wandb: bool
    ) -> "TrainingArgs":
        assert config.sft is not None, "SFT configuration is not set."
        cfg = config.sft
        dataset = cfg.dataset
        if isinstance(dataset, str):
            dataset = DatasetConfig(path=dataset)
        optimizer = cfg.optimizer
        if isinstance(optimizer, str):
            if optimizer == "adamw":
                optimizer = AdamWOptimizerConfig()
            elif optimizer == "lion":
                optimizer = LionOptimizerConfig()
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer}")
        assert checkpoint is not None, "Checkpoint must be provided for SFT."
        if checkpoint.is_dir():
            checkpoint = checkpoint / "model.pth"
            assert checkpoint.exists(), f"Checkpoint not found: {checkpoint}"
        return TrainingArgs(
            context_length=cfg.context_length,
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            grad_clip=cfg.grad_clip,
            warmup_steps=cfg.warmup_steps,
            accumulation_steps=cfg.accumulation_steps,
            gradient_checkpointing=cfg.gradient_checkpointing,
            optimizer=optimizer,
            dataset=dataset,
            checkpoint=checkpoint,
            wandb=wandb,
            type="sft",
            config=config,
            runid=runid,
        )

    def model_post_init(self, __context: Any) -> None:
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
        checkpoint: Path | None,
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
            TrainingArgs.pretrain(self.runid, config, wandb=use_wandb)
            if train_type == "pretrain"
            else TrainingArgs.sft(
                self.runid, config, checkpoint=checkpoint, wandb=use_wandb
            )
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
        # Load and save model configs
        self.config.save(self.save_dir / "config.yaml")
        (self.save_dir / "args.yaml").write_text(yaml.safe_dump(self.args.model_dump()))
        # Setup device and context
        self.ctx = (
            nullcontext()
            if self.args.device == "cpu"
            else torch.amp.autocast_mode.autocast("cuda")
        )
        # Initialize WandB if enabled
        assert self.config.name
        if self.args.wandb:
            wandb_proj = self.config.name + "-" + self.args.type
            wandb.init(
                project=wandb_proj, name=self.runid, config=self.args.model_dump()
            )
        # Initialize model, tokenizer, and data loader
        self.model, self.tokenizer = self.init_model()
        self.data_loader = self.init_data_loader()
        self.iter_per_epoch = len(self.data_loader)
        self.scaler = torch.amp.grad_scaler.GradScaler(
            "cuda", enabled=(self.args.dtype in ["float16", "bfloat16"])
        )
        self.optimizer = self.init_optimizer()
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.epochs * self.iter_per_epoch,
            eta_min=self.args.optimizer.learning_rate / 10,
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
        model = self.config.load_model()
        model = torch.compile(model, mode="default").to(self.args.device)  # type: ignore
        print(
            f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} M"
        )
        if self.args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
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

    def init_optimizer(self):
        opt = self.args.optimizer
        match opt.name:
            case "adamw":
                return optim.AdamW(
                    self.model.parameters(),
                    lr=opt.learning_rate,
                )
            case "lion":
                return Lion(
                    self.model.parameters(),
                    lr=opt.learning_rate,
                    weight_decay=opt.weight_decay,
                )
        raise ValueError(f"Unsupported optimizer: {opt.name}")

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

        state_dict = {k: v.half() for k, v in state_dict.items()}
        torch.save(state_dict, ckp)
        if epoch is None:
            self.model.save_pretrained(self.save_dir, safe_serialization=True)  # type: ignore
            self.tokenizer.save_pretrained(self.save_dir, save_jinja_files=False)
        self.model.train()

    def train(self):
        for epoch in range(self.args.epochs):
            if self.args.checkpoint_epoch is not None:
                epoch = self.args.checkpoint_epoch + 1 + epoch
            self.train_epoch(epoch)
        self.save_model()
