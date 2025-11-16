import os
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
from pixie.models import (
    DatasetConfig,
    Config,
    AdamWOptimizerConfig,
    LionOptimizerConfig,
)
from lion_pytorch import Lion

from pixie.train import pretrain, sft, dpo
from torch.utils.data import Dataset
import wandb
import pytorch_warmup as warmup
import git
import shutil
from safetensors.torch import load_model
from .. import utils
from datasets import load_dataset, Dataset


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
    type: Literal["pretrain", "sft", "dpo"]

    # Additional fields
    config: Config
    runid: str
    out_dir: str = "./out"
    wandb: bool = False
    log_interval: int = 100
    checkpoint: str | None = None
    max_steps: int | None = None
    name: str | None = None
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16" if torch.cuda.is_available() else "float32"
    checkpoint_epoch: int | None = None
    checkpoint_runid: str | None = None

    @staticmethod
    def create(
        runid: str,
        config: Config,
        checkpoint: Path | None,
        wandb: bool,
        type: Literal["pretrain", "sft", "dpo"],
    ) -> "TrainingArgs":
        match type:
            case "pretrain":
                assert config.pretrain is not None, "Pretrain configuration is not set."
                cfg = config.pretrain
            case "sft":
                assert config.sft is not None, "SFT configuration is not set."
                cfg = config.sft
            case "dpo":
                assert config.dpo is not None, "DPO configuration is not set."
                cfg = config.dpo
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
        if type != "pretrain":
            assert checkpoint is not None, "Checkpoint must be provided for SFT/DPO."
        else:
            assert (
                checkpoint is None
            ), "Checkpoint must not be provided for pretraining."
        if checkpoint and checkpoint.is_dir():
            checkpoint = checkpoint / "model.safetensors"
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
            checkpoint=str(checkpoint) if checkpoint else None,
            wandb=wandb,
            type=type,
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
        train_type: Literal["pretrain", "sft", "dpo"],
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
        print(f"Run ID: {self.runid}")

        self.config = config
        self.args = TrainingArgs.create(
            self.runid, config, checkpoint=checkpoint, wandb=use_wandb, type=train_type
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
        utils.save_config(self.config, self.save_dir / "config.yaml")
        args_to_save = self.args.model_dump()
        del args_to_save["config"]
        (self.save_dir / "args.yaml").write_text(yaml.safe_dump(args_to_save))
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
        self.tokenizer, self.model, self.loss = self.init_model()
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
        self.save_tokenizer()

    def init_model(self):
        tokenizer = self.config.model.load_tokenizer()
        model = utils.load_model(self.config.model)
        print(
            f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} M"
        )
        if self.args.checkpoint is not None:
            missing, unexpected = load_model(
                model, self.args.checkpoint, device=self.args.device
            )
            if missing or unexpected:
                raise ValueError(
                    f"Failed to load model: missing keys: {missing}, unexpected keys: {unexpected}"
                )
            if self.args.checkpoint_epoch is not None:
                print(
                    f"Loaded checkpoint from {self.args.checkpoint} (epoch: {self.args.checkpoint_epoch})"
                )
            else:
                print(f"Loaded checkpoint from {self.args.checkpoint}")
        model = torch.compile(model, mode="default").to(self.args.device)  # type: ignore
        if self.args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        match self.args.type:
            case "pretrain":
                loss = pretrain.PretrainLoss(self.args.device, model)
            case "sft":
                loss = sft.SFTLoss(self.args.device, model)
            case "dpo":
                ref_model = utils.load_model(self.config.model)
                assert self.args.checkpoint is not None
                missing, unexpected = load_model(
                    ref_model, self.args.checkpoint, device=self.args.device
                )
                assert (
                    not missing and not unexpected
                ), f"Failed to load reference model: missing keys: {missing}, unexpected keys: {unexpected}"
                ref_model = torch.compile(ref_model, mode="default").to(self.args.device)  # type: ignore
                ref_model.eval()
                ref_model.requires_grad_(False)
                assert self.args.config.dpo is not None
                loss = dpo.DPOLoss(
                    self.args.device, model, ref_model, beta=self.args.config.dpo.beta
                )
        loss = torch.compile(loss, mode="default")  # type: ignore
        return tokenizer, model, loss

    def init_data_loader(self):
        match self.args.type:
            case "pretrain":
                preprocess = pretrain.preprocess
            case "sft":
                preprocess = sft.preprocess
            case "dpo":
                preprocess = dpo.preprocess
        if isinstance(self.args.dataset, str):
            path = Path(self.args.dataset)
            if path.is_file():
                ds = DatasetConfig(path=str(path.parent), data_files=[str(path.name)])
            else:
                ds = DatasetConfig(path=self.args.dataset)
        else:
            ds = self.args.dataset
        dataset = load_dataset(
            ds.path,
            split=f"train[:{ds.ratio * 100}%]" if ds.ratio else "train",
            data_dir=ds.data_dir,
            data_files=ds.data_files,
        )
        assert isinstance(dataset, Dataset)
        dataset = dataset.shuffle(seed=42).map(
            preprocess,
            remove_columns=dataset.column_names,
            batched=True,
            num_proc=os.cpu_count(),
            fn_kwargs={"config": self.config, "tokenizer": self.tokenizer},
            # load_from_cache_file=False,
        )
        loader = DataLoader(
            dataset.with_format("torch"),  # type: ignore
            batch_size=self.args.batch_size,
        )
        return loader

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
        for step, batch in enumerate(self.data_loader):
            with self.ctx:
                loss = self.loss(batch)

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

        self.model.save_pretrained(self.save_dir, safe_serialization=True)
        if epoch is not None:
            # copy model.safetensors to model-{epoch}.safetensors
            latest_ckp = self.save_dir / f"model-{epoch}.safetensors"
            latest_ckp.unlink(missing_ok=True)
            shutil.copy(self.save_dir / "model.safetensors", latest_ckp)

        self.model.train()

    def save_tokenizer(self):
        self.tokenizer.save_pretrained(self.save_dir, save_jinja_files=False)

    def train(self):
        for epoch in range(self.args.epochs):
            if self.args.checkpoint_epoch is not None:
                epoch = self.args.checkpoint_epoch + 1 + epoch
            self.train_epoch(epoch)
        self.save_model()
