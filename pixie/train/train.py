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
from pixie.config import (
    DatasetConfig,
    Config,
    AdamWOptimizerConfig,
    LionOptimizerConfig,
)
from lion_pytorch import Lion
from .dataset import DuckDBDataset
from torch.utils.data import Dataset
import wandb
import torch.nn.functional as F
import pytorch_warmup as warmup
import git
import shutil
from safetensors.torch import load_model


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


def logits_to_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # logits: [batch_size, seq_len, vocab_size]
    # labels: [batch_size, seq_len]
    log_probs = F.log_softmax(logits, dim=2)
    probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return probs  # [batch_size, seq_len]


def dpo_loss(
    ref_probs: torch.Tensor, probs: torch.Tensor, mask: torch.Tensor, beta: float
) -> torch.Tensor:
    # ref_probs, probs: [batch_size, seq_len]
    # https://github.com/jingyaogong/minimind/issues/298
    seq_lengths = mask.sum(dim=1, keepdim=True)  # (batch_size, 1)
    ref_probs = (ref_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    probs = (probs * mask).sum(dim=1) / seq_lengths.squeeze()

    batch_size = ref_probs.shape[0]
    chosen_ref_probs = ref_probs[: batch_size // 2]
    reject_ref_probs = ref_probs[batch_size // 2 :]
    chosen_probs = probs[: batch_size // 2]
    reject_probs = probs[batch_size // 2 :]

    pi_logratios = chosen_probs - reject_probs
    ref_logratios = chosen_ref_probs - reject_ref_probs
    logits = pi_logratios - ref_logratios
    loss = (logits - 1 / (2 * beta)) ** 2
    return loss.mean()


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
        self.config.save(self.save_dir / "config.yaml")
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
        self.model, self.tokenizer, self.ref_model = self.init_model()
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
        tokenizer = self.config.load_tokenizer()
        model = self.config.load_model()
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
        if self.args.type == "dpo":
            ref_model = self.config.load_model()
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
        else:
            ref_model = None
        return model, tokenizer, ref_model

    def init_data_loader(self):
        train_ds: Dataset
        train_ds = DuckDBDataset(
            self.args.dataset.path,
            self.tokenizer,
            max_length=self.args.context_length,
            type=self.args.type,
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
        for step, batch in enumerate(self.data_loader):
            if self.args.type == "dpo":
                chosen_x = batch["chosen_x"].to(self.args.device)
                chosen_y = batch["chosen_y"].to(self.args.device)
                chosen_mask = batch["chosen_mask"].to(self.args.device)
                rejected_x = batch["rejected_x"].to(self.args.device)
                rejected_y = batch["rejected_y"].to(self.args.device)
                rejected_mask = batch["rejected_mask"].to(self.args.device)

                X = torch.cat([chosen_x, rejected_x], dim=0)
                Y = torch.cat([chosen_y, rejected_y], dim=0)
                loss_mask = torch.cat([chosen_mask, rejected_mask], dim=0)

                with self.ctx:
                    with torch.no_grad():
                        assert self.ref_model is not None
                        ref_out = self.ref_model(X)
                    ref_probs = logits_to_probs(ref_out.logits, Y) * loss_mask
                    out = self.model(X)
                    probs = logits_to_probs(out.logits, Y) * loss_mask
                    loss = dpo_loss(ref_probs, probs, loss_mask, beta=0.1)

            elif self.args.type == "sft":
                X, Y, loss_mask = batch
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
                    loss = (loss * loss_mask).sum() / loss_mask.sum()
                    # loss += res.aux_loss
            else:
                X, Y = batch
                X = X.to(self.args.device)
                Y = Y.to(self.args.device)

                with self.ctx:
                    out = self.model(X)
                    pad = self.tokenizer.pad_token_id
                    assert isinstance(pad, int)
                    loss = F.cross_entropy(
                        out.logits.view(-1, out.logits.size(-1)),
                        Y.view(-1),
                        ignore_index=pad,
                    )
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
