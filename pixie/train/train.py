from pathlib import Path
import os, time
from torch import Tensor
from transformers import TrainingArguments
from datasets import load_dataset, Dataset
from pixie import utils
from pixie.models._base import (
    AdamWOptimizerConfig,
    LionOptimizerConfig,
    DatasetConfig,
    Config,
    BaseTrainingConfig,
)
from pixie.train import pretrain, sft, dpo
from typing import Any
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers import PreTrainedTokenizerBase


def _create_runid_and_path(
    config: Config, type: str, tokenizer: PreTrainedTokenizerBase, dry_run: bool
) -> tuple[str, str]:
    assert config.name
    runid = config.name
    # if project is not None:
    #     self.runid += "-" + slugify(project)
    # get git branch name
    # try:
    #     repo = git.Repo(search_parent_directories=True)
    #     branch = repo.active_branch.name
    #     if branch != "main":
    #         self.runid += f"-{slugify(branch)}"
    # except Exception as e:
    #     ...
    runid += "-" + time.strftime("%Y%m%d-%H%M%S")
    print(f"Run ID: {runid}")
    os.environ["WANDB_PROJECT"] = f"{config.name}-{type}"
    os.environ["WANDB_NAME"] = runid
    if dry_run:
        path = Path("out/scratch")
    else:
        path = Path("out") / config.name / type / runid
    if not dry_run:
        path.mkdir(parents=True, exist_ok=True)
        utils.save_config(config, path / "config.yaml")
        tokenizer.save_pretrained(path, save_jinja_files=False)
    return runid, str(path)


def _load_dataset(
    dataset_config: DatasetConfig | str,
    preprocess_fn: Any,
    config: Config,
    tokenizer: Any,
) -> Dataset:
    if isinstance(dataset_config, str):
        path = Path(dataset_config)
        if path.is_file():
            ds = DatasetConfig(name=str(path.parent), data_files=[str(path.name)])
        else:
            ds = DatasetConfig(name=dataset_config)
    else:
        ds = dataset_config
    dataset = load_dataset(
        ds.name,
        split=(
            f"train[:{round(ds.ratio * 100)}%]" if ds.ratio is not None else "train"
        ),
        data_dir=ds.data_dir,
        data_files=ds.data_files,
    )
    assert isinstance(dataset, Dataset)
    dataset = dataset.shuffle(seed=42).map(
        preprocess_fn,
        remove_columns=dataset.column_names,
        batched=True,
        num_proc=os.cpu_count(),
        fn_kwargs={"config": config, "tokenizer": tokenizer},
        # load_from_cache_file=False,
    )
    return dataset


def _get_trainning_args(
    args: BaseTrainingConfig, model_save_dir: str, use_wandb: bool
) -> TrainingArguments:
    # assert isinstance(args.optimizer, AdamWOptimizerConfig)
    optim: AdamWOptimizerConfig | LionOptimizerConfig = (
        args.optimizer
        if isinstance(args.optimizer, (AdamWOptimizerConfig, LionOptimizerConfig))
        else (
            AdamWOptimizerConfig()
            if args.optimizer == "adamw"
            else LionOptimizerConfig()
        )
    )
    if isinstance(optim, AdamWOptimizerConfig):
        learning_rate = optim.learning_rate
        betas = optim.betas
        eps = optim.eps
        weight_decay = optim.weight_decay
    else:
        learning_rate = optim.learning_rate
        betas = optim.betas
        eps = None
        weight_decay = optim.weight_decay

    return TrainingArguments(
        output_dir=model_save_dir,
        overwrite_output_dir=True,
        # Training args
        do_train=True,
        fp16=True,
        per_device_train_batch_size=(
            args.batch_size if args.batch_size != "auto" else 8
        ),
        auto_find_batch_size=args.batch_size == "auto",
        gradient_accumulation_steps=args.accumulation_steps,
        max_grad_norm=args.grad_clip,
        warmup_steps=args.warmup_steps or 0,
        num_train_epochs=args.epochs,
        save_strategy="no",  # "steps",
        save_steps=5000,
        lr_scheduler_type="cosine",
        torch_compile=True,
        torch_compile_mode="default",
        half_precision_backend="cpu_amp",
        optim="adamw_torch" if optim.name == "adamw" else "lion_32bit",
        learning_rate=optim.learning_rate,
        weight_decay=weight_decay,
        adam_beta1=betas[0],
        adam_beta2=betas[1],
        adam_epsilon=eps if eps is not None else 1e-8,
        # Evaluation args
        do_eval=False,
        # Logging args
        logging_strategy="steps",
        logging_steps=100,
        report_to="wandb" if use_wandb else "none",
        # run_name=wandb
    )


def _count_tokens(dataset: Dataset) -> int:
    size = len(dataset[0]["input_ids"])
    total_tokens = size * len(dataset)
    return total_tokens


def train_pretrain(config: Config, use_wandb: bool, dry_run: bool):
    model = utils.load_model(config.model)
    tokenizer = config.model.load_tokenizer()
    runid, save_dir = _create_runid_and_path(config, "pretrain", tokenizer, dry_run)
    # prepare dataset
    assert config.pretrain is not None
    args = config.pretrain
    dataset = _load_dataset(args.dataset, pretrain.preprocess, config, tokenizer)
    # dataset = dataset.take(100)
    # count tokens
    tokens = _count_tokens(dataset)
    billion_tokens = tokens / 1_000_000_000
    print(f"Total tokens in dataset: {tokens} ({billion_tokens:.2f}B)")
    training_args = _get_trainning_args(args, save_dir, use_wandb)
    training_args.label_names = ["loss_mask"]
    trainer = pretrain.PretrainTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    if not dry_run:
        trainer.save_model(save_dir)


def train_sft(config: Config, ckpt: Path, use_wandb: bool, dry_run: bool):
    model = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True)
    tokenizer = config.model.load_tokenizer()
    print(f"Loaded checkpoint from {ckpt}")
    runid, save_dir = _create_runid_and_path(config, "sft", tokenizer, dry_run)
    # prepare dataset
    assert config.sft is not None
    args = config.sft
    dataset = _load_dataset(args.dataset, sft.preprocess, config, tokenizer)
    # dataset = dataset.take(100)
    # count tokens
    tokens = _count_tokens(dataset)
    billion_tokens = tokens / 1_000_000_000
    print(f"Total tokens in dataset: {tokens} ({billion_tokens:.2f}B)")
    training_args = _get_trainning_args(args, save_dir, use_wandb)
    training_args.label_names = ["loss_mask"]
    trainer = sft.SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(save_dir)


def train_dpo(config: Config, ckpt: Path, use_wandb: bool, dry_run: bool):
    model = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True)
    tokenizer = config.model.load_tokenizer()
    print(f"Loaded checkpoint from {ckpt}")
    ref_model = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True)
    runid, save_dir = _create_runid_and_path(config, "dpo", tokenizer, dry_run)
    # prepare dataset
    assert config.dpo is not None
    args = config.dpo
    dataset = _load_dataset(args.dataset, dpo.preprocess, config, tokenizer)
    # dataset = dataset.take(100)
    training_args = _get_trainning_args(args, save_dir, use_wandb)
    training_args.label_names = [
        "rejected_loss_mask",
        "chosen_loss_mask",
        "chosen_input_ids",
        "rejected_input_ids",
    ]
    trainer = dpo.DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(save_dir)
