from pathlib import Path
import os, time
from transformers import TrainingArguments
from datasets import load_dataset, Dataset
from pixie import utils
from pixie.models._config import (
    AdamWOptimizerConfig,
    DatasetConfig,
    Config,
    BaseTrainingConfig,
)
from pixie.train.pretrain import PretrainTrainer
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.dpo_trainer import DPOTrainer
from trl.trainer.dpo_config import DPOConfig
from trl.trainer.sft_config import SFTConfig
from pixie.train import pretrain
from safetensors.torch import load_model
from typing import Any
from transformers.modeling_utils import PreTrainedModel


def _create_runid_and_path(config: Config, type: str) -> tuple[str, str]:
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
    path = Path("out") / (config.name + "-x") / type / runid
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
            ds = DatasetConfig(path=str(path.parent), data_files=[str(path.name)])
        else:
            ds = DatasetConfig(path=dataset_config)
    else:
        ds = dataset_config
    dataset = load_dataset(
        ds.path,
        split=f"train[:{ds.ratio * 100}%]" if ds.ratio else "train",
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


def _load_dataset_raw(
    dataset_config: DatasetConfig | str,
) -> Dataset:
    if isinstance(dataset_config, str):
        path = Path(dataset_config)
        if path.is_file():
            ds = DatasetConfig(path=str(path.parent), data_files=[str(path.name)])
        else:
            ds = DatasetConfig(path=dataset_config)
    else:
        ds = dataset_config
    dataset = load_dataset(
        ds.path,
        split=f"train[:{ds.ratio * 100}%]" if ds.ratio else "train",
        data_dir=ds.data_dir,
        data_files=ds.data_files,
    )
    assert isinstance(dataset, Dataset)
    dataset = dataset.shuffle(seed=42)
    return dataset


def _get_trainning_args(
    args: BaseTrainingConfig, model_save_dir: str, use_wandb: bool
) -> TrainingArguments:
    assert isinstance(args.optimizer, AdamWOptimizerConfig)
    return TrainingArguments(
        output_dir=model_save_dir,
        overwrite_output_dir=True,
        # Training args
        do_train=True,
        fp16=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        max_grad_norm=args.grad_clip,
        warmup_steps=args.warmup_steps or 0,
        num_train_epochs=args.epochs,
        save_strategy="no",  # "steps",
        save_steps=5000,
        lr_scheduler_type="cosine",
        learning_rate=args.optimizer.learning_rate,
        torch_compile=True,
        torch_compile_mode="default",
        # Evaluation args
        do_eval=False,
        # Logging args
        logging_strategy="steps",
        logging_steps=100,
        report_to="wandb" if use_wandb else "none",
        # run_name=wandb
    )


def train_pretrain(config: Config, use_wandb: bool):
    model = utils.load_model(config.model)
    tokenizer = config.model.load_tokenizer()
    # prepare dataset
    assert config.pretrain is not None
    args = config.pretrain
    dataset = _load_dataset(args.dataset, pretrain.preprocess, config, tokenizer)
    runid, save_dir = _create_runid_and_path(config, "pretrain")
    training_args = _get_trainning_args(args, save_dir, use_wandb)
    training_args.label_names = ["loss_mask"]
    trainer = PretrainTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(save_dir)


def train_sft(config: Config, ckpt: Path, use_wandb: bool):
    model = utils.load_model(config.model)
    missing, unexpected = load_model(model, ckpt, device="cuda")
    if missing or unexpected:
        raise ValueError(
            f"Failed to load model: missing keys: {missing}, unexpected keys: {unexpected}"
        )
    print(f"Loaded checkpoint from {ckpt}")
    # prepare dataset
    assert config.sft is not None
    args = config.sft
    dataset = _load_dataset_raw(args.dataset)
    runid, save_dir = _create_runid_and_path(config, "sft")
    raw_training_args = _get_trainning_args(args, save_dir, use_wandb)
    training_args = SFTConfig(**raw_training_args.to_dict())
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(save_dir)


def train_dpo(config: Config, ckpt: Path, use_wandb: bool):
    model = utils.load_model(config.model)
    missing, unexpected = load_model(model, ckpt, device="cuda")
    if missing or unexpected:
        raise ValueError(
            f"Failed to load model: missing keys: {missing}, unexpected keys: {unexpected}"
        )
    print(f"Loaded checkpoint from {ckpt}")
    ref_model = utils.load_model(config.model)
    missing, unexpected = load_model(ref_model, ckpt, device="cuda")
    if missing or unexpected:
        raise ValueError(
            f"Failed to load reference model: missing keys: {missing}, unexpected keys: {unexpected}"
        )
    # prepare dataset
    assert config.sft is not None
    args = config.sft
    dataset = _load_dataset_raw(args.dataset)
    runid, save_dir = _create_runid_and_path(config, "dpo")
    raw_training_args = _get_trainning_args(args, save_dir, use_wandb)
    training_args = DPOConfig(**raw_training_args.to_dict())
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(save_dir)
