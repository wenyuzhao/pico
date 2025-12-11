from pathlib import Path
import os, time
import torch
from transformers import AutoTokenizer, TrainerControl, TrainerState, TrainingArguments
from datasets import load_dataset, Dataset, interleave_datasets, concatenate_datasets
from pixie import utils
from pixie.models._base import (
    AdamWOptimizerConfig,
    LionOptimizerConfig,
    DatasetConfig,
    MixedDatasets,
    Config,
    TrainingConfig,
)
from pixie.train import pretrain, sft, dpo
from typing import Any, override
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback
import os
from trl.trainer.dpo_config import DPOConfig

SEED = 42
ENABLE_DATASET_CACHE = os.environ.get("DATASET_CACHE", "1").lower() in ("1", "true")


def _create_runid_and_path(
    config: Config,
    type: str,
    tokenizer: PreTrainedTokenizerBase,
    dry_run: bool,
    project: str | None,
) -> tuple[str, str]:
    assert config.name
    runid = config.name + "-" + type
    if project:
        runid += "-" + project
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
    os.environ["WANDB_PROJECT"] = f"{config.name}"
    os.environ["WANDB_NAME"] = runid
    if dry_run:
        path = Path("out/scratch")
    else:
        path = Path("out") / config.name / type / runid
    if not dry_run:
        path.mkdir(parents=True, exist_ok=True)
        utils.save_config(config, path / "config.yaml")
        tokenizer.save_pretrained(path, save_jinja_files=True)
    return runid, str(path)


def _load_one_dataset(
    dataset_config: str | DatasetConfig,
    preprocess_fn: Any | None,
    filter_fn: Any | None,
    tokenizer: Any,
    max_length: int,
    force_no_shuffle: bool = False,
) -> Dataset:
    if isinstance(dataset_config, str):
        path = Path(dataset_config)
        if path.is_file():
            ds = DatasetConfig(path=str(path.parent), data_files=[str(path.name)])
        else:
            ds = DatasetConfig(path=dataset_config)
    else:
        ds = dataset_config
    max_length = ds.max_length or max_length
    dataset = load_dataset(
        path=ds.path,
        name=ds.name,
        split=ds.split or "train",
        data_dir=ds.data_dir,
        data_files=ds.data_files,
    )
    assert isinstance(dataset, Dataset), f"{type(dataset)}"
    if ds.shuffle and not force_no_shuffle:
        dataset = dataset.shuffle(seed=SEED)
    if ds.ratio is not None:
        assert 0.0 < ds.ratio and ds.ratio <= 1.0
        total_size = len(dataset)
        new_size = int(total_size * ds.ratio)
        dataset = dataset.select(range(new_size))
    if preprocess_fn is not None:
        dataset = dataset.map(
            preprocess_fn,
            remove_columns=dataset.column_names,
            batched=True,
            num_proc=os.cpu_count(),
            fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
            load_from_cache_file=ENABLE_DATASET_CACHE,
        )
    if filter_fn is not None:
        dataset = dataset.filter(
            filter_fn,
            num_proc=os.cpu_count(),
            load_from_cache_file=ENABLE_DATASET_CACHE,
        )
    return dataset


def _load_dataset(
    dataset_config: str | DatasetConfig | MixedDatasets | list[str | DatasetConfig],
    preprocess_fn: Any | None,
    filter_fn: Any | None,
    tokenizer: Any,
    max_length: int,
    is_pretrain: bool = False,
) -> Dataset:
    if not is_pretrain:
        assert tokenizer.chat_template
        assert (
            "endgeneration" in tokenizer.chat_template
        ), "chat template does not contain `{% generation %}` keyword."
    # Single dataset
    if not isinstance(dataset_config, MixedDatasets) and not isinstance(
        dataset_config, list
    ):
        return _load_one_dataset(
            dataset_config, preprocess_fn, filter_fn, tokenizer, max_length
        )
    # Mixed dataset
    if isinstance(dataset_config, list):
        dataset_config = MixedDatasets(datasets=dataset_config)
    datasets: list[Dataset] = []
    for ds_cfg in dataset_config.datasets:
        no_shuffle = isinstance(ds_cfg, str) or ds_cfg.ratio is None
        ds = _load_one_dataset(
            ds_cfg,
            preprocess_fn,
            filter_fn,
            tokenizer,
            max_length,
            force_no_shuffle=no_shuffle,
        )
        datasets.append(ds)
    if dataset_config.probabilities:
        ds = interleave_datasets(
            datasets,
            probabilities=dataset_config.probabilities,
            seed=SEED,
        )
    else:
        ds = concatenate_datasets(datasets)
    if dataset_config.shuffle:
        ds = ds.shuffle(seed=SEED)
    if dataset_config.ratio is not None:
        assert 0.0 < dataset_config.ratio and dataset_config.ratio <= 1.0
        total_size = len(ds)
        new_size = int(total_size * dataset_config.ratio)
        ds = ds.select(range(new_size))
    return ds


def _get_training_args(
    args: TrainingConfig, model_save_dir: str, use_wandb: bool
) -> TrainingArguments:
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
        betas = optim.betas
        eps = optim.eps
        weight_decay = optim.weight_decay
    else:
        betas = optim.betas
        eps = None
        weight_decay = optim.weight_decay

    return TrainingArguments(
        output_dir=model_save_dir,
        overwrite_output_dir=True,
        # Training args
        do_train=True,
        fp16=True,
        bf16=os.environ.get("USE_BF16", "0").lower() in ("1", "true"),
        per_device_train_batch_size=(
            args.batch_size if args.batch_size != "auto" else 8
        ),
        auto_find_batch_size=args.batch_size == "auto",
        gradient_accumulation_steps=args.accumulation_steps,
        max_grad_norm=args.grad_clip,
        warmup_steps=args.warmup_steps or 0,
        num_train_epochs=args.epochs,
        save_strategy="no" if not args.save_steps else "steps",
        save_steps=args.save_steps if args.save_steps else 1000,
        lr_scheduler_type="cosine",
        torch_compile=True,
        torch_compile_mode="default",
        half_precision_backend="cpu_amp",
        optim="adamw_torch_fused" if optim.name == "adamw" else "lion_32bit",
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


def train_pretrain(config: Config, use_wandb: bool, dry_run: bool, project: str | None):
    torch.manual_seed(SEED)
    model = utils.load_model(config.model)
    tokenizer = config.model.load_tokenizer()
    runid, save_dir = _create_runid_and_path(
        config, "pretrain", tokenizer, dry_run, project
    )
    # prepare dataset
    args = config.train["pretrain"]
    assert args is not None
    dataset = _load_dataset(
        dataset_config=args.dataset,
        preprocess_fn=pretrain.preprocess,
        filter_fn=None,
        tokenizer=tokenizer,
        max_length=args.max_length,
        is_pretrain=True,
    )
    # dataset = dataset.take(100)
    # count tokens
    tokens = _count_tokens(dataset)
    billion_tokens = tokens / 1_000_000_000
    print(f"Total tokens in dataset: {tokens} ({billion_tokens:.2f}B)")
    training_args = _get_training_args(args, save_dir, use_wandb)
    training_args.label_names = ["loss_mask"]
    trainer = pretrain.PretrainTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=[ManualSaveCallback()],
    )
    trainer.train()
    if not dry_run:
        trainer.save_model(save_dir)


def train_sft(
    config: Config,
    ckpt: Path,
    use_wandb: bool,
    dry_run: bool,
    project: str | None,
    key="sft",
):
    torch.manual_seed(SEED)
    model = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    print(f"Loaded checkpoint from {ckpt}")
    runid, save_dir = _create_runid_and_path(config, key, tokenizer, dry_run, project)
    # prepare dataset
    args = config.train[key]
    assert args is not None
    dataset = _load_dataset(
        dataset_config=args.dataset,
        preprocess_fn=sft.preprocess,
        filter_fn=sft.filter,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    # dataset = dataset.take(100)
    # count tokens
    tokens = _count_tokens(dataset)
    billion_tokens = tokens / 1_000_000_000
    print(f"Total tokens in dataset: {tokens} ({billion_tokens:.2f}B)")
    training_args = _get_training_args(args, save_dir, use_wandb)
    training_args.label_names = ["loss_mask"]
    trainer = sft.SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=[ManualSaveCallback()],
    )
    if args.think_tokens:
        tokens = []
        for t in args.think_tokens:
            tokens.extend(tokenizer(t).input_ids)
        trainer.think_tokens = tokens
    trainer.train()
    trainer.save_model(save_dir)


def train_dpo(
    config: Config, ckpt: Path, use_wandb: bool, dry_run: bool, project: str | None
):
    torch.manual_seed(SEED)
    model = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    print(f"Loaded checkpoint from {ckpt}")
    ref_model = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True)
    runid, save_dir = _create_runid_and_path(config, "dpo", tokenizer, dry_run, project)
    # prepare dataset
    args = config.train["dpo"]
    assert args is not None
    dataset = _load_dataset(
        dataset_config=args.dataset,
        preprocess_fn=None,
        filter_fn=None,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    # dataset = dataset.take(100)
    training_args = _get_training_args(args, save_dir, use_wandb)
    dpo_args = DPOConfig(
        **training_args.to_dict(),
        max_length=args.max_length,
        dataset_num_proc=os.cpu_count(),
    )
    trainer = dpo.DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[ManualSaveCallback()],
    )
    trainer.train()
    trainer.save_model(save_dir)


class ManualSaveCallback(TrainerCallback):
    def __init__(self) -> None:
        super().__init__()
        self.file = Path(__file__).parent / "SAVE"

    @override
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not self.file.exists() or not self.file.is_file():
            return
        print("Manual save triggered.")
        control.should_save = True
        self.file.unlink()
