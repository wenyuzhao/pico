from pathlib import Path
from typing import Annotated, Literal
import typer
from model.config import BaseTrainingConfig, Config
from model.models import BaseGPTModel
from model.run import run_model
from model.train.train import Trainer
import model.train.dataset as dataset


app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    context_settings=dict(help_option_names=["-h", "--help"]),
    pretty_exceptions_short=True,
    pretty_exceptions_show_locals=False,
)


@app.command(help="Prints information about the model.")
def info(config_name: str, verbose: bool = False):
    config = Config.load(f"configs/{config_name}.yaml")
    model = BaseGPTModel.load(config)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{config_name}]")
    print(f"Model: {config.model.name}")
    print(f"Parameters: {num_params / 1e6:.2f}M")
    print(f"Layers: {config.model.num_hidden_layers}")
    print(f"Hidden Size: {config.model.hidden_size}")
    print(f"Attention Heads: {config.model.num_attention_heads}")
    print(f"Feed Forward Size: {config.model.feed_forward_size}")
    print(
        f"Tokenizer: {config.model.tokenizer} (vocab size: {model.config.vocab_size})"
    )
    if verbose:
        print("\nFull Configuration:")
        print(config.model_dump())


@app.command()
def pretrain(
    config_name: str,
    project: Annotated[str | None, typer.Option("--project", "-p")] = None,
    checkpoint: Annotated[str | None, typer.Option("--checkpoint", "--ckpt")] = None,
    wandb: bool = False,
):
    config = Config.load(f"configs/{config_name}.yaml")
    trainer = Trainer(
        config,
        project=project,
        checkpoint=checkpoint,
        use_wandb=wandb,
        train_type="pretrain",
    )
    trainer.train()


dataset_app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    context_settings=dict(help_option_names=["-h", "--help"]),
    pretty_exceptions_short=True,
    pretty_exceptions_show_locals=False,
)
app.add_typer(dataset_app, name="dataset", help="Dataset related commands.")


@dataset_app.command(name="preprocess")
def dataset_preprocess(
    config_name: str,
    type: Annotated[Literal["pretrain", "sft"], typer.Option("--type", "-t")],
):
    config = Config.load(f"configs/{config_name}.yaml")
    train_cfg: BaseTrainingConfig
    match type:
        case "pretrain":
            assert config.pretrain is not None
            train_cfg = config.pretrain
        case "sft":
            assert config.sft is not None
            train_cfg = config.sft
    path = (
        train_cfg.dataset
        if isinstance(train_cfg.dataset, str)
        else train_cfg.dataset.path
    )
    cfg = dataset.DatasetLoaderConfig(
        tokenizer=config.model.tokenizer,
        max_length=train_cfg.context_length,
        add_special_tokens=True,
        type=type,
    )
    dataset.preprocess(path, cfg, type)


@dataset_app.command(name="remove")
def dataset_remove(
    db: str,
    files: Annotated[list[str], typer.Option("--files", "-f")],
):
    files2 = [Path(f) for f in files]
    dataset.force_delete_from_db(Path(db), files2)


@dataset_app.command(name="shuffle")
def dataset_shuffle(db: str):
    dataset.shuffle_dataset(Path(db))


@dataset_app.command(name="info")
def dataset_info(db: str, limit: int | None = None):
    dataset.show_dataset_stats(Path(db), limit)


@app.command(name="run")
def run(
    config_name: str,
    type: Annotated[Literal["pretrain", "sft"], typer.Option("--type", "-t")],
    prompt: Annotated[str | None, typer.Option("--prompt", "-p")] = None,
    checkpoint: Annotated[str | None, typer.Option("--checkpoint", "--ckpt")] = None,
    repl: bool = False,
):
    if not checkpoint:
        checkpoint = f"out/{config_name}/{type}/latest/model.pth"
    if repl:
        assert type == "sft", "REPL mode is only supported for 'sft' type."
        assert prompt is None, "Prompt should not be provided in REPL mode."
    else:
        assert prompt is not None, "Prompt must be provided if not in REPL mode."
    config = Config.load(f"configs/{config_name}.yaml")
    print(f"[RUN] Loading model from {checkpoint} ...")
    run_model(config, checkpoint, prompt, repl, type)


def main():
    app()
