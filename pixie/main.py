import os
from pathlib import Path
from typing import Annotated, Literal
import typer
from .config import Config
from .models import BaseGPTModel
from .run import run_model
from pixie.train.train import Trainer
import pixie.train.dataset as dataset
import warnings
import dotenv

warnings.filterwarnings("ignore")

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
    model = BaseGPTModel.load(config.model)
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
    proj: Annotated[str | None, typer.Option("--project", "-p")] = None,
    wandb: bool = False,
):
    dotenv.load_dotenv()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    config = Config.load(f"configs/{config_name}.yaml")
    trainer = Trainer(
        config, project=proj, checkpoint=None, use_wandb=wandb, train_type="pretrain"
    )
    trainer.train()


@app.command()
def sft(
    ckpt: Annotated[Path, typer.Option("--checkpoint", "--ckpt")],
    proj: Annotated[str | None, typer.Option("--project", "-p")] = None,
    wandb: bool = False,
):
    dotenv.load_dotenv()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    assert ckpt.is_dir()
    config = Config.load(ckpt / "config.yaml")
    assert config.sft is not None, "SFT configuration is missing."
    trainer = Trainer(
        config, project=proj, checkpoint=ckpt, use_wandb=wandb, train_type="sft"
    )
    trainer.train()


@app.command()
def dpo(
    ckpt: Annotated[Path, typer.Option("--checkpoint", "--ckpt")],
    proj: Annotated[str | None, typer.Option("--project", "-p")] = None,
    wandb: bool = False,
):
    dotenv.load_dotenv()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    assert ckpt.is_dir()
    config = Config.load(ckpt / "config.yaml")
    assert config.dpo is not None, "DPO configuration is missing."
    trainer = Trainer(
        config, project=proj, checkpoint=ckpt, use_wandb=wandb, train_type="dpo"
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
    path: Path,
    type: Annotated[Literal["pretrain", "sft", "dpo"], typer.Option("--type", "-t")],
    tokenizer: Annotated[str, typer.Option("--tokenizer", "--tok", "-k")],
    max_length: Annotated[int, typer.Option("--max-length", "--len", "-l")],
):
    cfg = dataset.DatasetLoaderConfig(
        tokenizer=tokenizer,
        max_length=max_length,
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
    model_path: Path,
    chat: Annotated[str | None, typer.Option("--chat")] = None,
    gen: Annotated[str | None, typer.Option("--gen")] = None,
    repl: bool = False,
):
    if (
        model_path.is_dir()
        and not (model_path / "model.safetensors").exists()
        and (model_path / "latest" / "model.safetensors").exists()
    ):
        model_path = model_path / "latest"
    if model_path.is_dir():
        assert (
            model_path / "model.safetensors"
        ).exists(), f"Model not found in {model_path}"
    else:
        assert model_path.suffix in [
            ".safetensors",
            ".pth",
            ".pt",
        ], "Invalid model file."
    run_type: Literal["chat", "gen", "repl"]
    if repl:
        assert chat is None, "--chat should not be provided in REPL mode."
        assert gen is None, "--gen should not be provided in REPL mode."
        prompt = None
        run_type = "repl"
    else:
        assert chat or gen, "Either --chat or --gen must be provided."
        assert not (chat and gen), "Only one of --chat or --gen can be provided."
        prompt = chat if chat else gen
        run_type = "chat" if chat else "gen"

    run_model(model_path, prompt, run_type)


def main():
    app()
