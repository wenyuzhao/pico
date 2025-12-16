import os
from pathlib import Path
from typing import Annotated, Literal
import typer
from .run import run_model
from pico.train.train import train_dpo, train_pretrain, train_sft
import warnings
import dotenv
from . import utils

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
    config = utils.load_config(f"configs/{config_name}.yaml")
    model = utils.load_model(config.model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{config_name}]")
    print(f"Model: {config.model.name}")
    print(f"Parameters: {num_params / 1e6:.2f}M")
    print(f"Layers: {config.model.num_hidden_layers}")
    print(f"Hidden Size: {config.model.hidden_size}")
    print(f"Attention Heads: {config.model.num_attention_heads}")
    print(f"Feed Forward Size: {config.model.feed_forward_size}")
    print(
        f"Tokenizer: {config.model.tokenizer} (vocab size: {model.args.get_vocab_size()})"
    )
    if verbose:
        print("\nFull Configuration:")
        print(config.model_dump())


train_app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    context_settings=dict(help_option_names=["-h", "--help"]),
    pretty_exceptions_short=True,
    pretty_exceptions_show_locals=False,
)


@train_app.command()
def pretrain(
    config_name: str,
    wandb: bool = False,
    dry_run: bool = False,
    project: Annotated[str | None, typer.Option("--project", "-p")] = None,
):
    dotenv.load_dotenv()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    config = utils.load_config(f"configs/{config_name}.yaml")
    assert "pretrain" in config.train, "Pretrain configuration is missing."
    train_pretrain(config, use_wandb=wandb, dry_run=dry_run, project=project)


@train_app.command()
def sft(
    ckpt: Annotated[Path, typer.Option("--checkpoint", "--ckpt", "-c")],
    wandb: bool = False,
    dry_run: bool = False,
    config: str | None = None,
    project: Annotated[str | None, typer.Option("--project", "-p")] = None,
):
    dotenv.load_dotenv()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    assert ckpt.is_dir()
    cfg = utils.load_config(config or (ckpt / "config.yaml"))
    assert "sft" in cfg.train, "SFT configuration is missing."
    train_sft(cfg, ckpt, use_wandb=wandb, dry_run=dry_run, key="sft", project=project)


@train_app.command()
def dpo(
    ckpt: Annotated[Path, typer.Option("--checkpoint", "--ckpt", "-c")],
    wandb: bool = False,
    dry_run: bool = False,
    config: str | None = None,
    project: Annotated[str | None, typer.Option("--project", "-p")] = None,
):
    dotenv.load_dotenv()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    assert ckpt.is_dir()
    cfg = utils.load_config(config or (ckpt / "config.yaml"))
    assert "dpo" in cfg.train, "DPO configuration is missing."
    train_dpo(cfg, ckpt, use_wandb=wandb, dry_run=dry_run, project=project)


@train_app.command()
def reason(
    ckpt: Annotated[Path, typer.Option("--checkpoint", "--ckpt", "-c")],
    wandb: bool = False,
    dry_run: bool = False,
    config: str | None = None,
    project: Annotated[str | None, typer.Option("--project", "-p")] = None,
):
    dotenv.load_dotenv()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    assert ckpt.is_dir()
    cfg = utils.load_config(config or (ckpt / "config.yaml"))
    assert "reason" in cfg.train, "Reason configuration is missing."
    train_sft(
        cfg, ckpt, use_wandb=wandb, dry_run=dry_run, key="reason", project=project
    )


app.add_typer(train_app, name="train", help="Training related commands.")


@app.command(name="export-onnx")
def export_onnx(
    ckpt: Annotated[Path, typer.Argument(...)],
    task: Annotated[str, typer.Option("--task")] = "text-generation",
):
    from pico.onnx import export_onnx

    export_onnx(ckpt, task)


dataset_app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    context_settings=dict(help_option_names=["-h", "--help"]),
    pretty_exceptions_short=True,
    pretty_exceptions_show_locals=False,
)
app.add_typer(dataset_app, name="dataset", help="Dataset related commands.")


@app.command(name="run")
def run(
    model_path: Path,
    chat: Annotated[str | None, typer.Option("--chat")] = None,
    gen: Annotated[str | None, typer.Option("--gen")] = None,
    repl: bool = False,
):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
