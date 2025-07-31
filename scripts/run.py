import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import torch
import warnings
from threading import Thread
from queue import Queue
from transformers.generation.streamers import TextStreamer
from model.pixie import Config, Pixie
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.generation.configuration_utils import GenerationConfig
from typing import Literal

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class Args:
    prompt: str | list[dict[str, str]]
    checkpoint: str
    type: Literal["base", "chat"] = "chat"
    temperature: float = 0.7
    top_p: float = 0.92
    max_tokens: int = 8192
    device: str = "cuda"
    repl: bool = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str, default=None, nargs="?")
    parser.add_argument("--checkpoint", "--ckpt", type=str, required=True)
    parser.add_argument(
        "--type",
        "-t",
        type=str,
        choices=["base", "chat"],
        default=None,
        help="Type of model to use: 'base' for pretraining, 'chat' for chat model.",
    )
    parser.add_argument("--repl", action="store_true", help="Run in REPL mode.")
    args = parser.parse_args()

    if not args.type:
        stem = Path(args.checkpoint).stem
        if "pretrain" in stem or "base" in stem:
            args.type = "base"
            assert not args.repl, "REPL mode is not supported for base model."
            assert args.prompt is not None, "Prompt is required for base model."
        else:
            args.type = "chat"
            if not args.repl:
                assert (
                    args.prompt is not None
                ), "Prompt is required for chat model unless in REPL mode."
    return Args(**args.__dict__)


def init_model(args: Args):
    checkpoint_file = Path(args.checkpoint)
    assert checkpoint_file.exists()
    assert checkpoint_file.suffix == ".pth"
    config = Config.from_pretrained(checkpoint_file.parent)
    tokenizer = Pixie.tokenizer(config)
    model = Pixie(config, compile=False)
    state_dict = {}
    for k, v in torch.load(args.checkpoint, map_location=args.device).items():
        k = k.replace("._orig_mod", "")
        state_dict[k] = v
    model.load_state_dict(state_dict, strict=True)
    print(
        f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} M"
    )
    return model.eval().to(args.device), tokenizer  # type: ignore


class CustomStreamer(TextStreamer):
    def __init__(self, tokenizer, queue):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.queue = queue
        self.tokenizer = tokenizer

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.queue.put(text)
        if stream_end:
            self.queue.put(None)


def get_inputs(tokenizer: PreTrainedTokenizerFast, args: Args):
    if args.type == "chat":
        messages = (
            args.prompt
            if isinstance(args.prompt, list)
            else [{"role": "user", "content": args.prompt}]
        )
        new_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        assert isinstance(new_prompt, str), "Prompt should be a string."
        inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(
            args.device
        )
    else:
        new_prompt = args.prompt
        assert isinstance(new_prompt, str), "Prompt should be a string."
        if tokenizer.bos_token is not None:
            assert isinstance(tokenizer.bos_token, str), "BOS token should be a string."
            new_prompt = tokenizer.bos_token + new_prompt
        inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(
            args.device
        )
    return inputs


def complete_streamed(tokenizer: PreTrainedTokenizerFast, model: Pixie, args: Args):
    inputs = get_inputs(tokenizer, args)
    queue = Queue()
    streamer = CustomStreamer(tokenizer, queue)

    def _generate():
        assert isinstance(model, Pixie), "Model should be an instance of Pixie."
        model.generate(
            inputs.input_ids,
            max_new_tokens=args.max_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=1.15,
            attention_mask=inputs.attention_mask,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )

    Thread(target=_generate).start()

    while True:
        text = queue.get()
        if text is None:
            break
        yield text


def complete(tokenizer: PreTrainedTokenizerFast, model: Pixie, args: Args):
    inputs = get_inputs(tokenizer, args)
    with torch.no_grad():
        generated_ids = model.generate(
            inputs["input_ids"],  # type: ignore
            max_length=inputs["input_ids"].shape[1] + args.max_tokens,  # type: ignore
            do_sample=True,
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p,
            temperature=args.temperature,
        )
        answer = tokenizer.decode(
            generated_ids[0][inputs["input_ids"].shape[1] :],  # type: ignore
            skip_special_tokens=True,
        )
        print(f"Answer: {answer}")
    return answer


def main():
    args = parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = init_model(args)
    if not args.repl:
        for t in complete_streamed(tokenizer, model, args):
            print(t, end="", flush=True)
        print()
    else:
        print("Running in REPL mode. Type 'exit' to quit.")
        args.prompt = []
        while True:
            user_input = input("> ").strip()
            if user_input.lower() == "exit":
                break
            args.prompt.append({"role": "user", "content": user_input})
            response = ""
            for t in complete_streamed(tokenizer, model, args):
                print(t, end="", flush=True)
                response += t
            print()
            args.prompt.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available. Please check your setup."
    main()
