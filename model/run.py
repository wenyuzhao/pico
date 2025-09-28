import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import torch
import warnings
from threading import Thread
from queue import Queue
from transformers.generation.streamers import TextStreamer
from model.config import Config
from model.models import BaseGPTModel
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.generation.configuration_utils import GenerationConfig
from typing import Literal

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class Args:
    prompt: str | list[dict[str, str]] | None
    checkpoint: str
    type: Literal["base", "chat"] = "chat"
    temperature: float = 0.7
    top_p: float = 0.92
    max_tokens: int = 8192
    device: str = "cuda"
    repl: bool = False


def init_model(config: Config, args: Args):
    checkpoint_file = Path(args.checkpoint)
    assert checkpoint_file.exists()
    assert checkpoint_file.suffix == ".pth"
    tokenizer = config.load_tokenizer()
    model = config.load_model()
    state_dict = {}
    for k, v in torch.load(args.checkpoint, map_location=args.device).items():
        k = k.replace("._orig_mod", "").replace("_orig_mod.", "")
        state_dict[k] = v
    model.load_state_dict(state_dict, strict=True)
    return model.eval().to(args.device), tokenizer  # type: ignore


class CustomStreamer(TextStreamer):
    def __init__(self, tokenizer, queue):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.queue = queue

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


def complete_streamed(
    tokenizer: PreTrainedTokenizerFast, model: BaseGPTModel, args: Args
):
    inputs = get_inputs(tokenizer, args)
    queue = Queue()
    streamer = CustomStreamer(tokenizer, queue)

    def _generate():
        assert isinstance(model, BaseGPTModel), "Model should be an instance of Pixie."
        model.generate(
            inputs.input_ids,
            generation_config=GenerationConfig(
                max_new_tokens=args.max_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=1.15,
                attention_mask=inputs.attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False,
            ),
            max_new_tokens=args.max_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=1.15,
            attention_mask=inputs.attention_mask,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,
            streamer=streamer,
        )

    Thread(target=_generate).start()

    while True:
        text = queue.get()
        if text is None:
            break
        yield text


def complete(tokenizer: PreTrainedTokenizerFast, model: BaseGPTModel, args: Args):
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


def run_model(
    config: Config,
    checkpoint: str,
    prompt: str | None,
    repl: bool,
    type: Literal["pretrain", "sft"],
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t = "chat" if type == "sft" else "base"
    args = Args(
        prompt=prompt,
        checkpoint=checkpoint,
        type=t,
        temperature=0.7,
        top_p=0.92,
        max_tokens=8192,
        device=device,
        repl=repl,
    )
    model, tokenizer = init_model(config, args)
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
