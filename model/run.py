import os
from pathlib import Path
from typing import Literal, TypedDict
import warnings
from threading import Thread
from queue import Queue
from transformers.generation.streamers import TextStreamer
from transformers import AutoTokenizer
from transformers.pipelines import pipeline

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class CustomStreamer(TextStreamer):
    def __init__(self, tokenizer: AutoTokenizer, queue: Queue):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.queue = queue
        self.tokenizer = tokenizer

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.queue.put(text)
        if stream_end:
            self.queue.put(None)


def complete_streamed(model: Path, messages: str | list[Message]):
    tokenizer = AutoTokenizer.from_pretrained(model)
    queue = Queue()
    streamer = CustomStreamer(tokenizer, queue)
    pl = pipeline(task="text-generation", model=str(model), streamer=streamer)

    def _generate():
        r = pl(messages)  # type: ignore

    Thread(target=_generate).start()

    while True:
        text = queue.get()
        if text is None:
            break
        yield text


def run_model(
    path: Path,
    prompt: str | None,
    type: Literal["chat", "gen", "repl"],
):

    if type != "repl":
        assert prompt is not None, "Prompt must be provided if not in REPL mode."
        messages: str | list[Message]
        if type == "chat":
            messages = [{"role": "user", "content": prompt}]
        else:
            assert isinstance(prompt, str)
            messages = prompt
        for t in complete_streamed(path, messages):
            print(t, end="", flush=True)
        print()
    else:
        print("Running in REPL mode. Type 'exit' to quit.")
        # args.prompt = []
        # while True:
        #     user_input = input("> ").strip()
        #     if user_input.lower() == "exit":
        #         break
        #     args.prompt.append({"role": "user", "content": user_input})
        #     response = ""
        #     for t in complete_streamed(tokenizer, model, args):
        #         print(t, end="", flush=True)
        #         response += t
        #     print()
        #     args.prompt.append({"role": "assistant", "content": response})
