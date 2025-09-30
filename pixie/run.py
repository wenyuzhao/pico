import os
from pathlib import Path
from typing import Literal, TypedDict
import warnings
from threading import Thread
from queue import Queue
from transformers.generation.streamers import TextStreamer
from transformers import AutoTokenizer
from transformers.pipelines import pipeline
from transformers.pipelines.text_generation import TextGenerationPipeline

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class CustomStreamer(TextStreamer):
    def __init__(self, tokenizer: AutoTokenizer, queue: Queue):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.queue = queue

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.queue.put(text)
        if stream_end:
            self.queue.put(None)


def complete_streamed(
    tok: AutoTokenizer, pl: TextGenerationPipeline, messages: str | list[Message]
):
    queue = Queue()
    streamer = CustomStreamer(tok, queue)

    def _generate():
        r = pl(messages, streamer=streamer)  # type: ignore

    t = Thread(target=_generate)
    t.start()

    while True:
        text = queue.get()
        if text is None:
            break
        yield text

    t.join()


def run_model(
    path: Path,
    prompt: str | None,
    type: Literal["chat", "gen", "repl"],
):
    tok = AutoTokenizer.from_pretrained(path)
    pl = pipeline(task="text-generation", model=str(path))

    if type != "repl":
        assert prompt is not None, "Prompt must be provided if not in REPL mode."
        messages: str | list[Message]
        if type == "chat":
            messages = [{"role": "user", "content": prompt}]
        else:
            assert isinstance(prompt, str)
            messages = prompt
        for t in complete_streamed(tok, pl, messages):
            print(t, end="", flush=True)
        print()
    else:
        print("Running in REPL mode. Type 'exit' to quit.")
        messages = []
        while True:
            user_input = input("> ").strip()
            if user_input.lower() == "exit":
                break
            if not user_input:
                continue
            messages.append({"role": "user", "content": user_input})
            response = ""
            for t in complete_streamed(tok, pl, messages):
                print(t, end="", flush=True)
                response += t
            print()
            messages.append({"role": "assistant", "content": response})
