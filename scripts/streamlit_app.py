import streamlit as st
from typing import Literal, TypedDict, cast
from threading import Thread
from queue import Queue
from transformers.generation.streamers import TextStreamer
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.pipelines import pipeline
from transformers.pipelines.text_generation import TextGenerationPipeline

REPO_ID = "wenyuzhao/pico-100m"
SYSTEM_MESSAGE = None

st.html(
    '<p><span style="font-size:1.7em; font-weight: 700"><a href="https://huggingface.co/wenyuzhao/pico-100m">Pico-100M</a> Chat</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<i><a href="https://github.com/wenyuzhao/pico">GitHub</a> | <a href="https://huggingface.co/wenyuzhao/pico-100m">HuggingFace</a></i><p>'
)


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


def chat_complete_stream(
    tok: AutoTokenizer, pl: TextGenerationPipeline, messages: list[Message]
):
    queue = Queue()
    streamer = CustomStreamer(tok, queue)

    def _generate():
        r = pl(messages, streamer=streamer)  # type: ignore

    t = Thread(target=_generate)
    t.start()

    while True:
        text: str = queue.get()
        if text is None:
            break
        yield cast(str, text)

    t.join()


with st.spinner("Loading model ...", show_time=True):
    if "tokenizer" not in st.session_state:
        st.session_state["tokenizer"] = AutoTokenizer.from_pretrained(REPO_ID)
    tok: AutoTokenizer = st.session_state["tokenizer"]
    if "pipeline" not in st.session_state:
        st.session_state["pipeline"] = pipeline(
            task="text-generation", model=REPO_ID, trust_remote_code=True
        )
    pl: TextGenerationPipeline = st.session_state["pipeline"]

if "messages" not in st.session_state:
    st.session_state["messages"] = []
    if SYSTEM_MESSAGE is not None:
        st.session_state["messages"].append(
            {"role": "system", "content": SYSTEM_MESSAGE}
        )
messages: list[Message] = st.session_state["messages"]

for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Your message ..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        response = st.write_stream(chat_complete_stream(tok, pl, messages))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
