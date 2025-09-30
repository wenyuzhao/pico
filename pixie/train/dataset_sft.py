import numpy as np
import pandas as pd
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from typing import Generator, TypedDict
from .dataset import BATCH_SIZE, DatasetLoaderConfig


class Message(TypedDict):
    role: str
    content: str


def create_chat_prompt(
    conversations: list[list[Message]], tokenizer: PreTrainedTokenizerFast
) -> list[str]:
    records: list[np.ndarray[dict[str, str]]] = (  # type: ignore
        [conversations] if isinstance(conversations[0], dict) else conversations
    )
    records: list[list[dict[str, str]]] = [r.tolist() if not isinstance(r, list) else r for r in records]  # type: ignore
    # Tokenize the conversations
    prompts = tokenizer.apply_chat_template(records, tokenize=False)
    assert isinstance(prompts, list), "Prompts should be a list."
    return prompts  # type: ignore


def get_conversations(df: pd.DataFrame) -> list[list[Message]]:
    conversations: list[list[Message]]

    if "conversations" in df.columns:
        conversations = df["conversations"].to_list()  # type: ignore
    elif "messages" in df.columns:
        conversations = df["messages"].to_list()  # type: ignore
    elif (
        "instruction" in df.columns and "input" in df.columns and "output" in df.columns
    ):
        conversations = []
        for instruction, input, output in zip(
            df["instruction"], df["input"], df["output"]
        ):
            if input and len(input) > 0:
                conversations.append(
                    [
                        {"role": "user", "content": instruction + "\n\n" + input},
                        {"role": "assistant", "content": output},
                    ]
                )
            else:
                conversations.append(
                    [
                        {"role": "user", "content": instruction},
                        {"role": "assistant", "content": output},
                    ]
                )
    else:
        raise ValueError("Invalid format")

    assert isinstance(conversations, list)
    assert isinstance(conversations[0], list)
    assert isinstance(conversations[0][0], dict)

    # Fix field names
    alt_role_keys = ["from"]
    role_mapping: dict[str, str] = {"human": "user", "gpt": "assistant"}
    alt_content_keys = ["value"]
    for msgs in conversations:
        for x in msgs:
            for key in alt_role_keys:
                if key in x and "role" not in x:
                    x["role"] = x[key]
                    del x[key]
            x["role"] = role_mapping.get(x["role"], x["role"])
            for key in alt_content_keys:
                if key in x and "content" not in x:
                    x["content"] = x[key]
                    del x[key]

    return conversations


def process_sft(df: pd.DataFrame, cfg: DatasetLoaderConfig) -> Generator[pd.DataFrame]:
    conversations: list[list[Message]] = get_conversations(df)

    tok = cfg._tokenizer
    samples = create_chat_prompt(conversations, tok)
    # Do it batched
    for i in range(0, len(samples), BATCH_SIZE):
        print(f"       . {i} / {len(samples)}", flush=True)
        max_index = min(i + BATCH_SIZE, len(samples))
        slice = samples[i:max_index]
        encoding = tok(
            slice,
            max_length=cfg.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=cfg.add_special_tokens,
        )
        all_input_ids = []
        all_attention_masks = []
        pad = tok.pad_token_id
        assert pad is not None, "Tokenizer must have a pad token."
        end = tok.eos_token_id
        assert end is not None, "Tokenizer must have an eos token."
        for input_ids, attention_mask in zip(
            encoding.input_ids, encoding.attention_mask
        ):
            input_ids = input_ids.squeeze().numpy()
            last_token = input_ids[-1]
            if last_token != end and last_token != pad:
                continue
            attention_mask = attention_mask.squeeze().numpy()
            # truncate
            input_ids = input_ids[: cfg.max_length]
            attention_mask = attention_mask[: cfg.max_length]
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
        df = pd.DataFrame(
            {"input_ids": all_input_ids, "attention_mask": all_attention_masks}
        )
        yield df
