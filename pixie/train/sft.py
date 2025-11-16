import torch
import pandas as pd
from typing import TypedDict, cast
from . import CHAT_TEMPLATES
from .pretrain import PretrainLoss
import torch
from typing import cast
from transformers import PreTrainedTokenizerFast
from typing import Any
from pixie.models._config import Config


class Message(TypedDict):
    role: str
    content: str


def _get_conversations(df: pd.DataFrame) -> list[list[Message]]:
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


def preprocess(
    data: dict[str, Any], config: Config, tokenizer: PreTrainedTokenizerFast
) -> dict[str, Any]:
    assert config.sft
    max_length = config.sft.context_length
    samples = _get_conversations(pd.DataFrame(data))
    tokens = tokenizer.apply_chat_template(
        cast(list[list[dict[str, str]]], samples),
        tokenize=True,
        # add_generation_prompt=True,
        return_assistant_tokens_mask=True,
        return_dict=True,
        chat_template=CHAT_TEMPLATES.get(tokenizer.name_or_path, ""),
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    tokens = cast(dict[str, torch.Tensor], tokens)
    xs = [torch.tensor(x[:-1], dtype=torch.long) for x in tokens["input_ids"].tolist()]
    ys = [torch.tensor(y[1:], dtype=torch.long) for y in tokens["input_ids"].tolist()]
    masks = [
        torch.tensor(m[1:], dtype=torch.long)
        for m in tokens["assistant_masks"].tolist()
    ]
    return {"x": xs, "y": ys, "loss_mask": masks}


class SFTLoss(PretrainLoss): ...
