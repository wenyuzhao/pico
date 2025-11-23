from typing import Any
from pixie.models._base import Config
import torch
from typing import cast, TypedDict
from transformers import PreTrainedTokenizerFast
from .pretrain import PretrainTrainer


class Message(TypedDict):
    role: str
    content: str


def _get_conversations(data: dict[str, Any]) -> list[list[dict[str, str]]]:
    conversations: list[list[dict[str, str]]]

    if "conversations" in data:
        conversations = data["conversations"]  # type: ignore
    elif "messages" in data:
        conversations = data["messages"]  # type: ignore
    elif "instruction" in data and "input" in data and "output" in data:
        conversations = []
        for instruction, input, output in zip(
            data["instruction"], data["input"], data["output"]
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
    data: dict[str, Any], tokenizer: PreTrainedTokenizerFast, max_length: int
) -> dict[str, torch.Tensor]:
    samples = _get_conversations(data)
    tokens = tokenizer.apply_chat_template(
        cast(list[list[dict[str, str]]], samples),
        tokenize=True,
        # add_generation_prompt=True,
        return_assistant_tokens_mask=True,
        return_dict=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    tokens = cast(dict[str, torch.Tensor], tokens)
    return {"input_ids": tokens["input_ids"], "loss_mask": tokens["assistant_masks"]}


class SFTTrainer(PretrainTrainer): ...
