import numpy as np
import pandas as pd
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from typing import Generator, TypedDict, cast
from .dataset import BATCH_SIZE, DatasetLoaderConfig


class Message(TypedDict):
    role: str
    content: str


def create_chat_prompt(
    samples: list[list[Message]], tokenizer: PreTrainedTokenizerFast
) -> list[str]:
    # Tokenize the conversations
    prompts = tokenizer.apply_chat_template(
        cast(list[list[dict[str, str]]], samples), tokenize=False
    )
    assert isinstance(prompts, list), "Prompts should be a list."
    return prompts  # type: ignore


def get_conversations(
    df: pd.DataFrame,
) -> tuple[list[list[Message]], list[list[Message]]]:
    assert "chosen" in df.columns, "DataFrame must contain 'chosen' column."
    assert "rejected" in df.columns, "DataFrame must contain 'rejected' column."

    chosen = [x.tolist() for x in df["chosen"].to_list()]
    rejected = [x.tolist() for x in df["rejected"].to_list()]

    return cast(list[list[Message]], chosen), cast(list[list[Message]], rejected)  # type: ignore


def process_batch(
    samples: list[str], cfg: DatasetLoaderConfig, tok: PreTrainedTokenizerFast
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    encoding = tok(
        samples,
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
    assert len(encoding.input_ids) == len(samples)
    for input_ids, attention_mask in zip(encoding.input_ids, encoding.attention_mask):
        input_ids = input_ids.squeeze().numpy()
        attention_mask = attention_mask.squeeze().numpy()
        # truncate
        input_ids = input_ids[: cfg.max_length]
        attention_mask = attention_mask[: cfg.max_length]
        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
    return all_input_ids, all_attention_masks


def process_dpo(df: pd.DataFrame, cfg: DatasetLoaderConfig) -> Generator[pd.DataFrame]:
    chosen, rejected = get_conversations(df)
    assert len(chosen) == len(rejected)
    tok = cfg._tokenizer
    chosen = create_chat_prompt(chosen, tok)
    rejected = create_chat_prompt(rejected, tok)
    assert len(chosen) == len(rejected)
    count = len(chosen)
    # Do it batched
    for i in range(0, count, BATCH_SIZE):
        print(f"       . {i} / {count}", flush=True)
        max_index = min(i + BATCH_SIZE, count)
        chosen_input_ids, chosen_attention_masks = process_batch(
            chosen[i:max_index], cfg, tok
        )
        rejected_input_ids, rejected_attention_masks = process_batch(
            rejected[i:max_index], cfg, tok
        )
        df = pd.DataFrame(
            {
                "chosen": chosen_input_ids,
                "chosen_mask": chosen_attention_masks,
                "rejected": rejected_input_ids,
                "rejected_mask": rejected_attention_masks,
            }
        )
        yield df
