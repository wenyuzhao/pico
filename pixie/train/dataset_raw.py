import numpy as np
import pandas as pd
from tokenizers import processors
from typing import Generator
from .dataset import BATCH_SIZE, DatasetLoaderConfig


def process_raw(df: pd.DataFrame, cfg: DatasetLoaderConfig) -> Generator[pd.DataFrame]:
    possible_text_columns = ["text", "content"]
    col_name: str | None = None
    for col in possible_text_columns:
        if col in df.columns:
            col_name = col
            break
    assert col_name is not None, "No text column found in the DataFrame."
    samples = df[col_name]
    tok = cfg._tokenizer
    if cfg.add_special_tokens:
        (bos, eos) = (tok.bos_token, tok.eos_token)
        (bos_id, eos_id) = (tok.bos_token_id, tok.eos_token_id)
        if bos and eos:
            single = f"{bos} $A {eos}"
            pair = f"{bos} $A {eos} $B:1 {eos}:1"
            special_tokens = [(bos, bos_id), (eos, eos_id)]
        elif not bos and eos:
            single = f"$A {eos}"
            pair = f"$A {eos} $B:1 {eos}:1"
            special_tokens = [(eos, eos_id)]
        else:
            raise ValueError(f"bos={bos}, eos={eos}")
        tok._tokenizer.post_processor = processors.Sequence(  # type: ignore
            [
                processors.TemplateProcessing(
                    single=single, pair=pair, special_tokens=special_tokens
                ),
            ]
        )
    # Do it batched
    for i in range(0, len(samples), BATCH_SIZE):
        print(f"       . {i} / {len(samples)}", flush=True)
        max_index = min(i + BATCH_SIZE, len(samples))
        slice = samples[i:max_index].to_list()
        if len(slice) == 0:
            continue
        encoding = tok(
            slice,
            max_length=cfg.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=cfg.add_special_tokens,
            return_overflowing_tokens=True,
        )
        all_input_ids = []
        all_attention_masks = []
        for input_ids, attention_mask in zip(
            encoding.input_ids, encoding.attention_mask
        ):
            input_ids = input_ids.squeeze().numpy()
            attention_mask = attention_mask.squeeze().numpy()
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
        df = pd.DataFrame(
            {"input_ids": all_input_ids, "attention_mask": all_attention_masks}
        )
        yield df


import duckdb
import torch
import pandas as pd
from typing import TypedDict, cast
from .dataset import DatasetLoaderConfig, DataPreprocessor, CHAT_TEMPLATES


class Message(TypedDict):
    role: str
    content: str


class RawDataPreprocessor(DataPreprocessor):
    def __init__(self, cfg: DatasetLoaderConfig):
        super().__init__(cfg)

    def init(self, conn: duckdb.DuckDBPyConnection):
        conn.execute(
            f"""
            CREATE TABLE dataset (
                input_ids INTEGER[], attention_mask INTEGER[],
                file VARCHAR, tokens INTEGER
            )
            """
        )

    def process_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        possible_text_columns = ["text", "content"]
        col_name: str | None = None
        for col in possible_text_columns:
            if col in df.columns:
                col_name = col
                break
        assert col_name is not None, "No text column found in the DataFrame."
        samples = df[col_name]
        tok = self.cfg._tokenizer
        data = tok(
            samples.to_list(),
            max_length=self.cfg.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
            return_overflowing_tokens=True,
        )
        data = cast(dict[str, torch.Tensor], data)
        attention_mask = data["attention_mask"].tolist()
        df = pd.DataFrame(
            {
                "input_ids": data["input_ids"].tolist(),
                "attention_mask": attention_mask,
                "tokens": [np.count_nonzero(x) for x in attention_mask],
            }
        )
        return df
