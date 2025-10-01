import duckdb
import numpy as np
import torch
import pandas as pd
from typing import TypedDict, cast
from .dataset import DatasetLoaderConfig, DataPreprocessor, CHAT_TEMPLATES


class Message(TypedDict):
    role: str
    content: str


class DPODataPreprocessor(DataPreprocessor):
    def __init__(self, cfg: DatasetLoaderConfig):
        super().__init__(cfg)

    def init(self, conn: duckdb.DuckDBPyConnection):
        conn.execute(
            f"""
            CREATE TABLE dataset (
                chosen INTEGER[], chosen_attention_mask INTEGER[], chosen_assistant_mask INTEGER[],
                rejected INTEGER[], rejected_attention_mask INTEGER[], rejected_assistant_mask INTEGER[],
                file VARCHAR, tokens INTEGER
            )
            """
        )

    def get_conversations(
        self,
        df: pd.DataFrame,
    ) -> tuple[list[list[Message]], list[list[Message]]]:
        assert "chosen" in df.columns, "DataFrame must contain 'chosen' column."
        assert "rejected" in df.columns, "DataFrame must contain 'rejected' column."

        chosen = [x.tolist() for x in df["chosen"].to_list()]
        rejected = [x.tolist() for x in df["rejected"].to_list()]

        return cast(list[list[Message]], chosen), cast(list[list[Message]], rejected)  # type: ignore

    def process_batch_impl(
        self, samples: list[list[Message]]
    ) -> dict[str, torch.Tensor]:
        tok = self.cfg._tokenizer
        r = tok.apply_chat_template(
            cast(list[list[dict[str, str]]], samples),
            tokenize=True,
            # add_generation_prompt=True,
            return_assistant_tokens_mask=True,
            return_dict=True,
            chat_template=CHAT_TEMPLATES.get(tok.name_or_path, ""),
            max_length=self.cfg.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=self.cfg.add_special_tokens,
        )
        return r  # type: ignore

    def process_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        chosen, rejected = self.get_conversations(df)
        assert len(chosen) == len(rejected)
        chosen = self.process_batch_impl(chosen)
        rejected = self.process_batch_impl(rejected)
        chosen_attention_mask = chosen["attention_mask"].tolist()
        rejected_attention_mask = rejected["attention_mask"].tolist()
        df = pd.DataFrame(
            {
                "chosen": chosen["input_ids"].tolist(),
                "chosen_attention_mask": chosen_attention_mask,
                "chosen_assistant_mask": chosen["assistant_masks"].tolist(),
                "rejected": rejected["input_ids"].tolist(),
                "rejected_attention_mask": rejected_attention_mask,
                "rejected_assistant_mask": rejected["assistant_masks"].tolist(),
                "tokens": [
                    np.count_nonzero(x) + np.count_nonzero(y)
                    for x, y in zip(chosen_attention_mask, rejected_attention_mask)
                ],
            }
        )
        return df
