import duckdb
import numpy as np
import torch
import pandas as pd
from typing import TypedDict, cast
from .dataset import DatasetLoaderConfig, DataPreprocessor, CHAT_TEMPLATES


class Message(TypedDict):
    role: str
    content: str


class SFTDataPreprocessor(DataPreprocessor):
    def __init__(self, cfg: DatasetLoaderConfig):
        super().__init__(cfg)

    def init(self, conn: duckdb.DuckDBPyConnection):
        conn.execute(
            f"""
            CREATE TABLE dataset (
                input_ids INTEGER[], attention_mask INTEGER[], assistant_mask INTEGER[],
                file VARCHAR, tokens INTEGER
            )
            """
        )

    def get_conversations(self, df: pd.DataFrame) -> list[list[Message]]:
        conversations: list[list[Message]]

        if "conversations" in df.columns:
            conversations = df["conversations"].to_list()  # type: ignore
        elif "messages" in df.columns:
            conversations = df["messages"].to_list()  # type: ignore
        elif (
            "instruction" in df.columns
            and "input" in df.columns
            and "output" in df.columns
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

    def process_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        samples = self.get_conversations(df)
        tok = self.cfg._tokenizer
        data = tok.apply_chat_template(
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
        data = cast(dict[str, torch.Tensor], data)
        attention_mask = data["attention_mask"].tolist()
        df = pd.DataFrame(
            {
                "input_ids": data["input_ids"].tolist(),
                "attention_mask": attention_mask,
                "assistant_mask": data["assistant_mask"].tolist(),
                "tokens": [np.count_nonzero(x) for x in attention_mask],
            }
        )
        return df
