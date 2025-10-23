import numpy as np
import pandas as pd
from typing import override
from .dataset import DuckDBDataset
import torch.nn.functional as F
import duckdb
import torch
import pandas as pd
from typing import cast
from .dataset import DataPreprocessor


class PretrainDataPreprocessor(DataPreprocessor):
    @override
    def init(self, conn: duckdb.DuckDBPyConnection):
        conn.execute(
            f"""
            CREATE TABLE dataset (
                input_ids INTEGER[], attention_mask INTEGER[],
                file VARCHAR, tokens INTEGER
            )
            """
        )

    @override
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


class PretrainDataset(DuckDBDataset):
    @override
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        result = self.conn.execute(
            "SELECT input_ids, attention_mask FROM dataset LIMIT 1 OFFSET ?",
            (index,),
        ).fetchone()
        if not result:
            raise IndexError(f"Index {index} out of range.")
        input_ids = result[0]
        attention_mask = result[1]
        assert (
            len(input_ids) == self.max_length
        ), f"Expected input_ids length {self.max_length}, got {len(input_ids)}"
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(attention_mask[1:], dtype=torch.long)
        return {"x": X, "y": Y, "loss_mask": loss_mask}


class PretrainLoss(torch.nn.Module):
    def __init__(self, device: str, model: torch.nn.Module):
        super().__init__()
        self.device = device
        self.model = model

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        X, Y = batch
        X = batch["x"].to(self.device)
        Y = batch["y"].to(self.device)
        mask = batch["loss_mask"].to(self.device)
        out = self.model(X)
        loss = F.cross_entropy(
            out.logits.view(-1, out.logits.size(-1)),
            Y.view(-1),
        )
        loss = (loss * mask).sum() / mask.sum()
        return loss
