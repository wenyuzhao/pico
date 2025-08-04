import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
import torch
import pandas as pd
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from tokenizers import processors
from typing import Generator, Literal
from slugify import slugify
from transformers import AutoTokenizer
import duckdb


class DuckDBDataset(Dataset):
    def __init__(
        self, path: str | Path, tokenizer: PreTrainedTokenizerFast, max_length: int
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Path {self.path} does not exist.")
        db_name = f".{slugify(self.tokenizer.name_or_path)}-{self.max_length}.db"
        if self.path.is_dir():
            db_path = self.path / db_name
        else:
            db_path = self.path.with_suffix(db_name)
        if not db_path.exists():
            raise FileNotFoundError(f"Database {db_path} does not exist.")
        self.conn = duckdb.connect(db_path)

        result = self.conn.execute("SELECT COUNT(*) FROM dataset").fetchone()
        assert result
        self.len = result[0]

        first_row = self.conn.execute(
            "SELECT input_ids FROM dataset LIMIT 1"
        ).fetchone()
        if not first_row:
            raise ValueError("Dataset is empty.")
        tokens_per_row = len(first_row[0])
        assert (
            tokens_per_row == self.max_length
        ), f"Expected {self.max_length} tokens per sample, got {tokens_per_row}."
        self.tokens = tokens_per_row * self.len

    def __len__(self):
        result = self.conn.execute("SELECT COUNT(*) FROM dataset").fetchone()
        return result[0] if result else 0

    def __getitem__(self, index: int):
        result = self.conn.execute(
            "SELECT input_ids, attention_mask FROM dataset LIMIT 1 OFFSET ?",
            (index,),
        ).fetchone()
        if not result:
            raise IndexError(f"Index {index} out of range.")
        input_ids, attention_mask = result
        assert len(input_ids) == self.max_length
        assert len(attention_mask) == self.max_length
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(attention_mask[1:], dtype=torch.long)
        return X, Y, loss_mask


BATCH_SIZE = 50000


@dataclass
class DatasetLoaderConfig:
    tokenizer: str = "jingyaogong/MiniMind2"
    max_length: int = 512
    add_special_tokens: bool = True
    type: Literal["pretrain", "sft"] = "pretrain"

    def __post_init__(self):
        tok = AutoTokenizer.from_pretrained(self.tokenizer)
        assert isinstance(tok, PreTrainedTokenizerFast)
        self._tokenizer = tok


def create_chat_prompt(
    conversations: list[list[dict[str, str]]],
    tokenizer: PreTrainedTokenizerFast,
) -> list[str]:
    alt_role_keys = ["from"]
    role_mapping: dict[str, str] = {
        "human": "user",
        "gpt": "assistant",
    }
    alt_content_keys = ["value"]
    records: list[np.ndarray[dict[str, str]]] = (  # type: ignore
        [conversations] if isinstance(conversations[0], dict) else conversations
    )
    records: list[list[dict[str, str]]] = [r.tolist() if not isinstance(r, list) else r for r in records]  # type: ignore
    # Fix record keys
    for msgs in records:
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
    # Tokenize the conversations
    prompts = tokenizer.apply_chat_template(records, tokenize=False)
    assert isinstance(prompts, list), "Prompts should be a list."
    return prompts  # type: ignore


def process_pretrain_data(
    df: pd.DataFrame, cfg: DatasetLoaderConfig
) -> Generator[pd.DataFrame]:
    samples = df["text"]
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
        print(f"{i} / {len(samples)}")
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


def process_sft_data(
    df: pd.DataFrame, cfg: DatasetLoaderConfig
) -> Generator[pd.DataFrame]:
    conversations: list[list[dict[str, str]]] = (
        df["conversations"] if "conversations" in df.columns else df["messages"]
    ).to_list()
    tok = cfg._tokenizer
    samples = create_chat_prompt(conversations, tok)
    # Do it batched
    for i in range(0, len(samples), BATCH_SIZE):
        print(f"{i} / {len(samples)}")
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


def preprocess(path: str, cfg: DatasetLoaderConfig, type: Literal["pretrain", "sft"]):
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Path {data_path} does not exist.")

    def processor(df: pd.DataFrame) -> Generator[pd.DataFrame]:
        if type == "pretrain":
            yield from process_pretrain_data(df, cfg)
        elif type == "sft":
            yield from process_sft_data(df, cfg)
        else:
            raise ValueError(f"Unsupported dataset type: {type}")

    def insert_into_db(file: Path, df: pd.DataFrame, conn: duckdb.DuckDBPyConnection):
        df = df[["input_ids", "attention_mask"]]
        # Add file name to the DataFrame
        df["file"] = [str(file)] * len(df)
        # Insert into database
        conn.execute("INSERT INTO dataset BY NAME SELECT * FROM df")

    def load_file(f: Path) -> pd.DataFrame:
        if f.suffix.lower() == ".parquet":
            return pd.read_parquet(f)
        elif f.suffix.lower() == ".jsonl":
            data = []
            with open(f, "r") as file:
                for line in file:
                    if line := line.strip():
                        data.append(json.loads(line))
            return pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported file type: {f.suffix}")

    def process_file(file: Path, conn: duckdb.DuckDBPyConnection) -> int:
        print(f"Processing {file} ...")
        df = load_file(file)
        samples = 0
        for seg in processor(df):
            samples += len(seg)
            insert_into_db(file, seg, conn)
        return samples

    # Open database
    db_name = f".{slugify(cfg.tokenizer)}-{cfg.max_length}.db"
    if data_path.is_dir():
        db_path = data_path / db_name
    else:
        db_path = data_path.with_suffix(db_name)
    # Remove existing database if it exists
    if db_path.exists():
        print(f"Removing existing database {db_path} ...")
        db_path.unlink()
    with duckdb.connect(db_path) as conn:
        max_len = cfg.max_length
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS dataset (input_ids INTEGER[{max_len}], attention_mask INTEGER[{max_len}], file VARCHAR)"
        )
        samples = 0
        # Load all jsonl or parquet
        if data_path.is_dir():
            for f in data_path.glob("**/*"):
                if (
                    f.is_file()
                    and f.suffix.lower() in [".parquet", ".jsonl"]
                    and not str(f).lower().endswith(".tokens.parquet")
                ):
                    samples += process_file(f, conn)
        else:
            assert data_path.is_file(), "Path must be a file or directory."
            assert data_path.suffix.lower() in [
                ".parquet",
                ".jsonl",
            ], "Unsupported file type."
            assert (
                not str(data_path).lower().endswith(".tokens.parquet")
            ), "Precomputed token files should not be processed here."
            samples += process_file(data_path, conn)
    print(f"Done. Processed {samples} samples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument(
        "--type", type=str, choices=["pretrain", "sft"], default="pretrain"
    )
    parser.add_argument("--tokenizer", type=str, default="jingyaogong/MiniMind2")
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()
    cfg = DatasetLoaderConfig(
        tokenizer=args.tokenizer,
        max_length=args.max_length,
        add_special_tokens=True,
        type=args.type,
    )
    preprocess(args.path, cfg, args.type)
