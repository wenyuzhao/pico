import argparse
from dataclasses import dataclass
import json
import os
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
import time

from model.config import DEFAULT_TOKENIZER, TRAINING_CONTEXT_LENGTH


class DuckDBDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int,
        limit: int | None = None,
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
        records = result[0]
        if limit is not None:
            if limit > records:
                raise ValueError(
                    f"Limit size {limit} is larger than the dataset size {records}."
                )
            # Count tokens
            self.len = limit
            r = self.conn.execute(
                "SELECT SUM(tokens) FROM (SELECT tokens FROM dataset LIMIT ?)", (limit,)
            ).fetchone()
            assert r, "No records found in the dataset."
            print(f"Using first {limit} records from the dataset ({r[0]} tokens).")
        else:
            self.len = records
            r = self.conn.execute("SELECT COUNT(*) FROM dataset").fetchone()
            assert r, "No records found in the dataset."
            print(f"Using the entire dataset with {self.len} records ({r[0]} tokens).")

        # verify vector length in the database
        r = self.conn.execute("SELECT input_ids FROM dataset LIMIT 1").fetchone()
        if not r:
            raise ValueError("Dataset is empty.")
        tokens_per_row = len(r[0])
        assert (
            tokens_per_row == self.max_length
        ), f"Expected {self.max_length} tokens per sample, got {tokens_per_row}."

    def __len__(self):
        return self.len

    def __getitem__(self, index: int):
        result = self.conn.execute(
            "SELECT input_ids, attention_mask FROM dataset LIMIT 1 OFFSET ?", (index,)
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


BATCH_SIZE = 20000


@dataclass
class DatasetLoaderConfig:
    tokenizer: str = DEFAULT_TOKENIZER
    max_length: int = TRAINING_CONTEXT_LENGTH
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
        # Trim entries: last entry must be from user
        # while len(msgs) > 0 and msgs[-1]["role"] != "user":
        #     msgs.pop()
    # records = [r for r in records if len(r) > 0]
    # Tokenize the conversations
    prompts = tokenizer.apply_chat_template(records, tokenize=False)
    assert isinstance(prompts, list), "Prompts should be a list."
    return prompts  # type: ignore


def process_pretrain_data(
    df: pd.DataFrame, cfg: DatasetLoaderConfig
) -> Generator[pd.DataFrame]:
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


PROJECT_ROOT = Path(__file__).parent.parent
assert (PROJECT_ROOT / "pyproject.toml").exists(), "Not in the project root."


def create_database(conn: duckdb.DuckDBPyConnection, max_len: int):
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS dataset (input_ids INTEGER[{max_len}], attention_mask INTEGER[{max_len}], file VARCHAR, tokens INTEGER)"
    )


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

    def insert_into_db(
        file: Path, df: pd.DataFrame, conn: duckdb.DuckDBPyConnection
    ) -> int:
        file_path = file.resolve().relative_to(PROJECT_ROOT)
        df = df[["input_ids", "attention_mask"]]
        # Add file name to the DataFrame
        df["file"] = [str(file_path)] * len(df)
        df["tokens"] = df["input_ids"].apply(lambda x: int(np.count_nonzero(x)))
        total_tokens = df["tokens"].sum()
        # Insert into database
        conn.execute("INSERT INTO dataset BY NAME SELECT * FROM df")
        return int(total_tokens)

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
        elif f.suffix.lower() == ".txt":
            assert (
                type == "pretrain"
            ), "Text files can only be processed for pretraining."
            # just read as a giant string
            text = f.read_text(encoding="utf-8")
            return pd.DataFrame({"text": [text]})
        else:
            raise ValueError(f"Unsupported file type: {f.suffix}")

    def process_file(file: Path, conn: duckdb.DuckDBPyConnection) -> tuple[int, int]:
        df = load_file(file)
        samples, tokens = 0, 0
        for seg in processor(df):
            samples += len(seg)
            tokens += insert_into_db(file, seg, conn)
        return samples, tokens

    # Open database
    db_name = f".{slugify(cfg.tokenizer)}-{cfg.max_length}.db"
    if data_path.is_dir():
        db_path = data_path / db_name
    else:
        db_path = data_path.with_suffix(db_name)
    with duckdb.connect(db_path) as conn:
        # 0. Create the database and table if not exists
        max_len = cfg.max_length
        create_database(conn, max_len)
        # 1. find out all files in the database
        print(f"1. Collecting files in the database ...", flush=True)
        all_files_in_db = conn.execute("SELECT DISTINCT file FROM dataset").fetchall()
        all_files_in_db = {Path(row[0]) for row in all_files_in_db}
        print(f"1. Found {len(all_files_in_db)} files in the database.", flush=True)
        # 2. find out all files in the directory
        print(f"2. Collecting files ...", flush=True)
        all_files_collected = set()
        if data_path.is_dir():
            for f in data_path.glob("**/*"):
                if f.is_file() and f.suffix.lower() in [".parquet", ".jsonl", ".txt"]:
                    all_files_collected.add(f.resolve().relative_to(PROJECT_ROOT))
        else:
            assert data_path.exists(), "Path must be a file or directory."
            all_files_collected = {data_path.resolve().relative_to(PROJECT_ROOT)}
        print(f"2. Found {len(all_files_collected)} files to process.", flush=True)
        # 3. Process files that are not in the database
        print(f"3. Processing files ...", flush=True)
        to_process = all_files_collected - all_files_in_db
        samples, tokens = 0, 0
        sorted_to_process = sorted(to_process, key=lambda x: str(x))
        for i, f in enumerate(sorted_to_process):
            print(f"    - ADD {f} ({i + 1} / {len(sorted_to_process)})", flush=True)
            file_samples, file_tokens = process_file(PROJECT_ROOT / f, conn)
            samples += file_samples
            tokens += file_tokens
            print(
                f"    - DONE. samples: {file_samples}, tokens: {file_tokens}",
                flush=True,
            )
        print(
            f"3. Processed {len(to_process)} files, total samples: {samples}, total tokens: {tokens}",
            flush=True,
        )
        # 4. Remove files that are in the database but not in the collected files
        print(f"4. Removing dangling files in the database ...", flush=True)
        to_remove = all_files_in_db - all_files_collected
        for f in to_remove:
            conn.execute("DELETE FROM dataset WHERE file = ?", (str(f),))
        print(f"4. Removed {len(to_remove)} files from the database.", flush=True)
        # 5. Display current total samples and tokens in the database
        result = conn.execute("SELECT COUNT(*), SUM(tokens) FROM dataset").fetchone()
        total_samples, total_tokens = result if result else (0, 0)
        print(f"\nDatabase Stats: {total_samples}", flush=True)
        print(f"  Files: {len(all_files_collected)}", flush=True)
        print(f"  Samples: {total_samples}", flush=True)
        print(f"  Tokens: {total_tokens} ({total_tokens / 1e9:.3f} B)", flush=True)


def force_delete_from_db(db: Path, files: list[Path]):
    with duckdb.connect(db) as conn:
        for f in files:
            f = f.resolve().relative_to(PROJECT_ROOT)
            print(f"Deleting {f} from database ...", flush=True)
            conn.execute("DELETE FROM dataset WHERE file = ?", (str(f),))
    print(f"Deleted {len(files)} files from database {db}.", flush=True)


def show_dataset_stats(db: Path, limit: int | None = None):
    if db.suffix != ".db":
        raise ValueError(f"Database file must have .db suffix, got {db.suffix}")
    with duckdb.connect(db) as conn:
        first = conn.execute("SELECT input_ids FROM dataset LIMIT 1").fetchone()
        if not first:
            print("Dataset is empty.", flush=True)
            return
        tokens_per_row = len(first[0])
        if limit is None:
            result = conn.execute(
                "SELECT COUNT(*), SUM(tokens), COUNT(DISTINCT file) FROM dataset"
            ).fetchone()
        else:
            result = conn.execute(
                "SELECT COUNT(*), SUM(tokens), COUNT(DISTINCT file) FROM (SELECT * FROM dataset LIMIT ?)",
                (limit,),
            ).fetchone()
        if not result:
            print("No records found in the dataset.", flush=True)
            return
        total_samples, total_tokens, total_files = result
        if limit is not None:
            print(f"Showing stats for the first {limit} samples.", flush=True)
        print(f"Total samples: {total_samples}", flush=True)
        print(f"Total tokens: {total_tokens} ({total_tokens / 1e9:.3f} B)", flush=True)
        print(f"Total files: {total_files}", flush=True)
        print(f"Tokens per sample: {tokens_per_row}", flush=True)


def shuffle_dataset(db: Path):
    if db.suffix != ".db":
        raise ValueError(f"Database file must have .db suffix, got {db.suffix}")
    start_time = time.time()
    with duckdb.connect(db) as conn:
        # Shuffle the dataset
        conn.execute(
            "CREATE OR REPLACE TABLE dataset AS SELECT * FROM dataset ORDER BY RANDOM()"
        )
    elapsed_time = time.time() - start_time
    print(f"Shuffled dataset in {db}, took {elapsed_time:.2f} seconds.", flush=True)


def main():
    os.environ["PYTHONUNBUFFERED"] = "1"
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    # subparser: process
    parser_process = subparsers.add_parser("process", help="Process dataset files.")
    parser_process.add_argument("path", type=str)
    parser_process.add_argument(
        "--type", type=str, choices=["pretrain", "sft"], default="pretrain"
    )
    parser_process.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER)
    parser_process.add_argument(
        "--max_length", type=int, default=TRAINING_CONTEXT_LENGTH
    )
    parser_process.set_defaults(func=preprocess)
    # subparser: remove
    parser_remove = subparsers.add_parser("remove")
    parser_remove.add_argument("db", type=str)
    parser_remove.add_argument("--files", "-f", type=str, nargs="+")
    parser_remove.set_defaults(func=force_delete_from_db)
    # subparser: shuffle
    parser_shuffle = subparsers.add_parser("shuffle", help="Shuffle the dataset.")
    parser_shuffle.add_argument("db", type=str)
    parser_shuffle.set_defaults(func=shuffle_dataset)
    # subparser: stats
    parser_stats = subparsers.add_parser("stats", help="Show dataset statistics.")
    parser_stats.add_argument("db", type=str)
    parser_stats.set_defaults(func=show_dataset_stats)
    parser_stats.add_argument("--limit", type=int, default=None)
    # parse and run
    args = parser.parse_args()
    if hasattr(args, "func"):
        if args.func == preprocess:
            cfg = DatasetLoaderConfig(
                tokenizer=args.tokenizer, max_length=args.max_length, type=args.type
            )
            preprocess(args.path, cfg, args.type)
        elif args.func == force_delete_from_db:
            db_path = Path(args.db)
            files = [Path(f) for f in args.files]
            force_delete_from_db(db_path, files)
        elif args.func == shuffle_dataset:
            shuffle_dataset(Path(args.db))
        elif args.func == show_dataset_stats:
            show_dataset_stats(Path(args.db), args.limit)


if __name__ == "__main__":
    main()
