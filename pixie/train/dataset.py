from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
import torch
import pandas as pd
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from typing import Generator, Literal
from slugify import slugify
from transformers import AutoTokenizer
import duckdb
import time


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
            r = self.conn.execute("SELECT SUM(tokens) FROM dataset").fetchone()
            assert r, "No records found in the dataset."
            print(f"Using the entire dataset with {self.len} records ({r[0]} tokens).")

        r = self.conn.execute("SELECT input_ids FROM dataset LIMIT 1").fetchone()
        if not r:
            raise ValueError("Dataset is empty.")
        self.is_dpo = not r[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index: int):
        if not self.is_dpo:
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
        else:
            result = self.conn.execute(
                "SELECT chosen, chosen_mask, rejected, rejected_mask FROM dataset LIMIT 1 OFFSET ?",
                (index,),
            ).fetchone()
            if not result:
                raise IndexError(f"Index {index} out of range.")
            chosen, chosen_mask, rejected, rejected_mask = result
            assert len(chosen) == self.max_length
            assert len(chosen_mask) == self.max_length
            assert len(rejected) == self.max_length
            assert len(rejected_mask) == self.max_length
            chosen_x = torch.tensor(chosen[:-1], dtype=torch.long)
            chosen_y = torch.tensor(chosen[1:], dtype=torch.long)
            chosen_loss_mask = torch.tensor(chosen_mask[1:], dtype=torch.long)
            rejected_x = torch.tensor(rejected[:-1], dtype=torch.long)
            rejected_y = torch.tensor(rejected[1:], dtype=torch.long)
            rejected_loss_mask = torch.tensor(rejected_mask[1:], dtype=torch.long)
            return {
                "chosen_x": chosen_x,
                "chosen_y": chosen_y,
                "chosen_mask": chosen_loss_mask,
                "rejected_x": rejected_x,
                "rejected_y": rejected_y,
                "rejected_mask": rejected_loss_mask,
            }


BATCH_SIZE = 20000


@dataclass
class DatasetLoaderConfig:
    tokenizer: str
    max_length: int
    add_special_tokens: bool = True
    type: Literal["pretrain", "sft", "dpo"] = "pretrain"

    def __post_init__(self):
        tok = AutoTokenizer.from_pretrained(self.tokenizer)
        assert isinstance(tok, PreTrainedTokenizerFast)
        self._tokenizer = tok


PROJECT_ROOT = Path(__file__).parent.parent.parent
assert (PROJECT_ROOT / "pyproject.toml").exists(), "Not in the project root."


def create_database(conn: duckdb.DuckDBPyConnection):
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS dataset (input_ids INTEGER[], attention_mask INTEGER[], chosen INTEGER[], chosen_mask INTEGER[], rejected INTEGER[], rejected_mask INTEGER[], file VARCHAR, tokens INTEGER)"
    )
    # clear any existing data
    conn.execute("DELETE FROM dataset")


def preprocess(
    data_path: Path, cfg: DatasetLoaderConfig, type: Literal["pretrain", "sft", "dpo"]
):
    if not data_path.exists():
        raise FileNotFoundError(f"Path {data_path} does not exist.")

    def processor(df: pd.DataFrame) -> Generator[pd.DataFrame]:
        if type == "pretrain":
            from .dataset_raw import process_raw

            yield from process_raw(df, cfg)
        elif type == "sft":
            from .dataset_sft import process_sft

            yield from process_sft(df, cfg)
        elif type == "dpo":
            from .dataset_dpo import process_dpo

            yield from process_dpo(df, cfg)
        else:
            raise ValueError(f"Unsupported dataset type: {type}")

    def insert_into_db(
        file: Path, df: pd.DataFrame, conn: duckdb.DuckDBPyConnection
    ) -> int:
        # Add empty columns for all keys if not existing
        keys = [
            "input_ids",
            "attention_mask",
            "chosen",
            "chosen_mask",
            "rejected",
            "rejected_mask",
        ]
        for key in keys:
            if key not in df.columns:
                df[key] = [None] * len(df)
        file_path = file.resolve().relative_to(PROJECT_ROOT)
        df = df[keys]
        # Add file name to the DataFrame
        df["file"] = [str(file_path)] * len(df)
        df["tokens"] = (
            df["input_ids"].apply(
                lambda x: int(np.count_nonzero(x if x is not None else []))
            )
            + df["chosen"].apply(
                lambda x: int(np.count_nonzero(x if x is not None else []))
            )
            + df["rejected"].apply(
                lambda x: int(np.count_nonzero(x if x is not None else []))
            )
        )
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
        create_database(conn)
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
