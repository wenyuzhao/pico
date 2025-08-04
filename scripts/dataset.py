import json
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
import torch
import pandas as pd
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from tokenizers import processors
from typing import Callable, Sequence
from slugify import slugify


def load_and_preprocess(
    tokenizer: str,
    path: str | Path | Sequence[str | Path],
    processor: Callable[[pd.DataFrame], pd.DataFrame],
    size: int,
):
    paths = [Path(path)] if isinstance(path, (Path, str)) else [Path(p) for p in path]
    for p in paths:
        if p.is_dir():
            precomputed = p / f".{tokenizer}-{size}.tokens.parquet"
            if precomputed.exists():
                continue
            files = []
            for p in p.glob("**/*"):
                if not p.is_file():
                    continue
                s = str(p).lower()
                if s.endswith((".parquet", ".jsonl")) and not s.endswith(
                    ".tokens.parquet"
                ):
                    if p.suffix.lower() == ".parquet":
                        files.append(pd.read_parquet(p))
                    elif p.suffix.lower() == ".jsonl":
                        data = []
                        with open(p, "r") as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    data.append(json.loads(line))
                        files.append(pd.DataFrame(data))
            df = pd.concat(files, ignore_index=True)
            df = processor(df)
            df.to_parquet(precomputed, index=False)

        elif p.is_file():
            assert p.suffix.lower() in [
                ".parquet",
                ".jsonl",
            ], f"Unsupported file type: {p.suffix}"
            s = str(p).lower()
            if s.endswith((".parquet", ".jsonl")) and not s.endswith(".tokens.parquet"):
                precomputed = p.with_suffix(f".{tokenizer}-{size}.tokens.parquet")
                if precomputed.exists():
                    continue
                if p.suffix.lower() == ".parquet":
                    df = pd.read_parquet(p)
                else:
                    data = []
                    with open(p, "r") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                data.append(json.loads(line))
                    df = pd.DataFrame(data)
                df = processor(df)
                df.to_parquet(precomputed, index=False)


def load_precomputed(
    tokenizer: str, path: str | Path | Sequence[str | Path], size: int
) -> pd.DataFrame:
    paths = [Path(path)] if isinstance(path, (Path, str)) else [Path(p) for p in path]
    files: list[pd.DataFrame] = []
    for p in paths:
        if p.is_dir():
            precomputed = p / f".{tokenizer}-{size}.tokens.parquet"
            assert precomputed.exists()
            files.append(pd.read_parquet(precomputed))
        elif p.is_file():
            s = str(p).lower()
            assert p.suffix.lower() in [
                ".parquet",
                ".jsonl",
            ], f"Unsupported file type: {p.suffix}"
            assert not s.endswith(
                ".tokens.parquet"
            ), "Precomputed token files should not be loaded here."
            precomputed = p.with_suffix(f".{tokenizer}-{size}.tokens.parquet")
            assert precomputed.exists()
            files.append(pd.read_parquet(precomputed))
    return pd.concat(files, ignore_index=True)


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


class PretrainDataset(Dataset):
    def __init__(
        self,
        data: str | Path | Sequence[str | Path],
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 512,
        add_special_tokens: bool = True,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        self.samples = self.load_data(data)

    def load_data(self, paths: str | Path | Sequence[str | Path]) -> pd.DataFrame:
        def processor(df: pd.DataFrame) -> pd.DataFrame:
            return self.tokenize_data(df["text"].tolist())

        slug = slugify(self.tokenizer.name_or_path)
        load_and_preprocess(slug, paths, processor, size=self.max_length)
        return load_precomputed(slug, paths, size=self.max_length)

    def tokenize_data(self, samples: list[str]) -> pd.DataFrame:
        if self.add_special_tokens:
            (bos, eos) = (self.tokenizer.bos_token, self.tokenizer.eos_token)
            (bos_id, eos_id) = (
                self.tokenizer.bos_token_id,
                self.tokenizer.eos_token_id,
            )
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
            self.tokenizer._tokenizer.post_processor = processors.Sequence(  # type: ignore
                [
                    processors.TemplateProcessing(
                        single=single, pair=pair, special_tokens=special_tokens
                    ),
                ]
            )
        # Do it batched
        batch_size = 50000
        tokenized_samples = []
        for i in range(0, len(samples), batch_size):
            print(f"{i} / {len(samples)}")
            max_index = min(i + batch_size, len(samples))
            slice = samples[i:max_index]
            if len(slice) == 0:
                continue
            encoding = self.tokenizer(
                slice,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                add_special_tokens=True,
                return_overflowing_tokens=True,
            )
            for input_ids, attention_mask in zip(
                encoding.input_ids, encoding.attention_mask
            ):
                input_ids = input_ids.squeeze().numpy()
                attention_mask = attention_mask.squeeze().numpy()
                tokenized_samples.append(
                    {"input_ids": input_ids, "attention_mask": attention_mask}
                )
        return pd.DataFrame(tokenized_samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples.iloc[index]
        input_ids = np.array(sample["input_ids"])
        loss_mask = np.array(sample["attention_mask"])
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        return X, Y, loss_mask


class SFTDataset(Dataset):
    def __init__(
        self,
        data: str | Path | Sequence[str | Path],
        tokenizer: PreTrainedTokenizerFast,
        max_length=1024,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bos_id: int | None = tokenizer.bos_token_id  # type: ignore
        self.eos_id: int = tokenizer.eos_token_id  # type: ignore
        self.samples = self.load_data(data)

    def load_data(self, path):
        def processor(df: pd.DataFrame) -> pd.DataFrame:
            col = "conversations" if "conversations" in df.columns else "messages"
            return self.__create_chat_prompt_and_tokenize(df[col].to_list())

        slug = slugify(self.tokenizer.name_or_path)
        load_and_preprocess(slug, path, processor, size=self.max_length)
        return load_precomputed(slug, path, size=self.max_length)

    def __create_chat_prompt_and_tokenize(
        self, conversations: list[list[dict[str, str]]]
    ) -> pd.DataFrame:
        samples = create_chat_prompt(conversations, self.tokenizer)
        # Do it batched
        batch_size = 50000
        tokenized_samples = []
        for i in range(0, len(samples), batch_size):
            print(f"{i} / {len(samples)}")
            max_index = min(i + batch_size, len(samples))
            slice = samples[i:max_index]
            encoding = self.tokenizer(
                slice,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                add_special_tokens=True,
            )
            for input_ids, attention_mask in zip(
                encoding.input_ids, encoding.attention_mask
            ):
                input_ids = input_ids.squeeze().numpy()
                attention_mask = attention_mask.squeeze().numpy()
                tokenized_samples.append(
                    {"input_ids": input_ids, "attention_mask": attention_mask}
                )
        print(f"Tokenized {len(tokenized_samples)} samples.")
        num_filtered = sum(
            1
            for sample in tokenized_samples
            if len(sample["input_ids"]) <= self.max_length
        )
        print(f"Including filtered {num_filtered} samples.")
        return pd.DataFrame(tokenized_samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples.iloc[index]
        input_ids = np.array(sample["input_ids"])
        attention_mask = np.array(sample["attention_mask"])
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        attention_mask = torch.tensor(attention_mask[1:], dtype=torch.long)
        return X, Y, attention_mask
