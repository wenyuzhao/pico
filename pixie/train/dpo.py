import torch
import pandas as pd
from typing import TypedDict, cast
from . import CHAT_TEMPLATES
from .pretrain import PretrainLoss
import torch
from typing import cast
from transformers import PreTrainedTokenizerFast
from typing import Any
from pixie.models._config import Config
import torch.nn.functional as F


class Message(TypedDict):
    role: str
    content: str


def _get_conversations(
    df: pd.DataFrame,
) -> tuple[list[list[Message]], list[list[Message]]]:
    assert "chosen" in df.columns, "DataFrame must contain 'chosen' column."
    assert "rejected" in df.columns, "DataFrame must contain 'rejected' column."

    chosen = [x if isinstance(x, list) else x.tolist() for x in df["chosen"].to_list()]
    rejected = [
        x if isinstance(x, list) else x.tolist() for x in df["rejected"].to_list()
    ]

    return cast(list[list[Message]], chosen), cast(list[list[Message]], rejected)  # type: ignore


def _process_batch_impl(
    samples: list[list[Message]], config: Config, tok: PreTrainedTokenizerFast
) -> dict[str, torch.Tensor]:
    assert config.dpo
    max_length = config.dpo.context_length
    r = tok.apply_chat_template(
        cast(list[list[dict[str, str]]], samples),
        tokenize=True,
        # add_generation_prompt=True,
        return_assistant_tokens_mask=True,
        return_dict=True,
        chat_template=CHAT_TEMPLATES.get(tok.name_or_path, ""),
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return r  # type: ignore


def preprocess(
    self, data: dict[str, Any], config: Config, tokenizer: PreTrainedTokenizerFast
) -> dict[str, Any]:
    chosen, rejected = self._get_conversations(pd.DataFrame(data))
    assert len(chosen) == len(rejected)
    chosen = self._process_batch_impl(chosen, config, tokenizer)
    rejected = self._process_batch_impl(rejected, config, tokenizer)
    chosen_x = [
        torch.tensor(x[:-1], dtype=torch.long) for x in chosen["input_ids"].tolist()
    ]
    chosen_y = [
        torch.tensor(y[1:], dtype=torch.long) for y in chosen["input_ids"].tolist()
    ]
    chosen_loss_mask = [
        torch.tensor(m[1:], dtype=torch.long)
        for m in chosen["assistant_masks"].tolist()
    ]
    rejected_x = [
        torch.tensor(x[:-1], dtype=torch.long) for x in rejected["input_ids"].tolist()
    ]
    rejected_y = [
        torch.tensor(y[1:], dtype=torch.long) for y in rejected["input_ids"].tolist()
    ]
    rejected_loss_mask = [
        torch.tensor(m[1:], dtype=torch.long)
        for m in rejected["assistant_masks"].tolist()
    ]
    return {
        "chosen_x": chosen_x,
        "chosen_y": chosen_y,
        "chosen_mask": chosen_loss_mask,
        "rejected_x": rejected_x,
        "rejected_y": rejected_y,
        "rejected_mask": rejected_loss_mask,
    }


def logits_to_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # logits: [batch_size, seq_len, vocab_size]
    # labels: [batch_size, seq_len]
    log_probs = F.log_softmax(logits, dim=2)
    probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return probs  # [batch_size, seq_len]


def dpo_loss(
    ref_probs: torch.Tensor, probs: torch.Tensor, mask: torch.Tensor, beta: float
) -> torch.Tensor:
    # ref_probs, probs: [batch_size, seq_len]
    # https://github.com/jingyaogong/minimind/issues/298
    seq_lengths = mask.sum(dim=1, keepdim=True)  # (batch_size, 1)
    ref_probs = (ref_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    probs = (probs * mask).sum(dim=1) / seq_lengths.squeeze()

    batch_size = ref_probs.shape[0]
    chosen_ref_probs = ref_probs[: batch_size // 2]
    reject_ref_probs = ref_probs[batch_size // 2 :]
    chosen_probs = probs[: batch_size // 2]
    reject_probs = probs[batch_size // 2 :]

    pi_logratios = chosen_probs - reject_probs
    ref_logratios = chosen_ref_probs - reject_ref_probs
    logits = pi_logratios - ref_logratios
    loss = (logits - 1 / (2 * beta)) ** 2
    return loss.mean()


class DPOLoss(torch.nn.Module):
    def __init__(
        self,
        device: str,
        model: torch.nn.Module,
        ref_model: torch.nn.Module,
        beta: float,
    ):
        super().__init__()
        self.device = device
        self.beta = beta
        self.model = model
        self.ref_model = ref_model

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        chosen_x = batch["chosen_x"].to(self.device)
        chosen_y = batch["chosen_y"].to(self.device)
        chosen_mask = batch["chosen_mask"].to(self.device)
        rejected_x = batch["rejected_x"].to(self.device)
        rejected_y = batch["rejected_y"].to(self.device)
        rejected_mask = batch["rejected_mask"].to(self.device)
        X = torch.cat([chosen_x, rejected_x], dim=0)
        Y = torch.cat([chosen_y, rejected_y], dim=0)
        loss_mask = torch.cat([chosen_mask, rejected_mask], dim=0)
        with torch.no_grad():
            assert self.ref_model is not None
            ref_out = self.ref_model(X)
        ref_probs = logits_to_probs(ref_out.logits, Y) * loss_mask
        out = self.model(X)
        probs = logits_to_probs(out.logits, Y) * loss_mask
        loss = dpo_loss(ref_probs, probs, loss_mask, beta=0.1)
        return loss
