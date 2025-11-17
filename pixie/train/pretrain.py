from typing import Any
from pixie.models._config import Config
import torch.nn.functional as F
import torch
from typing import cast
from transformers import PreTrainedTokenizerFast


def preprocess(
    data: dict[str, Any], config: Config, tokenizer: PreTrainedTokenizerFast
) -> dict[str, torch.Tensor]:
    assert config.pretrain
    max_length = config.pretrain.context_length
    possible_text_columns = ["text", "content"]
    col_name: str | None = None
    for col in possible_text_columns:
        if col in data:
            col_name = col
            break
    assert col_name is not None, "No text column found in the dataset."
    text = data[col_name]
    tokens = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False,
        return_overflowing_tokens=True,
    )
    tokens = cast(dict[str, torch.Tensor], tokens)
    return {"input_ids": tokens["input_ids"], "loss_mask": tokens["attention_mask"]}


class PretrainLoss(torch.nn.Module):
    def __init__(self, device: str, model: torch.nn.Module, accumulation_steps: int):
        super().__init__()
        self.device = device
        self.model = model
        self.accumulation_steps = accumulation_steps

    def forward(
        self, input_ids: torch.Tensor, loss_mask: torch.Tensor, **kwargs
    ) -> dict[str, torch.Tensor]:
        X = input_ids[:, :-1].to(self.device)
        Y = input_ids[:, 1:].contiguous().to(self.device)
        mask = loss_mask[:, 1:].to(self.device)
        out = self.model(X)
        loss = F.cross_entropy(
            out.logits.view(-1, out.logits.size(-1)),
            Y.view(-1),
            reduction="none",
        ).view(Y.size())
        loss = (loss * mask).sum() / mask.sum()
        loss = loss / self.accumulation_steps
        return {"loss": loss}
