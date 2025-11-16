from typing import Any
from pixie.models._config import Config
import torch.nn.functional as F
import torch
from typing import cast
from transformers import PreTrainedTokenizerFast


def preprocess(
    data: dict[str, Any], config: Config, tokenizer: PreTrainedTokenizerFast
) -> dict[str, list[torch.Tensor]]:
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
    xs = [torch.tensor(x[:-1], dtype=torch.long) for x in tokens["input_ids"].tolist()]
    ys = [torch.tensor(y[1:], dtype=torch.long) for y in tokens["input_ids"].tolist()]
    masks = [
        torch.tensor(m[1:], dtype=torch.long) for m in tokens["attention_mask"].tolist()
    ]
    return {"x": xs, "y": ys, "loss_mask": masks}


class PretrainLoss(torch.nn.Module):
    def __init__(self, device: str, model: torch.nn.Module):
        super().__init__()
        self.device = device
        self.model = model

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        X = batch["x"].to(self.device)
        Y = batch["y"].to(self.device)
        mask = batch["loss_mask"].to(self.device)
        out = self.model(X)
        loss = F.cross_entropy(
            out.logits.view(-1, out.logits.size(-1)),
            Y.view(-1),
            reduction="none",
        ).view(Y.size())
        loss = (loss * mask).sum() / mask.sum()
        return loss
