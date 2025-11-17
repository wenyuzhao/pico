from typing import Any, Optional, Union
from pixie.models._config import Config
import torch.nn.functional as F
import torch
from typing import cast
from transformers import PreTrainedTokenizerFast
from torch import nn
from transformers import Trainer


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


class PretrainTrainer(Trainer):
    @torch.compile
    def __compute_loss(
        self, out: Any, Y: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        loss = F.cross_entropy(
            out.logits.view(-1, out.logits.size(-1)),
            Y.view(-1),
            reduction="none",
        ).view(Y.size())
        loss = (loss * mask).sum() / mask.sum()
        loss = loss / self.args.gradient_accumulation_steps
        return loss

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        if self.model_accepts_loss_kwargs:
            kwargs = {}
            if num_items_in_batch is not None:
                kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **kwargs}

        input_ids = inputs["input_ids"]
        loss_mask = inputs["loss_mask"]
        X = input_ids[:, :-1]
        Y = input_ids[:, 1:].contiguous()
        mask = loss_mask[:, 1:]

        outputs = model(X)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        loss = self.__compute_loss(outputs, Y, mask)

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss
