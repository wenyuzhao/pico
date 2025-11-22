from typing import Any, Optional, Union
from pixie.models._base import Config
import torch.nn.functional as F
import torch
from typing import cast, TypedDict
from transformers import PreTrainedTokenizerFast
from torch import nn
from transformers import Trainer


class Message(TypedDict):
    role: str
    content: str


def _get_conversations(
    data: dict[str, Any],
) -> tuple[list[list[Message]], list[list[Message]]]:
    assert "chosen" in data
    assert "rejected" in data

    chosen = [x if isinstance(x, list) else x.tolist() for x in data["chosen"]]
    rejected = [x if isinstance(x, list) else x.tolist() for x in data["rejected"]]

    return chosen, rejected


def _process_batch_impl(
    samples: list[list[Message]], config: Config, tok: PreTrainedTokenizerFast
) -> dict[str, torch.Tensor]:
    args = config.train["dpo"]
    assert args is not None
    max_length = args.context_length
    assert tok.chat_template
    assert (
        "endgeneration" in tok.chat_template
    ), "chat template does not contain `{% generation %}` keyword."
    r = tok.apply_chat_template(
        cast(list[list[dict[str, str]]], samples),
        tokenize=True,
        # add_generation_prompt=True,
        return_assistant_tokens_mask=True,
        return_dict=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return r  # type: ignore


def preprocess(
    data: dict[str, Any], config: Config, tokenizer: PreTrainedTokenizerFast
) -> dict[str, torch.Tensor]:
    chosen, rejected = _get_conversations(data)
    assert len(chosen) == len(rejected)
    chosen = _process_batch_impl(chosen, config, tokenizer)
    rejected = _process_batch_impl(rejected, config, tokenizer)
    chosen_input_ids = chosen["input_ids"]
    chosen_loss_mask = chosen["assistant_masks"]
    rejected_input_ids = rejected["input_ids"]
    rejected_loss_mask = rejected["assistant_masks"]
    return {
        "chosen_input_ids": chosen_input_ids,
        "chosen_loss_mask": chosen_loss_mask,
        "rejected_input_ids": rejected_input_ids,
        "rejected_loss_mask": rejected_loss_mask,
    }


# @torch.compile
def logits_to_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # logits: [batch_size, seq_len, vocab_size]
    # labels: [batch_size, seq_len]
    log_probs = F.log_softmax(logits, dim=2)
    probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return probs  # [batch_size, seq_len]


# @torch.compile
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


class DPOTrainer(Trainer):
    def __init__(self, *args: Any, ref_model: nn.Module, beta: float, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.ref_model = torch.compile(ref_model, mode="default").to(self.args.device)  # type: ignore
        self.beta = beta

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

        chosen_input_ids, chosen_loss_mask = (
            inputs["chosen_input_ids"],
            inputs["chosen_loss_mask"],
        )
        rejected_input_ids, rejected_loss_mask = (
            inputs["rejected_input_ids"],
            inputs["rejected_loss_mask"],
        )
        chosen_x = chosen_input_ids[:, :-1]
        chosen_y = chosen_input_ids[:, 1:]
        chosen_mask = chosen_loss_mask[:, 1:]
        rejected_x = rejected_input_ids[:, :-1]
        rejected_y = rejected_input_ids[:, 1:]
        rejected_mask = rejected_loss_mask[:, 1:]
        X = torch.cat([chosen_x, rejected_x], dim=0)
        Y = torch.cat([chosen_y, rejected_y], dim=0)
        loss_mask = torch.cat([chosen_mask, rejected_mask], dim=0)

        with torch.no_grad():
            assert self.ref_model is not None
            ref_out = self.ref_model(X)
        ref_probs = logits_to_probs(ref_out.logits, Y) * loss_mask
        outputs = model(X)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        probs = logits_to_probs(outputs.logits, Y) * loss_mask
        loss = dpo_loss(ref_probs, probs, loss_mask, beta=self.beta)

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss
