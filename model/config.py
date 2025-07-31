from typing import Any
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers import AutoTokenizer
from transformers.configuration_utils import PretrainedConfig


class Config(PretrainedConfig):
    def __init__(
        self,
        tokenizer: str = "jingyaogong/MiniMind2",
        # tokenizer: str = "NousResearch/Llama-2-7b-hf",
        context_length: int = 512,
        hidden_size: int = 512,
        num_hidden_layers: int = 8,
        hidden_act: str = "silu",
        num_attention_heads: int = 8,
        num_kv_attention_heads: int = 2,
        dropout: float = 0.0,
        feed_forward_size: int = 1408,
        rope_theta: float = 1e6,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_act = hidden_act
        self.num_attention_heads = num_attention_heads
        self.num_kv_attention_heads = num_kv_attention_heads
        self.dropout = dropout
        self.feed_forward_size = feed_forward_size
        self.rope_theta = rope_theta

        tok = AutoTokenizer.from_pretrained(self.tokenizer)
        assert isinstance(tok, PreTrainedTokenizerFast)
        self.vocab_size = tok.vocab_size + len(tok.additional_special_tokens)
        num_kv_attention_heads = self.num_kv_attention_heads or self.num_attention_heads
        assert (
            self.num_attention_heads % num_kv_attention_heads == 0
        ), "num_attention_heads must be divisible by num_kv_attention_heads"
        assert (
            self.hidden_size % self.num_attention_heads == 0
        ), "hidden_size must be divisible by num_attention_heads"
        if self.feed_forward_size is None or self.feed_forward_size == 0:
            self.feed_forward_size = self.hidden_size * 4
        assert self.feed_forward_size > 0, "feed_forward_size must be positive"

    def to_dict(self):
        return self.__dict__
