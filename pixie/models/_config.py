from typing import Annotated, Any, Literal, Self
from pydantic import BaseModel, Field
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.configuration_utils import PretrainedConfig


class GenerationConfig(BaseModel):
    max_new_tokens: int
    temperature: float
    top_p: float
    repetition_penalty: float
    do_sample: bool
    use_cache: bool


class ModelConfig(BaseModel):
    name: str
    tokenizer: str
    hidden_size: int
    num_hidden_layers: int
    hidden_act: str
    num_attention_heads: int
    num_kv_attention_heads: int
    dropout: float
    feed_forward_size: int
    max_position_embeddings: int
    rope_theta: float
    generation: GenerationConfig | None = None

    def load_tokenizer(self) -> PreTrainedTokenizerFast:
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        assert isinstance(tokenizer, PreTrainedTokenizerFast)
        return tokenizer

    def get_vocab_size(self) -> int:
        tokenizer = self.load_tokenizer()
        return tokenizer.vocab_size + len(tokenizer.additional_special_tokens)

    def model_post_init(self, context: Any) -> None:
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

    @classmethod
    def cast(cls, data: dict[str, Any] | Self | PretrainedConfig) -> Self:
        if isinstance(data, cls):
            return data
        if isinstance(data, PretrainedConfig):
            data = data.to_dict()
        assert isinstance(data, dict)
        return cls(**data)


class DatasetConfig(BaseModel):
    path: str
    limit: int | None = None


class AdamWOptimizerConfig(BaseModel):
    name: Literal["adamw"] = "adamw"
    learning_rate: float = 5e-4


class LionOptimizerConfig(BaseModel):
    name: Literal["lion"] = "lion"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2


type OptimizerConfig = Annotated[
    AdamWOptimizerConfig | LionOptimizerConfig, Field(discriminator="name")
]


class BaseTrainingConfig(BaseModel):
    dataset: str | DatasetConfig
    context_length: int
    batch_size: int

    epochs: int = 1
    grad_clip: float = 1.0
    warmup_steps: int | None = 400
    accumulation_steps: int = 8
    gradient_checkpointing: bool = False
    optimizer: OptimizerConfig | Literal["adamw", "lion"] = "adamw"


class PretrainConfig(BaseTrainingConfig): ...


class SFTConfig(BaseTrainingConfig): ...


class DPOConfig(BaseTrainingConfig): ...


class Config(BaseModel):
    name: str | None = None
    model: ModelConfig
    pretrain: PretrainConfig | None = None
    sft: SFTConfig | None = None
    dpo: DPOConfig | None = None
