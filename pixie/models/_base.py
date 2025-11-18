from typing import Annotated, Any, Literal, Self, override
from pydantic import BaseModel, Field
from pathlib import Path
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.utils import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch import nn
import shutil


class GenerationConfig(BaseModel):
    max_new_tokens: int
    temperature: float
    top_p: float
    repetition_penalty: float
    do_sample: bool
    use_cache: bool


class RopeScaling(BaseModel):
    beta_fast: float = 32
    beta_slow: float = 1
    original_max_position_embeddings: int
    type: Literal["yarn"] = "yarn"


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
    rope_scaling: RopeScaling | None = None
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
    name: str
    ratio: float | None = None
    data_dir: str | None = None
    data_files: list[str] | None = None


class AdamWOptimizerConfig(BaseModel):
    name: Literal["adamw"] = "adamw"
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


class LionOptimizerConfig(BaseModel):
    name: Literal["lion"] = "lion"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.99)


type OptimizerConfig = Annotated[
    AdamWOptimizerConfig | LionOptimizerConfig, Field(discriminator="name")
]


class BaseTrainingConfig(BaseModel):
    dataset: str | DatasetConfig
    context_length: int
    batch_size: int | Literal["auto"]

    epochs: int = 1
    grad_clip: float = 1.0
    warmup_steps: int | None = 400
    accumulation_steps: int = 8
    gradient_checkpointing: bool = False
    optimizer: OptimizerConfig | Literal["adamw", "lion"] = "adamw"


class PretrainConfig(BaseTrainingConfig): ...


class SFTConfig(BaseTrainingConfig): ...


class DPOConfig(BaseTrainingConfig):
    beta: float = 0.1


class Config(BaseModel):
    name: str | None = None
    model: ModelConfig
    pretrain: PretrainConfig | None = None
    sft: SFTConfig | None = None
    dpo: DPOConfig | None = None


class BaseCasualLM[C: ModelConfig](PreTrainedModel, GenerationMixin):
    config_class = PretrainedConfig
    model_type: str

    def __init__(self, config: C):
        self.args: C = config
        tok = config.load_tokenizer()
        config_dict = config.model_dump()
        if "generation" in config_dict:
            del config_dict["generation"]
        super().__init__(self.config_class(**config_dict))
        assert self.generation_config
        gcfg = self.args.generation
        if gcfg is not None:
            self.generation_config.max_new_tokens = gcfg.max_new_tokens
            self.generation_config.do_sample = gcfg.do_sample
            self.generation_config.temperature = gcfg.temperature
            self.generation_config.top_p = gcfg.top_p
            self.generation_config.repetition_penalty = gcfg.repetition_penalty
            self.generation_config.pad_token_id = tok.pad_token_id
            self.generation_config.eos_token_id = tok.eos_token_id
            self.generation_config.use_cache = gcfg.use_cache
        self.out = CausalLMOutputWithPast()
        self.model: nn.Module

    @override
    def save_pretrained(self, *args, **kwargs):
        models_dir = Path(__file__).parent
        save_directory = args[0] if args else kwargs.get("save_directory", None)
        assert save_directory
        save_dir = Path(save_directory)
        # Copy model definition
        shutil.copyfile(models_dir / "_base.py", save_dir / "_base.py")
        if (models_dir / f"{self.model_type}.py").exists():
            src = models_dir / f"{self.model_type}.py"
        elif (models_dir / f"model.py").exists():
            src = models_dir / f"model.py"
        else:
            raise FileNotFoundError(f"Source file for {self.model_type} not found.")
        shutil.copyfile(src, save_dir / f"model.py")
        # Save model weights and config
        super().save_pretrained(*args, **kwargs)


MODELS: dict[str, tuple[type[BaseCasualLM], type[ModelConfig]]] = {}


class BasePretrainedConfig(PretrainedConfig):
    model_name: str
    has_no_defaults_at_init = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.auto_map = {
            "AutoConfig": f"model.{self.__class__.__name__}",
            "AutoModel": f"model.{self.model_name}",
            "AutoModelForCausalLM": f"model.{self.model_name}",
        }


def register_model[T: ModelConfig](name: str, config: type[T]):
    def _register(cls: type[BaseCasualLM]):
        if name in MODELS:
            raise ValueError(f"Model {name} is already registered.")
        MODELS[name] = (cls, config)
        cls.model_type = name
        cls.config_class.model_type = name
        return cls

    return _register
