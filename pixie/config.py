from pathlib import Path
from typing import Annotated, Any, TYPE_CHECKING, Literal, Self
from pydantic import BaseModel, Field
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers import AutoTokenizer
from transformers.configuration_utils import PretrainedConfig as _PretrainedConfig
import yaml

if TYPE_CHECKING:
    from .models import BaseGPTModel


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
    generation: GenerationConfig

    def get_vocab_size(self) -> int:
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        assert isinstance(tokenizer, PreTrainedTokenizerFast)
        return tokenizer.vocab_size + len(tokenizer.additional_special_tokens)

    def to_dict(self):
        return self.model_dump()

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

    def load_tokenizer(self) -> PreTrainedTokenizerFast:
        """
        Creates the tokenizer for the model.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        assert isinstance(tokenizer, PreTrainedTokenizerFast)
        return tokenizer

    @classmethod
    def cast(cls, data: dict[str, Any] | Self | _PretrainedConfig) -> Self:
        if isinstance(data, cls):
            return data
        if isinstance(data, _PretrainedConfig):
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

    @staticmethod
    def load(path: str | Path) -> "Config":
        from .models import MODELS

        data = load_yaml_and_resolve_imports(path)
        # Create model config
        model_name = data.get("model", {}).get("name", None)
        assert (
            model_name is not None
        ), "Model name must be specified in the configuration."
        _, ConfigType = MODELS[model_name]
        mc = ConfigType(**data.get("model", {}))
        del data["model"]
        data["model"] = mc
        config = Config(**data)
        if config.name is None:
            config.name = Path(path).stem
        return config

    def load_model(self) -> "BaseGPTModel":
        """
        Creates the model configuration.
        """
        from .models import BaseGPTModel

        return BaseGPTModel.load(self.model)

    def load_tokenizer(self) -> PreTrainedTokenizerFast:
        """
        Creates the tokenizer for the model.
        """
        return self.model.load_tokenizer()

    def save(self, path: str | Path):
        with open(path, "w") as f:
            yaml.safe_dump(self.model_dump(), f)


class PretrainedConfig(_PretrainedConfig):
    has_no_defaults_at_init = True


def merge_yaml(dict1: dict[str, Any], dict2: dict[str, Any]) -> dict[str, Any]:
    """
    Merges two dictionaries, with dict2 taking precedence over dict1.
    """
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_yaml(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml_and_resolve_imports(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if "import" in data:
        paths = data["import"] if isinstance(data["import"], list) else [data["import"]]
        for p in paths:
            import_path = Path(p)
            if not import_path.is_absolute():
                import_path = Path(path).parent / p
            imported_data = load_yaml_and_resolve_imports(import_path)
            data = merge_yaml(imported_data, data)
        del data["import"]

    if "override" in data:
        overrides = data["override"]
        for key, value in overrides.items():
            keys = key.split(".")
            d = data
            for k in keys[:-1]:
                if k not in d or not isinstance(d, dict):
                    raise KeyError(f"Key {key} not found in configuration.")
                d = d[k]
            if not isinstance(d, dict):
                raise KeyError(f"Key {key} not found in configuration.")
            d[keys[-1]] = value
        del data["override"]

    return data
