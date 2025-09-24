from pathlib import Path
from typing import Any, TYPE_CHECKING
from pydantic import BaseModel
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers import AutoTokenizer
from transformers.configuration_utils import PretrainedConfig as _PretrainedConfig
import yaml

if TYPE_CHECKING:
    from model.models import BaseGPTModel


class ModelConfig(BaseModel):
    name: str
    tokenizer: str = "microsoft/phi-4"
    hidden_size: int = 512
    num_hidden_layers: int = 8
    hidden_act: str = "silu"
    num_attention_heads: int = 8
    num_kv_attention_heads: int = 2
    dropout: float = 0.0
    feed_forward_size: int = 1408
    max_position_embeddings: int = 32768
    rope_theta: float = 1e6

    def get_vocab_size(self) -> int:
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        assert isinstance(tokenizer, PreTrainedTokenizerFast)
        return tokenizer.vocab_size + len(tokenizer.additional_special_tokens)


class DatasetConfig(BaseModel):
    path: str
    limit: int | None = None


class BaseTrainingConfig(BaseModel):
    dataset: str | DatasetConfig
    context_length: int
    batch_size: int
    epochs: int = 1
    learning_rate: float = 5e-4
    grad_clip: float = 1.0
    warmup_steps: int | None = 400


class PretrainConfig(BaseTrainingConfig): ...


class SFTConfig(BaseTrainingConfig): ...


class Config(BaseModel):
    name: str | None = None
    model: ModelConfig
    pretrain: PretrainConfig | None = None
    sft: SFTConfig | None = None

    @staticmethod
    def load(path: str | Path) -> "Config":
        data = load_yaml_and_resolve_imports(path)
        config = Config(**data)
        if config.name is None:
            config.name = Path(path).stem
        return config

    def load_model(self, compile: bool = False) -> "BaseGPTModel":
        """
        Creates the model configuration.
        """
        from model.models import BaseGPTModel

        return BaseGPTModel.load(self, compile=compile)

    def load_tokenizer(self) -> PreTrainedTokenizerFast:
        """
        Creates the tokenizer for the model.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model.tokenizer)
        assert isinstance(tokenizer, PreTrainedTokenizerFast)
        return tokenizer

    def save(self, path: str | Path):
        with open(path, "w") as f:
            yaml.safe_dump(self.model_dump(), f)


class PretrainedConfig(_PretrainedConfig):
    def __init__(
        self,
        config: ModelConfig,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.tokenizer = config.tokenizer
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_act = config.hidden_act
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_attention_heads = config.num_kv_attention_heads
        self.dropout = config.dropout
        self.feed_forward_size = config.feed_forward_size
        self.rope_theta = config.rope_theta
        self.vocab_size = config.get_vocab_size()
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
