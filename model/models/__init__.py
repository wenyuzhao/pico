from transformers.generation.utils import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from model.config import ModelConfig
from ..config import PretrainedConfig
from torch import nn


class BaseGPTModel(PreTrainedModel, GenerationMixin):
    def __init__(self, config: ModelConfig):
        self.model_config = config
        super().__init__(PretrainedConfig(config))
        self.out = CausalLMOutputWithPast()
        self.model: nn.Module

    @staticmethod
    def load(config: ModelConfig) -> "BaseGPTModel":
        assert config.name in MODELS, f"Unknown model name: {config.name}"
        Model, Config = MODELS[config.name]
        assert isinstance(config, Config)
        return Model(config)


MODELS: dict[str, tuple[type[BaseGPTModel], type[ModelConfig]]] = {}


def register_model(name: str, config: type[ModelConfig]):
    def _register(cls: type[BaseGPTModel]):
        if name in MODELS:
            raise ValueError(f"Model {name} is already registered.")
        MODELS[name] = (cls, config)

    return _register


from . import pixie, pixie_x
