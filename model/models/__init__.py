from transformers.generation.utils import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from model.config import Config, ModelConfig
from ..config import PretrainedConfig


class BaseGPTModel(PreTrainedModel, GenerationMixin):
    def __init__(self, config: ModelConfig):
        self.model_config = config
        super().__init__(PretrainedConfig(config))
        self.out = CausalLMOutputWithPast()

    @staticmethod
    def load(config: Config, compile: bool = False) -> "BaseGPTModel":
        match config.model.name:
            case "pixie":
                from .pixie import Pixie

                return Pixie(config.model, compile=compile)
            case _:
                raise ValueError(f"Unknown model name: {config.model.name}")
