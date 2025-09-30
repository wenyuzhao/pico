from transformers.generation.utils import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from model.config import ModelConfig
from ..config import PretrainedConfig
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM


class BaseGPTModel(PreTrainedModel, GenerationMixin):
    config_class = PretrainedConfig

    def __init__(self, config: ModelConfig):
        self.model_config = config
        tok = config.load_tokenizer()
        super().__init__(PretrainedConfig(**config.to_dict()))
        assert self.generation_config
        gcfg = self.model_config.generation
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

        class PretrainedConfig2(PretrainedConfig):
            model_type = name

        cls.config_class = PretrainedConfig2
        AutoConfig.register(name, PretrainedConfig2)
        AutoModelForCausalLM.register(PretrainedConfig2, cls)
        return cls

    return _register


from . import pixie, pixie_x
