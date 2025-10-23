import inspect
from pathlib import Path
from typing import override
from transformers.generation.utils import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.configuration_utils import PretrainedConfig as _PretrainedConfig
from ._config import *
from torch import nn


class BaseGPTModel[C: ModelConfig](PreTrainedModel, GenerationMixin):
    config_class = _PretrainedConfig
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
        # Copy model definition
        s = f"from pixie.models.{self.model_type} import *"
        save_directory = args[0] if args else kwargs.get("save_directory", None)
        assert save_directory
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "model.py").write_text(s)
        # Save model weights and config
        super().save_pretrained(*args, **kwargs)


MODELS: dict[str, tuple[type[BaseGPTModel], type[ModelConfig]]] = {}


def register_model[T: ModelConfig](name: str, config: type[T]):
    def _register(cls: type[BaseGPTModel]):
        if name in MODELS:
            raise ValueError(f"Model {name} is already registered.")
        MODELS[name] = (cls, config)

        class PretrainedConfigImpl(_PretrainedConfig):
            model_type = name
            has_no_defaults_at_init = True

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.auto_map = {
                    "AutoConfig": "model._Config",
                    "AutoModel": "model._Model",
                    "AutoModelForCausalLM": "model._Model",
                }

        cls.config_class = PretrainedConfigImpl
        cls.model_type = name
        caller = inspect.stack()[1]
        caller.frame.f_globals["_Config"] = cls.config_class
        caller.frame.f_globals["_Model"] = cls
        if "__all__" in caller.frame.f_globals:
            caller.frame.f_globals["__all__"].extend(["_Config", "_Model"])
        else:
            caller.frame.f_globals["__all__"] = ["_Config", "_Model"]

        return cls

    return _register
