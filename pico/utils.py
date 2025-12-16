from pico.models._base import Config, ModelConfig, BaseCasualLM, MODELS
from pathlib import Path
import yaml
from typing import Any


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


def load_config(path: str | Path) -> Config:
    data = load_yaml_and_resolve_imports(path)
    model_name = data.get("model", {}).get("name", None)
    assert model_name, "Model name must be specified in the configuration."
    # Create model config
    _, ConfigType = MODELS[model_name]
    mc = ConfigType(**data.get("model", {}))
    del data["model"]
    data["model"] = mc
    config = Config(**data)
    if config.name is None:
        config.name = Path(path).stem
    return config


def load_model(config: ModelConfig) -> BaseCasualLM:
    assert config.name in MODELS, f"Unknown model name: {config.name}"
    Model, Config = MODELS[config.name]
    assert isinstance(config, Config)
    return Model(config)


def save_config(config: Config, path: str | Path):
    with open(path, "w") as f:
        yaml.safe_dump(config.model_dump(), f)
