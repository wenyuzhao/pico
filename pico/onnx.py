from pathlib import Path
from optimum.exporters.onnx import main_export
from optimum.exporters.onnx.model_configs import (
    Phi3OnnxConfig,
    TextDecoderWithPositionIdsOnnxConfig,
    NormalizedTextConfig,
)

from transformers import AutoConfig


class CustomOnnxConfig(TextDecoderWithPositionIdsOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        common_inputs = super().inputs

        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
        }


def export_onnx(path: Path, task: str):
    assert path.is_dir()
    print(f"Exporting ONNX model for task '{task}' from checkpoint at {path}...")
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    custom_config = CustomOnnxConfig(
        config=config,
        task=task,
    )
    main_export(
        model_name_or_path=str(path),
        output=path / "onnx",
        task=task,
        trust_remote_code=True,
        custom_onnx_configs={"model": custom_config},
    )
    print(f"ONNX model exported to {path / 'onnx'}")
