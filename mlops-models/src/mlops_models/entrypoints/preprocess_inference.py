"""Load registered model metadata and pass the separate parameter path to preprocessing."""

from importlib import import_module
from pathlib import Path
import json


def execute(model_id, input_path, selection_path, output_dir, *, model_dir):
    parameter_path = Path(__file__).parents[1] / "models" / model_id / "parameters.json"
    config_path = Path(model_dir) / "model_metadata.json"
    config = json.loads(config_path.read_text())
    process_inference = import_module(f"mlops_models.models.{model_id}.inference.preprocess").process_inference
    return process_inference(
        Path(input_path), Path(selection_path), Path(output_dir), config, parameter_path,
    )
