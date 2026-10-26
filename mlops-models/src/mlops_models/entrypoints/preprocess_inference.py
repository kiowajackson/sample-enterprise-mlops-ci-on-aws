"""Pass registered metadata, parameter path and mode to the model's preprocessing router."""

from importlib import import_module
from pathlib import Path
import json


def execute(model_id, input_path, selection_path, output_dir, *, model_dir, analysis_mode="prediction_inference"):
    parameter_path = Path(__file__).parents[1] / "models" / model_id / "parameters.json"
    config_path = Path(model_dir) / "model_metadata.json"
    config = json.loads(config_path.read_text())
    preprocess = import_module(f"mlops_models.models.{model_id}.inference.preprocess")
    return preprocess.preprocess_data(
        Path(input_path), Path(selection_path), Path(output_dir), config, parameter_path,
        analysis_mode=analysis_mode,
    )
