"""Call the selected use case's batch prediction function."""

from importlib import import_module
import json
from pathlib import Path


def execute(model_id, input_path, model_dir, output_path):
    parameter_path = Path(__file__).parents[1] / "models" / model_id / "parameters.json"
    config_path = Path(model_dir) / "model_metadata.json"
    config = json.loads(config_path.read_text())
    predict_with_model = import_module(f"mlops_models.models.{model_id}.inference.predict").predict_with_model
    return predict_with_model(
        Path(input_path), Path(model_dir), Path(output_path), config, parameter_path,
    )
