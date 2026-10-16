"""Load model metadata and pass the separate parameter path to process_train."""

from importlib import import_module
from pathlib import Path
import json


def execute(model_id, input_path, selection_path, output_dir):
    parameter_path = Path(__file__).parents[1] / "models" / model_id / "parameters.json"
    config_path = Path(__file__).parents[1] / "models" / model_id / "model_metadata.json"
    config = json.loads(config_path.read_text())
    process_train = import_module(f"mlops_models.models.{model_id}.training.preprocess").process_train
    return process_train(
        Path(input_path), Path(selection_path), Path(output_dir), config, parameter_path,
    )
