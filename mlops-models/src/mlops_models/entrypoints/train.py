"""Load the selected use case's settings and call its train_model function."""

from importlib import import_module
import json
from pathlib import Path


def execute(model_id, input_path, model_dir, output_dir, *, selection_path, hyperparameters=None):
    selection_file = Path(selection_path)
    if not selection_file.is_file():
        raise FileNotFoundError(f"Selected-feature CSV is missing at the supplied path: {selection_file}")
    print(f"Training inputs: prepared={Path(input_path) / 'prepared'}, selection={selection_file}",
          flush=True)
    parameter_path = Path(__file__).parents[1] / "models" / model_id / "parameters.json"
    config_path = Path(__file__).parents[1] / "models" / model_id / "model_metadata.json"
    config = json.loads(config_path.read_text())
    train_model = import_module(f"mlops_models.models.{model_id}.training.train").train_model
    return train_model(
        Path(input_path) / "prepared",
        selection_file,
        Path(model_dir), Path(output_dir), config, parameter_path, hyperparameters=hyperparameters,
    )
