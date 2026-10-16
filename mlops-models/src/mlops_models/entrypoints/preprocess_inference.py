"""Pass GetData's snapshot and the approved model's metadata to preprocessing."""

from importlib import import_module
import csv
import json
from pathlib import Path


def execute(model_id, input_path, output_dir, *, model_dir):
    """Supply the same five arguments as training, using registered metadata."""
    parameter_path = Path(__file__).parents[1] / "models" / model_id / "parameters.json"
    config_path = Path(model_dir) / "model_metadata.json"
    model_parameters = json.loads((Path(model_dir) / "parameters.json").read_text())
    if model_parameters.get("model_id", model_id) != model_id:
        raise ValueError(f"Artifact belongs to {model_parameters['model_id']}, not selected model {model_id}.")
    config = json.loads(config_path.read_text())

    # Reconstruct training's selection-file input from this model's saved mapping.
    mapping = config["feature_mapping"]
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("model_metadata.json needs a nonempty feature_mapping.")
    selection_file = Path(output_dir) / "selection" / "selected_features.csv"
    selection_file.parent.mkdir(parents=True, exist_ok=True)
    with selection_file.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["features", "description"])
        writer.writerows(mapping.items())

    process = import_module(f"mlops_models.models.{model_id}.inference.preprocess").process_inference
    # The loader uploads written files; a customer implementation may return None.
    return process(
        Path(input_path), selection_file, Path(output_dir), config, parameter_path,
    )
