"""Pass explicit data, selection and model metadata to inference preprocessing."""

from pathlib import Path
import json

from mlops_models.entrypoints import load_step


def execute(model_id, input_path, selection_path, output_dir, *, model_dir):
    parameter_path = Path(__file__).parents[1] / "models" / model_id / "parameters.json"
    config = json.loads((Path(model_dir) / "model_metadata.json").read_text())
    process = load_step(model_id, "preprocess", "process_inference")
    return process(Path(input_path), Path(selection_path), Path(output_dir), config, parameter_path)
