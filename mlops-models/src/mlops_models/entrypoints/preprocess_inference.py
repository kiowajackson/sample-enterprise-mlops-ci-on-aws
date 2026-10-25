"""Load registered model metadata and pass the separate parameter path to preprocessing."""

from importlib import import_module
from itertools import islice
from pathlib import Path
import json


def execute(model_id, input_path, selection_path, output_dir, *, model_dir):
    # Temporary diagnostic: show the container's source before importing it.
    script = Path(__file__).resolve().parents[1] / "model_base/inference/preprocess.py"
    with script.open() as source:
        print(f"PREPROCESS SOURCE: {script}\n" + "".join(islice(source, 20)), flush=True)
    parameter_path = Path(__file__).parents[1] / "models" / model_id / "parameters.json"
    config_path = Path(model_dir) / "model_metadata.json"
    config = json.loads(config_path.read_text())
    process_inference = import_module(f"mlops_models.models.{model_id}.inference.preprocess").process_inference
    return process_inference(
        Path(input_path), Path(selection_path), Path(output_dir), config, parameter_path,
    )
