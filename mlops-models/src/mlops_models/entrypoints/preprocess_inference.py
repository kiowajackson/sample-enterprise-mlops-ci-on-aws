"""Pass GetData's snapshot and the approved model's metadata to preprocessing."""

from importlib import import_module
import json
from pathlib import Path


def execute(model_id, input_path, output_dir, *, model_dir, shard_rows=10000):
    """Read saved settings and invoke this use case's process_inference."""
    config = json.loads((Path(model_dir) / "parameters.json").read_text())
    if config.get("model_id", model_id) != model_id:
        raise ValueError(f"Artifact belongs to {config['model_id']}, not selected model {model_id}.")
    config["model_metadata"] = json.loads((Path(model_dir) / "model_metadata.json").read_text())
    process = import_module(f"mlops_models.models.{model_id}.inference.preprocess").process_inference
    output, rows = process(Path(input_path), Path(output_dir), config, shard_rows=shard_rows)
    print(json.dumps({"event": "success", "step": "Preprocess", "model": model_id, "rows": rows}), flush=True)
    return output
