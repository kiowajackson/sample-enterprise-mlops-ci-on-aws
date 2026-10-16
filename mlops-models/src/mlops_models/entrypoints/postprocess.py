"""Call the selected use case's prediction postprocessing function."""

from importlib import import_module
import json
import os
from pathlib import Path
import tarfile


def execute(model_id, input_path, output_dir, *, model_dir, data_directory, selection_file, chunk_rows=10000):
    """Pass the original snapshot, saved selection and scores to the model's analysis."""
    postprocess = import_module(f"mlops_models.models.{model_id}.inference.postprocess")
    output = postprocess.postprocess_predictions(
        Path(data_directory), Path(selection_file), Path(input_path), Path(output_dir),
        load_model_config(model_dir), chunk_rows=chunk_rows,
        prediction_format=os.getenv("MLOPS_PREDICTION_FORMAT", "csv"),
    )
    print(json.dumps({"event": "success", "step": "Postprocess", "model": model_id,
                      "output": str(output)}), flush=True)
    return output


def load_model_config(model_dir):
    """Read the configuration retained with the approved checkpoint."""
    model_dir = Path(model_dir)
    if model_dir.is_file():
        with tarfile.open(model_dir) as archive:
            config = json.load(archive.extractfile("parameters.json"))
            config["model_metadata"] = json.load(archive.extractfile("model_metadata.json"))
    else:
        config = json.loads((model_dir / "parameters.json").read_text())
        config["model_metadata"] = json.loads((model_dir / "model_metadata.json").read_text())
    return config
