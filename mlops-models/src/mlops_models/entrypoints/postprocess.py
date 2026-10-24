"""Call the selected use case's prediction postprocessing function."""

from importlib import import_module
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory


def execute(model_id, input_path, output_dir, *, model_dir, data_directory, selection_file,
            chunk_rows=10000, analysis_mode="prediction_inference"):
    """Pass the original snapshot, saved selection and scores to the model's analysis."""
    postprocess = import_module(f"mlops_models.models.{model_id}.inference.postprocess")
    prediction_format = os.getenv("MLOPS_PREDICTION_FORMAT", "csv")
    if prediction_format not in {"csv", "jsonlines"}:
        raise ValueError("MLOPS_PREDICTION_FORMAT must be csv or jsonlines.")
    if analysis_mode not in {"prediction_inference", "prediction_test"}:
        raise ValueError("analysis_mode must be prediction_inference or prediction_test.")
    print(json.dumps({"event": "postprocess_analysis", "analysis_mode": analysis_mode,
                      "prediction_format": prediction_format}), flush=True)
    with TemporaryDirectory(prefix="postprocess-inputs-") as temporary:
        # Ordinary Processing supplies CSVs unchanged. Only native Batch Transform
        # needs its JSON-lines response converted before entering the same analysis.
        if prediction_format == "jsonlines":
            from mlops_models.model_base.inference.postprocess import convert_transform_predictions
            convert_transform_predictions(input_path, temporary, model_dir, chunk_rows)
            input_path = temporary
        output = postprocess.postprocess_predictions(
            Path(data_directory), Path(selection_file), Path(input_path), Path(output_dir),
            Path(model_dir), analysis_mode=analysis_mode,
        )
    print(json.dumps({"event": "success", "step": "Postprocess", "model": model_id,
                      "output": str(output)}), flush=True)
    return output
