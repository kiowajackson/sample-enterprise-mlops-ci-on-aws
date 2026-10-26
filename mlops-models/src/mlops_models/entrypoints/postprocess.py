"""Call the selected use case's prediction postprocessing function."""

from importlib import import_module
from pathlib import Path


def execute(model_id, input_path, output_dir, *, model_dir, data_directory, selection_file,
            chunk_rows=10000, analysis_mode="prediction_inference"):
    """Pass the original snapshot, saved selection and scores to the model's analysis."""
    postprocess = import_module(f"mlops_models.models.{model_id}.inference.postprocess")
    print(f"Postprocess: {analysis_mode} for {model_id}", flush=True)
    output = postprocess.postprocess_predictions(
        Path(data_directory), Path(selection_file), Path(input_path), Path(output_dir),
        Path(model_dir), analysis_mode=analysis_mode, chunk_rows=chunk_rows,
    )
    print(f"Postprocess complete. Output: {output}", flush=True)
    return output
