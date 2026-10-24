"""Call the selected use case's drift comparison function."""

from importlib import import_module
from pathlib import Path


def execute(model_id, input_path, output_dir, *, model_dir, data_directory, selection_file):
    """Pass the saved snapshot, selection, scores and model to the comparison step."""
    compare_drift = import_module(f"mlops_models.models.{model_id}.inference.drift").compare_drift
    return compare_drift(
        Path(data_directory), Path(selection_file), Path(input_path), Path(output_dir), Path(model_dir),
    )
