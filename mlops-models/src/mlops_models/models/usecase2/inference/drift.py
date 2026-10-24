"""Inherit the default drift comparison, or customize it for this use case."""

from mlops_models.model_base.inference.drift import compare_drift as base_compare_drift


def compare_drift(data_directory, selection_file, prediction_directory, output_directory, model_directory):
    """Forward this execution's saved inputs and model to the comparison implementation."""
    return base_compare_drift(
        data_directory, selection_file, prediction_directory, output_directory, model_directory,
    )
