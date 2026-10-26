"""Inherit scoring preprocessing, or replace it with this model's implementation."""

from mlops_models.model_base.inference.preprocess import (
    preprocess_data as base_preprocess_data, process_inference, process_test,
)


def preprocess_data(data_directory, selection_file, output_directory, config, parameter_path,
                    *, analysis_mode="prediction_inference"):
    """Delegate routing to the base with this use case's replaceable functions."""
    return base_preprocess_data(
        data_directory, selection_file, output_directory, config, parameter_path,
        analysis_mode=analysis_mode, inference_fn=process_inference, test_fn=process_test,
    )
