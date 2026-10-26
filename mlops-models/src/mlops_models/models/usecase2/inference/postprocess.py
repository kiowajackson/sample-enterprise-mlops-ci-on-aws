"""Use shared input/output and routing with this use case's replaceable analysis."""

from mlops_models.model_base.inference.postprocess import (
    postprocess_predictions as base_postprocess_predictions, parameters,
)
from mlops_models.model_base.inference.postprocess import prediction_inference as example_inference
from mlops_models.model_base.inference.postprocess import prediction_test as example_test


def postprocess_predictions(data_directory, selection_file, prediction_directory, output_directory,
                            model_directory, *, analysis_mode="prediction_inference", chunk_rows=10000):
    """Delegate shared behavior to the base with this use case's analysis functions."""
    return base_postprocess_predictions(
        data_directory, selection_file, prediction_directory, output_directory, model_directory,
        analysis_mode=analysis_mode, chunk_rows=chunk_rows,
        inference_fn=prediction_inference, test_fn=prediction_test,
    )


def prediction_inference(data_path, selection_file, predictions, output_directory):
    """Replace this example call with the customer's prediction_inference.

    Read project/test YMID from the module-level parameters using the customer's keys. Pass
    data_path, selection_file and the required predictions[...] paths explicitly;
    set the customer's destination to output_directory.
    """
    example_inference(data_path, selection_file, predictions, output_directory)


def prediction_test(data_path, selection_file, predictions, output_directory):
    """Replace this placeholder with the customer's prediction_test, using the same inputs."""
    example_test(data_path, selection_file, predictions, output_directory)
