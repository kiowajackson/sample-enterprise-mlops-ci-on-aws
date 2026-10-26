"""Resolve inputs, run this use case's analysis, and write its outputs."""

from tempfile import TemporaryDirectory

from mlops_models.model_base.inference.postprocess import get_container_input, write_container_output, parameters
from mlops_models.model_base.inference.postprocess import prediction_inference as example_inference
from mlops_models.model_base.inference.postprocess import prediction_test as example_test


def postprocess_predictions(data_directory, selection_file, prediction_directory, output_directory,
                            model_directory, *, analysis_mode="prediction_inference", chunk_rows=10000):
    """Read inputs, initialize parameters, run the chosen analysis, and write outputs."""
    if analysis_mode not in {"prediction_inference", "prediction_test"}:
        raise ValueError("analysis_mode must be prediction_inference or prediction_test.")
    with TemporaryDirectory(prefix="postprocess-inputs-") as converted_directory:
        data_path, selection_path, predictions, output, _ = get_container_input(
            data_directory, selection_file, prediction_directory, output_directory, model_directory,
            converted_directory=converted_directory, chunk_rows=chunk_rows)
        if analysis_mode == "prediction_inference":
            prediction_inference(data_path, selection_path, predictions, output)
        else:
            prediction_test(data_path, selection_path, predictions, output)
        return write_container_output(
            output, data_path=data_path, selection_file=selection_path,
            predictions=predictions, parameters=parameters)


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
