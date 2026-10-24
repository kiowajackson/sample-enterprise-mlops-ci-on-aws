"""Resolve inputs, run this use case's analysis, and write its outputs."""

from mlops_models.model_base.inference.postprocess import get_container_input, write_container_output
from mlops_models.model_base.inference.postprocess import prediction_inference as example_inference
from mlops_models.model_base.inference.postprocess import prediction_test as example_test


def postprocess_predictions(data_directory, selection_file, prediction_directory, output_directory,
                            model_directory, *, analysis_mode="prediction_inference"):
    data_path, selection_path, predictions, output, parameters = get_container_input(
        data_directory, selection_file, prediction_directory, output_directory, model_directory)
    if analysis_mode == "prediction_inference":
        prediction_inference(data_path, selection_path, predictions, output, parameters)
    elif analysis_mode == "prediction_test":
        prediction_test(data_path, selection_path, predictions, output, parameters)
    else:
        raise ValueError("analysis_mode must be prediction_inference or prediction_test.")
    return write_container_output(
        output, data_path=data_path, selection_file=selection_path,
        predictions=predictions, parameters=parameters)


def prediction_inference(data_path, selection_file, predictions, output_directory, parameters):
    """Replace this example call with the customer's prediction_inference.

    Read project/test YMID from parameters using the customer's keys. Pass
    data_path, selection_file and the required predictions[...] paths explicitly;
    set the customer's destination to output_directory.
    """
    example_inference(data_path, selection_file, predictions, output_directory, parameters)


def prediction_test(data_path, selection_file, predictions, output_directory, parameters):
    """Replace this placeholder with the customer's prediction_test, using the same inputs."""
    example_test(data_path, selection_file, predictions, output_directory, parameters)
