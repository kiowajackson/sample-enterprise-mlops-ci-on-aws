"""Inherit scoring preprocessing, or replace it with this model's implementation."""

from mlops_models.model_base.inference.preprocess import process_inference, process_test


def preprocess_data(data_directory, selection_file, output_directory, config, parameter_path,
                    *, analysis_mode="prediction_inference"):
    """Route the requested mode to this model's preprocessing implementation."""
    print(f"Preprocess: {analysis_mode}", flush=True)
    if analysis_mode == "prediction_inference":
        return process_inference(data_directory, selection_file, output_directory, config, parameter_path)
    if analysis_mode == "prediction_test":
        return process_test(data_directory, selection_file, output_directory, config, parameter_path)
    raise ValueError("analysis_mode must be prediction_inference or prediction_test.")
