"""Format predictions for this use case's downstream consumers."""

from mlops_models.model_base.inference.postprocess import postprocess
from mlops_models.model_base.inference.postprocess import postprocess_predictions as base_postprocess_predictions


def use_case_postprocess(data_path, selected_features, result, config):
    """Take raw model scores and settings; return this use case's formatted prediction dataframe."""
    return postprocess(data_path, selected_features, result, config)


def postprocess_predictions(data_directory, selection_file, prediction_directory, output_directory, config, *, chunk_rows=10000, prediction_format="csv"):
    """Inherit container I/O while allowing this use case to replace its formatter."""
    return base_postprocess_predictions(
        data_directory, selection_file, prediction_directory, output_directory, config,
        format_predictions=use_case_postprocess, chunk_rows=chunk_rows, prediction_format=prediction_format,
    )
