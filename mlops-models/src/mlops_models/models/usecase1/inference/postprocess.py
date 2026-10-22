"""Run this use case's analysis with explicit file paths and an output directory."""

from mlops_models.model_base.inference.postprocess import postprocess
from mlops_models.model_base.inference.postprocess import postprocess_predictions as base_postprocess_predictions


def postprocess_predictions(data_directory, selection_file, prediction_directory, output_directory, config, *, chunk_rows=10000, prediction_format="csv"):
    """Pass postprocess directly as the analysis callable; replace its import to customize."""
    return base_postprocess_predictions(
        data_directory, selection_file, prediction_directory, output_directory, config,
        run_analysis=postprocess, chunk_rows=chunk_rows, prediction_format=prediction_format,
    )
