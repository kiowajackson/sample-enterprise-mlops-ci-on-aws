"""Run the selected model's test analysis from explicit mounted inputs."""

from pathlib import Path

from mlops_models.entrypoints import load_step


def execute(model_id, train_data_path, test_data_path, selection_file, predictions,
            output_directory):
    """Pass training/test folders, a selection CSV and Score/output folders.

    The model function appends data.csv to each data folder.
    """
    prediction_test = load_step(model_id, "postprocess", "prediction_test")
    return prediction_test(
        Path(train_data_path),
        Path(test_data_path),
        Path(selection_file),
        Path(predictions),
        Path(output_directory),
    )
