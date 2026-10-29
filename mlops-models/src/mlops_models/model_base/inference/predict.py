"""Use the registered checkpoint and its learned feature mapping to score rows."""

import json
from pathlib import Path

from ...frameworks.autogluon import gluonPredictor as ModelPredictor


def predict_with_model(input_directory, checkpoint_path, output_directory, config, parameter_path):
    """Pass inputs and a writable output directory to the model; retain its files."""
    parameters = json.loads(Path(parameter_path).read_text())
    user_test_data_path, feature_mapping, primary_key = get_container_input(input_directory, config, parameters)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    # Replace these calls with the customer's prediction code and its own writes.
    predictor = ModelPredictor(checkpoint_path, config, parameters)
    predictor.predict_file(user_test_data_path, feature_mapping, primary_key,
                           output_directory=output_directory)
    return write_container_output(output_directory)


def get_container_input(input_directory, config, parameters):
    """Return the prepared CSV path, saved feature mapping and primary key."""
    path = Path(input_directory) / "data.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Score needs Preprocess's prepared CSV: {path}")
    return path, config["feature_mapping"], parameters["primary_key"]


def write_container_output(output_directory):
    """Find and log every CSV, preserving subfolders for the loader's upload."""
    output = Path(output_directory)
    csv_files = sorted(path for path in output.rglob("*")
                       if path.is_file() and path.suffix.lower() == ".csv")
    if not csv_files:
        raise FileNotFoundError(f"Prediction did not write any CSV files under: {output}")
    for path in csv_files:
        print(json.dumps({"event": "prediction_output_csv", "path": str(path),
                          "relative_path": path.relative_to(output).as_posix()}), flush=True)
    return output
