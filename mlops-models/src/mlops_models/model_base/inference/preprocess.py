"""Read the data and selected-feature dataframes, prepare them, and write CSV."""

from pathlib import Path
import json

import pandas as pd


def preprocess_data(data_directory, selection_file, output_directory, config, parameter_path,
                    *, analysis_mode="prediction_inference", inference_fn=None, test_fn=None):
    """Route the requested mode, using supplied model functions or the base defaults."""
    print(f"Preprocess: {analysis_mode}", flush=True)
    if analysis_mode == "prediction_inference":
        process = process_inference if inference_fn is None else inference_fn
    elif analysis_mode == "prediction_test":
        process = process_test if test_fn is None else test_fn
    else:
        raise ValueError("analysis_mode must be prediction_inference or prediction_test.")
    return process(data_directory, selection_file, output_directory, config, parameter_path)


def process_inference(data_directory, selection_file, output_directory, config, parameter_path):
    """Accept metadata and a separate parameter path; publish output through files."""
    parameters = json.loads(Path(parameter_path).read_text())
    df, selected_features = get_container_input(data_directory, selection_file, parameters)
    columns = list(dict.fromkeys([parameters["primary_key"], *selected_features.iloc[:, 0]]))
    timestamp = parameters.get("timestamp")
    if timestamp in df.columns and timestamp not in columns:
        columns.insert(1, timestamp)
    # Training labels are retained when supplied; scoring data can be unlabeled.
    target = parameters.get("target_key")
    if target in df.columns and target not in columns:
        columns.append(target)
    write_container_output(df[columns], output_directory)


def process_test(data_directory, selection_file, output_directory, config, parameter_path):
    """Replace with the model's test preprocessing; the demo uses the same CSV preparation."""
    return process_inference(data_directory, selection_file, output_directory, config, parameter_path)


def get_container_input(data_directory, selection_file, config):
    """Read raw data and selected features from the two local job inputs."""
    df = pd.read_csv(Path(data_directory) / "data.csv",
                     dtype={config["primary_key"]: str},
                     keep_default_na=False, na_values=["", r"\N"])
    return df, pd.read_csv(selection_file)


def write_container_output(df, output_directory):
    """Save prepared/data.csv locally; load_source.py uploads prepared/ to S3."""
    output = Path(output_directory) / "prepared"
    output.mkdir(parents=True, exist_ok=True)
    df.to_csv(output / "data.csv", index=False)
