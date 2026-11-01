"""Preprocess prepared training, test or inference inputs using explicit paths."""

import json
from pathlib import Path
import pandas as pd


def process_train(data_directory, selection_file, output_directory, config, parameter_path):
    return prepare_data(data_directory, selection_file, output_directory, parameter_path)


def process_test(data_directory, selection_file, output_directory, config, parameter_path):
    return prepare_data(data_directory, selection_file, output_directory, parameter_path)


def process_inference(data_directory, selection_file, output_directory, config, parameter_path):
    return prepare_data(data_directory, selection_file, output_directory, parameter_path)


def prepare_data(data_directory, selection_file, output_directory, parameter_path):
    """Example preprocessing; replace the relevant function with customer code."""
    parameters = json.loads(Path(parameter_path).read_text())
    data = pd.read_csv(Path(data_directory) / "data.csv",
                       dtype={parameters["primary_key"]: str},
                       keep_default_na=False, na_values=["", r"\N"])
    selection = pd.read_csv(selection_file)
    columns = list(dict.fromkeys([parameters["primary_key"], *selection.iloc[:, 0]]))
    timestamp = parameters.get("timestamp")
    if timestamp in data.columns and timestamp not in columns:
        columns.insert(1, timestamp)
    target = parameters.get("target_key")
    if target in data.columns and target not in columns:
        columns.append(target)
    output = Path(output_directory) / "prepared"
    output.mkdir(parents=True, exist_ok=True)
    data[columns].to_csv(output / "data.csv", index=False)
    return output
