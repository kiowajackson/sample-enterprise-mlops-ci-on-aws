"""Use the registered checkpoint and its learned feature mapping to score rows."""

from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor


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


class ModelPredictor:
    """Replace model loading/prediction together with the model's trainer."""

    def __init__(self, checkpoint_path, config, parameters):
        self.checkpoint_path = Path(checkpoint_path)
        self.config = deepcopy(config)
        self.parameters = deepcopy(parameters)
        self.primary_key = self.parameters["primary_key"]
        self.feature_mapping = self.config.get("feature_mapping")
        if not isinstance(self.feature_mapping, dict) or not self.feature_mapping or any(
            not isinstance(name, str) or not name or not isinstance(mapped, str) or not mapped
            for name, mapped in self.feature_mapping.items()
        ):
            raise ValueError("model_metadata.json must contain a nonempty feature_mapping: "
                             "{input_column: model_column}.")
        if len(set(self.feature_mapping.values())) != len(self.feature_mapping):
            raise ValueError("feature_mapping must not map multiple input columns to one model column.")
        # A copied AML.path is at the archive root. The demo explicitly records predictor/.
        model_path = (self.checkpoint_path / self.config.get("predictor_path", ".")).resolve()
        if not model_path.is_relative_to(self.checkpoint_path.resolve()):
            raise ValueError("predictor_path must stay inside the registered model archive.")
        self.predictor = TabularPredictor.load(str(model_path))
        expected = self.predictor.features()
        # The fitted predictor may have discarded constant or redundant inputs.
        if not set(expected).issubset(self.feature_mapping.values()):
            raise ValueError("feature_mapping does not cover the loaded predictor's required features.")
        print(json.dumps({"event": "prediction_model_loaded", "model_path": str(model_path),
                          "metadata_path": str(self.checkpoint_path / "model_metadata.json"),
                          "features": len(self.feature_mapping), "primary_key": self.primary_key,
                          "task": self.parameters["task_type"]}), flush=True)

    def predict_file(self, user_test_data_path, feature_mapping, primary_key, *, output_directory):
        """File-based integration point for a Processing prediction job.

        Replace this body with the model's code using these three inputs.
        Set its output_dir to output_directory and keep its existing CSV writes.
        self.checkpoint_path points to the approved model's local artifacts.
        No return value is needed. Files are published without rewriting their data.
        The demo has no explanation algorithm; its messaging contains identifiers only.
        """
        df = pd.read_csv(user_test_data_path, dtype={primary_key: str},
                         keep_default_na=False, na_values=["", r"\N"])
        predictions = self.predict_with_best_model(df[[primary_key, *feature_mapping]])
        messaging = df[[primary_key]].copy()
        output_dir = Path(output_directory)
        predictions.to_csv(output_dir / "predictions.csv", index=False)
        messaging.to_csv(output_dir / "messaging.csv", index=False)
        predictions.to_csv(output_dir / "messaging_final.csv", index=False)

    def predict_with_best_model(self, df, *, output_dir=None) -> pd.DataFrame:
        """Return one dataframe containing the original identifiers and model scores."""
        required = [self.primary_key, *self.feature_mapping]
        missing = [name for name in required if name not in df]
        if missing:
            raise ValueError(f"Scoring input is missing {len(missing)} required columns: {missing[:20]}. "
                             "Column names are case-sensitive; apply the training normalization "
                             "in the model's predictor.")
        # Replace this selection/rename with the model's saved inference transformations.
        model_input = df[list(self.feature_mapping)].rename(columns=self.feature_mapping)
        if output_dir is not None:
            output = Path(output_dir)
            output.mkdir(parents=True, exist_ok=True)
            model_input.to_csv(output / "X.csv", index_label="row_index")
            df[[self.primary_key]].to_csv(output / "rows.csv", index_label="row_index")
        if self.parameters["task_type"] == "regression":
            values = self.predictor.predict(model_input).to_numpy()
        elif self.parameters["task_type"] == "classification":
            if self.predictor.problem_type != "binary":
                raise ValueError("This score adapter supports binary classification; "
                                 "supply a multiclass prediction/output adapter for other tasks.")
            values = self.predictor.predict_proba(model_input, as_multiclass=True)[
                self.predictor.positive_class
            ].to_numpy()
        else:
            raise ValueError("task_type must be classification or regression.")
        return pd.DataFrame({
            self.primary_key: df[self.primary_key].astype(str).to_numpy(),
            "score": values,
        })


def predict_model(df, checkpoint_path, *, predictor_cls=ModelPredictor, output_dir=None):
    """Score an evaluation dataframe without the batch-file handoff."""
    checkpoint_path = Path(checkpoint_path)
    config = json.loads((checkpoint_path / "model_metadata.json").read_text())
    parameters = json.loads((checkpoint_path / "parameters.json").read_text())
    return predictor_cls(checkpoint_path, config, parameters).predict_with_best_model(df, output_dir=output_dir)
