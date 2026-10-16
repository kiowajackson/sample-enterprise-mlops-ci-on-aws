"""Use the registered checkpoint and its learned feature mapping to score rows."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from autogluon.tabular import TabularPredictor


def predict_with_model(input_directory, checkpoint_path, output_directory, *, predictor_cls):
    """Read each mounted shard, score it with one loaded model, and write its scores."""
    paths = sorted(Path(input_directory).rglob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No scoring shards in {input_directory}.")
    predictor = predictor_cls(checkpoint_path)
    with TemporaryDirectory(prefix="prediction-input-") as temporary:
        user_data_path = Path(temporary) / "data.csv"
        for path in paths:
            df = get_container_input(path)
            df.to_csv(user_data_path, index=False)
            result = predictor.predict_file(user_data_path)
            if len(result) != len(df):
                raise ValueError(f"Prediction returned {len(result)} rows for {len(df)} input rows.")
            write_container_output(result, Path(output_directory) / (path.name + ".out"))
    return Path(output_directory)


def get_container_input(path):
    """Read one JSON-lines shard that SageMaker downloaded into the container."""
    with Path(path).open() as stream:
        return pd.DataFrame([json.loads(line) for line in stream if line.strip()])


def write_container_output(result, path):
    """Save raw scores locally; load_source.py uploads the output directory to S3."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_json(path, orient="records", lines=True, double_precision=15)
    return path


class ModelPredictor:
    """Replace model loading/prediction together with the model's trainer."""

    def __init__(self, checkpoint_path):
        self.checkpoint_path = Path(checkpoint_path)
        self.config = json.loads((self.checkpoint_path / "parameters.json").read_text())
        self.metadata = json.loads((self.checkpoint_path / "model_metadata.json").read_text())
        self.primary_key = self.config["primary_key"]
        self.feature_mapping = self.metadata.get("feature_mapping")
        if not isinstance(self.feature_mapping, dict) or not self.feature_mapping or any(
            not isinstance(name, str) or not name or not isinstance(mapped, str) or not mapped
            for name, mapped in self.feature_mapping.items()
        ):
            raise ValueError("model_metadata.json must contain a nonempty feature_mapping: "
                             "{input_column: model_column}.")
        if len(set(self.feature_mapping.values())) != len(self.feature_mapping):
            raise ValueError("feature_mapping must not map multiple input columns to one model column.")
        # A copied AML.path is at the archive root. The demo explicitly records predictor/.
        model_path = (self.checkpoint_path / self.metadata.get("predictor_path", ".")).resolve()
        if not model_path.is_relative_to(self.checkpoint_path.resolve()):
            raise ValueError("predictor_path must stay inside the registered model archive.")
        self.predictor = TabularPredictor.load(str(model_path))
        expected = self.predictor.features()
        if set(expected) != set(self.feature_mapping.values()):
            raise ValueError("feature_mapping model columns do not match the loaded predictor's features.")
        print(json.dumps({"event": "prediction_model_loaded", "model_path": str(model_path),
                          "metadata_path": str(self.checkpoint_path / "model_metadata.json"),
                          "features": len(self.feature_mapping), "primary_key": self.primary_key,
                          "task": self.config["task_type"]}), flush=True)

    def predict_file(self, user_data_path) -> pd.DataFrame:
        """File-based integration point for a Processing prediction job.

        Replace this body with the model's code using user_data_path,
        self.feature_mapping, self.primary_key and self.checkpoint_path.
        Return a dataframe with the original primary key and a numeric score.
        """
        df = pd.read_csv(user_data_path, dtype={self.primary_key: str},
                         keep_default_na=False, na_values=["", r"\N"])
        return self.predict_with_best_model(df)

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
        if self.config["task_type"] == "regression":
            values = self.predictor.predict(model_input).to_numpy()
        elif self.config["task_type"] == "classification":
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
    return predictor_cls(checkpoint_path).predict_with_best_model(df, output_dir=output_dir)
