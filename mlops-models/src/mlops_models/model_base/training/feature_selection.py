"""Shared feature-selection script: read a dataframe, fit, export CSV/PDF."""

from pathlib import Path
import json

import pandas as pd
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def run_feature_selection(data_directory, output_directory, config, *, selector_cls):
    """The callable equivalent of the feature-selection script's main block."""
    df = get_container_input(data_directory, config)
    selector = selector_cls(config)
    selector.fit(df)
    return write_container_output(selector, output_directory)


def get_container_input(data_directory, config):
    """Read the raw CSV downloaded into this job by SageMaker."""
    return pd.read_csv(Path(data_directory) / "data.csv",
                       dtype={config["primary_key"]: str},
                       keep_default_na=False, na_values=["", r"\N"])


def write_container_output(selector, output_directory):
    """Export locally into selection/; load_source.py uploads that folder to S3."""
    output = Path(output_directory) / "selection"
    output.mkdir(parents=True, exist_ok=True)
    selector.export_features_to_csv(output / "selected_features.csv")
    selector.export_to_pdf(output / "random_forest_plots.pdf")
    # The demo selector measures its fitted matrix; custom selectors may supply their own metrics.
    if getattr(selector, "metrics", None) is not None:
        (output / "metrics.json").write_text(json.dumps(selector.metrics, indent=2) + "\n")
        print("Feature selection:", json.dumps(selector.metrics), flush=True)
    return output


class FeatureSelector:
    """Replace this demo forest and exporters with the model's feature selector."""

    def __init__(self, config):
        self.config = config

    def fit(self, df):
        """Fit a small forest and retain the selected feature importances."""
        config = self.config
        if config["target_key"] not in df.columns:
            raise ValueError(f"Training target {config['target_key']!r} is missing from the input CSV. "
                             "Set target_key in this use case's parameters.json to the actual column name.")
        if config["task_type"] not in {"classification", "regression"}:
            raise ValueError("task_type must be classification or regression.")
        settings = config.get("feature_selection", {})
        sample = df.sample(n=min(len(df), settings.get("sample_rows", 1000)),
                           random_state=config.get("seed", 42))
        candidates = sample.drop(columns=[
            config["primary_key"], config["target_key"], config.get("timestamp", ""),
        ], errors="ignore")
        features = config.get("features", {
            "numeric": candidates.select_dtypes(include="number").columns.tolist(),
            "categorical": candidates.select_dtypes(exclude="number").columns.tolist(),
        })
        numeric = sample[features["numeric"]].fillna(0).astype(float)
        categorical = sample[features["categorical"]].fillna("__missing__").astype(str)
        encoded = pd.DataFrame({name: pd.factorize(categorical[name], sort=True)[0]
                                for name in categorical}, index=sample.index)
        values = pd.concat([numeric, encoded], axis=1)
        estimator = (RandomForestClassifier if config["task_type"] == "classification"
                     else RandomForestRegressor)
        model_rf = estimator(n_estimators=settings.get("n_estimators", 40),
                             random_state=config.get("seed", 42), n_jobs=config.get("num_cpus", 2))
        model_rf.fit(values, sample[config["target_key"]])
        self.selected = pd.Series(model_rf.feature_importances_, index=values.columns).sort_values(
            ascending=False, kind="stable").head(settings.get("top_k", 50))
        self.metrics = {
            "input_rows": len(df), "selection_rows": len(sample),
            "input_predictors": len(candidates.columns),
            "candidate_features": len(values.columns), "selected_features": len(self.selected),
        }

    def export_features_to_csv(self, path):
        pd.DataFrame({"feature": self.selected.index}).to_csv(path, index=False)

    def export_to_pdf(self, path):
        figure = Figure(figsize=(9, 6), layout="constrained")
        axes = figure.subplots()
        self.selected.head(30).sort_values().plot.barh(ax=axes)
        axes.set_xlabel("Random forest feature importance")
        figure.savefig(path)
