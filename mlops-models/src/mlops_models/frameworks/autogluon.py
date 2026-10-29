"""AutoGluon training, prediction and evaluation metric conversion."""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
from shutil import copyfile

import pandas as pd
from autogluon.tabular import TabularPredictor
from autogluon.core.metrics import get_metric


def evaluation_report(performance, parameters, data_usage):
    """Format this demo's AutoGluon scores; model packaging does not require them."""
    if not isinstance(performance, dict) or not performance:
        raise ValueError("Pass the nonempty performance dictionary returned by AML.evaluate(...).")
    if not data_usage:
        raise ValueError("Describe the scored data, for example validation or prepared_input_smoke_check.")
    if parameters["task_type"] not in {"classification", "regression"}:
        raise ValueError("task_type must be classification or regression.")
    aliases = {"root_mean_squared_error": "rmse", "mean_squared_error": "mse",
               "mean_absolute_error": "mae"}
    raw, metrics = {}, {}
    for name, score in performance.items():
        value = float(score)
        if isinstance(score, bool) or not math.isfinite(value):
            raise ValueError(f"Evaluation metric {name} must be a finite number, got {score!r}.")
        metric = get_metric(name)
        raw[name] = value
        metrics[aliases.get(name, name)] = float(metric.convert_score_to_original(value))
    return {"model_id": parameters["model_id"], "task": parameters["task_type"],
            "data_usage": data_usage, "metric_source": "autogluon.TabularPredictor.evaluate",
            "raw_metrics": raw, "metrics": metrics}


class gluonTraining:
    """Accept input paths, then own the demo's entire training implementation."""

    def __init__(self, user_feature_path, user_data_path, config, parameters):
        self.feature_path = Path(user_feature_path)
        self.data_path = Path(user_data_path)
        self.config = deepcopy(config)
        self.parameters = deepcopy(parameters)

    def train_model(self, checkpoint_path, output):
        """Demo fit; replace this method with the model's training implementation."""
        parameters = self.parameters
        final_model = os.getenv("MLOPS_TRAINING_MODE", "1") == "2"
        df = pd.concat([pd.read_csv(path, dtype={parameters["primary_key"]: str},
                                    keep_default_na=False, na_values=["", r"\N"])
                        for path in sorted(self.data_path.glob("*.csv"))], ignore_index=True)
        selected = pd.read_csv(self.feature_path)["feature"].tolist()
        parameters["features"] = {
            "numeric": df[selected].select_dtypes(include="number").columns.tolist(),
            "categorical": df[selected].select_dtypes(exclude="number").columns.tolist(),
        }
        checkpoint = Path(checkpoint_path)
        checkpoint.mkdir(parents=True, exist_ok=True)
        classification = parameters["task_type"] == "classification"
        objective = parameters.get("tuning", {}).get("objective", {
            "name": "validation:log_loss" if classification else "validation:rmse",
        })
        metric = objective.get("report_metric", objective["name"].split(":")[-1])
        predictor = TabularPredictor(
            label=parameters["target_key"], problem_type="binary" if classification else "regression",
            eval_metric="root_mean_squared_error" if metric == "rmse" else metric,
            path=str(checkpoint / "predictor"),
            learner_kwargs={"cache_data": final_model, **({"positive_class": 1} if classification else {})},
        )
        predictor.fit(
            df[selected + [parameters["target_key"]]],
            hyperparameters={"RF": {
                **parameters.get("autogluon", {}).get(parameters["task_type"], {"n_estimators": 40}),
                "random_state": parameters.get("seed", 42),
            }},
            time_limit=parameters.get("time_limit_seconds", 60),
            num_cpus=parameters.get("num_cpus", 2), num_gpus=0,
            num_bag_folds=0, num_stack_levels=0, fit_weighted_ensemble=False, refit_full=False,
            raise_on_no_models_fitted=True,
        )
        validation_model = predictor.model_best
        if final_model:
            # Refit the selected model on training + validation rows. Its parent
            # keeps the validation score; do not report that as a final-model score.
            predictor.refit_full(model=validation_model, set_best_to_refit_full=True,
                                 num_cpus=parameters.get("num_cpus", 2), num_gpus=0)
            predictor.save_space()  # Do not package the temporary cached training rows.
        predictor.save()
        (checkpoint / "parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
        copyfile(self.feature_path, checkpoint / "selected_features.csv")
        # This is a lifecycle smoke check, not an independent test dataset.
        # Other trainers can keep their own reports; this JSON is optional.
        performance = predictor.evaluate(df[selected + [parameters["target_key"]]])
        (checkpoint / "evaluation.json").write_text(json.dumps(
            evaluation_report(performance, parameters, "prepared_input_smoke_check"),
            indent=2, allow_nan=False,
        ) + "\n")
        metadata = {"selected_model": predictor.model_best, "rows": {"input": len(df)},
                    "feature_mapping": {name: name for name in selected},
                    "predictor_path": "predictor",
                    "evaluation": {"usage": "prepared_input_smoke_check"}}
        if os.getenv("TUNING_OBJECTIVE"):
            score = predictor.leaderboard().set_index("model").loc[validation_model, "score_val"]
            value = float(predictor.eval_metric.convert_score_to_original(score))
            if not math.isfinite(value):
                raise ValueError(f"The tuning objective {metric} must be finite.")
            print(f"tuning_objective={value:.17g};", flush=True)
            metadata["tuning"] = {"metric": metric, "value": value, "split": "validation",
                                  "model": validation_model,
                                  "source": "autogluon"}
        return checkpoint, metadata


class gluonPredictor:
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
