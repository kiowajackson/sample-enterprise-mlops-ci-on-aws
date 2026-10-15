"""Runnable demo trainer; model adapters may replace this class with their own."""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
from shutil import copyfile, copytree, rmtree, ignore_patterns

import pandas as pd

from autogluon.tabular import TabularPredictor
from autogluon.core.metrics import get_metric


def train_model(data_directory, selection_file, model_dir, output_directory, config,
                *, trainer_cls, hyperparameters=None):
    """Read input paths, call the model trainer, and retain its checkpoint/metadata."""
    feature_path, data_path = get_container_input(data_directory, selection_file)
    if hyperparameters:
        config = resolve_trial_parameters(config, hyperparameters)
    output = Path(output_directory)
    trainer = trainer_cls(feature_path, data_path, config)
    checkpoint, metadata, performance = trainer.train_model(output / "checkpoint", output)
    return write_container_output(
        checkpoint, metadata, model_dir, output, config=config, performance=performance,
        data_usage=metadata["evaluation"]["usage"],
    )


def get_container_input(data_directory, selection_file):
    """Return the selected-feature CSV path and prepared-data directory."""
    return Path(selection_file), Path(data_directory)


def write_container_output(checkpoint, metadata, model_dir, output_directory, *,
                           config, performance, data_usage):
    """Save the checkpoint, metadata and its AutoGluon evaluation results for upload."""
    if not isinstance(metadata, dict):
        raise TypeError(f"Expected metadata dictionary, got {type(metadata).__name__}")
    metadata_json = json.dumps(metadata, indent=2, default=os.fspath) + "\n"
    report_json = json.dumps(
        evaluation_report(performance, config, data_usage), indent=2, allow_nan=False,
    ) + "\n"
    checkpoint, model_dir = Path(checkpoint), Path(model_dir)
    if not checkpoint.is_dir() or not any(checkpoint.iterdir()):
        raise FileNotFoundError(f"Training produced no model artifact: {checkpoint}")
    if model_dir.resolve().is_relative_to(checkpoint.resolve()):
        raise ValueError("Save the model outside the working checkpoint directory.")
    if model_dir.exists() and any(model_dir.iterdir()):
        raise FileExistsError(f"Choose an empty model directory: {model_dir}")
    # Keep the trainer's effective settings for inference; supply them if missing.
    settings_json = json.dumps(config, indent=2, default=os.fspath) + "\n"
    copytree(checkpoint, model_dir, dirs_exist_ok=True)
    (model_dir / "model_metadata.json").write_text(metadata_json)
    (model_dir / "evaluation.json").write_text(report_json)
    if not (model_dir / "parameters.json").exists():
        (model_dir / "parameters.json").write_text(settings_json)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "training.json").write_text(metadata_json)
    source = Path(__file__).resolve().parents[3]
    copytree(source / "mlops_models", model_dir / "code/mlops_models",
             ignore=ignore_patterns("__pycache__", "*.pyc"))
    if (source / "source.json").is_file():
        copyfile(source / "source.json", model_dir / "code/source.json")
    rmtree(checkpoint)
    return model_dir


def evaluation_report(performance, config, data_usage):
    """Keep raw AutoGluon scores and convert losses to their original positive scale."""
    if not isinstance(performance, dict) or not performance:
        raise ValueError("Pass the nonempty performance dictionary returned by AML.evaluate(...).")
    if not data_usage:
        raise ValueError("Describe the scored data, for example validation or prepared_input_smoke_check.")
    if config["task_type"] not in {"classification", "regression"}:
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
    return {"model_id": config["model_id"], "task": config["task_type"],
            "data_usage": data_usage, "metric_source": "autogluon.TabularPredictor.evaluate",
            "raw_metrics": raw, "metrics": metrics}


class ModelTrainer:
    """Accept input paths, then own the demo's entire training implementation."""

    def __init__(self, user_feature_path, user_data_path, config):
        self.feature_path = Path(user_feature_path)
        self.data_path = Path(user_data_path)
        self.config = deepcopy(config)

    def train_model(self, checkpoint_path, output):
        """Demo fit; replace this method with the model's training implementation."""
        config = self.config
        df = pd.concat([pd.read_csv(path, dtype={config["primary_key"]: str},
                                    keep_default_na=False, na_values=["", r"\N"])
                        for path in sorted(self.data_path.glob("*.csv"))], ignore_index=True)
        selected = pd.read_csv(self.feature_path)["feature"].tolist()
        config["features"] = {
            "numeric": df[selected].select_dtypes(include="number").columns.tolist(),
            "categorical": df[selected].select_dtypes(exclude="number").columns.tolist(),
        }
        checkpoint = Path(checkpoint_path)
        checkpoint.mkdir(parents=True, exist_ok=True)
        classification = config["task_type"] == "classification"
        objective = config.get("tuning", {}).get("objective", {
            "name": "validation:log_loss" if classification else "validation:rmse",
        })
        metric = objective.get("report_metric", objective["name"].split(":")[-1])
        predictor = TabularPredictor(
            label=config["target_key"], problem_type="binary" if classification else "regression",
            eval_metric="root_mean_squared_error" if metric == "rmse" else metric,
            path=str(checkpoint / "predictor"),
            learner_kwargs={"cache_data": False, **({"positive_class": 1} if classification else {})},
        )
        predictor.fit(
            df[selected + [config["target_key"]]],
            hyperparameters={"RF": {
                **config.get("autogluon", {}).get(config["task_type"], {"n_estimators": 40}),
                "random_state": config.get("seed", 42),
            }},
            time_limit=config.get("time_limit_seconds", 60),
            num_cpus=config.get("num_cpus", 2), num_gpus=0,
            num_bag_folds=0, num_stack_levels=0, fit_weighted_ensemble=False, refit_full=False,
            raise_on_no_models_fitted=True,
        )
        predictor.save()
        (checkpoint / "parameters.json").write_text(json.dumps(config, indent=2) + "\n")
        copyfile(self.feature_path, checkpoint / "selected_features.csv")
        # This is a lifecycle smoke check, not an independent test dataset.
        # Customer trainers return the performance from their own validation call.
        performance = predictor.evaluate(df[selected + [config["target_key"]]])
        metadata = {"selected_model": predictor.model_best, "rows": {"input": len(df)},
                    "evaluation": {"usage": "prepared_input_smoke_check"}}
        if os.getenv("TUNING_OBJECTIVE"):
            score = predictor.leaderboard().set_index("model").loc[predictor.model_best, "score_val"]
            value = float(predictor.eval_metric.convert_score_to_original(score))
            if not math.isfinite(value):
                raise ValueError(f"The tuning objective {metric} must be finite.")
            print(f"tuning_objective={value:.17g};", flush=True)
            metadata["tuning"] = {"metric": metric, "value": value, "split": "validation",
                                  "source": "autogluon"}
        return checkpoint, metadata, performance


def resolve_trial_parameters(config, parameters):
    """Take string-valued trial parameters; return model settings with the trial's typed overrides.

    - Decode JSON numbers, booleans and objects; preserve ordinary strings.
    - Apply named feature sets and feature-selection overrides before fitting.
    - Apply remaining values to the model's algorithm parameters.
    """
    resolved = deepcopy(config)
    framework, task = resolved.get("framework", "autogluon"), resolved["task_type"]
    overrides = resolved.setdefault(framework, {}).setdefault(task, {})
    for name, value in parameters.items():
        if name.startswith("_tuning_"):
            continue
        try:
            value = json.loads(value) if isinstance(value, str) else value
        except json.JSONDecodeError:
            pass
        if name == "feature_set":
            resolved["features"] = deepcopy(resolved["feature_sets"][value])
            resolved["selected_feature_set"] = value
        elif name.startswith("feature_selection."):
            resolved["feature_selection"][name.split(".", 1)[1]] = value
        else:
            overrides[name] = value
    return resolved
