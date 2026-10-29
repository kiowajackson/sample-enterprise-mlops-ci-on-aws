"""Read training inputs, call the selected trainer and package its outputs."""

from copy import deepcopy
import json
import os
from pathlib import Path
from shutil import copyfile, copytree, rmtree, ignore_patterns

from ...frameworks.autogluon import gluonTraining as ModelTraining

from ...storage import training_metadata


def train_model(data_directory, selection_file, model_dir, output_directory, parameter_path,
                *, trainer_cls, hyperparameters=None):
    """Read settings and optional metadata, train, and retain the new checkpoint/metadata."""
    feature_path, data_path = get_container_input(data_directory, selection_file)
    parameter_path = Path(parameter_path)
    parameters = json.loads(parameter_path.read_text())
    parameters["model_id"] = parameter_path.parent.name
    # Keep metadata loading here, before the model's own training implementation.
    config = training_metadata(parameter_path)
    mode = os.getenv("MLOPS_TRAINING_MODE", "1")
    if mode not in {"1", "2"}:
        raise ValueError(f"Unknown TrainingMode code: {mode!r}. Expected '1' or '2'.")
    print(f"Training mode: {mode}", flush=True)
    if hyperparameters:
        parameters = resolve_trial_parameters(parameters, hyperparameters)
    output = Path(output_directory)
    trainer = trainer_cls(feature_path, data_path, config, parameters)
    checkpoint, metadata = trainer.train_model(output / "checkpoint", output)
    return write_container_output(
        checkpoint, metadata, model_dir, output, parameters=parameters,
    )


def get_container_input(data_directory, selection_file):
    """Return the selected-feature CSV path and prepared-data directory."""
    return Path(selection_file), Path(data_directory)


def write_container_output(checkpoint, metadata, model_dir, output_directory, *, parameters):
    """Preserve the trainer's files and add metadata/settings/source for upload."""
    if not isinstance(metadata, dict):
        raise TypeError(f"Expected metadata dictionary, got {type(metadata).__name__}")
    metadata_json = json.dumps(metadata, indent=2, default=os.fspath) + "\n"
    checkpoint, model_dir = Path(checkpoint), Path(model_dir)
    if not checkpoint.is_dir() or not any(checkpoint.iterdir()):
        raise FileNotFoundError(f"Training produced no model artifact: {checkpoint}")
    if model_dir.resolve().is_relative_to(checkpoint.resolve()):
        raise ValueError("Save the model outside the working checkpoint directory.")
    if model_dir.exists() and any(model_dir.iterdir()):
        raise FileExistsError(f"Choose an empty model directory: {model_dir}")
    # Keep the trainer's effective settings for inference; supply them if missing.
    settings_json = json.dumps(parameters, indent=2, default=os.fspath) + "\n"
    copytree(checkpoint, model_dir, dirs_exist_ok=True)
    (model_dir / "model_metadata.json").write_text(metadata_json)
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


def resolve_trial_parameters(parameters, hyperparameters):
    """Take string-valued trial parameters; return model settings with the trial's typed overrides.

    - Decode JSON numbers, booleans and objects; preserve ordinary strings.
    - Apply named feature sets and feature-selection overrides before fitting.
    - Apply remaining values to the model's algorithm parameters.
    """
    resolved = deepcopy(parameters)
    framework, task = resolved.get("framework", "autogluon"), resolved["task_type"]
    overrides = resolved.setdefault(framework, {}).setdefault(task, {})
    for name, value in hyperparameters.items():
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
