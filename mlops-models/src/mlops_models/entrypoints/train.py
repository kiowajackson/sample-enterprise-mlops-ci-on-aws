"""Pass input paths to the selected use case's train_model function."""

from pathlib import Path

from mlops_models.entrypoints import load_step


def execute(model_id, input_path, model_dir, output_dir, *, selection_path, hyperparameters=None):
    selection_file = Path(selection_path)
    if not selection_file.is_file():
        raise FileNotFoundError(f"Selected-feature CSV is missing at the supplied path: {selection_file}")
    print(f"Training inputs: prepared={Path(input_path) / 'prepared'}, selection={selection_file}",
          flush=True)
    parameter_path = Path(__file__).parents[1] / "models" / model_id / "parameters.json"
    train_model = load_step(model_id, "training.train", "train_model")
    trainer = load_step(model_id, "training.train", "ModelTraining", default=None)
    return train_model(
        Path(input_path) / "prepared",
        selection_file,
        Path(model_dir), Path(output_dir), parameter_path,
        trainer_cls=trainer, hyperparameters=hyperparameters,
    )
