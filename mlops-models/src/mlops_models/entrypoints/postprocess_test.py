"""Run the selected model's test analysis from explicit mounted inputs."""

from pathlib import Path

from mlops_models.entrypoints import load_step


def execute(model_id, input_path, output_dir, *, model_dir, data_directory, selection_file,
            train_data_directory=None):
    prediction_test = load_step(model_id, "postprocess", "prediction_test")
    parameter_path = Path(__file__).parents[1] / "models" / model_id / "parameters.json"
    training = {}
    if train_data_directory is not None:
        train_data_file = Path(train_data_directory) / "data.csv"
        if not train_data_file.is_file():
            raise FileNotFoundError(f"Postprocess test needs the raw training CSV: {train_data_file}")
        training["train_data_file"] = train_data_file
    return prediction_test(
        Path(data_directory), Path(selection_file), Path(input_path), Path(output_dir),
        Path(model_dir), parameter_path=parameter_path, **training,
    )
