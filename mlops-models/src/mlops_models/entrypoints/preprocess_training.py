"""Pass optional model metadata and the parameter path to process_train."""

from pathlib import Path
from mlops_models.storage import training_metadata

from mlops_models.entrypoints import load_step


def execute(model_id, input_path, selection_path, output_dir):
    parameter_path = Path(__file__).parents[1] / "models" / model_id / "parameters.json"
    config = training_metadata(parameter_path)
    process_train = load_step(model_id, "preprocess", "process_train")
    return process_train(
        Path(input_path), Path(selection_path), Path(output_dir), config, parameter_path,
    )
