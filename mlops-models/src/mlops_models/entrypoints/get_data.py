"""Select the requested data function; the model code owns source selection."""

from pathlib import Path
from mlops_models.entrypoints import load_step


def execute(model_id, input_path, output_dir, *, kind="training", model_dir=None):
    functions = {"training": "download_train_data", "testing": "download_test_data",
                 "inference": "download_inference_data"}
    download = load_step(model_id, "get_data", functions[kind])
    return download(model_id, Path(input_path) if input_path is not None else None, Path(output_dir), model_dir=model_dir)
