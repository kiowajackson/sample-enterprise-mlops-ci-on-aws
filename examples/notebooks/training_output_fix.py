# Replace write_container_output in your existing training script with this function.
# Copy the imports below too. Keep your existing call to write_container_output.
# Destination: mlops-models/src/mlops_models/model_base/training/train.py
# The first argument is the directory containing the saved model files.
# Paste this into train.py; it uses that file's location to package the source code.

import json
import os
from pathlib import Path
from shutil import copyfile, copytree, ignore_patterns, rmtree


def write_container_output(checkpoint, metadata, model_dir, output_directory):
    """Save model, metadata and source locally; load_source.py archives/uploads to S3."""
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
    copytree(checkpoint, model_dir, dirs_exist_ok=True)
    (model_dir / "model-metadata.json").write_text(metadata_json)
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
