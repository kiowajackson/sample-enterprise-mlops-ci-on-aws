"""Temporary training-path diagnostics; no training runs when this file is imported.

At the very start of train_model, before any training or path reassignment:
    package_dir = at_train_model_entry(model_dir, output_directory, __file__)

Immediately before the existing write_container_output call:
    before_write_container_output(model_dir, package_dir, checkpoint, output_directory)

Copy these helpers into the training file, or copy their diagnostic statements
into the two positions above. Keep the original package_dir through training.
"""

import os
from pathlib import Path


# 1. At the start of train_model: capture and print the incoming paths.
def at_train_model_entry(model_dir, output_directory, source_file):
    package_dir = Path(model_dir)
    print("TRAIN ENTRY", {
        "source": str(source_file),
        "received_model_dir": str(model_dir),
        "loader_model_dir": os.environ.get("SM_MODEL_DIR"),
        "output_directory": str(output_directory),
    }, flush=True)
    return package_dir


# 2. Immediately before write_container_output: inspect the final path values.
def before_write_container_output(model_dir, package_dir, checkpoint, output_directory):
    for name, value in {
        "model_dir": model_dir,
        "package_dir": package_dir,
        "checkpoint": checkpoint,
        "output_directory": output_directory,
    }.items():
        path = Path(value)
        print(name, str(path),
              sorted(p.name for p in path.iterdir())[:20]
              if path.is_dir() else "<not a directory>",
              flush=True)

    expected = Path(os.environ["SM_MODEL_DIR"])
    assert Path(package_dir).resolve() == expected.resolve(), (
        f"Wrong package destination: {package_dir}; loader expects {expected}"
    )
