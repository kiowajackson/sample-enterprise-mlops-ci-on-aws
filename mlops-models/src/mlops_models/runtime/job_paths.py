"""Standard SageMaker mount paths used by the pipeline-step dispatcher."""

import os


def job_paths(step):
    """Return standard SageMaker paths; explicit dispatcher arguments can override them."""
    if step == "train":
        return ("/opt/ml/input/data", os.getenv("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"),
                os.getenv("SM_MODEL_DIR", "/opt/ml/model"))
    if step == "serve":
        return (None, None, os.getenv("SM_MODEL_DIR", "/opt/ml/model"))
    inputs = {"select-features": "/opt/ml/processing/raw",
              "preprocess": "/opt/ml/processing/raw",
              "preprocess-training": "/opt/ml/processing/raw",
              "evaluate": "/opt/ml/processing/test/test.csv",
              "score": "/opt/ml/processing/prepared",
              "predict": "/opt/ml/processing/prepared",
              "postprocess": "/opt/ml/processing/input"}
    outputs = {"evaluate": "/opt/ml/processing/evaluation"}
    return (inputs.get(step, "/opt/ml/processing/input"),
            outputs.get(step, "/opt/ml/processing/output"),
            "/opt/ml/processing/model/model.tar.gz")


def selected_features_path(step, input_path=None):
    """Processing and Training mount the same selection artifact under different roots."""
    from pathlib import Path
    root = (Path(input_path or job_paths("train")[0]) / "selection" if step == "train"
            else Path("/opt/ml/processing/selection"))
    return str(root / "selected_features.csv")
