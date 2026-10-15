"""Publish the scores already calculated by training; do not load the predictor."""

import json
import math
from pathlib import Path


def get_container_input(checkpoint_path):
    """Read the evaluation report packaged with this exact checkpoint."""
    path = Path(checkpoint_path) / "evaluation.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Training did not package {path}. Pass performance to write_container_output "
            "and rerun Train; an older cached artifact cannot supply these scores."
        )
    return json.loads(path.read_text())


def write_container_output(report, output):
    """Write the report locally; load_source.py uploads it to the evaluation S3 output."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def evaluate(source, checkpoint_path, output, config):
    """Read saved training scores, validate their model/task and publish the report."""
    if source is not None:
        raise ValueError("Evaluate reads the model's saved evaluation.json; do not pass a scoring dataset.")
    report = get_container_input(checkpoint_path)
    if report["model_id"] != config["model_id"]:
        raise ValueError(
            f"Artifact belongs to {report['model_id']}, not selected model {config['model_id']}."
        )
    if report["task"] != config["task_type"]:
        raise ValueError("The saved evaluation task differs from the selected model's task_type.")
    if not report.get("data_usage") or not report.get("metrics"):
        raise ValueError("The saved evaluation must identify its scored data and contain metrics.")
    for name, value in report["metrics"].items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Saved evaluation metric {name} must be a finite number.")
    return write_container_output(report, output)
