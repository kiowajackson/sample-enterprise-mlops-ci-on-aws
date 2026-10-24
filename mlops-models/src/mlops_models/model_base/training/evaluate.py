"""Publish optional saved scores and model details; do not load the predictor."""

import json
import math
from pathlib import Path


def get_container_input(model_dir):
    """Read an optional evaluation report packaged with this exact checkpoint."""
    path = Path(model_dir) / "evaluation.json"
    if not path.is_file():
        print(f"No structured evaluation report at {path}; publishing model details without metrics.",
              flush=True)
        return {}
    report = json.loads(path.read_text())
    if not isinstance(report, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    metrics = report.get("metrics")
    print(json.dumps({
        "event": "saved_evaluation_loaded", "path": str(path),
        "model_id": report.get("model_id"), "task": report.get("task"),
        "data_usage": report.get("data_usage"),
        "metric_names": list(metrics) if isinstance(metrics, dict) else [],
        "metrics_type": type(metrics).__name__,
    }), flush=True)
    return report


def write_container_output(report, output_dir):
    """Write the report locally; load_source.py uploads it to the evaluation S3 output."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def evaluate(input_path, model_dir, output_dir, config):
    """Publish model details and any saved scores; no scores are required by default."""
    if input_path is not None:
        raise ValueError("Evaluate reads the model's saved evaluation.json; do not pass a scoring dataset.")
    report = get_container_input(model_dir)
    path = Path(model_dir) / "evaluation.json"
    report.setdefault("model_id", config["model_id"])
    report.setdefault("task", config["task_type"])
    report.setdefault("metrics", {})
    if report.get("model_id") != config["model_id"]:
        raise ValueError(
            f"{path}: model_id is {report.get('model_id')!r}, expected {config['model_id']!r}."
        )
    if report.get("task") != config["task_type"]:
        raise ValueError("The saved evaluation task differs from the selected model's task_type.")
    if not isinstance(report["metrics"], dict):
        raise ValueError(f"{path}: metrics must be a dictionary when supplied.")
    if report["metrics"] and not report.get("data_usage"):
        raise ValueError(
            f"{path}: data_usage is missing or empty. Training must describe which data was "
            "scored, for example validation. Evaluate reads saved scores; it does not compute them."
        )
    for name, value in report["metrics"].items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Saved evaluation metric {name} must be a finite number.")
    if not report["metrics"]:
        print("No numeric evaluation metrics supplied; configured quality thresholds still apply.",
              flush=True)
    return write_container_output(report, output_dir)
