"""Publish saved training metrics and check thresholds before model registration."""

from importlib import import_module
import json
from pathlib import Path
import tarfile
import math
from shutil import rmtree

import os


def execute(model_id, model_dir, output_dir, *, input_path):
    """Take a checkpoint; publish its saved metrics and return the quality decision.

    - Unpack the model archive when supplied.
    - Load the selected use case's parameters and saved-report evaluation function.
    - Check the saved scores against thresholds and publish the report.
    """
    print(json.dumps({"event": "event_received", "step": "Evaluate",
                      "model": model_id, "input": str(input_path)}), flush=True)

    # Locate the saved checkpoint containing evaluation.json.
    model_dir, test_file, work = prepare_evaluation_files(
        model_dir, input_path, output_dir,
    )

    try:
        report = evaluate_saved_model(model_id, model_dir, test_file, output_dir)
    finally:
        remove_unpacked_inputs(work)

    print(json.dumps({"event": "success", "step": "Evaluate", "model": model_id,
                      "output": str(output_dir), **report}), flush=True)

    return report


def prepare_evaluation_files(model_path, test_file, output_dir):
    """Take checkpoint and test-data locations; unpack archives and return local paths."""
    work = Path(output_dir) / "unpacked"
    work.mkdir(parents=True, exist_ok=False)
    model_dir = Path(model_path)
    if model_dir.is_file():
        model_dir = work / "checkpoint"
        model_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(model_path) as archive:
            archive.extractall(model_dir, filter="data")
    return model_dir, test_file, work


def remove_unpacked_inputs(work):
    """Take the temporary evaluation directory and remove its unpacked inputs."""
    rmtree(work)


def evaluate_saved_model(model_id, model_dir, test_file, output):
    """Publish saved scores using the selected use case's current quality thresholds."""
    model_folder = Path(__file__).parents[1] / "models" / model_id
    config = json.loads((model_folder / "parameters.json").read_text())
    config["model_id"] = model_id
    evaluate = import_module(
        f"mlops_models.models.{model_id}.training.evaluate"
    ).use_case_evaluate
    report = evaluate(test_file, model_dir, output, config)
    report["checkpoint_uri"] = os.getenv("CHECKPOINT_URI", str(model_dir))
    identity_path = model_dir / "code/source.json"
    model_source = json.loads(identity_path.read_text()) if identity_path.is_file() else {}
    report["model_source_commit"] = model_source.get("commit", "unknown")
    report["model_source_dirty"] = str(model_source.get("dirty", True)).lower()
    report["evaluation_source_commit"] = os.getenv("SOURCE_COMMIT", "unknown")
    report["quality"] = check_quality_thresholds(report, config.get("quality_gate", {}))
    (Path(output) / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def check_quality_thresholds(report, thresholds):
    """Take metrics and configured minimum/maximum bounds and return the registration decision and reasons."""
    checks = []
    for metric, limits in thresholds.items():
        value = report["metrics"].get(metric)
        if not limits or set(limits) - {"min", "max"}:
            raise ValueError(f"Set min and/or max for quality_gate.{metric}.")
        passed = isinstance(value, (float, int)) and math.isfinite(value)
        passed = passed and ("min" not in limits or value >= limits["min"])
        passed = passed and ("max" not in limits or value <= limits["max"])
        checks.append({"metric": metric, "value": value, "limits": limits, "passed": bool(passed)})
    return {"configured": bool(thresholds), "passed": all(check["passed"] for check in checks),
            "checks": checks}


if __name__ == "__main__":
    # Reserved for local setup and debugging.
    pass
