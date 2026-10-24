"""Publish model details and optional metrics, then apply configured thresholds."""

from importlib import import_module
import hashlib
import json
from pathlib import Path
import tarfile
import math
from tempfile import TemporaryDirectory

import os


def execute(model_id, model_dir, output_dir, *, input_path):
    """Take a checkpoint; publish its saved metrics and return the quality decision.

    - Unpack the model archive when supplied.
    - Load the selected use case's parameters and saved-report evaluation function.
    - Check the saved scores against thresholds and publish the report.
    """
    print(json.dumps({"event": "event_received", "step": "Evaluate",
                      "model": model_id, "model_artifact": str(model_dir),
                      "checkpoint_uri": os.getenv("CHECKPOINT_URI", ""),
                      "input_path": str(input_path)}), flush=True)

    # Locate the saved checkpoint and its optional evaluation.json.
    with TemporaryDirectory(prefix="evaluate-model-") as temporary:
        model_dir = Path(model_dir)
        if model_dir.is_file():
            with tarfile.open(model_dir) as archive:
                model_dir = Path(temporary)
                archive.extractall(model_dir, filter="data")
        report = evaluate_saved_model(model_id, model_dir, output_dir, input_path=input_path)

    print(json.dumps({"event": "success", "step": "Evaluate", "model": model_id,
                      "output": str(output_dir), **report}), flush=True)

    return report


def evaluate_saved_model(model_id, model_dir, output_dir, *, input_path):
    """Assemble the evaluation report and files consumed by Registry and tracking.

    1. Load this use case's settings and evaluation function.
    2. Call that function, then add source details and the quality-gate result.
    3. Copy the model's metadata/settings/run record and write the completed report.

    The default function passes through any scores saved during training. Missing
    scores are allowed unless a configured quality threshold needs them. It does
    not load the predictor or score another dataset; input_path is None for it.
    model_dir is the checkpoint directory; evaluation.json is optional.
    """
    # 1. Read the use-case settings and import its evaluation function.
    model_folder = Path(__file__).parents[1] / "models" / model_id
    config = json.loads((model_folder / "parameters.json").read_text())
    config["model_id"] = model_id
    evaluate = import_module(
        f"mlops_models.models.{model_id}.training.evaluate"
    ).use_case_evaluate

    # 2. Delegate to the use case and receive its report dictionary.
    # The default function also writes a report; we add fields and save it below.
    report = evaluate(input_path, model_dir, output_dir, config)

    # Add the model location and the code versions used for training/evaluation.
    report["checkpoint_uri"] = os.getenv("CHECKPOINT_URI", str(model_dir))
    identity_path = model_dir / "code/source.json"
    model_source = json.loads(identity_path.read_text()) if identity_path.is_file() else {}
    report["model_source_commit"] = model_source.get("commit", "unknown")
    report["model_source_dirty"] = str(model_source.get("dirty", True)).lower()
    report["model_source_uri"] = model_source.get("uri", "")
    report["model_source_digest"] = model_source.get("sha256", "")
    report["evaluation_source_commit"] = os.getenv("SOURCE_COMMIT", "unknown")

    # Compare the returned scores with configured limits; this is the quality gate.
    report["quality"] = check_quality_thresholds(report, config.get("quality_gate", {}))

    # 3. Copy supporting files into this job's output directory for later upload.
    # Validate metadata as JSON, but copy its original bytes so its checksum matches.
    metadata = (model_dir / "model_metadata.json").read_bytes()
    json.loads(metadata)
    (Path(output_dir) / "model_metadata.json").write_bytes(metadata)

    # These are the settings saved with the trained model. The config above holds
    # the selected evaluator's settings; they are separate files and can differ.
    # Older model packages may omit this file, so record whether it is available.
    parameter_file = model_dir / "parameters.json"
    report["effective_parameters_available"] = parameter_file.is_file()
    if parameter_file.is_file():
        parameters = parameter_file.read_bytes()
        json.loads(parameters)
        (Path(output_dir) / "parameters.json").write_bytes(parameters)

    # The loader wrote _execution.json when training finished: code/settings/run
    # details identifying what produced the model. Older packages may lack it.
    # Here provenance is a local Path; is_file() only checks whether it exists.
    provenance = model_dir / "_execution.json"
    if provenance.is_file():
        (Path(output_dir) / "training_execution.json").write_bytes(provenance.read_bytes())

    # Save the final report, including the metadata checksum for Registry lookup.
    # load_source.py uploads these local output files to S3 after this returns.
    report["model_metadata_sha256"] = hashlib.sha256(metadata).hexdigest()
    (Path(output_dir) / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
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
