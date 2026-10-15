"""Paste this whole file into one notebook cell and set EVALUATION_JOB_NAME.

Inspect the exact model archive supplied to an Evaluate processing job.
Print its file layout and the settings/evaluation-data contract it contains.
Save a manifest/report locally. Do not extract files or load executable models.
No AWS resources are changed and no evaluation score is invented.
"""

EVALUATION_JOB_NAME = ""

import csv
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import tempfile
from urllib.parse import urlparse

import boto3


def trace_evaluation_inputs(job_name):
    if not job_name.strip():
        raise ValueError("Set EVALUATION_JOB_NAME to the Evaluate processing job's name.")
    session = boto3.Session()
    job = session.client("sagemaker").describe_processing_job(ProcessingJobName=job_name)
    bindings = [item["S3Input"] for item in job["ProcessingInputs"]
                if item.get("S3Input", {}).get("LocalPath") == "/opt/ml/processing/model"]
    if len(bindings) != 1:
        raise ValueError("Expected exactly one model input mounted at /opt/ml/processing/model.")
    uri = bindings[0]["S3Uri"]
    location = urlparse(uri)
    if location.scheme != "s3" or not location.netloc or not location.path.endswith(".tar.gz"):
        raise ValueError(f"Expected the pipeline's explicit model.tar.gz S3 URI, got {uri!r}.")
    environment = job.get("Environment", {})
    print("Evaluation job:", job_name, "| status:", job["ProcessingJobStatus"])
    print("Job failure:", job.get("FailureReason", ""))
    print("ModelId:", environment.get("MODEL_ID"))
    print("Model archive supplied to Evaluate:", uri)
    print("Configured checkpoint URI:", environment.get("CHECKPOINT_URI"))
    print("Model code source:", environment.get("MLOPS_MODEL_CODE_SOURCE", "artifact"))
    print("Evaluation input mode:", environment.get("MLOPS_EVALUATION_INPUT", "explicit test input"))
    print("Model mount:", bindings[0]["LocalPath"])
    print("The loader unpacks this archive into /tmp/mlops-step-<job>/model for reading.")
    outputs = {item["OutputName"]: item["S3Output"]["S3Uri"]
               for item in job["ProcessingOutputConfig"]["Outputs"]}
    print("Evaluation output destinations:", outputs)

    with tempfile.TemporaryFile() as download:
        session.client("s3").download_fileobj(location.netloc, location.path.lstrip("/"), download)
        download.seek(0)
        with tarfile.open(fileobj=download, mode="r:gz") as archive:
            files = {PurePosixPath(member.name).as_posix(): member
                     for member in archive.getmembers() if member.isfile()}
            names = sorted(files)
            roots = sorted({PurePosixPath(name).parts[0] for name in names})
            print("Archive top-level entries:", roots[:40])
            print("Archive file count:", len(names), "| first 60:", names[:60])
            settings = {}
            for filename in ("parameters.json", "model_metadata.json"):
                present = filename in files
                other_locations = [name for name in names
                                   if PurePosixPath(name).name == filename and name != filename]
                print(filename, "| regular file at archive root:", present,
                      "| other locations:", other_locations)
                if present:
                    with archive.extractfile(files[filename]) as stream:
                        settings[filename] = json.load(stream)
                    if not isinstance(settings[filename], dict):
                        raise ValueError(f"{filename} must contain a JSON object.")
                    print(filename, "| top-level keys:", list(settings[filename])[:40])

            parameters = settings.get("parameters.json")
            if parameters is not None:
                print("Saved task/identifier/target:", {
                    key: parameters.get(key) for key in ("task_type", "primary_key", "target_key")})
            else:
                print("The current Evaluate entrypoint requires parameters.json at the archive root.")
                print("This file supplies task/target/identifier settings; model_metadata.json is separate.")
                print("The generic training writer saves parameters when called with config=config.")

            metadata = settings.get("model_metadata.json", {})
            evaluation = metadata.get("evaluation")
            dataset = None
            if isinstance(evaluation, dict) and isinstance(evaluation.get("data_file"), str):
                data_file = PurePosixPath(evaluation["data_file"])
                relative = not data_file.is_absolute() and ".." not in data_file.parts
                name = data_file.as_posix()
                dataset = {"data_file": name, "usage": evaluation.get("usage"),
                           "regular_file_present": relative and name in files}
                if dataset["regular_file_present"]:
                    with io.TextIOWrapper(archive.extractfile(files[name]), encoding="utf-8-sig",
                                          newline="") as stream:
                        columns = next(csv.reader(stream))
                    dataset["columns"] = columns
                    print("Saved evaluation CSV:", name, "| columns:", len(columns), "| first 20:", columns[:20])
                    if parameters is not None:
                        for key in ("primary_key", "target_key"):
                            value = parameters.get(key)
                            print(key, "| present exactly:", value in columns,
                                  "| case matches:", [column for column in columns
                                                      if isinstance(value, str) and column.casefold() == value.casefold()])
                print("Saved evaluation-data reference:",
                      {key: dataset[key] for key in ("data_file", "usage", "regular_file_present")})
            else:
                print("No evaluation.data_file is declared in model_metadata.json.")
                print("The current use-case evaluator needs that reference when input mode is model-artifact.")
            predictor_locations = [name for name in names if PurePosixPath(name).name == "predictor.pkl"]
            print("AutoGluon predictor.pkl locations:", predictor_locations)
            print("The placeholder predictor expects predictor/; a custom loader must match the saved layout.")

    output = Path.cwd() / ".runs/evaluation-input-trace" / job_name
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "evaluation_job": job_name, "model_id": environment.get("MODEL_ID"), "model_uri": uri,
        "checkpoint_uri": environment.get("CHECKPOINT_URI"),
        "model_code_source": environment.get("MLOPS_MODEL_CODE_SOURCE", "artifact"),
        "evaluation_input_mode": environment.get("MLOPS_EVALUATION_INPUT"),
        "outputs": outputs, "archive_files": names,
        "parameters_at_root": "parameters.json" in settings,
        "metadata_at_root": "model_metadata.json" in settings,
        "settings_keys": {name: list(value) for name, value in settings.items()},
        "saved_evaluation": dataset, "predictor_locations": predictor_locations,
    }
    (output / "artifact-files.txt").write_text("\n".join(names) + "\n")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Saved complete artifact manifest:", (output / "artifact-files.txt").resolve())
    print("Saved evaluation-input report:", (output / "report.json").resolve())
    print("The archive was inspected without extraction or model execution; no data rows were printed.")
    return report


if __name__ == "__main__":
    trace_evaluation_inputs(EVALUATION_JOB_NAME)
