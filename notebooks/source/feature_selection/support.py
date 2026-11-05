"""Prepare local FeatureSelection inputs and export the same call for VS Code."""

from contextlib import contextmanager
import importlib
from inspect import getsource
import json
import os
from pathlib import Path
import sys
from tempfile import mkdtemp, TemporaryDirectory
from textwrap import indent
from time import perf_counter
from types import SimpleNamespace
from urllib.parse import urlsplit

import boto3
from botocore.config import Config


def _processing_job(sm, execution_arn):
    visited = set()
    current = execution_arn
    while current not in visited:
        visited.add(current)
        steps = [
            step
            for page in sm.get_paginator("list_pipeline_execution_steps").paginate(
                PipelineExecutionArn=current)
            for step in page["PipelineExecutionSteps"]
            if step["StepName"] == "FeatureSelection"
        ]
        if len(steps) != 1:
            raise ValueError(f"No unique FeatureSelection step in {current}.")
        step = steps[0]
        cached_from = step.get("CacheHitResult", {}).get("SourcePipelineExecutionArn")
        if cached_from:
            current = cached_from
            continue
        job_arn = step.get("Metadata", {}).get("ProcessingJob", {}).get("Arn")
        if job_arn:
            return current, sm.describe_processing_job(
                ProcessingJobName=job_arn.rsplit("/", 1)[-1])
        raise ValueError("FeatureSelection has no Processing job yet; choose an execution that reached it.")
    raise ValueError("Repeated FeatureSelection cache reference: " + current)


def _download_raw(s3, uri, destination):
    address = urlsplit(uri)
    if address.scheme != "s3" or not address.netloc or not address.path.strip("/"):
        raise ValueError("The job's raw input is not an S3 URI: " + uri)
    key = address.path.lstrip("/")
    metadata = s3.head_object(Bucket=address.netloc, Key=key)
    identity = {
        "uri": uri,
        "etag": metadata["ETag"],
        "bytes": metadata["ContentLength"],
        "version_id": metadata.get("VersionId"),
    }
    receipt = destination.with_suffix(".input.json")
    try:
        previous = json.loads(receipt.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        previous = None
    if previous == identity and destination.is_file() and destination.stat().st_size == identity["bytes"]:
        print(f"Reusing raw input: {destination}", flush=True)
        return
    print(f"Downloading {uri} ({identity['bytes']:,} bytes)", flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="download-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "data.csv"
        s3.download_file(address.netloc, key, str(staged))
        if staged.stat().st_size != identity["bytes"]:
            raise ValueError("Downloaded raw input size differs from the S3 object.")
        staged.replace(destination)
    receipt.write_text(json.dumps(identity, indent=2) + "\n")


def _prepare_imports(repository, model_id):
    source = repository / "mlops-models/src"
    for name, module in tuple(sys.modules.items()):
        if name == "mlops_models" or name.startswith("mlops_models."):
            origin = getattr(module, "__file__", None)
            if origin and not Path(origin).resolve().is_relative_to(source):
                raise RuntimeError("Restart the kernel: mlops_models was imported from another checkout.")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    importlib.invalidate_caches()
    # Reload the base before its subclass so reruns see edits in either file.
    for name in (
        "mlops_models.model_base.training.feature_selection",
        f"mlops_models.models.{model_id}.training.feature_selection",
    ):
        if name in sys.modules:
            importlib.reload(sys.modules[name])


def _vscode_code(model_id, execution_arn):
    """A standalone main block that resolves the editable model/execution inputs."""
    pipeline_arn = execution_arn.rsplit("/execution/", 1)[0]
    assignments = (
        "# Change these two values to test another model and execution.\n"
        f"MODEL_ID = {model_id!r}\n"
        f"EXECUTION_ID = {execution_arn!r}\n"
    )
    imports_and_lookup = r'''
from pathlib import Path
from tempfile import mkdtemp
from urllib.parse import urlsplit
import json
import os
import boto3
from botocore.config import Config

'''
    # Embed the same cache-aware lookup; the pasted block needs no helper file.
    imports_and_lookup += getsource(_processing_job) + "\n"
    imports_and_lookup += (
        "# A bare ID uses the original pipeline; a full ARN can select another one.\n"
        "execution_arn = EXECUTION_ID.strip()\n"
        "if not execution_arn:\n"
        '    raise ValueError("Set EXECUTION_ID to an execution ID or its full ARN.")\n'
        'if not execution_arn.startswith("arn:"):\n'
        '    if any(character in execution_arn for character in "/: \\t\\n"):\n'
        '        raise ValueError("Supply an execution ID or full ARN, not a console URL.")\n'
        f"    execution_arn = {pipeline_arn + '/execution/'!r} + execution_arn\n"
    )
    setup_and_call = r'''
parts = execution_arn.split(":", 5)
if (len(parts) != 6 or parts[2] != "sagemaker"
        or not parts[5].startswith("pipeline/") or "/execution/" not in parts[5]):
    raise ValueError("Supply a SageMaker pipeline execution ARN.")

# This block belongs to entrypoints/feature_selection.py, where execute is defined.
# Read the current checkout's parameters; do not reuse notebook-local paths.
model_directory = Path(__file__).resolve().parents[1] / "models" / MODEL_ID
if not MODEL_ID.isidentifier() or not (model_directory / "parameters.json").is_file():
    raise ValueError("Set MODEL_ID to a model in this checkout with parameters.json.")
session = boto3.Session(region_name=parts[3])
client_options = Config(connect_timeout=5, read_timeout=15, retries={"total_max_attempts": 2})
sm = session.client("sagemaker", config=client_options)
print("Inspecting execution:", execution_arn, flush=True)
producer_arn, job = _processing_job(sm, execution_arn)
recorded = job.get("Environment", {})
if recorded.get("MODEL_ID") != MODEL_ID:
    raise ValueError(f"This job belongs to {recorded.get('MODEL_ID')!r}, not {MODEL_ID!r}.")
raw = [item["S3Input"] for item in job["ProcessingInputs"]
       if item.get("InputName") == "raw" and "S3Input" in item]
if len(raw) != 1:
    raise ValueError("The FeatureSelection job must have one raw S3 input.")
raw_uri = raw[0]["S3Uri"].rstrip("/")
if not raw_uri.endswith(".csv"):
    raw_uri += "/data.csv"
address = urlsplit(raw_uri)
if address.scheme != "s3" or not address.netloc or not address.path.strip("/"):
    raise ValueError("The job's raw input is not an S3 URI: " + raw_uri)
print("Processing job:", job["ProcessingJobName"], "|", job["ProcessingJobStatus"], flush=True)
if producer_arn != execution_arn:
    print("Cached FeatureSelection producer:", producer_arn, flush=True)
print("Current checkout parameters:", model_directory / "parameters.json", flush=True)

work_directory = Path(mkdtemp(prefix="feature-selection-"))
data_directory = work_directory / "raw"
output_directory = work_directory / "output"
data_directory.mkdir()
output_directory.mkdir()
(output_directory / "job.json").write_text(json.dumps(job, indent=2, default=str) + "\n")
(output_directory / "parameters.used.json").write_bytes(
    (model_directory / "parameters.json").read_bytes()
)
print("Download:", raw_uri, flush=True)
session.client("s3", config=client_options).download_file(
    address.netloc, address.path.lstrip("/"), str(data_directory / "data.csv")
)
selection = recorded.get("MLOPS_TRAINING_SELECTION", recorded.get("MLOPS_TRAINING_MODE", "1"))
mode = {"Preliminary models with validation": "1", "Final model with all data": "2"}.get(selection, selection)
environment = {
    "MODEL_ID": MODEL_ID,
    "MLOPS_TRAINING_MODE": mode,
    "MLOPS_TRAINING_SELECTION": selection,
    "MLOPS_MODEL_METADATA_URI": recorded.get("MLOPS_MODEL_METADATA_URI", "none"),
    "MLOPS_TRACKING_URI": "",
    "MLOPS_WRITE_RUN_REPORT": "0",
    "MLOPS_LOG_ARTIFACTS": "0",
    "MLOPS_MODEL_DIR": str(model_directory),
    "SM_MODEL_DIR": str(model_directory),
    "SM_OUTPUT_DATA_DIR": str(output_directory),
    "SM_OUTPUT_DIR": str(output_directory),
}
previous = {key: os.environ.get(key) for key in environment}
result = None
try:
    os.environ.update(environment)
    result = execute(
        model_id=MODEL_ID,
        input_path=data_directory,
        output_dir=output_directory,
    )
    print("Returned value:", result)
finally:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    print("Inputs and outputs retained at:", work_directory, flush=True)
'''
    code = (
        "# Paste at the bottom of mlops-models/src/mlops_models/entrypoints/feature_selection.py.\n"
        "# Replace its previous local main block. Open the repository root in VS Code.\n"
        "# Use this checkout's Python interpreter with the project installed editable,\n"
        "# or the existing module-based F5 configuration. See README.md: Run a step locally.\n"
        'if __name__ == "__main__":\n'
        + indent(assignments + imports_and_lookup + setup_and_call, "    ")
    )
    compile(code, "feature_selection.py", "exec")
    return code


def prepare(repository, model_id, execution_id):
    """Resolve/download inputs without running selection; prepare the VS Code block."""
    started = perf_counter()
    repository = Path(repository).resolve()
    parameter_path = repository / "mlops-models/src/mlops_models/models" / model_id / "parameters.json"
    if not model_id.isidentifier() or not parameter_path.is_file():
        raise ValueError("Set MODEL_ID to a model in this checkout with parameters.json.")
    execution_id = execution_id.strip()
    if not execution_id:
        raise ValueError("Set EXECUTION_ID to a training execution ID or its full ARN.")
    if execution_id.startswith("arn:"):
        parts = execution_id.split(":", 5)
        if len(parts) != 6 or parts[2] != "sagemaker" or not parts[5].startswith("pipeline/") or "/execution/" not in parts[5]:
            raise ValueError("Supply a SageMaker pipeline execution ARN.")
        session = boto3.Session(region_name=parts[3])
        execution_arn = execution_id
    else:
        if any(character in execution_id for character in "/: \t\n"):
            raise ValueError("Supply a bare execution ID or the full execution ARN, not a console URL.")
        session = boto3.Session()
        execution_arn = None
    client_options = Config(connect_timeout=5, read_timeout=15, retries={"total_max_attempts": 2})
    sm = session.client("sagemaker", config=client_options)
    if execution_arn is None:
        config = json.loads((repository / "notebooks/config.json").read_text())
        namespace = config.get("namespace", "").strip()
        if not namespace:
            raise ValueError("Set namespace in notebooks/config.json, or use the full execution ARN.")
        pipeline = sm.describe_pipeline(PipelineName=f"{namespace}-training")
        execution_arn = pipeline["PipelineArn"] + "/execution/" + execution_id
    print("Inspecting execution:", execution_arn, flush=True)
    producer_arn, job = _processing_job(sm, execution_arn)
    recorded = job.get("Environment", {})
    if recorded.get("MODEL_ID") != model_id:
        raise ValueError(f"This job belongs to {recorded.get('MODEL_ID')!r}, not {model_id!r}.")
    raw = [item["S3Input"] for item in job["ProcessingInputs"]
           if item.get("InputName") == "raw" and "S3Input" in item]
    if len(raw) != 1:
        raise ValueError("The FeatureSelection job must have one raw S3 input.")
    uri = raw[0]["S3Uri"].rstrip("/")
    raw_uri = uri if uri.endswith(".csv") else uri + "/data.csv"
    job_name = job["ProcessingJobName"]
    if Path(job_name).name != job_name or job_name in {".", ".."}:
        raise ValueError("Unexpected Processing job name.")
    work = repository / ".runs/step-debug/feature-selection" / job_name
    data_directory = work / "raw"
    print("Processing job:", job_name, "|", job["ProcessingJobStatus"], flush=True)
    if producer_arn != execution_arn:
        print("Cached FeatureSelection producer:", producer_arn, flush=True)
    print("Raw input:", raw_uri, flush=True)
    work.mkdir(parents=True, exist_ok=True)
    output_directory = Path(mkdtemp(prefix="run-", dir=work))
    selection = recorded.get("MLOPS_TRAINING_SELECTION", recorded.get("MLOPS_TRAINING_MODE", "1"))
    mode = {"Preliminary models with validation": "1", "Final model with all data": "2"}.get(selection, selection)
    environment = {
        "MODEL_ID": model_id,
        "MLOPS_TRAINING_MODE": mode,
        "MLOPS_TRAINING_SELECTION": selection,
        "MLOPS_MODEL_METADATA_URI": recorded.get("MLOPS_MODEL_METADATA_URI", "none"),
        "MLOPS_TRACKING_URI": "",
        "MLOPS_WRITE_RUN_REPORT": "0",
        "MLOPS_LOG_ARTIFACTS": "0",
        "MLOPS_MODEL_DIR": str(parameter_path.parent),
        "SM_MODEL_DIR": str(parameter_path.parent),
        "SM_OUTPUT_DATA_DIR": str(output_directory),
        "SM_OUTPUT_DIR": str(output_directory),
    }
    vscode_code = _vscode_code(model_id, execution_arn)
    try:
        # Save the export before downloading or importing the model's dependencies.
        (output_directory / "vscode_main.py").write_text(vscode_code, encoding="utf-8")
        (output_directory / "job.json").write_text(json.dumps(job, indent=2, default=str) + "\n")
        (output_directory / "parameters.used.json").write_bytes(parameter_path.read_bytes())
        s3 = session.client("s3", config=client_options)
        _download_raw(s3, raw_uri, data_directory / "data.csv")
    except BaseException:
        print(vscode_code, flush=True)
        raise
    print("Current checkout parameters:", parameter_path, flush=True)
    print("Local output directory:", output_directory, flush=True)
    print(f"Inputs ready in {perf_counter() - started:.1f}s; run the next cell.", flush=True)
    return SimpleNamespace(
        repository=repository, model_id=model_id, data_directory=data_directory,
        output_directory=output_directory, environment=environment, vscode_code=vscode_code,
    )


@contextmanager
def local_environment(step):
    """Set local paths and environment; the notebook imports and calls its entrypoint."""
    previous = {key: os.environ.get(key) for key in step.environment}
    started = perf_counter()
    try:
        os.environ.update(step.environment)
        parameter_path = Path(step.environment["MLOPS_MODEL_DIR"]) / "parameters.json"
        (step.output_directory / "parameters.used.json").write_bytes(parameter_path.read_bytes())
        print("Metadata input:", step.environment["MLOPS_MODEL_METADATA_URI"], flush=True)
        _prepare_imports(step.repository, step.model_id)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        print(f"Local call ended after {perf_counter() - started:.1f}s; "
              f"outputs retained at {step.output_directory}", flush=True)
