"""Download the actual inputs of one Processing post-processing job for local use."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


def execution_arn(execution, pipeline_arn=""):
    """Accept a full execution ARN, or an execution ID with its pipeline ARN."""
    value = execution.strip()
    if not value:
        raise ValueError("Set INFERENCE_EXECUTION to an inference execution ARN or ID.")
    if not value.startswith("arn:"):
        if not re.fullmatch(r"[A-Za-z0-9-]+", value) or not pipeline_arn:
            raise ValueError("For an execution ID, run notebook 02 first or supply the full execution ARN.")
        value = pipeline_arn.rstrip("/") + "/execution/" + value
    if not re.fullmatch(
        r"arn:aws[a-z-]*:sagemaker:[a-z0-9-]+:\d{12}:pipeline/[^/]+/execution/[^/]+", value
    ):
        raise ValueError("Expected a SageMaker pipeline execution ARN.")
    return value


def postprocess_job(sm, arn):
    """Follow the selected Postprocess step to its real job, including a cache hit."""
    seen = set()
    while arn not in seen:
        seen.add(arn)
        steps = [
            step
            for page in sm.get_paginator("list_pipeline_execution_steps").paginate(
                PipelineExecutionArn=arn
            )
            for step in page["PipelineExecutionSteps"]
            if step["StepName"] == "Postprocess"
        ]
        if len(steps) != 1:
            raise ValueError(f"{arn}: no single Postprocess step. Choose a Processing inference execution.")
        step = steps[0]
        job_arn = step.get("Metadata", {}).get("ProcessingJob", {}).get("Arn")
        if job_arn:
            # Failed jobs are useful here: their inputs are exactly what we want to debug.
            return sm.describe_processing_job(ProcessingJobName=job_arn.rsplit("/", 1)[-1])
        source = step.get("CacheHitResult", {}).get("SourcePipelineExecutionArn")
        if not source:
            raise ValueError(f"Postprocess has no job yet (status: {step['StepStatus']}).")
        arn = source
    raise ValueError("Postprocess cache references form a cycle.")


def download_input(s3, uri, destination):
    """Copy an exact S3 object or all objects below a directory prefix."""
    if not uri.startswith("s3://") or "/" not in uri[5:]:
        raise ValueError(f"Expected an S3 input location: {uri}")
    bucket, key = uri[5:].split("/", 1)
    if not key:
        raise ValueError("An input must name an object or folder, not the bucket root.")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        s3.head_object(Bucket=bucket, Key=key)
    except ClientError as error:
        if error.response["Error"]["Code"] not in {"404", "NoSuchKey", "NotFound"}:
            raise
        prefix = key.rstrip("/") + "/"
        objects = [
            (item["Key"], item["Key"][len(prefix):])
            for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
            for item in page.get("Contents", [])
            if not item["Key"].endswith("/")
        ]
    else:
        objects = [(key, key.rsplit("/", 1)[-1])]
    if not objects:
        raise FileNotFoundError(f"No input files found at {uri}")
    for object_key, relative in objects:
        name = PurePosixPath(relative)
        if name.is_absolute() or ".." in name.parts or not name.name:
            raise ValueError(f"Unsafe S3 filename: {relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        # Fresh directories prevent combining files from different executions.
        with s3.get_object(Bucket=bucket, Key=object_key)["Body"] as source, target.open("xb") as output:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                output.write(block)


def prepare_postprocess(execution, destination, *, pipeline_arn="", session=None):
    """Download matching job inputs; return explicit paths without importing model code."""
    arn = execution_arn(execution, pipeline_arn)
    region = arn.split(":")[3]
    session = session or boto3.Session(region_name=region)
    limits = Config(connect_timeout=10, read_timeout=30,
                    retries={"mode": "standard", "total_max_attempts": 3})
    sm = session.client("sagemaker", region_name=region, config=limits)
    s3 = session.client("s3", region_name=region, config=limits)
    job = postprocess_job(sm, arn)
    environment = job.get("Environment", {})
    # Match the runner: command-line values override environment defaults.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", default=environment.get("MODEL_ID", ""))
    parser.add_argument("--analysis-mode", default=environment.get("MLOPS_ANALYSIS_MODE", "prediction_inference"))
    args, _ = parser.parse_known_args(job.get("AppSpecification", {}).get("ContainerArguments", []))
    model_id, mode = args.model, args.analysis_mode
    if not model_id or mode not in {"prediction_test", "prediction_inference"}:
        raise ValueError("The job must identify the model and a supported analysis mode.")
    if environment.get("MLOPS_PREDICTION_FORMAT", "csv") != "csv":
        raise ValueError("This local example expects the Processing CSV prediction route.")

    mounts = {
        item["S3Input"]["LocalPath"].rstrip("/"): item["S3Input"]["S3Uri"]
        for item in job.get("ProcessingInputs", []) if "S3Input" in item
    }
    folders = {"raw": "/opt/ml/processing/raw", "selection": "/opt/ml/processing/selection",
               "input": "/opt/ml/processing/input", "model": "/opt/ml/processing/model"}
    for folder, mount in folders.items():
        if mount not in mounts:
            raise ValueError(f"Postprocess input {mount} is missing; available mounts: {sorted(mounts)}")
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    for folder, mount in folders.items():
        print(f"{mounts[mount]} -> {destination / folder}", flush=True)
        download_input(s3, mounts[mount], destination / folder)

    model = destination / "model"
    if not (model / "parameters.json").is_file():
        archives = list(model.glob("*.tar.gz"))
        if len(archives) != 1:
            raise ValueError(f"Expected one model archive in {model}; found {len(archives)}.")
        with tarfile.open(archives[0]) as archive:
            archive.extractall(model, filter="data")
    paths = {
        "data_directory": str(destination / "raw"),
        "selection_file": str(destination / "selection/selected_features.csv"),
        "prediction_directory": str(destination / "input"),
        "output_directory": str(destination / "output"),
        "model_directory": str(model),
    }
    for path in [destination / "raw/data.csv", Path(paths["selection_file"]),
                 model / "parameters.json", model / "model_metadata.json"]:
        if not path.is_file():
            raise FileNotFoundError(f"Downloaded inputs are missing {path}")
    expected = environment.get("MLOPS_MODEL_METADATA_SHA256")
    if expected and hashlib.sha256((model / "model_metadata.json").read_bytes()).hexdigest() != expected:
        raise ValueError("The downloaded model metadata does not match the checksum recorded by this job.")
    if not any(p.is_file() and p.suffix.lower() == ".csv"
               for p in (destination / "input").rglob("*")):
        raise FileNotFoundError("The downloaded predictions contain no CSV files.")
    Path(paths["output_directory"]).mkdir()
    inputs = {**paths, "model_id": model_id, "analysis_mode": mode,
              "execution_arn": arn, "job_arn": job["ProcessingJobArn"]}
    (destination / "paths.json").write_text(json.dumps(inputs, indent=2) + "\n")
    print("Local inputs ready:", destination / "paths.json", flush=True)
    return inputs
