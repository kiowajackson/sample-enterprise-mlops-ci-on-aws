"""Paste into a notebook to compare an edited file with one execution's source.

Set EXECUTION_ARN, STEP_NAME and the absolute LOCAL_SOURCE_FILE you are editing.
Set SOURCE_RELATIVE_FILE to that file's path inside the uploaded source folder.
Reads SageMaker metadata and S3 only. For artifact mode, streams the model archive
until it finds the source file; it does not extract or execute anything.
"""

EXECUTION_ARN = ""
STEP_NAME = "Score"  # Use Preprocess to inspect inference preprocessing.
LOCAL_SOURCE_FILE = ""  # Absolute path to the file open in your editor.
SOURCE_RELATIVE_FILE = "mlops_models/model_base/inference/predict.py"

import hashlib
from pathlib import Path, PurePosixPath
import tarfile
from urllib.parse import urlparse

import boto3

local_file = Path(LOCAL_SOURCE_FILE).expanduser()
if not EXECUTION_ARN or not local_file.is_absolute() or not local_file.is_file():
    raise ValueError("Set EXECUTION_ARN and an existing absolute LOCAL_SOURCE_FILE.")
relative_file = PurePosixPath(SOURCE_RELATIVE_FILE)
if relative_file.is_absolute() or ".." in relative_file.parts or relative_file.parts[0] != "mlops_models":
    raise ValueError("SOURCE_RELATIVE_FILE must start with mlops_models/ and stay inside it.")

session = boto3.Session(region_name=EXECUTION_ARN.split(":")[3])
sm, s3 = session.client("sagemaker"), session.client("s3")
execution = sm.describe_pipeline_execution(PipelineExecutionArn=EXECUTION_ARN)
print("Execution:", EXECUTION_ARN)
print("Pipeline version used:", execution.get("PipelineVersionId", "not returned"))
print("Execution status:", execution["PipelineExecutionStatus"])

producer = EXECUTION_ARN
visited = set()
while True:
    if producer in visited:
        raise ValueError("Repeated execution in the recorded reuse chain.")
    visited.add(producer)
    pages = sm.get_paginator("list_pipeline_execution_steps").paginate(PipelineExecutionArn=producer)
    matches = [step for page in pages for step in page["PipelineExecutionSteps"]
               if step["StepName"] == STEP_NAME]
    if len(matches) != 1:
        raise ValueError(f"Expected one {STEP_NAME} step in {producer}; found {len(matches)}.")
    step = matches[0]
    reused = step.get("CacheHitResult") or step.get("SelectiveExecutionResult") or {}
    if not reused.get("SourcePipelineExecutionArn"):
        break
    reason = "Cache hit" if step.get("CacheHitResult") else "Selective reuse"
    producer = reused["SourcePipelineExecutionArn"]
    print(reason + " — outputs came from:", producer)

job_arn = step.get("Metadata", {}).get("ProcessingJob", {}).get("Arn")
if not job_arn:
    raise ValueError(f"{STEP_NAME} has no recorded Processing job yet: {step['StepStatus']}")
job = sm.describe_processing_job(ProcessingJobName=job_arn.rsplit("/", 1)[-1])
environment = job.get("Environment", {})
mode = environment.get("MLOPS_MODEL_CODE_SOURCE", "artifact")
print("Producing job:", job_arn)
print("Job created:", job["CreationTime"])
print("Code source:", mode)
print("Job cache revision:", environment.get("MLOPS_STEP_REVISION", "not set"))

if mode == "pipeline":
    source_input = next(item["S3Input"] for item in job["ProcessingInputs"]
                        if item["InputName"] == "source")
    code_uri = source_input["S3Uri"].rstrip("/") + "/" + str(relative_file)
    location = urlparse(code_uri)
    with s3.get_object(Bucket=location.netloc, Key=location.path.lstrip("/"))["Body"] as stream:
        job_file = stream.read()
    print("Source file selected by job:", code_uri)
elif mode == "artifact":
    model_input = next(item["S3Input"] for item in job["ProcessingInputs"]
                       if item["InputName"] == "checkpoint")
    model_uri = model_input["S3Uri"]
    member_path = PurePosixPath("code") / relative_file
    location = urlparse(model_uri)
    job_file = None
    with s3.get_object(Bucket=location.netloc, Key=location.path.lstrip("/"))["Body"] as stream:
        with tarfile.open(fileobj=stream, mode="r|gz") as archive:
            for member in archive:
                if member.isfile() and PurePosixPath(member.name) == member_path:
                    with archive.extractfile(member) as source:
                        job_file = source.read()
                    break
    if job_file is None:
        raise FileNotFoundError(f"{member_path} was not found in {model_uri}")
    print("Source file selected by job:", model_uri, "::", member_path)
else:
    raise ValueError(f"Unexpected code-source mode: {mode!r}")

local_sha = hashlib.sha256(local_file.read_bytes()).hexdigest()
job_sha = hashlib.sha256(job_file).hexdigest()
print("Local file:", local_file.resolve())
print("Local file SHA-256:", local_sha)
print("Job source file SHA-256:", job_sha)
print("FILE MATCH" if local_sha == job_sha else "FILE DIFFERS — this job selected different source bytes")
print("This compares the recorded job inputs as stored in S3 now.")
print("For the loader's effective code directory, find source_loaded in this job's logs.")
