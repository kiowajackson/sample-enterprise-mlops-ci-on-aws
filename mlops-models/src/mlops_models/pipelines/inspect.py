"""Inspect native jobs and collect, write and read their artifact reports."""

import json
import os
from pathlib import Path
import time
from botocore.config import Config
from mlops_models.storage import s3_location, read_existing_json

COLLECTION_LIMITS = {"seconds": 120, "log_pages": 20, "log_bytes": 8 * 1024 * 1024, "objects": 1000}
READ_CONFIG = Config(connect_timeout=5, read_timeout=10, retries={"mode": "standard", "total_max_attempts": 2})


def collection_limits(overrides=None):
    """Resolve collection limits and reject unknown keys or nonpositive values."""
    limits = {**COLLECTION_LIMITS, **(overrides or {})}
    if limits.keys() != COLLECTION_LIMITS.keys() or any(type(value) is not int or value < 1 for value in limits.values()):
        raise ValueError("Collection limits must use the supported keys with positive integer values.")
    return limits


def wait_for_pipeline(session, execution_arn, *, timeout=3600, interval=15):
    """Wait for a pipeline to reach a terminal state and return its description, or raise on timeout."""
    client = session.client("sagemaker")
    deadline, previous = time.monotonic() + timeout, None
    while True:
        state = client.describe_pipeline_execution(PipelineExecutionArn=execution_arn)
        status = state["PipelineExecutionStatus"]
        if status != previous:
            print(f"{execution_arn}: {status}", flush=True)
            previous = status
        if status not in {"Executing", "Stopping"}:
            return state
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Pipeline is still active; inspect or stop {execution_arn}.")
        time.sleep(min(interval, remaining))


def inspect_managed(session, execution_arn, *, logs=False):
    """Take an execution ARN and return its pipeline, step and job details.

    - Read the execution state and its steps from SageMaker.
    - Fetch the processing, training and transform job descriptions.
    - Print their CloudWatch logs when requested.
    """
    client = session.client("sagemaker", config=READ_CONFIG)
    deadline, complete = time.monotonic() + 120, True
    description = client.describe_pipeline_execution(PipelineExecutionArn=execution_arn)
    steps = []
    for page in client.get_paginator("list_pipeline_execution_steps").paginate(
            PipelineExecutionArn=execution_arn):
        steps.extend(page["PipelineExecutionSteps"])
        if time.monotonic() >= deadline:
            complete = False
            break
    jobs, tuning = {}, {}
    kinds = {
        "TrainingJob": ("describe_training_job", "TrainingJobName", "/aws/sagemaker/TrainingJobs"),
        "ProcessingJob": ("describe_processing_job", "ProcessingJobName", "/aws/sagemaker/ProcessingJobs"),
        "TransformJob": ("describe_transform_job", "TransformJobName", "/aws/sagemaker/TransformJobs"),
    }
    for step in steps:
        if "TuningJob" in step.get("Metadata", {}):
            name = step["Metadata"]["TuningJob"]["Arn"].rsplit("/", 1)[-1]
            details, trials, finished = inspect_tuning(client, name, deadline)
            tuning[step["StepName"]] = details
            jobs.update({f"{step['StepName']}-{name}": job for name, job in trials.items()})
            complete = complete and finished
        for kind, (operation, argument, log_group) in kinds.items():
            if kind not in step.get("Metadata", {}):
                continue
            if time.monotonic() >= deadline:
                complete = False
                break
            job_name = step["Metadata"][kind]["Arn"].rsplit("/", 1)[-1]
            jobs[step["StepName"]] = getattr(client, operation)(**{argument: job_name})
            if logs:
                paginator = session.client("logs", config=READ_CONFIG).get_paginator("filter_log_events")
                start = int(jobs[step["StepName"]]["CreationTime"].timestamp() * 1000)
                displayed = 0
                for number, page in enumerate(paginator.paginate(
                    logGroupName=log_group, logStreamNamePrefix=job_name,
                    startTime=start, endTime=int(time.time() * 1000),
                ), start=1):
                    for event in page["events"]:
                        displayed += len(event["message"].encode())
                        if displayed > COLLECTION_LIMITS["log_bytes"]:
                            break
                        print(f"[{step['StepName']}] {event['message']}")
                    if (number >= COLLECTION_LIMITS["log_pages"] or displayed > COLLECTION_LIMITS["log_bytes"]
                            or time.monotonic() >= deadline):
                        print("Log display limit reached; inspect the remaining events in CloudWatch.")
                        break
    return {"mode": "managed", "execution": description, "steps": steps, "jobs": jobs, "tuning": tuning,
            "inspection_complete": complete}



def inspect_tuning(client, name, deadline):
    """Read the parent and every trial status, including failures, with bounded job inspection."""
    parent = client.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=name)
    summaries, jobs, complete = [], {}, True
    paginator = client.get_paginator("list_training_jobs_for_hyper_parameter_tuning_job")
    for page in paginator.paginate(HyperParameterTuningJobName=name):
        summaries.extend(page["TrainingJobSummaries"])
        if time.monotonic() >= deadline or len(summaries) > 20:
            complete = False
            break
    for trial in summaries[:20]:
        if time.monotonic() >= deadline:
            complete = False
            break
        job_name = trial["TrainingJobName"]
        jobs[job_name] = client.describe_training_job(TrainingJobName=job_name)
    return {"job": parent, "trials": summaries, "inspection_complete": complete}, jobs, complete


def read_parameters(client, execution_arn):
    """Take an execution ARN and return the parameter values SageMaker used for that run."""
    parameters, token = {}, None
    while True:
        request = {"PipelineExecutionArn": execution_arn}
        if token:
            request["NextToken"] = token
        response = client.list_pipeline_parameters_for_execution(**request)
        parameters.update({item["Name"]: item["Value"] for item in response["PipelineParameters"]})
        token = response.get("NextToken")
        if not token:
            return parameters


def object_inventory(s3, bucket, prefix, *, limit, deadline):
    """Inspect bounded object metadata; return the inventory and whether uninspected objects may remain."""
    inventory = []
    request = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": min(limit + 2, 1000)}
    while time.monotonic() < deadline:
        page = s3.list_objects_v2(**request)
        for item in page.get("Contents", []):
            if len(inventory) >= limit or time.monotonic() >= deadline:
                return inventory, True
            head = s3.head_object(Bucket=bucket, Key=item["Key"])
            inventory.append({"path": item["Key"][len(prefix):], "bytes": head["ContentLength"],
                              "etag": head["ETag"], "version_id": head.get("VersionId"),
                              "sha256": head.get("Metadata", {}).get("sha256")})
        if not page.get("IsTruncated"):
            return inventory, False
        request["ContinuationToken"] = page["NextContinuationToken"]
    return inventory, True


def inspect_run(session, result, output_dir=None, *, limits=None):
    """Read a run's native jobs, registry records and existing S3 objects; optionally save one local report.

    - Resolve the run folder from its executed parameters.
    - Read the registered packages and inventory the files already written by SageMaker.
    - Return resource links without uploading logs, service descriptions or duplicate artifacts.
    """
    limits = collection_limits(limits)
    client, s3 = session.client("sagemaker"), session.client("s3")
    execution = result["execution"]
    arn = execution["PipelineExecutionArn"]
    parameters = read_parameters(client, arn)
    root = (parameters["OutputDataRoot"].rstrip("/") + "/" + parameters["ModelId"] + "/runs"
            if "OutputDataRoot" in parameters else parameters["OutputDataUri"].rstrip("/"))
    uri = root + "/" + arn.rsplit("/", 1)[-1]
    bucket, prefix = s3_location(uri + "/")
    objects, truncated = object_inventory(s3, bucket, prefix, limit=limits["objects"],
                                          deadline=time.monotonic() + limits["seconds"])
    packages = [client.describe_model_package(ModelPackageName=step["Metadata"]["RegisterModel"]["Arn"])
                for step in result["steps"] if "RegisterModel" in step.get("Metadata", {})]
    report = {"execution_arn": arn, "status": execution["PipelineExecutionStatus"],
              "artifact_uri": uri, "parameters": parameters, "registry": packages,
              "objects": objects, "jobs": result["jobs"], "tuning": result.get("tuning", {}),
              "collection": {"status": "partial" if truncated or not result.get("inspection_complete", True) else "complete", "read_only": True}}
    if output_dir is not None:
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "run.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    return report


# Existing execution outputs, including the original producers of cached steps.
MAX_ARTIFACT_BYTES = 10 * 1024 * 1024
MAX_OBJECTS = 1000


def _json(value):
    return json.dumps(value, sort_keys=True, default=str, allow_nan=False)



def _read_json(s3, uri, *, optional=False):
    bucket, key = s3_location(uri)
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
    except s3.exceptions.ClientError as error:
        if optional and error.response["Error"]["Code"] in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    with response["Body"] as stream:
        content = stream.read(MAX_ARTIFACT_BYTES + 1)
    if len(content) > MAX_ARTIFACT_BYTES:
        raise ValueError(f"Tracking JSON exceeds {MAX_ARTIFACT_BYTES} bytes: {uri}")
    return json.loads(content)



def _put_json(s3, uri, value, kms_key="", **conditions):
    bucket, key = s3_location(uri)
    encryption = {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": kms_key} if kms_key else {}
    return s3.put_object(Bucket=bucket, Key=key, Body=(_json(value) + "\n").encode(),
                         ContentType="application/json", **encryption, **conditions)



def _steps(sm, arn):
    return [step for page in sm.get_paginator("list_pipeline_execution_steps").paginate(
        PipelineExecutionArn=arn) for step in page["PipelineExecutionSteps"]]



def add_record(report, entry, record):
    """Consume the producer's declared files; no step names or report filenames."""
    entry["source_identity"] = record
    if record.get("record_version") != 1:
        entry["artifact_record_available"] = False
        return
    entry["artifact_record_available"] = True
    producer = entry["producer_execution_arn"]
    for artifact in record.get("artifacts", []):
        report["artifacts"].append({**artifact, "step": entry["name"],
                                    "producer_execution_arn": producer})
    for name, value in record.get("reports", {}).items():
        report["reports"][entry["name"] + "/" + name] = value
    for dataset in record.get("datasets", []):
        report["datasets"].append({**dataset, "producer_execution_arn": producer})
    for receipt in record.get("registry", []):
        if not any(item["receipt"]["package"] == receipt["package"] for item in report["registry"]):
            report["registry"].append({"receipt": receipt})


def run_report_uri(execution_arn, parameters):
    return (parameters["OutputDataRoot"].rstrip("/") + "/" + parameters["ModelId"] +
            "/runs/" + execution_arn.rsplit("/", 1)[-1] + "/run.json")


def read_run_report(session, execution_arn):
    """Read a successful execution's standard report without knowing its steps."""
    sm = session.client("sagemaker")
    state = sm.describe_pipeline_execution(PipelineExecutionArn=execution_arn)
    if state["PipelineExecutionStatus"] != "Succeeded":
        raise ValueError("ReferenceExecutionArn must identify a successful execution.")
    parameters = read_parameters(sm, execution_arn)
    uri = run_report_uri(execution_arn, parameters)
    report = _read_json(session.client("s3"), uri, optional=True)
    if not report or report.get("report_version") != 1:
        raise ValueError("Reference execution has no job-written run report. Run inference with the updated pipeline.")
    if report["execution"]["PipelineExecutionArn"] != execution_arn:
        raise ValueError("The saved report belongs to a different execution.")
    return report


def collect_execution(session, execution_arn, *, step_names=None, use_saved_report=True):
    """Read a job-written report, refreshing native status. Otherwise collect producer records.

    Native job descriptions locate records using the declared outputs. Adding or
    renaming a model step does not require a new inspection rule. Old executions
    without artifact records retain native job details but have no file inventory.
    """
    sm = session.client("sagemaker", region_name=execution_arn.split(":")[3], config=READ_CONFIG)
    s3 = session.client("s3", config=READ_CONFIG)
    execution = sm.describe_pipeline_execution(PipelineExecutionArn=execution_arn)
    execution.pop("ResponseMetadata", None)
    parameters = read_parameters(sm, execution_arn)
    native = _steps(sm, execution_arn)
    saved = None
    if use_saved_report and step_names is None and parameters.get("OutputDataRoot"):
        saved = _read_json(s3, run_report_uri(execution_arn, parameters), optional=True)
        if saved and saved["execution"]["PipelineExecutionArn"] != execution_arn:
            raise ValueError("Run report belongs to a different execution.")
        states = {step["StepName"]: step for step in native}
        # Partial caller snapshots and a retried job need fresh producer records.
        reusable = (saved and saved.get("report_version") == 1
                    and saved.get("report_phase") in {"outputs_prepared", "finalized"}
                    and {step["name"] for step in saved["steps"]} == set(states)
                    and all(step.get("attempt", 0) == states[step["name"]].get("AttemptCount", 0)
                            for step in saved["steps"])
                    and all(step.get("metadata", {}) == states[step["name"]].get("Metadata", {})
                            for step in saved["steps"] if not step.get("cached"))
                    and not (saved.get("report_phase") == "finalized"
                             and execution["PipelineExecutionStatus"] == "Executing"))
        if reusable:
            if saved["execution"]["PipelineExecutionArn"] != execution_arn:
                raise ValueError("Run report belongs to a different execution.")
            saved["execution"] = execution
            for step in saved["steps"]:
                state = states[step["name"]]
                step.update(status=state["StepStatus"], failure=state.get("FailureReason", ""),
                            start=state.get("StartTime"), end=state.get("EndTime"))
            return saved
    by_execution = {execution_arn: native}
    result = {"execution": execution, "parameters": parameters, "steps": [],
              "reports": {}, "artifacts": [], "registry": [], "datasets": []}
    # A later conditional job can require fresh records. Preserve Postprocess's
    # explicit input mapping so this execution can still be used as a reference.
    if saved and saved.get("report_version") == 1 and "inference" in saved:
        result["inference"] = saved["inference"]
    operations = {
        "ProcessingJob": ("describe_processing_job", "ProcessingJobName"),
        "TrainingJob": ("describe_training_job", "TrainingJobName"),
        "TransformJob": ("describe_transform_job", "TransformJobName"),
    }
    for step in native:
        name = step["StepName"]
        if step_names is not None and name not in step_names:
            continue
        producer, original, seen = execution_arn, step, set()
        while source := original.get("CacheHitResult", {}).get("SourcePipelineExecutionArn"):
            if source in seen or len(seen) >= 20:
                raise ValueError("Invalid or excessive cached-producer chain.")
            seen.add(source)
            if source not in by_execution:
                by_execution[source] = _steps(sm, source)
            matches = [item for item in by_execution[source] if item["StepName"] == name]
            if len(matches) != 1:
                raise ValueError(f"Cannot resolve cached producer for {name} in {source}.")
            original, producer = matches[0], source
        entry = {"name": name, "status": step["StepStatus"], "cached": producer != execution_arn,
                 "producer_execution_arn": producer, "failure": step.get("FailureReason", ""),
                 "start": step.get("StartTime"), "end": step.get("EndTime"),
                 "attempt": step.get("AttemptCount", 0), "metadata": original.get("Metadata", {})}
        result["steps"].append(entry)
        for kind, (operation, argument) in operations.items():
            metadata = original.get("Metadata", {})
            if kind not in metadata:
                continue
            arn = metadata[kind]["Arn"]
            job = getattr(sm, operation)(**{argument: arn.rsplit("/", 1)[-1]})
            job.pop("ResponseMetadata", None)
            entry.update(job_arn=arn, job_type=kind, job=job)
            env = job.get("Environment", {})
            entry["source_revision"] = env.get("MLOPS_STEP_REVISION", env.get("MLOPS_PREPARATION_REVISION", ""))
            records = []
            if kind == "ProcessingJob":
                entry["image"] = job["AppSpecification"]["ImageUri"]
                entry["outputs"] = {item["OutputName"]: item["S3Output"]["S3Uri"].rstrip("/") + "/"
                                    for item in job.get("ProcessingOutputConfig", {}).get("Outputs", [])}
                records = [uri + "_execution.json" for uri in entry["outputs"].values()]
            elif kind == "TrainingJob":
                entry["image"] = job["AlgorithmSpecification"]["TrainingImage"]
                entry["model_artifact"] = (job["OutputDataConfig"]["S3OutputPath"].rstrip("/") +
                    "/application/model.tar.gz" if env.get("MLOPS_TRAINING_OUTPUT_UPLOAD") == "application"
                    else job.get("ModelArtifacts", {}).get("S3ModelArtifacts", ""))
                if entry["model_artifact"]:
                    records = [entry["model_artifact"].rsplit("/", 1)[0] + "/_execution.json"]
            else:
                entry["model_name"] = job["ModelName"]
                entry["transform_output"] = job["TransformOutput"]["S3OutputPath"]
            if step["StepStatus"] != "Succeeded":
                continue
            for uri in records:
                record = _read_json(s3, uri, optional=True)
                if record:
                    add_record(result, entry, record)
                    break  # One producer record describes all of its declared outputs.
    for item in result["registry"]:
        package = sm.describe_model_package(ModelPackageName=item["receipt"]["package"])
        package.pop("ResponseMetadata", None)
        item["package"] = package
    return result


def write_run_index(session, report, *, receipt=None, kms_key=""):
    """Save the processing report; finalization refreshes native status in the same file."""
    uri = run_report_uri(report["execution"]["PipelineExecutionArn"], report["parameters"])
    s3 = session.client("s3", config=READ_CONFIG)
    previous = _read_json(s3, uri, optional=True) or {}
    index = {**report, "report_version": 1, "mlflow": receipt or previous.get("mlflow")}
    if report["execution"]["PipelineExecutionStatus"] in {"Succeeded", "Failed", "Stopped"}:
        index["report_phase"] = "finalized"
    else:
        index.setdefault("report_phase", "status_snapshot")
    if previous.get("native_lineage_uri"):
        index["native_lineage_uri"] = previous["native_lineage_uri"]
    _put_json(s3, uri, index, kms_key)
    return uri


def write_inference_report(output, *, data_path, selection_file, predictions, config):
    """Assemble the final inference report from records emitted by its producers."""
    if os.getenv("MLOPS_WRITE_RUN_REPORT") != "1":
        return None
    import boto3
    from mlops_models.runtime.load_source import processing_record

    job = json.loads(Path("/opt/ml/config/processingjobconfig.json").read_text())
    pipeline = os.environ["MLOPS_PIPELINE_ARN"]
    arn = pipeline + "/execution/" + os.environ["PIPELINE_EXECUTION_ID"]
    session = boto3.Session(region_name=pipeline.split(":")[3])
    # Upstream records contain their actual artifacts; native metadata supplies
    # statuses and cached producer links, with no model-step name conventions.
    report = collect_execution(session, arn, use_saved_report=False)
    local_root = Path(output)
    output_root = Path(job["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["LocalPath"])
    own = processing_record(job, {
        "step": os.environ["MLOPS_STEP_NAME"],
        "source": json.loads((Path(__file__).parents[2] / "source.json").read_text()),
        "producer_execution_id": os.environ["PIPELINE_EXECUTION_ID"],
        "model_code_source": os.getenv("MLOPS_MODEL_CODE_SOURCE", "artifact"),
    }, output_root, local_root)
    entry = next(step for step in report["steps"] if step["name"] == own["step"])
    add_record(report, entry, own)
    report["inference"] = inference_artifacts(
        job, data_path=data_path, selection_file=selection_file, predictions=predictions, config=config)
    # This report-producing job is still running. The caller adds terminal status.
    report["report_phase"] = "outputs_prepared"
    return write_run_index(session, report, kms_key=job["ProcessingOutputConfig"].get("KmsKeyId", ""))


def inference_artifacts(job, *, data_path, selection_file, predictions, config):
    """Give the comparison its semantic inputs where the processing code owns them."""
    mounts = [(Path(item["S3Input"]["LocalPath"]), item["S3Input"]["S3Uri"])
              for item in job["ProcessingInputs"] if "S3Input" in item]

    def locate(path):
        path = Path(path)
        matches = [(mount, uri) for mount, uri in mounts if path.is_relative_to(mount)]
        if not matches:
            raise ValueError(f"No declared Processing input contains {path}.")
        mount, uri = max(matches, key=lambda item: len(item[0].parts))
        producer_file = mount / "_execution.json"
        producer = json.loads(producer_file.read_text()) if producer_file.exists() else {}
        return {"uri": uri.rstrip("/") + "/" + path.relative_to(mount).as_posix(),
                "producer_execution_arn": producer.get("producer_execution_arn", "")}

    return {
        "model_id": os.environ["MODEL_ID"],
        "environment": os.environ["MLOPS_ENVIRONMENT"],
        "model_package_arn": os.environ["MLOPS_MODEL_PACKAGE"],
        "data": locate(data_path), "selection": locate(selection_file),
        "predictions": {name: locate(path) for name, path in predictions.items()},
        "primary_key": config["primary_key"], "target": config.get("target_key"),
    }


def index_execution(session, execution_arn, *, kms_key=""):
    """Save native status and producer records without requiring MLflow."""
    report = collect_execution(session, execution_arn)
    return write_run_index(session, report, kms_key=kms_key)
