"""Copy details of an existing SageMaker pipeline execution into an MLflow run.

source.configure_tracking() declares the tracking Processing jobs. When a pipeline
executes, those jobs run this module's main() with that execution's ARN:

    main() -> sync_execution() -> collect_execution() -> publish_execution()

collect_execution reads SageMaker job details and S3 output files into a report.
publish_execution creates or updates the corresponding MLflow run from that report.
Neither starts the training/inference pipeline. Notebook 02 can also import
sync_execution() and supply an existing execution ARN to refresh its MLflow record.

A SageMaker pipeline execution and an MLflow run are separate records. The S3
run.json receipt stores their association so repeated updates reuse the same run.
"""

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from uuid import uuid4

from botocore.config import Config
from mlops_models.pipelines.inspect import read_parameters
from mlops_models.storage import s3_location

READ_CONFIG = Config(connect_timeout=5, read_timeout=15,
                     retries={"mode": "standard", "total_max_attempts": 2})
MAX_ARTIFACT_BYTES = 10 * 1024 * 1024
MAX_OBJECTS = 1000
TRACKING_STEPS = {"StartTracking", "TrackTraining", "TrackTunedTraining",
                  "TrackInference", "TrackTransform"}
TERMINAL = {"Succeeded": "FINISHED", "Failed": "FAILED", "Stopped": "KILLED"}


def validate_tracking(settings):
    """Validate an explicitly supplied managed tracking service; empty means disabled."""
    if not settings:
        return None
    uri = settings.get("uri", "")
    if not re.fullmatch(
        r"arn:aws(?:-us-gov|-cn)?:sagemaker:[a-z0-9-]+:\d{12}:"
        r"mlflow-(?:app|tracking-server)/[A-Za-z0-9-]+", uri
    ):
        raise ValueError("tracking.uri must be the ARN of an existing SageMaker MLflow service.")
    prefix = settings.get("experiment_prefix", "models")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./-]{0,127}", prefix) or ".." in prefix.split("/"):
        raise ValueError("Use an ordinary experiment_prefix without parent traversal.")
    return {**settings, "uri": uri, "experiment_prefix": prefix.rstrip("/")}


@contextmanager
def managed_client(uri):
    """Use the current AWS credential chain, restoring the caller's MLflow URI."""
    import mlflow
    import sagemaker_mlflow  # noqa: F401; installs SigV4/client plugins

    previous = mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri(uri)
        yield mlflow.MlflowClient(tracking_uri=uri)
    finally:
        mlflow.set_tracking_uri(previous)


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


@contextmanager
def _lease(s3, uri, kms_key):
    """Allow one writer at a time using a temporary S3 lock file.

    Create the lock or take over an expired one, yield to the caller, then remove
    our lock. This protects run creation when a job and notebook sync overlap.
    """
    token = uuid4().hex
    payload = {"owner": token, "expires": time.time() + 900}
    try:
        response = _put_json(s3, uri, payload, kms_key, IfNoneMatch="*")
    except s3.exceptions.ClientError as error:
        if error.response["Error"]["Code"] not in {"PreconditionFailed", "412", "ConditionalRequestConflict"}:
            raise
        bucket, key = s3_location(uri)
        previous = s3.get_object(Bucket=bucket, Key=key)
        with previous["Body"] as stream:
            lock = json.load(stream)
        if lock["expires"] > time.time():
            raise RuntimeError("This execution is already being synchronized; retry after its publisher finishes.") from error
        response = _put_json(s3, uri, payload, kms_key, IfMatch=previous["ETag"])
    try:
        yield
    finally:
        bucket, key = s3_location(uri)
        try:
            s3.delete_object(Bucket=bucket, Key=key, IfMatch=response["ETag"])
        except s3.exceptions.ClientError as error:
            if error.response["Error"]["Code"] not in {"PreconditionFailed", "412", "NoSuchKey", "404"}:
                raise


def _steps(sm, arn):
    return [step for page in sm.get_paginator("list_pipeline_execution_steps").paginate(
        PipelineExecutionArn=arn) for step in page["PipelineExecutionSteps"]]


def collect_execution(session, execution_arn):
    """Read an existing execution and return a report dictionary; make no writes.

    1. Read the execution's status, parameters and step list from SageMaker.
    2. Resolve each step to its actual job, following cached producers when needed.
    3. Read saved output records, Registry references and reports from S3.
    """
    # 1. Read the execution selected by the caller and prepare an empty report.
    sm = session.client("sagemaker", region_name=execution_arn.split(":")[3], config=READ_CONFIG)
    s3 = session.client("s3", config=READ_CONFIG)
    execution = sm.describe_pipeline_execution(PipelineExecutionArn=execution_arn)
    execution.pop("ResponseMetadata", None)
    parameters = read_parameters(sm, execution_arn)
    by_execution = {execution_arn: _steps(sm, execution_arn)}
    result = {"execution": execution, "parameters": parameters, "steps": [],
              "reports": {}, "artifacts": [], "registry": [], "datasets": []}

    # Map each recorded job type to the SageMaker API used to describe that job.
    operations = {
        "ProcessingJob": ("describe_processing_job", "ProcessingJobName"),
        "TrainingJob": ("describe_training_job", "TrainingJobName"),
        "TransformJob": ("describe_transform_job", "TransformJobName"),
    }

    # 2. Inspect model steps. Tracking jobs themselves are excluded from the report.
    for step in by_execution[execution_arn]:
        name = step["StepName"]
        if name in TRACKING_STEPS:
            continue

        # A cache hit points to an earlier execution. Follow it to the job that
        # actually produced the files, while retaining this execution's step status.
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
                 "attempt": step.get("AttemptCount", 0)}
        result["steps"].append(entry)

        # Fetch that producer job's image, source revision and output locations.
        metadata = original.get("Metadata", {})
        for kind, (operation, argument) in operations.items():
            if kind not in metadata:
                continue
            arn = metadata[kind]["Arn"]
            job = getattr(sm, operation)(**{argument: arn.rsplit("/", 1)[-1]})
            job.pop("ResponseMetadata", None)
            entry.update(job_arn=arn, job_type=kind, job=job)
            env = job.get("Environment", {})
            entry["source_revision"] = env.get("MLOPS_STEP_REVISION", env.get("MLOPS_PREPARATION_REVISION", ""))
            outputs = []
            if kind == "ProcessingJob":
                outputs = [(item["OutputName"], item["S3Output"]["S3Uri"])
                           for item in job.get("ProcessingOutputConfig", {}).get("Outputs", [])]
                if name in {"RegisterModel", "RegisterTunedModel"}:
                    registration = env["REGISTRATION_OUTPUT_URI"]
                    if not any(uri.rstrip("/") == registration.rstrip("/") for _, uri in outputs):
                        outputs.append(("registry", registration))
            elif kind == "TrainingJob":
                entry["image"] = job["AlgorithmSpecification"]["TrainingImage"]
                entry["model_artifact"] = (job["OutputDataConfig"]["S3OutputPath"].rstrip("/") +
                                          "/application/model.tar.gz" if
                                          env.get("MLOPS_TRAINING_OUTPUT_UPLOAD") == "application"
                                          else job.get("ModelArtifacts", {}).get("S3ModelArtifacts", ""))
            else:
                entry["model_name"] = job["ModelName"]
                entry["transform_output"] = job["TransformOutput"]["S3OutputPath"]
            if kind == "ProcessingJob":
                entry["image"] = job["AppSpecification"]["ImageUri"]

            # 3. Read the files from successful outputs. Failed jobs may leave
            # partial files, so their job/failure details are retained without these.

            if step["StepStatus"] != "Succeeded":
                continue
            for output_name, prefix in outputs:
                prefix = prefix.rstrip("/") + "/"
                entry.setdefault("outputs", {})[output_name] = prefix
                # This loader-written file says which code/settings produced the output.
                provenance = _read_json(s3, prefix + "_execution.json", optional=True)
                if provenance:
                    entry["source_identity"] = provenance

                # GetData separately records the input dataset's origin.
                if name == "GetData":
                    data = _read_json(s3, prefix + "provenance.json", optional=True)
                    if data:
                        result["datasets"].append({"uri": prefix + "data.csv", "provenance": data,
                                                   "producer_execution_arn": producer})

                # Read the exact registered model version from the step's receipt.
                if name in {"RegisterModel", "RegisterTunedModel", "ResolveModel"}:
                    receipt = _read_json(s3, prefix + "package.json")
                    package = sm.describe_model_package(ModelPackageName=receipt["package"])
                    package.pop("ResponseMetadata", None)
                    result["registry"].append({"receipt": receipt, "package": package})

                # Inventory report/analysis files and read the known JSON reports.
                # CSVs and plots are listed here; publish_execution copies them later.
                if name in {"Evaluate", "EvaluateTuned", "Postprocess", "PostprocessTransform"}:
                    bucket, key = s3_location(prefix)
                    count = 0
                    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=key):
                        for item in page.get("Contents", []):
                            count += 1
                            if count > MAX_OBJECTS:
                                raise ValueError(f"Tracking output inventory exceeds {MAX_OBJECTS} objects: {prefix}")
                            relative = item["Key"][len(key):]
                            if not relative or relative.endswith("/"):
                                continue
                            artifact = {"step": name, "uri": f"s3://{bucket}/{item['Key']}",
                                        "relative_path": relative, "size": item["Size"],
                                        "etag": item["ETag"], "producer_execution_arn": producer}
                            result["artifacts"].append(artifact)
                            if relative in {"evaluation.json", "parameters.json", "model_metadata.json",
                                            "summary.json", "training_execution.json"}:
                                result["reports"][f"{name}/{relative}"] = _read_json(s3, artifact["uri"])
    return result


def _experiment(client, name):
    from mlflow.exceptions import MlflowException
    existing = client.get_experiment_by_name(name)
    if existing:
        return existing.experiment_id
    try:
        return client.create_experiment(name)
    except MlflowException as error:
        if error.error_code != "RESOURCE_ALREADY_EXISTS":
            raise
        return client.get_experiment_by_name(name).experiment_id


def _find_run(client, experiment_id, arn):
    runs = client.search_runs([experiment_id], f"tags.`pipeline.execution_arn` = '{arn}'", max_results=2)
    if len(runs) > 1:
        raise ValueError(f"Multiple tracking runs already claim execution {arn}; reconcile them before retrying.")
    return runs[0] if runs else None


def _milliseconds(value):
    return int(value.timestamp() * 1000) if hasattr(value, "timestamp") else int(time.time() * 1000)


def publish_execution(client, session, report, settings, *, phase="sync", receipt_uri):
    """Write collect_execution's report to MLflow and return its S3 receipt.

    1. Find or create the MLflow run associated with this pipeline execution.
    2. Assemble its status, parameters, metrics and model/job links.
    3. Write changed values, dataset references and output artifacts to MLflow.
    4. Update run status and save the receipt used by subsequent syncs.
    """
    from mlflow.entities import Dataset, DatasetInput, InputTag, Metric, Param, RunTag

    # 1. Find the experiment for this model/environment, then its execution's run.
    execution, parameters = report["execution"], report["parameters"]
    arn = execution["PipelineExecutionArn"]
    model = parameters["ModelId"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", model):
        raise ValueError("The execution must have an ordinary ModelId.")
    environment = parameters.get("Environment", settings.get("environment", "dev"))
    experiment_id = _experiment(client, f"{settings['experiment_prefix']}/{environment}/{model}")
    kind = settings["kind"]
    s3 = session.client("s3", config=READ_CONFIG)
    # The receipt contains the MLflow run ID from a previous update, if one exists.
    receipt = _read_json(s3, receipt_uri, optional=True)
    if receipt:
        if receipt["execution_arn"] != arn or receipt["tracking_uri"] != settings["uri"]:
            raise ValueError("Tracking receipt belongs to another execution or service.")
        run = client.get_run(receipt["run_id"])
        if run.info.experiment_id != experiment_id or run.data.tags.get("pipeline.execution_arn") != arn:
            raise ValueError("Tracking receipt and MLflow run identity differ.")
    else:
        # Recover a CreateRun response lost after the server committed it.
        run = _find_run(client, experiment_id, arn)
        if run is None:
            run = client.create_run(experiment_id, start_time=_milliseconds(execution["CreationTime"]),
                                    run_name=f"{kind}-{arn.rsplit('/', 1)[-1]}",
                                    tags={"pipeline.execution_arn": arn, "model.id": model,
                                          "environment": environment, "workflow": kind})
        receipt = {"execution_arn": arn, "tracking_uri": settings["uri"],
                   "experiment_id": experiment_id, "run_id": run.info.run_id}
        _put_json(s3, receipt_uri, receipt, settings.get("kms_key", ""))
    run_id = run.info.run_id

    # 2. Assemble values from the collected report before sending them to MLflow.
    # A final tracking job can observe completed model steps while its own pipeline
    # is still Executing; preserve that distinction in the tags below.
    status = execution["PipelineExecutionStatus"]
    business_complete = bool(report["steps"]) and all(
        step["status"] in {"Succeeded", "Skipped"} for step in report["steps"])
    failed = any(step["status"] == "Failed" for step in report["steps"])
    mlflow_status = TERMINAL.get(status, "FAILED" if failed else
                                 "FINISHED" if phase == "finish" and business_complete else "RUNNING")
    tags = {"pipeline.status": status, "pipeline.failure": execution.get("FailureReason", ""),
            "tracking.phase": phase, "tracking.receipt_uri": receipt_uri,
            "tracking.finalized_before_pipeline_terminal": str(
                mlflow_status == "FINISHED" and status not in TERMINAL).lower()}

    # Identify each actual producer, including reused jobs and their code versions.
    for step in report["steps"]:
        prefix = "step." + step["name"]
        tags[prefix + ".producer_execution_arn"] = step["producer_execution_arn"]
        tags[prefix + ".cached"] = str(step["cached"]).lower()
        for source_key in ("job_arn", "image", "source_revision"):
            if step.get(source_key):
                tags[prefix + "." + source_key] = step[source_key]
        source = step.get("source_identity", {}).get("source", {})
        if source.get("commit"):
            tags[prefix + ".source_commit"] = source["commit"]
            tags[prefix + ".source_dirty"] = str(source.get("dirty", "unknown")).lower()

    # Collect effective settings and the metrics already saved by model code.
    metrics = {"steps.cached": sum(step["cached"] for step in report["steps"]),
               "steps.failed": sum(step["status"] == "Failed" for step in report["steps"])}
    params = {"pipeline." + key: str(value) for key, value in parameters.items()}
    for step in report["steps"]:
        if step["name"] in {"Score", "Postprocess", "PostprocessTransform"}:
            identity = step.get("source_identity", {})
            for binding in ("source_parameters", "checkpoint_parameters"):
                params.update({step["name"] + "." + binding + "." + key:
                               item if isinstance(item, str) else _json(item)
                               for key, item in identity.get(binding, {}).items()})
    for name, value in report["reports"].items():
        if name.endswith("/evaluation.json"):
            metrics.update({"evaluation." + key: number for key, number in value["metrics"].items()
                            if type(number) in (int, float) and math.isfinite(number)})
            tags.update({"evaluation.data_usage": value["data_usage"],
                         "source.model_commit": value.get("model_source_commit", "unknown"),
                         "source.model_dirty": value.get("model_source_dirty", "unknown")})
        if name.endswith("/parameters.json"):
            prefix = "training." if name.startswith("Evaluate") else "inference."
            params.update({prefix + key: item if isinstance(item, str) else _json(item)
                           for key, item in value.items()})
        if name.endswith("/summary.json") and type(value.get("rows")) is int:
            metrics["prediction.rows"] = value["rows"]

    # Associate this run with its Registry version and, for inference, its training run.
    for reference in report["registry"]:
        package = reference["package"]
        tags["registry.package_arn"] = package["ModelPackageArn"]
        tags["registry.approval_at_sync"] = package["ModelApprovalStatus"]
        container = package["InferenceSpecification"]["Containers"][0]
        tags.update({"model.checkpoint_uri": container["ModelDataUrl"], "model.image_uri": container["Image"]})
        training_arn = package.get("MetadataProperties", {}).get("GeneratedBy", "")
        if kind == "inference" and training_arn:
            tags["model.training_execution_arn"] = training_arn
            training = _find_run(client, experiment_id, training_arn)
            if training:
                tags["model.training_run_id"] = training.info.run_id

    # 3. Send only new/changed fields so repeated syncs do not repeat metric history.
    # Large parameter values remain complete in the JSON artifacts.
    new_params = []
    for key, value in params.items():
        if len(value) > 6000:
            continue
        if key in run.data.params and run.data.params[key] != value:
            raise ValueError(f"Recorded parameter changed for the same execution: {key}")
        if key not in run.data.params:
            new_params.append(Param(key, value))
    timestamp = _milliseconds(execution.get("LastModifiedTime", execution["CreationTime"]))
    new_metrics = [Metric(key, value, timestamp, 0) for key, value in metrics.items()
                   if run.data.metrics.get(key) != value]
    new_tags = [RunTag(key, str(value)) for key, value in tags.items() if run.data.tags.get(key) != str(value)]
    for offset in range(0, max(len(new_params), len(new_metrics), len(new_tags)), 100):
        client.log_batch(run_id, params=new_params[offset:offset+100],
                         metrics=new_metrics[offset:offset+100], tags=new_tags[offset:offset+100],
                         synchronous=True)

    # Add references to the input datasets; the data itself stays in S3.
    for dataset in report["datasets"]:
        provenance = dataset["provenance"]
        # Digest of the provenance record is explicit; do not claim a full-data hash.
        digest = hashlib.sha256(_json(provenance).encode()).hexdigest()[:32]
        context = "training" if kind == "training" else "inference"
        logged = getattr(getattr(run, "inputs", None), "dataset_inputs", []) or []
        if not any(item.dataset.digest == digest and json.loads(item.dataset.source).get("uri") == dataset["uri"]
                   for item in logged):
            entity = Dataset(name=f"{model}-{context}", digest=digest, source_type="s3",
                             source=_json({"uri": dataset["uri"]}))
            client.log_inputs(run_id, datasets=[DatasetInput(entity, [
                InputTag("mlflow.data.context", context),
                InputTag("digest.kind", "provenance_sha256"),
                InputTag("producer.execution_arn", dataset["producer_execution_arn"]),
            ])])

    # Store the complete report, then copy eligible small output files to MLflow.
    # JSON normalization converts datetime objects while retaining their values.
    snapshot = json.loads(_json(report))
    digest = hashlib.sha256(_json(snapshot).encode()).hexdigest()
    if receipt.get("snapshot_sha256") != digest:
        client.log_dict(run_id, snapshot, "lineage/execution.json")
        for name, value in report["reports"].items():
            client.log_dict(run_id, value, "reports/" + name)
        with tempfile.TemporaryDirectory(prefix="mlops-tracking-") as folder:
            for artifact in report["artifacts"]:
                relative = Path(artifact["relative_path"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Artifact path escapes its declared output.")
                if artifact["size"] > MAX_ARTIFACT_BYTES or relative.suffix.lower() not in {
                    ".csv", ".png", ".pdf", ".txt"
                }:
                    continue
                bucket, key = s3_location(artifact["uri"])
                response = s3.get_object(Bucket=bucket, Key=key, IfMatch=artifact["etag"])
                with response["Body"] as stream:
                    content = stream.read(MAX_ARTIFACT_BYTES + 1)
                if len(content) > MAX_ARTIFACT_BYTES:
                    raise ValueError("Artifact grew after collection; retry with a stable output.")
                local = Path(folder) / artifact["step"] / relative
                local.parent.mkdir(parents=True, exist_ok=True)
                local.write_bytes(content)
                client.log_artifact(run_id, str(local), "outputs/" + artifact["step"] +
                                    ("/" + relative.parent.as_posix() if relative.parent != Path(".") else ""))

    # 4. Update run status and retain its ID/snapshot in S3 for the next sync.
    if mlflow_status == "RUNNING":
        if run.info.status != "RUNNING":
            client.update_run(run_id, status="RUNNING")
    else:
        client.set_terminated(run_id, status=mlflow_status,
                              end_time=timestamp if status in TERMINAL else int(time.time() * 1000))
    receipt.update(receipt_uri=receipt_uri, snapshot_sha256=digest, status=mlflow_status, pipeline_status=status)
    _put_json(s3, receipt_uri, receipt, settings.get("kms_key", ""))
    return receipt


def sync_execution(session, execution_arn, settings, *, phase="sync", client=None):
    """Refresh the MLflow record for one SageMaker execution that already exists.

    execution_arn comes from the tracking job's environment or a notebook caller.
    settings identifies the tracking service, experiment prefix and S3 output root.

    1. Check these inputs and choose the receipt/lock paths for this execution.
    2. Reserve the writer lock and collect the existing jobs and saved outputs.
    3. Connect to MLflow, write the collected report and return the saved receipt.
    """
    # 1. Validate the supplied execution and destinations; do not start any jobs.
    settings = validate_tracking(settings)
    if settings is None:
        raise ValueError("Supply tracking settings to synchronize an execution.")
    if phase not in {"start", "finish", "sync"} or settings.get("kind") not in {"training", "inference"}:
        raise ValueError("Choose a tracking phase and workflow kind.")
    if not re.fullmatch(r"arn:aws(?:-us-gov|-cn)?:sagemaker:[a-z0-9-]+:\d{12}:pipeline/[A-Za-z0-9-]+/execution/[A-Za-z0-9-]+", execution_arn):
        raise ValueError("Supply the full SageMaker pipeline execution ARN.")
    root = settings["s3_root"].rstrip("/")
    _, prefix = s3_location(root)
    if not prefix or ".." in prefix.split("/"):
        raise ValueError("Tracking needs an application S3 prefix, not the bucket root.")

    # Give the execution one stable receipt path for repeated job/notebook updates.
    key = hashlib.sha256((settings["uri"] + "/" + execution_arn).encode()).hexdigest()
    receipt_uri = root + "/tracking/" + key + "/run.json"
    s3 = session.client("s3", config=READ_CONFIG)

    # 2. Prevent overlapping writers, then READ the execution into a report dictionary.
    with _lease(s3, receipt_uri.removesuffix("run.json") + "lock.json", settings.get("kms_key", "")):
        report = collect_execution(session, execution_arn)
        if phase == "finish":
            # A dependency can complete before the execution-step API reflects it.
            for attempt in range(6):
                if report["execution"]["PipelineExecutionStatus"] in TERMINAL or (
                    report["steps"] and all(step["status"] in {"Succeeded", "Skipped"}
                                            for step in report["steps"])
                ):
                    break
                if attempt == 5:
                    raise RuntimeError("Final tracking cannot confirm completed model steps; synchronize again.")
                time.sleep(2)
                report = collect_execution(session, execution_arn)

        # 3. WRITE that report into its MLflow run. Tests can supply a local client;
        # normal jobs connect to the configured managed tracking service.
        if client is not None:
            return publish_execution(client, session, report, settings, phase=phase, receipt_uri=receipt_uri)
        with managed_client(settings["uri"]) as connected:
            return publish_execution(connected, session, report, settings, phase=phase, receipt_uri=receipt_uri)


def main():
    """Command-line entry point executed by the tracking Processing jobs.

    Read the phase and environment supplied by source.configure_tracking(), update
    the MLflow run for that existing pipeline execution, then print the receipt.
    """
    import argparse
    import boto3

    # Read the job's requested phase; it is start, finish or a later explicit sync.
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["start", "finish", "sync"], required=True)
    args = parser.parse_args()

    # The pipeline definition supplies these values, including the current
    # execution ARN. The ARN identifies the execution whose details we will read.
    settings = {
        name: os.environ["MLOPS_TRACKING_" + name.upper()]
        for name in ("uri", "experiment_prefix", "s3_root", "environment", "kind")
    }
    # Managed jobs may omit empty environment values; an output KMS key is optional.
    settings["kms_key"] = os.environ.get("MLOPS_TRACKING_KMS_KEY", "")

    # Delegate the read -> collect -> MLflow write work, then print its result.
    result = sync_execution(boto3.Session(), os.environ["PIPELINE_EXECUTION_ARN"], settings, phase=args.phase)
    print(_json({"event": "execution_tracked", **result}), flush=True)


if __name__ == "__main__":
    main()
