"""Log from the jobs that do the work; the completion service finalizes pipeline status.

job_tracking() supplies one shared run ID inside each executing container.
log_job_outputs() publishes local reports directly. sync_execution() is called
by the completion Lambda after terminal events, including cache-only and failed runs.
This module declares or launches no SageMaker jobs.
"""

from contextlib import contextmanager, nullcontext
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
from urllib.parse import urlparse

MAX_ARTIFACT_BYTES = 10 * 1024 * 1024

READ_CONFIG = Config(connect_timeout=5, read_timeout=15,
                     retries={"mode": "standard", "total_max_attempts": 2})
TERMINAL = {"Succeeded": "FINISHED", "Failed": "FAILED", "Stopped": "KILLED"}
COPY_SUFFIXES = {".csv", ".png", ".pdf", ".txt", ".html", ".json"}

def s3_location(uri):
    location = urlparse(uri)
    if location.scheme != "s3" or not location.netloc:
        raise ValueError("Expected an s3://bucket/key URI.")
    return location.netloc, location.path.lstrip("/")


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
def managed_client(uri, *, client_version=""):
    """Use the current AWS credential chain, restoring the caller's MLflow URI."""
    try:
        import mlflow
        import sagemaker_mlflow  # noqa: F401; installs SigV4/client plugins
    except ImportError as error:
        requirement = f"mlflow-skinny=={client_version}" if client_version else "mlflow-skinny"
        raise RuntimeError(
            f"MLflow logging requires {requirement} and sagemaker-mlflow in this environment. "
            "Install compatible clients through the approved notebook/image build."
        ) from error
    if client_version and mlflow.__version__ != client_version:
        raise RuntimeError(
            f"MLflow client is {mlflow.__version__}; this tracking binding requires "
            f"mlflow-skinny=={client_version}. Update the notebook environment and model image "
            "through the approved build, then restart the kernel and republish."
        )

    previous = mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri(uri)
        yield mlflow.MlflowClient(tracking_uri=uri)
    finally:
        mlflow.set_tracking_uri(previous)


def check_tracking(session, settings, experiment_names):
    """Verify the notebook caller with a small retained run in each model experiment."""
    settings = validate_tracking(settings)
    if settings is None:
        return None
    if not experiment_names:
        raise ValueError("Supply the model experiment names to check.")
    uri = settings["uri"]
    region, resource = uri.split(":")[3], uri.split(":", 5)[5]
    service, name = resource.split("/", 1)
    result = {"uri": uri, "service": service, "server_version": None, "experiments": []}
    expected_version = ""
    if service == "mlflow-tracking-server":
        # Use the ARN's Region; the notebook default can point elsewhere.
        sm = session.client("sagemaker", region_name=region, config=READ_CONFIG)
        server = sm.describe_mlflow_tracking_server(TrackingServerName=name)
        if server["TrackingServerArn"] != uri:
            raise ValueError("The described tracking server does not match tracking_uri.")
        if server.get("IsActive") != "Active":
            raise ValueError(
                f"MLflow tracking server {name} is not active "
                f"({server.get('TrackingServerStatus', 'unknown')}). Ask its owner to start it."
            )
        result["server_version"] = server["MlflowVersion"]
        # AWS's documented Tracking Server/client combinations.
        expected_version = {"2.13": "2.13.2", "2.16": "2.16.2", "3.0": "3.0.0"}.get(
            ".".join(server["MlflowVersion"].split(".")[:2]), ""
        )
        if not expected_version:
            raise ValueError(
                f"Tracking Server version {server['MlflowVersion']} has no reviewed client mapping. "
                "Verify AWS's compatibility guidance before updating check_tracking."
            )
    # Apps use the existing plugin connection. This avoids requiring App management
    # APIs that are absent from the pinned Boto3 SDK; the write/read check still runs.
    with managed_client(uri, client_version=expected_version) as client:
        import mlflow
        result["client_version"] = mlflow.__version__
        client.search_experiments(max_results=1)
        for experiment_name in experiment_names:
            experiment_id = _experiment(client, experiment_name)
            experiment = client.get_experiment(experiment_id)
            run = client.create_run(
                experiment_id, run_name="connection-check",
                tags={"mlops.purpose": "connection-check"},
            )
            run_id = run.info.run_id
            stage = "write run metadata"
            try:
                _log_values(client, run_id, metrics={"connection_check": 1},
                            params={"purpose": "connection-check"},
                            tags={"mlops.purpose": "connection-check"})
                client.set_tag(run_id, "mlops.check", "notebook")
                stage = "upload artifact"
                payload = {"run_id": run_id, "purpose": "connection-check"}
                client.log_dict(run_id, payload, "connection-check.json")
                stage = "read artifact"
                with tempfile.TemporaryDirectory() as directory:
                    downloaded = client.download_artifacts(run_id, "connection-check.json", directory)
                    if json.loads(Path(downloaded).read_text()) != payload:
                        raise ValueError("Downloaded artifact differs from the uploaded check.")
                stage = "finish and read run"
                client.set_terminated(run_id, status="FINISHED")
                saved = client.get_run(run_id)
                if (saved.info.status != "FINISHED"
                        or saved.data.metrics.get("connection_check") != 1
                        or saved.data.params.get("purpose") != "connection-check"):
                    raise ValueError("Run status, metric or parameter was not retained.")
            except Exception as error:
                try:
                    client.set_terminated(run_id, status="FAILED")
                except Exception:
                    pass  # Preserve the original write/read failure.
                raise RuntimeError(
                    f"MLflow notebook check could not {stage} in {experiment_name} "
                    f"(run {run_id}): {error}"
                ) from error
            result["experiments"].append({
                "name": experiment_name, "experiment_id": experiment_id, "run_id": run_id,
                "artifact_location": experiment.artifact_location,
            })
    return result


@contextmanager
def _lease(s3, uri, kms_key):
    """Allow one writer at a time using a temporary S3 lock file.

    Create the lock or take over an expired one, yield to the caller, then remove
    our lock. This protects run creation when a job and completion invocation overlap.
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


def analysis_metrics(reports):
    """Read optional metrics.json files; analysis owns calculation, units and label availability."""
    result = {}
    for path, document in reports.items():
        if not path.endswith("/metrics.json"):
            continue
        step = path.split("/", 1)[0]
        prefix = {"FeatureSelection": "selection", "Monitor": "monitoring"}.get(step, "prediction")
        if not isinstance(document, dict):
            raise ValueError(f"{path} must contain a JSON object of metric names and numbers.")
        for name, value in document.items():
            if value is None:
                continue  # Unavailable is not zero.
            if not isinstance(name, str) or type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{path}: metric {name!r} must be a finite number or null.")
            result[f"{prefix}.{name}"] = value
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


def ensure_run(client, session, execution_arn, model, settings, *, created=None):
    """Find/create the execution's run; serialize creation only, not job logging."""
    key = hashlib.sha256((settings["uri"] + "/" + execution_arn).encode()).hexdigest()
    receipt_uri = settings["s3_root"].rstrip("/") + "/tracking/" + key + "/run.json"
    s3 = session.client("s3", config=READ_CONFIG)
    receipt = _read_json(s3, receipt_uri, optional=True)
    if receipt:
        if receipt["execution_arn"] != execution_arn or receipt["tracking_uri"] != settings["uri"]:
            raise ValueError("Tracking receipt belongs to another execution or service.")
        return receipt
    # A Studio execution can initialize its run in its first uncached job. A
    # concurrent completion invocation must not create a second run for it.
    with _lease(s3, receipt_uri.removesuffix("run.json") + "lock.json", settings.get("kms_key", "")):
        receipt = _read_json(s3, receipt_uri, optional=True)
        if receipt:
            return receipt
        experiment_id = _experiment(
            client, f"{settings['experiment_prefix']}/{settings.get('environment', 'dev')}/{model}")
        run = _find_run(client, experiment_id, execution_arn)
        if run is None:
            run = client.create_run(
                experiment_id, start_time=_milliseconds(created),
                run_name=f"{settings['kind']}-{execution_arn.rsplit('/', 1)[-1]}",
                tags={"pipeline.execution_arn": execution_arn, "model.id": model,
                      "environment": settings.get("environment", "dev"),
                      "workflow": settings["kind"], "tracking.mode": "direct"})
        receipt = {"execution_arn": execution_arn, "tracking_uri": settings["uri"],
                   "experiment_id": experiment_id, "run_id": run.info.run_id,
                   "receipt_uri": receipt_uri}
        _put_json(s3, receipt_uri, receipt, settings.get("kms_key", ""))
        return receipt


@contextmanager
def job_tracking(model, output_uri=""):
    """Set MLflow context inside the running process; do not end the shared run."""
    if not os.getenv("MLOPS_TRACKING_URI"):
        yield None
        return
    import boto3

    execution_id = os.getenv("PIPELINE_EXECUTION_ID", "")
    if not execution_id and "/runs/" in output_uri:
        execution_id = output_uri.rsplit("/runs/", 1)[1].split("/")[0]
    pipeline = os.environ["MLOPS_TRACKING_PIPELINE_ARN"]
    if not execution_id:
        raise ValueError("Direct logging needs this job's execution ID from its output configuration.")
    execution_arn = pipeline + "/execution/" + execution_id
    settings = {key: os.getenv("MLOPS_TRACKING_" + name, "") for key, name in (
        ("uri", "URI"), ("experiment_prefix", "EXPERIMENT_PREFIX"), ("s3_root", "S3_ROOT"),
        ("environment", "ENVIRONMENT"), ("kind", "KIND"), ("kms_key", "KMS_KEY"))}
    session = boto3.Session(region_name=pipeline.split(":")[3])
    previous = {key: os.environ.get(key) for key in ("MLFLOW_TRACKING_URI", "MLFLOW_RUN_ID")}
    with managed_client(settings["uri"], client_version=os.getenv("MLOPS_TRACKING_CLIENT_VERSION", "")) as client:
        receipt = ensure_run(client, session, execution_arn, model, settings)
        os.environ.update(MLFLOW_TRACKING_URI=settings["uri"], MLFLOW_RUN_ID=receipt["run_id"])
        step = os.environ["MLOPS_TRACKING_STEP"]
        client.set_tag(receipt["run_id"], "step." + step + ".status", "Running")
        print(json.dumps({"event": "mlflow_job_logging", "step": step,
                          "run_id": receipt["run_id"], "execution_arn": execution_arn}), flush=True)
        try:
            yield {"client": client, "step": step, **receipt}
        except Exception as error:
            client.set_tag(receipt["run_id"], "step." + step + ".status", "Failed")
            client.set_tag(receipt["run_id"], "step." + step + ".failure", str(error)[:4000])
            raise
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def _log_values(client, run_id, *, metrics=None, params=None, tags=None):
    """Use explicit run IDs so logging never closes another step's shared run."""
    from mlflow.entities import Metric, Param, RunTag
    run = client.get_run(run_id)
    timestamp = int(time.time() * 1000)
    client.log_batch(
        run_id,
        metrics=[Metric(name, float(value), timestamp, 0) for name, value in (metrics or {}).items()
                 if run.data.metrics.get(name) != value],
        params=[Param(name, str(value)) for name, value in (params or {}).items()
                if name not in run.data.params],
        tags=[RunTag(name, str(value)) for name, value in (tags or {}).items()],
        synchronous=True)


def report_metrics(step, reports):
    """Log metrics produced by model code; never substitute missing metrics with zero."""
    result = analysis_metrics({step + "/" + name: value for name, value in reports.items()})
    rows = reports.get("summary.json", {}).get("rows")
    if step in {"Postprocess", "PostprocessTransform"} and type(rows) is int and rows >= 0:
        result["prediction.rows"] = rows
    evaluation = reports.get("evaluation.json", {}).get("metrics", {})
    for name, value in evaluation.items():
        if value is None:
            continue
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"Evaluation metric {name!r} must be finite or null.")
        result[("training." if step == "Train" else "evaluation.") + name] = value
    for name, value in reports.get("monitoring/metrics.json", {}).items():
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"Monitoring metric {name!r} must be finite.")
        result["monitoring." + name] = value
    return result


def monitoring_tags(summary):
    """Link the comparison and both model/data identities without another MLflow run."""
    if not summary:
        return {}
    tags = {"monitoring." + key: value if isinstance(value, str) else _json(value)
            for key, value in summary["context"].items()}
    tags.update({"monitoring.drift_status": summary["drift_status"],
                 "monitoring.quality_status": summary["quality_status"]})
    return tags


def log_job_outputs(tracking, directories, record):
    """Upload this job's local reports directly, preserving their folder structure."""
    if tracking is None:
        return
    client, run_id, step = tracking["client"], tracking["run_id"], tracking["step"]
    reports = {}
    # Data and model binaries stay in their declared S3 outputs. Report-producing
    # steps publish their small CSVs/plots; other steps publish their metadata only.
    report_step = os.getenv("MLOPS_LOG_ARTIFACTS") == "1"
    for directory in dict.fromkeys(Path(value) for value in directories):
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.is_symlink() or path.name == "_execution.json":
                continue
            relative = path.relative_to(directory)
            if relative.as_posix() in {"monitoring/metrics.json", "monitoring/summary.json"}:
                value = json.loads(path.read_text())
                reports[relative.as_posix()] = value
                client.log_dict(run_id, value, f"reports/{step}/{relative.as_posix()}")
            elif path.name in {"evaluation.json", "metrics.json", "parameters.json",
                             "model_metadata.json", "summary.json", "package.json",
                             "provenance.json"} and len(relative.parts) == 1:
                value = json.loads(path.read_text())
                reports[path.name] = value
                client.log_dict(run_id, value, f"reports/{step}/{path.name}")
            elif report_step and path.suffix.lower() in COPY_SUFFIXES and path.stat().st_size <= MAX_ARTIFACT_BYTES:
                # Training includes a saved model/code tree. Only its top-level
                # reports are relevant here; do not copy framework internals.
                if os.getenv("MLOPS_TRAINING_MODEL_URI") and len(relative.parts) > 1:
                    continue
                parent = relative.parent.as_posix()
                artifact_path = "outputs/" + step + ("/" + parent if parent != "." else "")
                client.log_artifact(run_id, str(path), artifact_path)
    tags = {"step." + step + ".status": "Succeeded",
            "step." + step + ".producer_execution_arn": tracking["execution_arn"],
            "step." + step + ".cached": "false"}
    tags.update(monitoring_tags(reports.get("monitoring/summary.json")))
    package = reports.get("package.json")
    if package:
        tags.update({"model.package_arn": package["package"],
                     "model.artifact_uri": package["checkpoint"], "model.image_uri": package["image"]})
    if os.getenv("MLOPS_TRAINING_MODEL_URI"):
        tags["model.artifact_uri"] = os.environ["MLOPS_TRAINING_MODEL_URI"]
    training = record.get("model_training", {}).get("mlflow")
    if training:
        tags["model.training_run_id"] = training["run_id"]
    if "provenance.json" in reports and record.get("outputs"):
        uri = next(iter(record["outputs"].values())).rstrip("/") + "/data.csv"
        log_dataset(client, run_id, uri, reports["provenance.json"], tracking["execution_arn"])
    _log_values(client, run_id, metrics=report_metrics(step, reports), tags=tags,
                params={f"step.{step}.{key}": _json(value)
                        for key, value in record.get("source_parameters", {}).items()})
    client.log_dict(run_id, record, "execution/" + step + ".json")


def log_dataset(client, run_id, uri, provenance, producer):
    """Retain the existing dataset reference without creating a new identity scheme."""
    from mlflow.entities import Dataset, DatasetInput, InputTag
    digest = hashlib.sha256(_json(provenance).encode()).hexdigest()[:36]
    run = client.get_run(run_id)
    existing = getattr(getattr(run, "inputs", None), "dataset_inputs", []) or []
    if any(item.dataset.digest == digest and json.loads(item.dataset.source).get("uri") == uri
           for item in existing):
        return
    entity = Dataset(name="input-batch", digest=digest, source_type="s3", source=_json({"uri": uri}))
    client.log_inputs(run_id, datasets=[DatasetInput(entity, [
        InputTag("mlflow.data.context", "input"), InputTag("producer.execution_arn", producer)])])


def sync_execution(session, execution_arn, settings, *, client=None):
    """Finalize from the completion service; copy cached reports and link the original producers.

    Jobs already publish live metrics/artifacts. This reads native final status,
    handles steps that never ran, and completes cached-output/Registry navigation.
    Called automatically for completed, failed and stopped executions.
    """
    from mlops_models.pipelines.inspect import collect_execution, write_run_index

    settings = validate_tracking(settings)
    if settings is None:
        raise ValueError("Supply tracking settings to synchronize an execution.")
    report = collect_execution(session, execution_arn)
    parameters, execution = report["parameters"], report["execution"]
    settings = {**settings, "environment": parameters.get("Environment", settings.get("environment", "dev"))}
    connection = nullcontext(client) if client is not None else managed_client(settings["uri"])
    with connection as connected:
        receipt = ensure_run(connected, session, execution_arn, parameters["ModelId"],
                             settings, created=execution["CreationTime"])
        run_id = receipt["run_id"]
        tags = {"pipeline.status": execution["PipelineExecutionStatus"],
                "pipeline.failure": execution.get("FailureReason", ""),
                "tracking.receipt_uri": receipt["receipt_uri"]}
        metrics = {"steps.cached": sum(step["cached"] for step in report["steps"]),
                   "steps.failed": sum(step["status"] == "Failed" for step in report["steps"])}
        for step in report["steps"]:
            prefix = "step." + step["name"]
            tags.update({prefix + ".cached": str(step["cached"]).lower(),
                         prefix + ".producer_execution_arn": step["producer_execution_arn"],
                         prefix + ".status": step["status"]})
            for key in ("job_arn", "image", "source_revision"):
                if step.get(key):
                    tags[prefix + "." + key] = step[key]
            producer_tracking = step.get("source_identity", {}).get("mlflow")
            if producer_tracking:
                tags[prefix + ".producer_run_id"] = producer_tracking["run_id"]
            trained = step.get("source_identity", {}).get("model_training", {}).get("mlflow")
            if trained:
                tags["model.training_run_id"] = trained["run_id"]
        for path, value in report["reports"].items():
            step, name = path.split("/", 1)
            metrics.update(report_metrics(step, {name: value}))
            if name == "monitoring/summary.json":
                tags.update(monitoring_tags(value))
            connected.log_dict(run_id, value, "reports/" + path)
        for reference in report["registry"]:
            package, description = reference["receipt"], reference["package"]
            container = description["InferenceSpecification"]["Containers"][0]
            tags.update({"model.package_arn": package["package"],
                         "model.artifact_uri": container["ModelDataUrl"], "model.image_uri": container["Image"]})
            if settings["kind"] == "inference" and "model.training_run_id" not in tags:
                producer = description.get("MetadataProperties", {}).get("GeneratedBy")
                trained = _find_run(connected, receipt["experiment_id"], producer) if producer else None
                if trained:
                    tags["model.training_run_id"] = trained.info.run_id
        for dataset in report["datasets"]:
            log_dataset(connected, run_id, dataset["uri"], dataset["provenance"],
                        dataset["producer_execution_arn"])
        _log_values(connected, run_id, metrics=metrics, tags=tags,
                    params={"pipeline." + key: str(value) for key, value in parameters.items()})
        # A cached job does not execute its logger. Make its small report files
        # available in this run, while preserving explicit original-producer tags.
        cached = {step["name"] for step in report["steps"] if step["cached"]}
        with tempfile.TemporaryDirectory(prefix="cached-reports-") as folder:
            for artifact in report["artifacts"]:
                relative = Path(artifact["relative_path"])
                if (not artifact.get("publish_to_mlflow") or artifact["step"] not in cached
                        or artifact["size"] > MAX_ARTIFACT_BYTES
                        or relative.suffix.lower() not in COPY_SUFFIXES or relative.name == "_execution.json"):
                    continue
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Artifact path escapes its output.")
                local = Path(folder) / artifact["step"] / relative
                local.parent.mkdir(parents=True, exist_ok=True)
                bucket, key = s3_location(artifact["uri"])
                session.client("s3").download_file(bucket, key, str(local))
                parent = relative.parent.as_posix()
                connected.log_artifact(run_id, str(local), "outputs/" + artifact["step"] +
                                       ("/" + parent if parent != "." else ""))
        connected.log_dict(run_id, json.loads(_json(report)), "execution.json")
        status = TERMINAL.get(execution["PipelineExecutionStatus"], "RUNNING")
        if status != "RUNNING":
            connected.set_terminated(run_id, status=status, end_time=_milliseconds(execution.get("LastModifiedTime")))
        elif connected.get_run(run_id).info.status != "RUNNING":
            connected.update_run(run_id, status="RUNNING")
        receipt.update(status=status, pipeline_status=execution["PipelineExecutionStatus"])
        receipt["index_uri"] = write_run_index(session, report, receipt={
            key: receipt[key] for key in ("tracking_uri", "experiment_id", "run_id", "receipt_uri")},
            kms_key=settings.get("kms_key", ""))
        _put_json(session.client("s3"), receipt["receipt_uri"], receipt, settings.get("kms_key", ""))
        return receipt
