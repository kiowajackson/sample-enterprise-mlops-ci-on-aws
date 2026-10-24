"""Execution records written by producing jobs and assembled by final processing."""

import json
import os
from pathlib import Path

MAX_REPORT_BYTES = 10 * 1024 * 1024
REPORT_SUFFIXES = {".csv", ".png", ".jpg", ".jpeg", ".svg", ".html", ".json", ".txt", ".pdf"}


def describe_outputs(record, directories):
    """Record declared output files and small JSON reports before upload.

    directories maps an output name to (local directory, actual S3 destination).
    Data/model files remain in S3; artifact publication is an explicit job setting.
    """
    artifacts, reports, datasets, registry = [], {}, [], []
    publish = os.getenv("MLOPS_LOG_ARTIFACTS") == "1"
    for name, (directory, uri) in directories.items():
        directory, uri = Path(directory), uri.rstrip("/") + "/"
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.is_symlink() or path.name in {"_execution.json", "run.json"}:
                continue
            relative = path.relative_to(directory).as_posix()
            artifact = {
                "output": name, "relative_path": relative, "uri": uri + relative,
                "size": path.stat().st_size,
                "publish_to_mlflow": publish and path.suffix.lower() in REPORT_SUFFIXES
                                     and path.stat().st_size <= MAX_REPORT_BYTES,
            }
            artifacts.append(artifact)
            if path.suffix.lower() != ".json" or artifact["size"] > MAX_REPORT_BYTES:
                continue
            document = json.loads(path.read_text())
            reports[relative] = document
            # These are the existing output contracts, independent of step names.
            if relative == "provenance.json":
                datasets.append({"uri": uri + "data.csv", "provenance": document})
            elif relative == "package.json":
                registry.append(document)
    return {**record, "record_version": 1, "artifacts": artifacts, "reports": reports,
            "datasets": datasets, "registry": registry}


def processing_record(job, record, output_root, local_root):
    """Describe a Processing job from its own configuration and local outputs."""
    pipeline = os.getenv("MLOPS_PIPELINE_ARN", os.getenv("MLOPS_TRACKING_PIPELINE_ARN", ""))
    producer = os.getenv("PIPELINE_EXECUTION_ID", "")
    outputs = job["ProcessingOutputConfig"]["Outputs"]
    directories = {
        item["OutputName"]: (
            Path(local_root) / Path(item["S3Output"]["LocalPath"]).relative_to(output_root),
            item["S3Output"]["S3Uri"])
        for item in outputs
    }
    result = describe_outputs({
        **record, "step": os.getenv("MLOPS_STEP_NAME", record.get("step", "")),
        "producer_execution_arn": pipeline + "/execution/" + producer if pipeline and producer else "",
        "inputs": {item["InputName"]: item["S3Input"]["S3Uri"]
                   for item in job.get("ProcessingInputs", []) if "S3Input" in item},
        "outputs": {item["OutputName"]: item["S3Output"]["S3Uri"] for item in outputs},
        "image": job["AppSpecification"]["ImageUri"],
    }, directories)
    return result


def read_run_report(session, execution_arn):
    """Read a successful execution's standard report without knowing its steps."""
    from mlops_models.pipelines.inspect import read_parameters, _read_json

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


def run_report_uri(execution_arn, parameters):
    return (parameters["OutputDataRoot"].rstrip("/") + "/" + parameters["ModelId"] +
            "/runs/" + execution_arn.rsplit("/", 1)[-1] + "/run.json")


def write_inference_report(output, *, data_path, selection_file, predictions, config):
    """Assemble the final inference report from records emitted by its producers."""
    if os.getenv("MLOPS_WRITE_RUN_REPORT") != "1":
        return None
    import boto3
    from mlops_models.pipelines.inspect import collect_execution, write_run_index

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
    from mlops_models.pipelines.inspect import add_record
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
