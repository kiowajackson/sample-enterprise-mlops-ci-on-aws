"""Read saved inference inputs, compare a reference batch and publish reports."""

import json
import os
from pathlib import Path
import tarfile

import pandas as pd


def compare_drift(data_directory, selection_file, prediction_directory, output_directory, model_directory):
    """Resolve the step's inputs, run comparison processing, then publish its outputs."""
    data, selection, predictions, config = get_container_input(
        data_directory, selection_file, prediction_directory, model_directory)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    compare_batches(data, selection, predictions, output, config)
    from mlops_models.pipelines.inspect import write_inference_report
    write_inference_report(output, data_path=data, selection_file=selection,
                           predictions=predictions, config=config)
    return write_container_output(output)


def get_container_input(data_directory, selection_file, prediction_directory, model_directory):
    """Resolve the current files and configuration saved with the selected model."""
    data = Path(data_directory) / "data.csv"
    if not data.is_file():
        raise FileNotFoundError(f"DriftComparison needs the original input snapshot: {data}")
    selection = Path(selection_file)
    if not selection.is_file():
        raise FileNotFoundError(f"DriftComparison needs the saved selected-feature CSV: {selection}")
    source = Path(prediction_directory)
    predictions = {path.relative_to(source).as_posix(): path for path in sorted(source.rglob("*"))
                   if path.is_file() and path.suffix.lower() == ".csv"}
    if not predictions:
        raise FileNotFoundError(f"DriftComparison found no prediction CSV files under: {source}")
    model = Path(model_directory)
    if model.is_file():
        with tarfile.open(model) as archive:
            config = json.load(archive.extractfile("parameters.json"))
            config["model_metadata"] = json.load(archive.extractfile("model_metadata.json"))
    else:
        config = json.loads((model / "parameters.json").read_text())
        config["model_metadata"] = json.loads((model / "model_metadata.json").read_text())
    return data, selection, predictions, config


def write_container_output(output_directory):
    """Log comparison files; the loader uploads them with their subfolders intact."""
    output = Path(output_directory)
    files = sorted(path for path in output.rglob("*") if path.is_file())
    if not files:
        raise FileNotFoundError(f"Drift comparison wrote no files under: {output}")
    for path in files:
        print(json.dumps({"event": "drift_output", "path": str(path),
                          "relative_path": path.relative_to(output).as_posix()}), flush=True)
    return output


def read_batch(data, predictions, *, primary_key, features, target, score_column):
    """Align raw features and actual scores by record key, never by row position."""
    options = {"dtype": {primary_key: str}, "keep_default_na": False, "na_values": ["", r"\N"]}
    raw, scores = pd.read_csv(data, **options), pd.read_csv(predictions, **options)
    columns = list(dict.fromkeys([primary_key, *features, *([target] if target in raw else [])]))
    raw = raw[columns]
    scores = scores[[primary_key, score_column]].rename(columns={score_column: "_monitor_score"})
    if "_monitor_score" in raw or raw[primary_key].isna().any() or scores[primary_key].isna().any():
        raise ValueError("Monitoring needs non-null record keys and a free _monitor_score column.")
    batch = raw.merge(scores, on=primary_key, how="outer", validate="one_to_one", indicator=True)
    if batch.empty or not batch["_merge"].eq("both").all():
        raise ValueError("Monitoring input records and predictions must have exactly matching keys.")
    return batch.drop(columns="_merge")


def reference_outputs(session, execution_arn, model_id, environment):
    """Read the prior inference's declared artifacts, independent of its step names."""
    from mlops_models.pipelines.inspect import read_run_report
    report = read_run_report(session, execution_arn)
    artifacts = report.get("inference")
    if not artifacts:
        raise ValueError("Reference execution must have an inference artifact report.")
    if artifacts["model_id"] != model_id or artifacts["environment"] != environment:
        raise ValueError("Reference execution must use the same ModelId and environment.")
    return artifacts


def compare_batches(data_path, selection_file, predictions, output, config):
    """Compare saved batches; the pipeline condition skips this job without a reference."""
    reference_arn = os.environ.get("MLOPS_REFERENCE_EXECUTION_ARN", "")
    pipeline_arn = os.environ["MLOPS_PIPELINE_ARN"]
    execution_arn = pipeline_arn + "/execution/" + os.environ["PIPELINE_EXECUTION_ID"]
    if (not reference_arn.startswith(pipeline_arn.split(":pipeline/")[0] + ":pipeline/") or
            reference_arn.count("/execution/") != 1 or reference_arn == execution_arn):
        raise ValueError("ReferenceExecutionArn must be a different execution in this account and region.")
    import boto3
    from mlops_models.pipelines.inspect import inference_artifacts
    from mlops_models.monitoring.report import write_report

    settings = json.loads(os.environ["MLOPS_MONITORING_CONFIG"])
    session = boto3.Session(region_name=pipeline_arn.split(":")[3])
    previous = reference_outputs(session, reference_arn, os.environ["MODEL_ID"], os.environ["MLOPS_ENVIRONMENT"])
    job = json.loads(Path("/opt/ml/config/processingjobconfig.json").read_text())
    declared_artifacts = inference_artifacts(
        job, data_path=data_path, selection_file=selection_file, predictions=predictions, config=config)
    selection = pd.read_csv(selection_file)
    features = selection["features"].tolist()
    if not features or len(features) != len(set(features)):
        raise ValueError("Monitoring needs a nonempty, distinct saved feature selection.")
    filename = settings.get("prediction_file", "messaging_final.csv")
    path = Path(filename)
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".csv":
        raise ValueError("monitoring.prediction_file must be a relative CSV path inside Score output.")
    options = {"primary_key": config["primary_key"], "features": features,
               "target": config.get("target_key"), "score_column": settings.get("score_column", "score")}
    current = read_batch(data_path, predictions[filename], **options)
    declared = config.get("features", {})
    categorical = [name for name in features if name in declared.get("categorical", []) or
                   (name not in declared.get("numeric", []) and not pd.api.types.is_numeric_dtype(current[name]))]
    numerical = [name for name in features if name not in categorical]
    context = {
        "model_id": declared_artifacts["model_id"], "environment": declared_artifacts["environment"],
        "execution_arn": execution_arn,
        "model_package_arn": declared_artifacts["model_package_arn"],
        "current_data_uri": declared_artifacts["data"]["uri"],
        "current_predictions_uri": declared_artifacts["predictions"][filename]["uri"],
        "summary_uri": job["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"].rstrip("/") +
                       "/monitoring/summary.json",
        "selected_features": features,
    }
    context.update({
        "reference_execution_arn": reference_arn,
        "reference_data_uri": previous["data"]["uri"],
        "reference_predictions_uri": previous["predictions"][filename]["uri"],
        "reference_data_producer_execution_arn": previous["data"]["producer_execution_arn"],
        "reference_score_producer_execution_arn": previous["predictions"][filename]["producer_execution_arn"],
        "reference_model_package_arn": previous["model_package_arn"],
    })
    context["same_model_version"] = previous["model_package_arn"] == context["model_package_arn"]
    s3 = session.client("s3")
    def object_body(uri):
        bucket, key = uri.removeprefix("s3://").split("/", 1)
        return s3.get_object(Bucket=bucket, Key=key)["Body"]
    with object_body(context["reference_data_uri"]) as data, object_body(context["reference_predictions_uri"]) as scores:
        reference = read_batch(data, scores, **options)
    task = config["task_type"]
    summary = write_report(
        reference, current, Path(output) / "monitoring", numerical=numerical, categorical=categorical,
        context=context, threshold=settings.get("threshold", 0.2), drift_share=settings.get("drift_share", 0.5),
        task_type=task, target=config.get("target_key"),
        probability="_monitor_score" if task == "classification" else None,
        prediction="_monitor_score" if task == "regression" else None,
        decision_threshold=config.get("threshold", 0.5))
    print(json.dumps({"event": "inference_monitoring", "drift_status": summary["drift_status"],
                      "quality_status": summary["quality_status"], "context": context}), flush=True)
    return summary
