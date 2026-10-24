"""Evidently batch reports -> S3/MLflow -> optional existing notification destination."""

import json
import math
import mimetypes
import os
from pathlib import Path
import shutil
import time

from mlops_models.storage import s3_location


def write_report(reference, current, output, *, numerical, categorical=(), context=None,
                 quality_metrics=None, threshold=0.2, drift_share=0.5, task_type=None,
                 target=None, prediction=None, probability=None, decision_threshold=0.5):
    """Describe a batch, optionally compare a reference and assess labelled quality.

    Each labelled batch must already have predictions aligned to its rows.
    No model is fitted or scored here. Missing labels mean unavailable quality,
    not zero quality. PSI thresholds are editable examples for the model owner.
    """
    os.environ["DO_NOT_TRACK"] = "1"
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset, DataSummaryPreset

    columns = list(numerical) + list(categorical)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("Supply distinct numerical/categorical feature columns.")
    if threshold <= 0 or not 0 < drift_share <= 1:
        raise ValueError("Use a positive PSI threshold and a drift share in (0, 1].")
    definition = DataDefinition(numerical_columns=list(numerical), categorical_columns=list(categorical))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Reused preview folders must not retain reports from an earlier reference/labels.
    for name in ("data_drift", "prediction_drift", "model_quality"):
        previous = output / ("evidently_" + name)
        if previous.exists():
            shutil.rmtree(previous)
    reference_data = (Dataset.from_pandas(reference[columns], data_definition=definition)
                      if reference is not None else None)
    current_data = Dataset.from_pandas(current[columns], data_definition=definition)
    quality = Report([DataSummaryPreset()]).run(reference_data=reference_data, current_data=current_data)
    save_snapshot(quality, output / "evidently_data_quality")
    snapshot, scores, drifted, detected = quality, {}, [], None
    metrics = {"current_rows": len(current)}
    if reference is not None:
        snapshot = Report([DataDriftPreset(method="psi", threshold=threshold, drift_share=drift_share)]).run(
            reference_data=reference_data, current_data=current_data)
        save_snapshot(snapshot, output / "evidently_data_drift")
        scores = drift_scores(snapshot, columns)
        drifted = sorted(column for column, value in scores.items() if value >= threshold)
        fraction = len(drifted) / len(columns)
        detected = fraction >= drift_share
        metrics.update(reference_rows=len(reference), columns_drifted=len(drifted), drifted_fraction=fraction)
    frames = {"current": current}
    if reference is not None:
        frames["reference"] = reference
    scored = probability or prediction
    for name, frame in frames.items():
        frame = frame[columns]
        metrics[f"data_quality.{name}.missing_cells"] = int(frame.isna().sum().sum())
        metrics[f"data_quality.{name}.duplicate_rows"] = int(frame.duplicated().sum())
    prediction_drift = None
    if scored:
        for name, frame in frames.items():
            if name == "reference" and scored not in frame:
                continue
            values = frame[scored]
            if values.empty or values.isna().any() or not all(math.isfinite(value) for value in values):
                raise ValueError(f"{name} predictions must be nonempty and finite.")
            for statistic in ("mean", "min", "max"):
                metrics[f"predictions.{name}.{statistic}"] = float(getattr(values, statistic)())
        if reference is not None and scored in reference:
            score_definition = DataDefinition(numerical_columns=[scored], categorical_columns=[])
            prediction_report = Report([DataDriftPreset(method="psi", threshold=threshold)]).run(
                reference_data=Dataset.from_pandas(reference[[scored]], data_definition=score_definition),
                current_data=Dataset.from_pandas(current[[scored]], data_definition=score_definition))
            save_snapshot(prediction_report, output / "evidently_prediction_drift")
            score = drift_scores(prediction_report, [scored])[scored]
            metrics["prediction_drift.score"] = score
            prediction_drift = score >= threshold
    quality_status = "unavailable"
    if task_type is not None:
        measured, quality_status = write_model_quality(
            reference, current, output, task_type=task_type, target=target,
            prediction=prediction, probability=probability, decision_threshold=decision_threshold)
        metrics.update(measured)
    for name, value in (quality_metrics or {}).items():
        if value is None:
            continue
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"Quality metric {name!r} must be a finite number or null.")
        metrics["quality." + name] = value
    summary = {
        "context": context or {}, "generated_at_ms": int(time.time() * 1000),
        "drift_status": "compared" if reference is not None else "not_evaluated_no_reference",
        "drift_detected": detected, "drifted_columns": drifted,
        "prediction_drift_detected": prediction_drift,
        "scores": scores, "method": "psi", "threshold": threshold, "drift_share": drift_share,
        "quality_status": quality_status if quality_status != "unavailable" else
                          "provided" if any(key.startswith("quality.") for key in metrics) else "unavailable",
        "interpretation": "Feature/prediction drift requests review; it does not establish model-quality degradation.",
    }
    # Without a reference, the convenient entry point shows current data quality.
    snapshot.save_html(str(output / "report.html"))
    document = json.loads(snapshot.json())
    for name, value in (("evidently.json", document), ("summary.json", summary), ("metrics.json", metrics)):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return summary


def drift_scores(snapshot, columns):
    """Require an actual score for each compared column; missing is not healthy."""
    scores = {metric["config"]["column"]: metric["value"] for metric in json.loads(snapshot.json())["metrics"]
              if metric["metric_name"].startswith("ValueDrift")}
    if set(scores) != set(columns) or any(
        type(value) not in (int, float) or not math.isfinite(value) for value in scores.values()
    ):
        raise ValueError("Evidently did not return a finite drift score for every requested column.")
    return scores


def save_snapshot(snapshot, directory):
    directory.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(directory / "report.html"))
    (directory / "report.json").write_text(snapshot.json())


def write_model_quality(reference, current, output, *, task_type, target, prediction,
                        probability, decision_threshold):
    """Produce Evidently quality reports and explicit current-minus-reference metrics."""
    import numpy as np
    from sklearn import metrics as sk
    from evidently import DataDefinition, Dataset, Report, BinaryClassification, Regression
    from evidently.presets import ClassificationPreset, RegressionPreset

    if task_type not in {"classification", "regression"}:
        raise ValueError("Model quality supports binary classification or regression.")
    if task_type == "classification" and not 0 <= decision_threshold <= 1:
        raise ValueError("Use a classification decision threshold in [0, 1].")
    if not target or target not in current or current[target].isna().any():
        return {}, "unavailable"
    scored = probability if task_type == "classification" and probability else prediction
    if not scored or scored not in current:
        raise ValueError("Provide the prediction/probability column for model quality.")
    frames = {"current": current.copy()}
    if reference is not None and target in reference and scored in reference and not reference[target].isna().any():
        frames["reference"] = reference.copy()
    values, datasets = {}, {}
    for name, frame in frames.items():
        y, score = frame[target].to_numpy(), frame[scored].to_numpy()
        if not np.isfinite(score).all():
            raise ValueError(f"{name} predictions must be finite.")
        if task_type == "classification":
            if not set(np.unique(y)).issubset({0, 1}):
                raise ValueError("The example classification report requires binary 0/1 labels.")
            if probability:
                if not ((score >= 0) & (score <= 1)).all():
                    raise ValueError("Classification probabilities must lie in [0, 1].")
                frame["_monitor_prediction"] = (score >= decision_threshold).astype(int)
                labels = frame["_monitor_prediction"]
                schema = BinaryClassification(target=target, prediction_probas=probability,
                                              prediction_labels="_monitor_prediction", pos_label=1)
            else:
                labels = score
                schema = BinaryClassification(target=target, prediction_labels=prediction, pos_label=1)
            measured = {
                "accuracy": sk.accuracy_score(y, labels),
                "precision": sk.precision_score(y, labels, zero_division=0),
                "recall": sk.recall_score(y, labels, zero_division=0),
                "f1": sk.f1_score(y, labels, zero_division=0),
            }
            if probability:
                measured["log_loss"] = sk.log_loss(y, score, labels=[0, 1])
                measured["brier"] = sk.brier_score_loss(y, score)
                if len(np.unique(y)) == 2:
                    measured["roc_auc"] = sk.roc_auc_score(y, score)
            definition = DataDefinition(numerical_columns=[], categorical_columns=[], classification=[schema])
        else:
            if not np.isfinite(y).all():
                raise ValueError(f"{name} regression labels must be finite.")
            measured = {"rmse": float(np.sqrt(sk.mean_squared_error(y, score))),
                        "mae": sk.mean_absolute_error(y, score), "r2": sk.r2_score(y, score)}
            definition = DataDefinition(numerical_columns=[], categorical_columns=[],
                                        regression=[Regression(target=target, prediction=prediction)])
        values[name] = {key: float(value) for key, value in measured.items() if np.isfinite(value)}
        datasets[name] = Dataset.from_pandas(frame, data_definition=definition)
    preset = ClassificationPreset(probas_threshold=decision_threshold) if task_type == "classification" else RegressionPreset()
    snapshot = Report([preset]).run(reference_data=datasets.get("reference"), current_data=datasets["current"])
    save_snapshot(snapshot, Path(output) / "evidently_model_quality")
    result = {f"model_quality.{batch}.{key}": value for batch, measured in values.items()
              for key, value in measured.items()}
    for key, baseline in values.get("reference", {}).items():
        if key in values["current"]:
            result["model_quality.delta." + key] = values["current"][key] - baseline
    return result, "reference_and_current" if "reference" in values else "current_only"


def publish_report(session, output, destination, *, client=None, run_id=None,
                   cloudwatch_namespace="", sns_topic_arn="", kms_key=""):
    """Upload one report, optionally attach it to an MLflow run and publish a drift signal.

    Destination is a unique report prefix. The caller owns the MLflow run's lifecycle.
    SNS publication is at least once: consumers can deduplicate on summary_uri.
    """
    output = Path(output)
    summary = json.loads((output / "summary.json").read_text())
    metrics = json.loads((output / "metrics.json").read_text())
    bucket, prefix = s3_location(destination.rstrip("/") + "/")
    if not prefix:
        raise ValueError("Supply a report prefix, not a bucket root.")
    if bool(client) != bool(run_id):
        raise ValueError("Supply both the MLflow client and run ID, or neither.")
    context = summary["context"]
    if cloudwatch_namespace and not all(context.get(key) for key in ("model_id", "environment")):
        raise ValueError("CloudWatch publication needs model_id and environment in the report context.")
    root = f"s3://{bucket}/{prefix}"
    files = [path.relative_to(output).as_posix() for path in sorted(output.rglob("*"))
             if path.is_file() and path.name != "receipt.json"]
    s3 = session.client("s3")
    encryption = {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": kms_key} if kms_key else {}
    for name in files:
        s3.upload_file(str(output / name), bucket, prefix + name, ExtraArgs={
            "ContentType": mimetypes.guess_type(name)[0] or "application/octet-stream", **encryption,
        })
    receipt = {"summary_uri": root + "summary.json", "report_uri": root + "report.html",
               "context": context, "drift_detected": summary["drift_detected"]}
    if client is not None:
        from mlflow.entities import Metric
        # Use a separate analysis run per report, or attach to the corresponding inference run.
        client.log_batch(run_id, metrics=[
            Metric("monitoring." + name, value, summary["generated_at_ms"], 0)
            for name, value in metrics.items()
        ], synchronous=True)
        for name in files:
            relative = Path(name)
            artifact_path = ("monitoring" if relative.parent == Path(".")
                             else relative.parent.as_posix())
            client.log_artifact(run_id, str(output / name), artifact_path)
        client.set_tag(run_id, "monitoring.summary_uri", receipt["summary_uri"])
        for name, value in context.items():
            client.set_tag(run_id, "monitoring." + name, str(value))
        receipt["mlflow_run_id"] = run_id
    if cloudwatch_namespace and "drifted_fraction" in metrics:
        session.client("cloudwatch").put_metric_data(
            Namespace=cloudwatch_namespace, MetricData=[{
                "MetricName": "DriftedFraction", "Value": metrics["drifted_fraction"], "Unit": "None",
                "Dimensions": [{"Name": name, "Value": str(context[name])}
                               for name in ("model_id", "environment")],
            }],
        )
        receipt["cloudwatch_namespace"] = cloudwatch_namespace
    if sns_topic_arn and summary["drift_detected"]:
        payload = {**receipt, "event": "batch_drift_review", "metrics": metrics,
                   "drifted_columns": summary["drifted_columns"],
                   "quality_status": summary["quality_status"],
                   "interpretation": summary["interpretation"]}
        receipt["sns_message_id"] = session.client("sns").publish(
            TopicArn=sns_topic_arn, Message=json.dumps(payload, allow_nan=False),
        )["MessageId"]
    content = json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    (output / "receipt.json").write_text(content)
    s3.put_object(Bucket=bucket, Key=prefix + "receipt.json", Body=content.encode(),
                  ContentType="application/json", **encryption)
    return receipt


def main():
    """Calculate reports in a managed job using the same image as model jobs."""
    import argparse
    import tempfile
    import boto3
    import pandas as pd
    from mlops_models.pipelines.tracking import job_tracking

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/opt/ml/processing/config/config.json")
    parser.add_argument("--reference", default="/opt/ml/processing/reference")
    parser.add_argument("--current", default="/opt/ml/processing/current")
    args = parser.parse_args()
    def csv_file(value):
        path = Path(value)
        files = [path] if path.is_file() else sorted(path.glob("*.csv"))
        if len(files) != 1:
            raise ValueError(f"Expected one mounted batch CSV at {path}; found {len(files)}.")
        return files[0]

    config = json.loads(Path(args.config).read_text())
    session = boto3.Session(region_name=config["region"])
    execution_id = os.environ["PIPELINE_EXECUTION_ID"]
    destination = config["output_root"].rstrip("/") + "/" + execution_id
    context = {**config.get("context", {}), "pipeline_execution_id": execution_id,
               "reference_uri": os.environ.get("REFERENCE_DATA_URI", ""),
               "current_uri": os.environ.get("CURRENT_DATA_URI", "")}
    with tempfile.TemporaryDirectory(prefix="batch-monitoring-") as temporary:
        with job_tracking(context["model_id"], destination) as tracking:
            write_report(pd.read_csv(csv_file(args.reference)), pd.read_csv(csv_file(args.current)), temporary,
                         context=context, **config["report"])
            receipt = publish_report(
                session, temporary, destination,
                client=tracking["client"] if tracking else None,
                run_id=tracking["run_id"] if tracking else None,
                sns_topic_arn=config.get("sns_topic_arn", ""), kms_key=config.get("kms_key", ""))
            if tracking:
                record = {"step": "Monitor", "producer_execution_id": execution_id,
                          "mlflow": {key: tracking[key] for key in
                                     ("run_id", "tracking_uri", "experiment_id", "execution_arn")}}
                content = json.dumps(record, indent=2).encode()
                bucket, key = s3_location(destination + "/_execution.json")
                encryption = ({"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": config["kms_key"]}
                              if config.get("kms_key") else {})
                session.client("s3").put_object(Bucket=bucket, Key=key, Body=content, **encryption)
            print(json.dumps(receipt), flush=True)


def configure_drift_alarm(cloudwatch, *, name, namespace, model_id, environment,
                          threshold, topic_arn, report_uri):
    """Create/update one optional CloudWatch alarm on the published drift fraction.

    Use a dedicated name for the example and delete it after review. The description
    points to the report being demonstrated. Production report lookup/notification
    routing belongs to the platform's receiving process.
    """
    cloudwatch.put_metric_alarm(
        AlarmName=name,
        AlarmDescription=json.dumps({"report_uri": report_uri, "model_id": model_id,
                                     "environment": environment}),
        Namespace=namespace, MetricName="DriftedFraction",
        Dimensions=[{"Name": "model_id", "Value": model_id},
                    {"Name": "environment", "Value": environment}],
        Statistic="Maximum", Period=60, EvaluationPeriods=1, DatapointsToAlarm=1,
        Threshold=threshold, ComparisonOperator="GreaterThanOrEqualToThreshold",
        TreatMissingData="missing", AlarmActions=[topic_arn],
    )


if __name__ == "__main__":
    main()
