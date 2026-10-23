"""Small batch comparison -> S3/MLflow -> optional metric and notification pattern."""

import json
import math
import mimetypes
import os
from pathlib import Path
import time

from mlops_models.storage import s3_location


def write_report(reference, current, output, *, numerical, categorical=(), context=None,
                 quality_metrics=None, threshold=0.2, drift_share=0.5):
    """Compare supplied frames using PSI and retain the model owner's optional quality metrics.

    No model is fitted or scored here. Select the input columns and threshold with
    the model owner. Data drift is a review signal, not evidence of model failure.
    """
    os.environ["DO_NOT_TRACK"] = "1"
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    columns = list(numerical) + list(categorical)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("Supply distinct numerical/categorical feature columns.")
    if threshold <= 0 or not 0 < drift_share <= 1:
        raise ValueError("Use a positive PSI threshold and a drift share in (0, 1].")
    definition = DataDefinition(numerical_columns=list(numerical), categorical_columns=list(categorical))
    snapshot = Report([DataDriftPreset(method="psi", threshold=threshold, drift_share=drift_share)]).run(
        reference_data=Dataset.from_pandas(reference[columns], data_definition=definition),
        current_data=Dataset.from_pandas(current[columns], data_definition=definition),
    )
    document = json.loads(snapshot.json())
    scores = {metric["config"]["column"]: metric["value"] for metric in document["metrics"]
              if metric["metric_name"].startswith("ValueDrift")}
    # The report API may omit an unusable column; do not count it as no drift.
    if set(scores) != set(columns) or any(
        type(value) not in (int, float) or not math.isfinite(value) for value in scores.values()
    ):
        raise ValueError("Evidently did not return a finite drift score for every requested column.")
    drifted = sorted(column for column, value in scores.items() if value >= threshold)
    fraction = len(drifted) / len(columns)
    metrics = {"reference_rows": len(reference), "current_rows": len(current),
               "columns_drifted": len(drifted), "drifted_fraction": fraction}
    for name, value in (quality_metrics or {}).items():
        if value is None:
            continue
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"Quality metric {name!r} must be a finite number or null.")
        metrics["quality." + name] = value
    summary = {
        "context": context or {}, "generated_at_ms": int(time.time() * 1000),
        "drift_detected": fraction >= drift_share, "drifted_columns": drifted,
        "scores": scores, "method": "psi", "threshold": threshold, "drift_share": drift_share,
        "quality_status": "provided" if any(key.startswith("quality.") for key in metrics) else "unavailable",
        "interpretation": "Feature drift requests review; it does not establish model-quality degradation.",
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(output / "report.html"))
    for name, value in (("evidently.json", document), ("summary.json", summary), ("metrics.json", metrics)):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return summary


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
    files = ["report.html", "evidently.json", "summary.json", "metrics.json"]
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
            client.log_artifact(run_id, str(output / name), "monitoring")
        client.set_tag(run_id, "monitoring.summary_uri", receipt["summary_uri"])
        for name, value in context.items():
            client.set_tag(run_id, "monitoring." + name, str(value))
        receipt["mlflow_run_id"] = run_id
    if cloudwatch_namespace:
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
