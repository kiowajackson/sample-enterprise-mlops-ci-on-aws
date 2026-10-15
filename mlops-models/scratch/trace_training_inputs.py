"""Paste this whole file into a notebook cell and set TRAINING_JOB_NAME.

Read the S3 inputs recorded on that training job, including a cached producer's
selection file. Save the selection CSV and a report locally under .runs/.
Only column names and selection metadata are printed; no prepared data rows.
This makes no AWS changes and does not choose another input if a read fails.
"""

TRAINING_JOB_NAME = ""

import csv
import io
import json
from pathlib import Path
from urllib.parse import urlparse

import boto3


def s3_location(uri):
    location = urlparse(uri)
    if location.scheme != "s3" or not location.netloc or not location.path.strip("/"):
        raise ValueError(f"Expected an S3 object or prefix: {uri}")
    return location.netloc, location.path.lstrip("/")


def read_object(s3, uri):
    bucket, key = s3_location(uri)
    with s3.get_object(Bucket=bucket, Key=key)["Body"] as body:
        return body.read()


def trace_training_inputs(training_job_name):
    if not training_job_name.strip():
        raise ValueError("Set TRAINING_JOB_NAME to the failed training job's name, then run this cell.")
    session = boto3.Session()
    job = session.client("sagemaker").describe_training_job(TrainingJobName=training_job_name)
    s3 = session.client("s3")
    channels = {}
    for channel in job["InputDataConfig"]:
        if channel["ChannelName"] in {"prepared", "selection", "source"}:
            source = channel["DataSource"]["S3DataSource"]
            if source["S3DataType"] != "S3Prefix":
                raise ValueError(f"Expected the pipeline's S3Prefix input: {channel['ChannelName']}")
            channels[channel["ChannelName"]] = source["S3Uri"]

    model_id = job["Environment"]["MODEL_ID"]
    selection_uri = channels["selection"].rstrip("/") + "/selected_features.csv"
    parameters_uri = channels["source"].rstrip("/") + f"/mlops_models/models/{model_id}/parameters.json"
    print("Training job:", training_job_name, "| status:", job["TrainingJobStatus"])
    print("ModelId:", model_id)
    print("Job failure:", job.get("FailureReason", ""))
    print("Training output root:", job["OutputDataConfig"]["S3OutputPath"])
    print("Selected-feature CSV:", selection_uri)
    print("Prepared-data prefix:", channels["prepared"])
    print("Training parameters:", parameters_uri)
    print("These are the job's recorded inputs; a cached selection can belong to an earlier execution.")

    selection_bytes = read_object(s3, selection_uri)
    rows = list(csv.reader(io.StringIO(selection_bytes.decode("utf-8-sig"))))
    if not rows or not rows[0] or any(not row for row in rows[1:]):
        raise ValueError(f"Selected-feature CSV has an empty header or record: {selection_uri}")
    selected = [row[0] for row in rows[1:]]
    print("Selection CSV headers:", rows[0])
    print("Feature-name comparisons use the first column:", repr(rows[0][0]))
    print("Selected-feature rows:", len(selected))
    print("First 20 selection records:", rows[1:21])
    print("The full selection CSV will be saved locally.")

    parameters = json.loads(read_object(s3, parameters_uri))
    target = parameters["target_key"]
    print("Configured target_key:", repr(target))
    print("Training source's feature_selection settings:", parameters.get("feature_selection", {}))
    print("A target need not be a selected predictor; it must be present in the prepared data.")

    bucket, prefix = s3_location(channels["prepared"])
    prepared = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for item in page.get("Contents", []):
            if not item["Key"].lower().endswith(".csv"):
                continue
            with io.TextIOWrapper(s3.get_object(Bucket=bucket, Key=item["Key"])["Body"],
                                  encoding="utf-8-sig", newline="") as stream:
                columns = next(csv.reader(stream))
            normalized_columns = {name.casefold() for name in columns}
            result = {
                "s3_uri": f"s3://{bucket}/{item['Key']}",
                "columns": columns,
                "target_present_exactly": target in columns,
                "target_case_matches": [name for name in columns if name.casefold() == target.casefold()],
                "selected_features_missing_exactly": [name for name in selected if name not in columns],
                "selected_features_missing_ignoring_case": [
                    name for name in selected if name.casefold() not in normalized_columns
                ],
            }
            prepared.append(result)
            print("Prepared CSV:", result["s3_uri"])
            print("Column count:", len(columns), "| first 20:", columns[:20])
            print("Target present exactly:", result["target_present_exactly"],
                  "| case matches:", result["target_case_matches"])
            print("Selected names missing exactly:", result["selected_features_missing_exactly"][:20])
            print("Selected names missing ignoring case:", result["selected_features_missing_ignoring_case"][:20])
    if not prepared:
        raise FileNotFoundError(f"No prepared CSV files at the recorded input prefix: {channels['prepared']}")

    report = {
        "training_job": training_job_name, "model_id": model_id, "inputs": channels,
        "selection_uri": selection_uri, "selection_headers": rows[0],
        "selection_rows": len(selected), "selected_features": selected,
        "parameters_uri": parameters_uri, "target_key": target,
        "training_feature_selection_settings": parameters.get("feature_selection", {}),
        "prepared": prepared,
    }
    output = Path.cwd() / ".runs/training-input-trace" / training_job_name
    output.mkdir(parents=True, exist_ok=True)
    (output / "selected_features.csv").write_bytes(selection_bytes)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Saved selected-feature CSV:", (output / "selected_features.csv").resolve())
    print("Saved full input report:", (output / "report.json").resolve())
    print("The parameters above belong to training's source, not necessarily a cached selector's source.")
    return report


if __name__ == "__main__":
    trace_training_inputs(TRAINING_JOB_NAME)
