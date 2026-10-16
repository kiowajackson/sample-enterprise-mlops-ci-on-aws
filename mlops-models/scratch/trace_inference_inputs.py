"""Paste this entire file into a notebook cell; set the two Processing job names.

Use the failed Score job and the Preprocess job that produced its input. For a
cache hit, use the original producer. Inspect the recorded S3 paths, source
parameters and CSV headers; save the deployed model scripts locally for review.
No data rows are printed and no AWS resources are changed. This cell traces
Processing with model_code_source="pipeline"; it does not inspect model archives.
"""

SCORE_JOB_NAME = ""
PREPROCESS_JOB_NAME = ""

import csv
import hashlib
import io
import json
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError


def s3_location(uri):
    location = urlparse(uri)
    if location.scheme != "s3" or not location.netloc or not location.path.strip("/"):
        raise ValueError(f"Expected an S3 URI: {uri}")
    return location.netloc, location.path.lstrip("/")


def read_object(s3, uri):
    bucket, key = s3_location(uri)
    with s3.get_object(Bucket=bucket, Key=key)["Body"] as stream:
        data = stream.read(2 * 1024 * 1024 + 1)
    if len(data) > 2 * 1024 * 1024:
        raise ValueError(f"Diagnostic settings/source file exceeds 2 MiB: {uri}")
    return data


def csv_header(s3, uri):
    bucket, key = s3_location(uri)
    with io.TextIOWrapper(s3.get_object(Bucket=bucket, Key=key)["Body"],
                          encoding="utf-8-sig", newline="") as stream:
        return next(csv.reader(stream))


def input_at(job, local_path):
    matches = [item["S3Input"] for item in job["ProcessingInputs"]
               if item.get("S3Input", {}).get("LocalPath", "").rstrip("/") == local_path]
    if len(matches) != 1 or matches[0].get("S3DataType") != "S3Prefix":
        raise ValueError(f"Expected one S3Prefix input mounted at {local_path}. Inspect the recorded job.")
    if matches[0].get("S3CompressionType", "None") != "None":
        raise ValueError("This diagnostic expects uncompressed CSV inputs.")
    return matches[0]["S3Uri"].rstrip("/")


def source_settings(s3, job):
    environment = job["Environment"]
    mode = environment.get("MLOPS_MODEL_CODE_SOURCE")
    if mode != "pipeline":
        raise ValueError(
            f"{job['ProcessingJobName']} uses code source {mode!r}. This cell reads pipeline "
            "source only; archived code requires inspecting that exact model archive."
        )
    sources = [item["S3Input"]["S3Uri"] for item in job["ProcessingInputs"]
               if item["InputName"] == "source"]
    if len(sources) != 1:
        raise ValueError("Expected exactly one recorded source input.")
    model_id = environment["MODEL_ID"]
    prefix = sources[0].rstrip("/")
    uri = prefix + f"/mlops_models/models/{model_id}/parameters.json"
    parameters = json.loads(read_object(s3, uri))
    key = parameters["primary_key"]
    if not isinstance(key, str) or not key:
        raise ValueError(f"Expected a nonempty primary_key in {uri}")
    return {"model_id": model_id, "source_uri": prefix, "parameters_uri": uri,
            "primary_key": key, "task_type": parameters.get("task_type")}


def key_summary(columns, key):
    return {"key": key, "exact": key in columns,
            "case_matches": [name for name in columns if name.casefold() == key.casefold()],
            "trim_matches": [name for name in columns if name.strip() == key.strip()],
            "column_count": len(columns), "columns": columns}


def diagnose(raw, prepared, preprocessing_key, scoring_key):
    if preprocessing_key != scoring_key:
        return "Preprocess and Score use different primary_key settings. Compare their recorded parameters."
    if prepared["exact"]:
        return ("The prepared CSV contains Score's configured key. Inspect the path and key actually "
                "used inside ModelPredictor, then any feature-only selection, rename, set_index or "
                "index_col before predict_with_best_model. The CSV header does not prove those internals.")
    if prepared["case_matches"]:
        return "The prepared CSV has a case-only match. Align key and column normalization in model code."
    if prepared["trim_matches"]:
        return "The prepared CSV has a whitespace-only match. Align the key and column names."
    if raw["exact"]:
        return "The raw CSV contains the key but the prepared CSV does not. Inspect preprocessing's selection and CSV write."
    return "The key is absent from both raw and prepared CSVs. Check the input snapshot and configured key."


def trace_inference_inputs(score_job_name, preprocess_job_name):
    if not score_job_name.strip() or not preprocess_job_name.strip():
        raise ValueError("Set SCORE_JOB_NAME and PREPROCESS_JOB_NAME, then run the whole cell.")
    session = boto3.Session()
    sm, s3 = session.client("sagemaker"), session.client("s3")
    score = sm.describe_processing_job(ProcessingJobName=score_job_name)
    prep = sm.describe_processing_job(ProcessingJobName=preprocess_job_name)
    prepared_prefix = input_at(score, "/opt/ml/processing/prepared")
    producers = [item["S3Output"]["S3Uri"].rstrip("/")
                 for item in prep["ProcessingOutputConfig"]["Outputs"] if item["OutputName"] == "prepared"]
    if producers != [prepared_prefix]:
        raise ValueError("That Preprocess job did not produce Score's input. For caching, use the original producer.")
    score_settings, prep_settings = source_settings(s3, score), source_settings(s3, prep)
    if score_settings["model_id"] != prep_settings["model_id"]:
        raise ValueError("Preprocess and Score have different ModelIds.")
    raw_prefix = input_at(prep, "/opt/ml/processing/raw")
    raw_uri, prepared_uri = raw_prefix + "/data.csv", prepared_prefix + "/data.csv"
    key = score_settings["primary_key"]
    raw = key_summary(csv_header(s3, raw_uri), key)
    prepared = key_summary(csv_header(s3, prepared_uri), key)
    report = {"score_job": score_job_name, "preprocess_job": preprocess_job_name,
              "failure": score.get("FailureReason", ""), "score_settings": score_settings,
              "preprocess_settings": prep_settings, "raw_uri": raw_uri, "prepared_uri": prepared_uri,
              "raw": raw, "prepared": prepared}
    provenance_uri = raw_prefix + "/provenance.json"
    try:
        provenance = json.loads(read_object(s3, provenance_uri))
    except ClientError as error:
        if error.response["Error"]["Code"] not in {"NoSuchKey", "404"}:
            raise
        report["original_input_note"] = "No GetData provenance file exists at the recorded raw prefix."
    else:
        report["get_data_provenance_uri"] = provenance_uri
        if provenance.get("data_source") == "csv_direct" and provenance.get("input_uri", "").startswith("s3://"):
            original_uri = provenance["input_uri"]
            report["original_uri"] = original_uri
            report["original"] = key_summary(csv_header(s3, original_uri), key)
    output = Path.cwd() / ".runs/inference-input-trace" / score_job_name
    output.mkdir(parents=True, exist_ok=True)
    report["saved_source"] = {}
    for stage, settings, relative in (
        ("preprocess", prep_settings, "mlops_models/model_base/inference/preprocess.py"),
        ("predict", score_settings, "mlops_models/model_base/inference/predict.py"),
    ):
        uri = settings["source_uri"] + "/" + relative
        data = read_object(s3, uri)
        path = output / (stage + "-deployed.py")
        path.write_bytes(data)
        report["saved_source"][stage] = {"uri": uri, "path": str(path.resolve()),
                                         "sha256": hashlib.sha256(data).hexdigest()}
    report["finding"] = diagnose(raw, prepared, prep_settings["primary_key"], key)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Score job:", score_job_name, "|", score["ProcessingJobStatus"])
    print("Preprocess job:", preprocess_job_name, "| output matches Score's input")
    print("ModelId:", score_settings["model_id"])
    print("Score primary_key:", repr(key), "| Preprocess primary_key:", repr(prep_settings["primary_key"]))
    print("Score parameters:", score_settings["parameters_uri"])
    print("Preprocess parameters:", prep_settings["parameters_uri"])
    for stage in ("original", "raw", "prepared"):
        if stage not in report:
            continue
        summary = report[stage]
        print(stage, report[stage + "_uri"])
        print("  columns:", summary["column_count"], "| key present:", summary["exact"],
              "| case matches:", summary["case_matches"], "| whitespace matches:", summary["trim_matches"])
        print("  first 20 column names:", summary["columns"][:20])
    print("Finding:", report["finding"])
    print("Full header report and deployed source:", output.resolve())
    print("No rows were printed. These are recorded job URIs read now; overwritten S3 objects can differ from job-time bytes.")
    return report


if __name__ == "__main__":
    inference_input_report = trace_inference_inputs(SCORE_JOB_NAME, PREPROCESS_JOB_NAME)
