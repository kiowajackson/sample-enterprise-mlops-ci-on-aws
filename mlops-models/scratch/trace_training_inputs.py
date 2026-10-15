"""Paste this whole file into a notebook cell and set the two job names.

Use the training job and the FeatureSelection processing job that produced its
selection CSV. For cached selection, use the original producing job. The cell
checks that job's output matches the training input before tracing its raw CSV.
Read metadata/sample paths from notebooks/config.json in this checkout.
Compare metadata → mapping/sample → S3 CSV → raw → selection → prepared columns.
Save reports and the supplied selector source locally under .runs/.
No data rows are printed, code is not executed, and no AWS resources are changed.
"""

TRAINING_JOB_NAME = ""
FEATURE_SELECTION_JOB_NAME = ""

import csv
import hashlib
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


def read_csv_header(s3, uri):
    bucket, key = s3_location(uri)
    with io.TextIOWrapper(s3.get_object(Bucket=bucket, Key=key)["Body"],
                          encoding="utf-8-sig", newline="") as stream:
        return next(csv.reader(stream))


def trace_metadata(selection_input):
    root = next((path for path in (Path.cwd(), *Path.cwd().parents)
                 if (path / "mlops-models/src").is_dir()), None)
    if root is None:
        raise ValueError("Run this cell from the project or its notebooks folder.")
    config_path = root / "notebooks/config.json"
    config = json.loads(config_path.read_text())
    metadata_path = Path(config["metadata_csv"]).expanduser()
    if not metadata_path.is_absolute():
        metadata_path = root / metadata_path
    with metadata_path.open(encoding="utf-8-sig", newline="") as stream:
        names = [row["column_name"].strip() for row in csv.DictReader(stream)]
    unique = {}
    for name in names:
        unique.setdefault(name.lower(), name)
    report = {
        "config_path": str(config_path), "metadata_path": str(metadata_path.resolve()),
        "metadata_definitions": len(names), "unique_names": list(unique.values()),
        "duplicates_removed": len(names) - len(unique),
    }
    print("Notebook config:", config_path)
    print("Input metadata:", metadata_path.resolve())
    print("Metadata feature definitions (rows):", len(names),
          "| unique names after strip/lower deduplication:", len(unique),
          "| duplicate definitions:", report["duplicates_removed"])
    print("Local files use the current notebook config; S3 inputs use the recorded jobs.")

    if config["csv_input"].startswith("s3://"):
        print("Config points directly to S3; no local generation folder is specified.")
        report["generation"] = None
        return report
    sample_path = Path(config["csv_input"]).expanduser()
    if not sample_path.is_absolute():
        sample_path = root / sample_path
    mapping_path = sample_path.parent / "feature-mapping.csv"
    details_path = sample_path.parent / "source-metadata.json"
    with sample_path.open(encoding="utf-8-sig", newline="") as stream:
        sample_columns = next(csv.reader(stream))
    with mapping_path.open(encoding="utf-8-sig", newline="") as stream:
        mapped = [row["feature"] for row in csv.DictReader(stream)]
    details = json.loads(details_path.read_text())
    metadata_hash_matches = (
        hashlib.sha256(metadata_path.read_bytes()).hexdigest() == details["metadata_sha256"])
    # Notebook 01 generates two key columns followed by the mapped feature names.
    generated_keys = sample_columns[:2]
    key_names = {name.lower() for name in generated_keys}
    expected_features = [name for key, name in unique.items() if key not in key_names]
    original = selection_input["original_csv"]
    report["generation"] = {
        "sample_path": str(sample_path.resolve()), "sample_columns": sample_columns,
        "mapping_path": str(mapping_path.resolve()), "mapped_names": mapped,
        "source_metadata_path": str(details_path.resolve()),
        "metadata_hash_matches_generation": metadata_hash_matches,
        "generated_key_columns": generated_keys, "expected_features": expected_features,
        "metadata_matches_mapping_exactly": expected_features == mapped,
        "mapping_matches_sample_exactly": generated_keys + mapped == sample_columns,
        "local_sample_matches_s3_header_exactly": (
            sample_columns == original["columns"] if original is not None else None),
    }
    generation = report["generation"]
    print("Metadata hash matches sample generation:", metadata_hash_matches)
    print("Generated key columns:", generated_keys)
    print("Expected feature names after deduplication/excluding generated keys:", len(expected_features))
    print("Feature mapping:", mapping_path.resolve(), "| feature count:", len(mapped))
    print("Metadata feature names/order/case match mapping:", generation["metadata_matches_mapping_exactly"])
    print("Local generated sample:", sample_path.resolve(), "| column count:", len(sample_columns))
    print("Mapping plus generated keys match sample:", generation["mapping_matches_sample_exactly"])
    print("Local sample header matches recorded S3 input:", generation["local_sample_matches_s3_header_exactly"])
    if not metadata_hash_matches:
        print("The configured metadata differs from the generation record; do not treat them as one generation.")
    return report


def trace_selection_inputs(sm, s3, job_name, model_id, selection_prefix, output):
    job = sm.describe_processing_job(ProcessingJobName=job_name)
    outputs = {item["OutputName"]: item["S3Output"]["S3Uri"]
               for item in job["ProcessingOutputConfig"]["Outputs"]}
    if outputs["selection"].rstrip("/") != selection_prefix.rstrip("/"):
        raise ValueError(
            f"This FeatureSelection job writes {outputs['selection']}, but training reads "
            f"{selection_prefix}. Set FEATURE_SELECTION_JOB_NAME to the actual producer; "
            "for a cache hit, use the original job. No alternate artifact will be read."
        )
    if job["Environment"]["MODEL_ID"] != model_id:
        raise ValueError("The FeatureSelection job and training job have different ModelIds.")
    inputs = {item["InputName"]: item["S3Input"]["S3Uri"]
              for item in job["ProcessingInputs"] if item["InputName"] in {"raw", "source"}}
    raw_uri = inputs["raw"].rstrip("/") + "/data.csv"
    provenance_uri = inputs["raw"].rstrip("/") + "/provenance.json"
    source = inputs["source"].rstrip("/")
    parameters_uri = source + f"/mlops_models/models/{model_id}/parameters.json"
    parameters = json.loads(read_object(s3, parameters_uri))
    provenance = json.loads(read_object(s3, provenance_uri))
    columns = read_csv_header(s3, raw_uri)
    target = parameters["target_key"]
    print("FeatureSelection job:", job_name, "| status:", job["ProcessingJobStatus"])
    print("FeatureSelection output matches the training selection input:", outputs["selection"])
    print("Raw CSV entering FeatureSelection:", raw_uri)
    print("Raw column count:", len(columns), "| first 20:", columns[:20])
    print("FeatureSelection parameters:", parameters_uri)
    print("FeatureSelection target_key:", repr(target))
    print("Raw target present exactly:", target in columns,
          "| case matches:", [name for name in columns if name.casefold() == target.casefold()])
    print("FeatureSelection settings:", parameters.get("feature_selection", {}))
    features = parameters.get("features")
    print("FeatureSelection explicit 'features' setting (first 2000 characters):",
          json.dumps(features)[:2000])
    if isinstance(features, dict) and all(isinstance(names, list) for names in features.values()):
        print("Explicit candidate entries:", sum(len(names) for names in features.values()),
              "| counts by group:", {group: len(names) for group, names in features.items()})
    elif isinstance(features, list):
        print("Explicit candidate entries:", len(features))
    print("These settings are supplied to the job; the custom selector decides how to use them.")

    original = None
    if provenance.get("data_source") == "csv_direct":
        original_uri = provenance["input_uri"]
        original_columns = read_csv_header(s3, original_uri)
        original = {"s3_uri": original_uri, "columns": original_columns,
                    "header_matches_raw_exactly": original_columns == columns}
        print("Original CSV recorded by GetData:", original_uri)
        print("Original CSV column count:", len(original_columns))
        print("Original and raw CSV headers match exactly:", original["header_matches_raw_exactly"])
    else:
        print("GetData source mode:", provenance.get("data_source", "not recorded"),
              "| no original CSV is recorded for this mode.")

    output.mkdir(parents=True, exist_ok=True)
    (output / "feature-selection-parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    source_files = {}
    for filename, relative in (
        ("feature-selection-base.py", "mlops_models/model_base/training/feature_selection.py"),
        ("feature-selection-usecase.py", f"mlops_models/models/{model_id}/training/feature_selection.py"),
    ):
        uri = source + "/" + relative
        (output / filename).write_bytes(read_object(s3, uri))
        source_files[filename] = uri
        print("Saved supplied selector source:", (output / filename).resolve())
    return {
        "job_name": job_name, "inputs": inputs, "selection_output": outputs["selection"],
        "raw_uri": raw_uri, "raw_columns": columns, "provenance_uri": provenance_uri,
        "original_csv": original, "parameters_uri": parameters_uri, "parameters": parameters,
        "source_files": source_files,
    }


def trace_training_inputs(training_job_name, feature_selection_job_name):
    if not training_job_name.strip() or not feature_selection_job_name.strip():
        raise ValueError("Set TRAINING_JOB_NAME and FEATURE_SELECTION_JOB_NAME, then run this cell.")
    session = boto3.Session()
    sm = session.client("sagemaker")
    job = sm.describe_training_job(TrainingJobName=training_job_name)
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

    output = Path.cwd() / ".runs/training-input-trace" / training_job_name
    selection_input = trace_selection_inputs(
        sm, s3, feature_selection_job_name, model_id, channels["selection"], output)
    metadata_input = trace_metadata(selection_input)

    bucket, prefix = s3_location(channels["prepared"])
    prepared = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for item in page.get("Contents", []):
            if not item["Key"].lower().endswith(".csv"):
                continue
            columns = read_csv_header(s3, f"s3://{bucket}/{item['Key']}")
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
        "feature_selection": selection_input, "metadata": metadata_input, "prepared": prepared,
    }
    (output / "selected_features.csv").write_bytes(selection_bytes)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Saved selected-feature CSV:", (output / "selected_features.csv").resolve())
    print("Saved full input report:", (output / "report.json").resolve())
    print("Column-count trace:", {
        "metadata_definitions": metadata_input["metadata_definitions"],
        "metadata_unique_names": len(metadata_input["unique_names"]),
        "mapped_features": (len(metadata_input["generation"]["mapped_names"])
                            if metadata_input["generation"] is not None else None),
        "local_sample": (len(metadata_input["generation"]["sample_columns"])
                         if metadata_input["generation"] is not None else None),
        "original_csv": (len(selection_input["original_csv"]["columns"])
                         if selection_input["original_csv"] is not None else None),
        "raw_entering_selection": len(selection_input["raw_columns"]),
        "selected_predictors": len(selected),
        "prepared_csvs": [len(item["columns"]) for item in prepared],
    })
    print("The parameters above belong to training's source, not necessarily a cached selector's source.")
    print("The separate FeatureSelection parameters/source belong to its producing job.")
    print("These are current S3 objects; replacing an object after a job ran changes what this cell reads.")
    return report


if __name__ == "__main__":
    trace_training_inputs(TRAINING_JOB_NAME, FEATURE_SELECTION_JOB_NAME)
