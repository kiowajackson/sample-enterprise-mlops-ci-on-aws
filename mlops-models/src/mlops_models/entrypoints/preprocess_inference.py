"""Preprocess batch-inference data using the approved model's saved configuration."""

import io
import json
import os
from pathlib import Path
import re
import time
from urllib.parse import urlparse

import boto3
import pandas as pd
import shutil

FEATURE_STORE_QUERY = """
SELECT {columns}
FROM (
    SELECT *, row_number() OVER (
        PARTITION BY {keys} ORDER BY api_invocation_time DESC, write_time DESC
    ) AS row_rank
    FROM {database}.{table}
    WHERE delivery_id = ? AND api_invocation_time <= from_iso8601_timestamp(?)
)
WHERE row_rank = 1 AND NOT is_deleted
ORDER BY {keys}
"""


def execute(model_id, input_path, output_dir, *, model_dir, shard_rows=10000):
    """Snapshot scoring rows and write bounded JSON-lines shards."""
    config = json.loads((Path(model_dir) / "model-config.json").read_text())
    if os.getenv("MLOPS_DATA_SOURCE") == "csv_direct":
        # SageMaker downloads the supplied CSV; the existing output uploader retains it.
        path = Path(input_path)
        if path.is_dir():
            files = sorted(path.glob("*.csv"))
            if len(files) != 1:
                raise ValueError(f"csv_direct expects one flat CSV in {path}; found {len(files)}.")
            path = files[0]
        raw = Path(output_dir) / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, raw / "data.csv")
        (raw / "provenance.json").write_text(json.dumps({
            "data_source": "csv_direct", "model_id": model_id,
            "input_uri": os.getenv("MLOPS_CSV_INPUT_URI", str(path)),
            "pipeline_execution_id": os.getenv("PIPELINE_EXECUTION_ID"),
        }, indent=2) + "\n")
    elif os.getenv("MLOPS_FEATURE_STORE_CONFIG_URI"):
        df, provenance = get_container_input(os.environ["MLOPS_FEATURE_STORE_CONFIG_URI"],
                                             model_id, config)
        raw = write_container_output(df, provenance, output_dir)
    else:
        # Direct Python development supplies a local CSV instead of querying AWS.
        raw = Path(output_dir) / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(input_path, raw / "data.csv")
    output, rows = write_batch_input(get_container_chunks(raw, config, shard_rows), output_dir, config)
    print(json.dumps({"event": "success", "step": "Preprocess", "model": model_id, "rows": rows}), flush=True)
    return output


def get_container_chunks(source, config, chunksize):
    """Read the local raw snapshot in bounded chunks."""
    columns = list(dict.fromkeys([
        config["identifier"], *config["features"]["numeric"], *config["features"]["categorical"],
        *config.get("inference_required_columns", []),
    ]))
    with pd.read_csv(Path(source) / "data.csv", dtype={config["identifier"]: str},
                     keep_default_na=False, na_values=["", r"\N"],
                     chunksize=chunksize, usecols=columns) as reader:
        yield from reader


def write_batch_input(chunks, output, config):
    """Take input chunks and saved settings; write selected columns and return the shard directory and row count."""
    output = Path(output) / "shards"
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use an empty batch shard output directory.")
    columns = list(dict.fromkeys([
        config["identifier"], *config["features"]["numeric"], *config["features"]["categorical"],
        *config.get("inference_required_columns", []),
    ]))
    rows = 0
    for number, df in enumerate(chunks):
        df[columns].to_json(output / f"part-{number:06d}.jsonl",
                               orient="records", lines=True, double_precision=15)
        rows += len(df)
    if rows == 0:
        raise ValueError("Batch input contains no rows.")
    return output, rows


def get_container_input(store_config_uri, model_id, config):
    """Query existing feature groups and return this model's dataframe."""
    session = boto3.Session()
    s3 = session.client("s3")
    settings, binding = read_json(s3, store_config_uri)
    receipt, delivery = resolve_delivery(s3, settings, os.getenv("MLOPS_FEATURE_STORE_DELIVERY_ID", ""))
    keys = [config["identifier"], config["timestamp"]]
    features = [name for names in config["features"].values() for name in names]
    features = list(dict.fromkeys([*features, *config.get("inference_required_columns", [])]))
    universe_root = os.getenv("MLOPS_FEATURE_STORE_UNIVERSE_ROOT", "").rstrip("/")
    wanted = features
    dfs, queries = [], []
    for group in settings["groups"].values():
        columns = [name for name in group["columns"] if name in keys or name in wanted]
        if not set(columns) - set(keys):
            continue
        df, query_info = read_group(session, settings, group, columns, receipt)
        dfs.append(df.set_index(keys))
        queries.append(query_info)
    df = pd.concat(dfs, axis=1).reset_index()[keys + wanted]
    if universe_root:
        bucket, key = (universe_root + "/" + model_id + "/universe.csv").removeprefix("s3://").split("/", 1)
        content = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        population = pd.read_csv(io.BytesIO(content), dtype={keys[0]: str}, keep_default_na=False)
        population[keys[1]] = pd.to_datetime(population[keys[1]], utc=True).map(
            lambda value: value.isoformat().replace("+00:00", "Z"))
        columns = keys
        df = population[columns].merge(df, on=keys, how="left", validate="one_to_one", indicator=True)
        if not df.pop("_merge").eq("both").all():
            raise ValueError("Some population ID/date pairs have no matching Feature Store row.")
    provenance = {"binding": binding, "delivery": receipt, "receipt_source": delivery,
                  "queries": queries, "model_id": model_id, "selected_features": features,
                  "pipeline_execution_id": os.getenv("PIPELINE_EXECUTION_ID"),
                  "universe_uri": universe_root + "/" + model_id + "/universe.csv" if universe_root else None}
    return df, provenance


def write_container_output(df, provenance, output_dir):
    """Write the raw snapshot; load_source.py uploads it under this execution."""
    output = Path(output_dir) / "raw"
    output.mkdir(parents=True, exist_ok=True)
    df.to_csv(output / "data.csv", index=False)
    provenance["records"] = len(df)
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return output


def read_group(session, settings, group, columns, receipt):
    """Execute one query and read its CSV result; ingestion must already be complete."""
    keys = [settings["record_id"], settings["event_time"]]
    database = group["database"]
    quote = lambda name: '"' + name.replace('"', '""') + '"'
    sql = FEATURE_STORE_QUERY.format(
        columns=", ".join(quote(name) + " AS " + quote(name) for name in columns),
        keys=", ".join(map(quote, keys)), database=quote(database), table=quote(group["table"]),
    )
    parameters = [f"'{receipt['delivery_id']}'", f"'{receipt['completed_at']}'"]
    query_id, location = run_query(session, {**settings, "database": database,
                                            "catalog": group.get("catalog", "AwsDataCatalog")}, sql, parameters)
    bucket, key = location.removeprefix("s3://").split("/", 1)
    content = session.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
    types = {"String": str, "Integral": "Int64", "Fractional": float}
    df = pd.read_csv(io.BytesIO(content), dtype={name: types[group["columns"][name]] for name in columns},
                     keep_default_na=False, na_values=[""])
    return df, {"group": group["name"], "query_id": query_id, "sql": sql, "parameters": parameters}


def read_json(s3, uri):
    """Read one explicit JSON object and retain its location/version for provenance."""
    location = urlparse(uri)
    if location.scheme != "s3" or not location.netloc or not location.path.strip("/"):
        raise ValueError("Supply an s3://bucket/key JSON object URI.")
    response = s3.get_object(Bucket=location.netloc, Key=location.path.lstrip("/"))
    return json.loads(response["Body"].read()), {"uri": uri, "version_id": response.get("VersionId")}


def resolve_delivery(s3, settings, delivery_id=""):
    """Resolve a requested receipt, or the most recently published completed receipt."""
    bucket, root = settings["bucket"], settings["prefix"] + "deliveries/"
    if delivery_id:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", delivery_id):
            raise ValueError("Use the delivery ID printed by ingestion.")
        key = root + delivery_id + "/ingestion.json"
    else:
        pages = s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=root)
        completed = (item for page in pages for item in page.get("Contents", [])
                     if item["Key"].endswith("/ingestion.json"))
        latest = max(completed, key=lambda item: (item["LastModified"], item["Key"]), default=None)
        if latest is None:
            raise ValueError("No completed Feature Store ingestion is available.")
        key = latest["Key"]
    receipt, identity = read_json(s3, f"s3://{bucket}/{key}")
    return receipt, identity


def run_query(session, settings, sql, parameters):
    """Take SQL, bound values and an Athena workgroup; return the completed query ID and CSV location."""
    if len(sql.encode()) > 262144:
        raise ValueError("This group's query exceeds Athena's query length limit; use smaller mapped groups.")
    client = session.client("athena")
    output = {}
    if settings.get("query_output"):
        output["ResultConfiguration"] = {"OutputLocation": settings["query_output"]}
        if settings.get("kms_key"):
            output["ResultConfiguration"]["EncryptionConfiguration"] = {
                "EncryptionOption": "SSE_KMS", "KmsKey": settings["kms_key"]}
    identifier = client.start_query_execution(
        QueryString=sql, WorkGroup=settings["workgroup"], ExecutionParameters=parameters,
        QueryExecutionContext={"Database": settings["database"], "Catalog": settings.get("catalog", "AwsDataCatalog")},
        **output,
    )["QueryExecutionId"]
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        result = client.get_query_execution(QueryExecutionId=identifier)["QueryExecution"]
        status = result["Status"]
        if status["State"] == "SUCCEEDED":
            return identifier, result["ResultConfiguration"]["OutputLocation"]
        if status["State"] in {"FAILED", "CANCELLED"}:
            raise RuntimeError(status.get("StateChangeReason", status["State"]))
        time.sleep(2)
    client.stop_query_execution(QueryExecutionId=identifier)
    raise TimeoutError(f"Athena query {identifier} exceeded the wait limit.")
