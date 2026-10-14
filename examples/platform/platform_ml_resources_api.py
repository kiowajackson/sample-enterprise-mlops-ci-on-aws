"""Create the registry/Feature Store resources directly with boto3.

Fill in the required settings, then run: python platform_ml_resources_api.py
In a notebook: %run platform_ml_resources_api.py
Requires boto3 and the platform operator's AWS credentials. CDK is not required.
Uses 04A's reviewed feature-mapping.csv; no records are ingested or IAM granted.
Matching existing groups are reused. Different feature schemas/settings stop.
Return the generated platform-resources.json to the application team.
"""

import csv
import json
from pathlib import Path
import re
import time

import boto3
from botocore.exceptions import ClientError


# Required settings: use the actual mapping and the same key names as 04A.
AWS_REGION = "ca-central-1"
MAPPING_CSV = "feature-mapping.csv"
RECORD_ID = ""
EVENT_TIME = ""
OFFLINE_ROLE_ARN = ""             # Feature Store's S3 writer role, not the ingestion caller.
OFFLINE_S3_URI = ""               # Approved s3://bucket/prefix/ for offline storage.

# Optional settings.
KMS_KEY_ARN = ""
REGISTRY_ENVIRONMENTS = ("dev", "prod")  # Select one if accounts are separate.
FEATURE_ENVIRONMENT = "dev"
OUTPUT_JSON = "platform-resources.json"

TYPES = {"binary": "Integral", "integer": "Integral", "continuous": "Fractional", "string": "String"}
RESERVED = {"is_deleted", "write_time", "api_invocation_time", "delivery_id", "row_rank"}


def feature_requests(mapping_csv, *, record_id, event_time, offline_role_arn,
                     offline_s3_uri, kms_key_arn=None, environment="dev"):
    """Read the mapping and prepare every request before creating resources."""
    with Path(mapping_csv).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["feature", "group", "type"]:
            raise ValueError("Expected feature-mapping.csv with: feature,group,type.")
        rows = list(reader)
    if not rows or not record_id or not event_time or record_id.lower() == event_time.lower():
        raise ValueError("Supply a nonempty mapping and distinct record/event-time keys.")
    if not offline_role_arn or not offline_s3_uri.startswith("s3://"):
        raise ValueError("Fill in OFFLINE_ROLE_ARN and OFFLINE_S3_URI.")
    if environment not in {"dev", "prod"} or {record_id.lower(), event_time.lower()} & RESERVED:
        raise ValueError("Use dev/prod and non-reserved record/event-time keys.")
    shared = {record_id: "String", event_time: "String", "delivery_id": "String"}
    seen = {record_id.lower(), event_time.lower(), *RESERVED}
    schemas = {}
    for row in rows:
        feature, key, kind = row["feature"], row["group"], row["type"]
        if not feature or feature.lower() in seen:
            raise ValueError(f"Duplicate, key or reserved feature: {feature}")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", key or "") or kind not in TYPES:
            raise ValueError(f"Invalid group/type in mapping: {row}")
        seen.add(feature.lower())
        schemas.setdefault(key, dict(shared))[feature] = TYPES[kind]

    storage = {"S3Uri": offline_s3_uri}
    if kms_key_arn:
        storage["KmsKeyId"] = kms_key_arn
    requests = {}
    for key, columns in sorted(schemas.items()):
        name = f"wealth-{key.replace('_', '-')}-features-{environment}"
        if len(columns) > 2500 or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", name):
            raise ValueError(f"Split oversized groups or shorten their names: {key}")
        requests[key] = {
            "FeatureGroupName": name, "RecordIdentifierFeatureName": record_id,
            "EventTimeFeatureName": event_time, "RoleArn": offline_role_arn,
            "FeatureDefinitions": [{"FeatureName": f, "FeatureType": t} for f, t in columns.items()],
            "OnlineStoreConfig": {"EnableOnlineStore": False},
            "OfflineStoreConfig": {"S3StorageConfig": storage, "DisableGlueTableCreation": False},
        }
    return requests


def describe_existing(describe, **name):
    try:
        return describe(**name)
    except ClientError as error:
        code = error.response["Error"]["Code"]
        message = error.response["Error"]["Message"].lower()
        if code in {"ResourceNotFound", "ResourceNotFoundException"}:
            return None
        # SageMaker may report a missing model package group as ValidationException.
        if code == "ValidationException" and next(iter(name.values())).lower() in message:
            if "does not exist" in message or "not found" in message:
                return None
        raise


def wait_ready(describe, name, status_field, ready_status, timeout=900):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = describe(**name)
        status = result[status_field]
        if status == ready_status:
            return result
        if status not in {"Creating", "Pending", "InProgress"}:
            raise RuntimeError(f"{name}: {status}; {result.get('FailureReason', '')}")
        print("Waiting:", next(iter(name.values())), status, flush=True)
        time.sleep(10)
    raise TimeoutError(f"Timed out waiting for {name}; rerun to resume.")


def check_feature_group(actual, requested):
    """Verify existing or newly created resources against the requested schema."""
    for field in ("RecordIdentifierFeatureName", "EventTimeFeatureName", "RoleArn"):
        if actual.get(field) != requested[field]:
            raise ValueError(f"{requested['FeatureGroupName']}: different {field}.")
    definitions = lambda group: {f["FeatureName"]: f["FeatureType"] for f in group["FeatureDefinitions"]}
    if definitions(actual) != definitions(requested) or any(f.get("CollectionType") for f in actual["FeatureDefinitions"]):
        raise ValueError(f"{requested['FeatureGroupName']}: different feature definitions.")
    offline = actual.get("OfflineStoreConfig", {})
    storage = offline.get("S3StorageConfig", {})
    expected = requested["OfflineStoreConfig"]["S3StorageConfig"]
    if storage.get("S3Uri", "").rstrip("/") != expected["S3Uri"].rstrip("/"):
        raise ValueError("Existing offline S3 location differs.")
    if "KmsKeyId" in expected and storage.get("KmsKeyId") != expected["KmsKeyId"]:
        raise ValueError("Existing offline KMS key differs.")
    if actual.get("OnlineStoreConfig", {}).get("EnableOnlineStore") or offline.get("DisableGlueTableCreation"):
        raise ValueError("Expected offline-only storage with Glue table creation enabled.")
    if actual.get("OfflineStoreStatus", {}).get("Status") in {"Blocked", "Disabled"}:
        raise RuntimeError(f"Offline store is not usable: {actual['OfflineStoreStatus']}")
    if not all(offline.get("DataCatalogConfig", {}).get(k) for k in ("Catalog", "Database", "TableName")):
        raise RuntimeError("Glue catalog bindings are not available yet; rerun after creation completes.")


def create_resources(sm, requests, registry_environments=("dev", "prod")):
    if not registry_environments or set(registry_environments) - {"dev", "prod"}:
        raise ValueError("Select dev and/or prod registry environments.")
    receipt = {"model_package_groups": {}, "feature_groups": {}}
    for environment in registry_environments:
        for model in ("iclv", "active-trader"):
            name = f"{model}-sm-{environment}"
            args = {"ModelPackageGroupName": name}
            if describe_existing(sm.describe_model_package_group, **args) is None:
                sm.create_model_package_group(**args)
            group = wait_ready(sm.describe_model_package_group, args, "ModelPackageGroupStatus", "Completed")
            receipt["model_package_groups"][name] = group["ModelPackageGroupArn"]
            print("Registry ready:", name, flush=True)
    # Wait for each group before creating another; avoid the concurrency limit.
    for key, request in requests.items():
        args = {"FeatureGroupName": request["FeatureGroupName"]}
        if describe_existing(sm.describe_feature_group, **args) is None:
            sm.create_feature_group(**request)
        group = wait_ready(sm.describe_feature_group, args, "FeatureGroupStatus", "Created")
        check_feature_group(group, request)
        receipt["feature_groups"][key] = {
            "name": group["FeatureGroupName"], "arn": group["FeatureGroupArn"],
            "record_id": group["RecordIdentifierFeatureName"],
            "event_time": group["EventTimeFeatureName"],
            "offline_store": group["OfflineStoreConfig"],
        }
        print("Feature group ready:", group["FeatureGroupName"], flush=True)
    return receipt


def main():
    requests = feature_requests(
        MAPPING_CSV, record_id=RECORD_ID, event_time=EVENT_TIME,
        offline_role_arn=OFFLINE_ROLE_ARN, offline_s3_uri=OFFLINE_S3_URI,
        kms_key_arn=KMS_KEY_ARN, environment=FEATURE_ENVIRONMENT,
    )
    session = boto3.Session(region_name=AWS_REGION)
    caller = session.client("sts").get_caller_identity()
    print("Account:", caller["Account"], "| Region:", AWS_REGION, "| Caller:", caller["Arn"])
    print("Feature groups:", {r["FeatureGroupName"]: len(r["FeatureDefinitions"]) for r in requests.values()})
    receipt = create_resources(session.client("sagemaker"), requests, REGISTRY_ENVIRONMENTS)
    receipt.update(account=caller["Account"], region=AWS_REGION)
    Path(OUTPUT_JSON).write_text(json.dumps(receipt, indent=2) + "\n")
    print("Return these bindings:", Path(OUTPUT_JSON).resolve())


if __name__ == "__main__":
    main()
