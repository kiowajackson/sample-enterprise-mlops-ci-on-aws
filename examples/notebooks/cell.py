"""Paste into a cell after 04A's inputs, or run with %run -i cell.py.

Uses bucket, role_arn and optional namespace, kms_key, athena_workgroup.
Checks calls independently, creates one tiny OFFLINE group, writes one dummy
record, and requests deletion of that group. Its offline S3 data / Glue table
are retained at the printed locations. No model or processing job is started.
"""

import json
import time
from datetime import datetime, timezone
from uuid import uuid4

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

fs_session = globals().get("session") or boto3.Session()
fs_clients = {}
fs_permission_results = []
fs_sdk_config = Config(connect_timeout=5, read_timeout=20,
                       retries={"mode": "standard", "total_max_attempts": 1})
fs_bucket = str(globals().get("bucket", "")).strip().removeprefix("s3://").rstrip("/")
fs_role = str(globals().get("role_arn", "")).strip()
fs_namespace = str(globals().get("namespace", "source-dev"))
fs_probe_name = f"{fs_namespace[:35]}-access-{uuid4().hex[:8]}"
fs_prefix = f"{fs_namespace}/features/permission-checks/{fs_probe_name}/"


def fs_check(service, operation, **request):
    """Keep API denials separate from missing resources, setup and service errors."""
    iam_service = "sagemaker" if service == "sagemaker-featurestore-runtime" else service
    action = iam_service + ":" + "".join(word.title() for word in operation.split("_"))
    if action == "s3:ListObjectsV2":
        action = "s3:ListBucket"
    try:
        if service not in fs_clients:
            fs_clients[service] = fs_session.client(service, config=fs_sdk_config)
        response = getattr(fs_clients[service], operation)(**request)
    except ClientError as error:
        detail = error.response["Error"]
        code = detail["Code"]
        status = ("DENIED" if code in {"AccessDenied", "AccessDeniedException",
                  "AccessForbidden", "UnauthorizedOperation"} else "ERROR")
        result = {"action": action, "status": status, "code": code,
                  "message": detail["Message"],
                  "request_id": error.response.get("ResponseMetadata", {}).get("RequestId")}
    except Exception as error:
        result = {"action": action, "status": "ERROR", "message": str(error)}
    else:
        result = {"action": action, "status": "OK"}
        fs_permission_results.append(result)
        print(json.dumps(result), flush=True)
        return response
    fs_permission_results.append(result)
    print(json.dumps(result), flush=True)
    return None


def fs_not_tested(action, reason):
    result = {"action": action, "status": "NOT_TESTED", "message": reason}
    fs_permission_results.append(result)
    print(json.dumps(result), flush=True)


fs_identity = fs_check("sts", "get_caller_identity")
print("Caller:", fs_identity["Arn"] if fs_identity else "could not resolve")
print("Region:", fs_session.region_name)
print("Offline writer role:", fs_role or "not supplied")
print("API checks use the notebook caller; supplying role_arn does not assume it.")
print("Probe group:", fs_probe_name)
print("Offline probe prefix:", f"s3://{fs_bucket}/{fs_prefix}")

fs_listing = fs_check("sagemaker", "list_feature_groups", MaxResults=10)
fs_known_groups = globals().get("groups", {})
fs_existing = next((item["name"] for item in fs_known_groups.values()), None)
if not fs_existing and fs_listing and fs_listing.get("FeatureGroupSummaries"):
    fs_existing = fs_listing["FeatureGroupSummaries"][0]["FeatureGroupName"]
if fs_existing:
    # Read only: no records are written to an existing group.
    fs_check("sagemaker", "describe_feature_group", FeatureGroupName=fs_existing)

# These calls still run if Feature Store itself is denied.
fs_check("athena", "get_work_group", WorkGroup=globals().get("athena_workgroup", "primary"))
fs_check("glue", "get_database", Name="sagemaker_featurestore")
if fs_bucket and "/" not in fs_bucket:
    fs_check("s3", "list_objects_v2", Bucket=fs_bucket, Prefix=fs_prefix, MaxKeys=1)

fs_created = None
fs_description = None
if not fs_bucket or "/" in fs_bucket or not fs_role:
    fs_not_tested("sagemaker:CreateFeatureGroup", "Set bucket and role_arn in 04A's inputs.")
else:
    fs_storage = {"S3Uri": f"s3://{fs_bucket}/{fs_prefix}"}
    if globals().get("kms_key"):
        fs_storage["KmsKeyId"] = kms_key
    fs_created = fs_check(
        "sagemaker", "create_feature_group",
        FeatureGroupName=fs_probe_name,
        RecordIdentifierFeatureName="record_id", EventTimeFeatureName="event_time",
        FeatureDefinitions=[
            {"FeatureName": "record_id", "FeatureType": "String"},
            {"FeatureName": "event_time", "FeatureType": "String"},
            {"FeatureName": "value", "FeatureType": "Fractional"},
        ],
        RoleArn=fs_role, OnlineStoreConfig={"EnableOnlineStore": False},
        OfflineStoreConfig={"S3StorageConfig": fs_storage, "DisableGlueTableCreation": False},
    )

try:
    if fs_created is not None:
        fs_deadline = time.monotonic() + 120
        while time.monotonic() < fs_deadline:
            fs_description = fs_check("sagemaker", "describe_feature_group", FeatureGroupName=fs_probe_name)
            if not fs_description:
                break
            print("Group:", fs_description["FeatureGroupStatus"],
                  "| Offline:", fs_description.get("OfflineStoreStatus"),
                  "| Failure:", fs_description.get("FailureReason"), flush=True)
            if fs_description["FeatureGroupStatus"] != "Creating":
                break
            time.sleep(5)
    if fs_description and fs_description["FeatureGroupStatus"] == "Created":
        fs_check("sagemaker-featurestore-runtime", "put_record",
                 FeatureGroupName=fs_probe_name, TargetStores=["OfflineStore"], Record=[
                     {"FeatureName": "record_id", "ValueAsString": "permission-probe"},
                     {"FeatureName": "event_time",
                      "ValueAsString": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
                     {"FeatureName": "value", "ValueAsString": "1.0"},
                 ])
        fs_catalog = fs_description.get("OfflineStoreConfig", {}).get("DataCatalogConfig", {})
        if fs_catalog:
            print("Probe catalog retained:", json.dumps(fs_catalog))
            fs_check("glue", "get_table", DatabaseName=fs_catalog["Database"], Name=fs_catalog["TableName"])
    else:
        fs_not_tested("sagemaker:PutRecord", "The new probe group was not confirmed ready.")
finally:
    if fs_created is not None:
        fs_deleted = fs_check("sagemaker", "delete_feature_group", FeatureGroupName=fs_probe_name)
        print("Group deletion requested." if fs_deleted is not None
              else f"Group cleanup failed: {fs_probe_name}")
    else:
        fs_not_tested("sagemaker:DeleteFeatureGroup", "No group creation was confirmed.")

print("\nSummary (OK means that API call succeeded):")
for fs_result in fs_permission_results:
    print(fs_result["status"], fs_result["action"], fs_result.get("message", ""))
print("This does not verify asynchronous offline delivery or Athena SELECT access to feature data.")
print("An ERROR such as ResourceNotFound does not establish permission. Keep the exact messages above.")
print("If creation succeeded, offline probe S3 files / Glue table may remain at the printed locations.")
