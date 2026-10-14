"""Sample CDK resource definitions to call from the platform's existing stack.

Input: 04A's feature-mapping.csv (feature,group,type), after schema review.
Use the same record_id and event_time configured in 04A.
The offline role is the Feature Store service role; ingestion uses its own role.
This module provisions schemas only. It does not ingest records or grant IAM.
For direct API creation instead, use platform_ml_resources_api.py.
Resources created by that script must be imported into CDK before CDK manages
them; deploying this example afterward does not automatically adopt them.

Example inside a Stack:
    registry = add_model_groups(self)  # Both environments in this account.
    features = add_feature_groups(
        self, "feature-mapping.csv",
        record_id=record_id, event_time=event_time,
        offline_role_arn=offline_role_arn, offline_s3_uri=offline_s3_uri,
        kms_key_arn=kms_key_arn,
    )
For separate dev/prod accounts, pass environments=("dev",) or ("prod",).
"""

import csv
from pathlib import Path
import re

from aws_cdk import RemovalPolicy
from aws_cdk import aws_sagemaker as sagemaker


TYPES = {
    "binary": "Integral", "integer": "Integral",
    "continuous": "Fractional", "string": "String",
}
RESERVED = {"is_deleted", "write_time", "api_invocation_time", "delivery_id", "row_rank"}


def add_model_groups(scope, environments=("dev", "prod")):
    """Create the ICLV and Active Trader registry groups."""
    groups = {}
    for environment in environments:
        for model in ("iclv", "active-trader"):
            name = f"{model}-sm-{environment}"
            group = sagemaker.CfnModelPackageGroup(
                scope, name, model_package_group_name=name,
            )
            group.apply_removal_policy(RemovalPolicy.RETAIN)
            if groups:
                group.node.add_dependency(list(groups.values())[-1])
            groups[name] = group
    return groups


def add_feature_groups(scope, mapping_csv, *, record_id, event_time,
                       offline_role_arn, offline_s3_uri, kms_key_arn=None,
                       environment="dev"):
    """Create shared offline groups; preserve the CSV's feature names and types."""
    with Path(mapping_csv).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["feature", "group", "type"]:
            raise ValueError("Expected 04A's feature-mapping.csv: feature,group,type.")
        rows = list(reader)
    if not rows or not record_id or not event_time or record_id.lower() == event_time.lower():
        raise ValueError("Supply a nonempty mapping and distinct record/event-time keys.")
    if not offline_role_arn or not offline_s3_uri.startswith("s3://"):
        raise ValueError("Supply the platform offline role ARN and S3 location.")
    if {record_id.lower(), event_time.lower()} & RESERVED:
        raise ValueError("Record/event-time keys must not use reserved feature names.")

    # Three shared fields leave room for 2,497 mapped features per group.
    # delivery_id is required by the current ingestion and snapshot reader.
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
        columns = schemas.setdefault(key, dict(shared))
        columns[feature] = TYPES[kind]
    for key, columns in schemas.items():
        name = f"wealth-{key.replace('_', '-')}-features-{environment}"
        if len(columns) > 2500 or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", name):
            raise ValueError(f"Split oversized groups or shorten their names: {key}")

    storage = {"S3Uri": offline_s3_uri}
    if kms_key_arn:
        storage["KmsKeyId"] = kms_key_arn
    groups = {}
    for key, columns in sorted(schemas.items()):
        name = f"wealth-{key.replace('_', '-')}-features-{environment}"
        group = sagemaker.CfnFeatureGroup(
            scope, name, feature_group_name=name,
            record_identifier_feature_name=record_id,
            event_time_feature_name=event_time,
            feature_definitions=[
                sagemaker.CfnFeatureGroup.FeatureDefinitionProperty(
                    feature_name=feature, feature_type=kind,
                )
                for feature, kind in columns.items()
            ],
            role_arn=offline_role_arn,
            online_store_config={"EnableOnlineStore": False},
            offline_store_config={
                "S3StorageConfig": storage,
                "DisableGlueTableCreation": False,
            },
        )
        group.apply_removal_policy(RemovalPolicy.RETAIN)
        # Avoid exceeding the concurrent feature-group creation limit.
        if groups:
            group.node.add_dependency(list(groups.values())[-1])
        groups[key] = group
    return groups
