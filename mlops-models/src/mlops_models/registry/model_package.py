"""Register versions in existing groups; resolve approved versions for inference.

The pipeline-step dispatcher calls these functions for ResolveModel or
ValidatePackage. The registration Processing job invokes this module directly.
It never creates groups and writes its registration receipt directly to S3.
"""

import json
import os
from pathlib import Path
import time
from uuid import NAMESPACE_URL, uuid5


def register_existing_group(evaluation, settings, *, client, s3):
    """Create one pending version for this execution, retaining the evaluated artifact."""
    group = client.describe_model_package_group(ModelPackageGroupName=settings["MODEL_PACKAGE_GROUP"])
    image, checkpoint = settings["IMAGE_URI"], evaluation["checkpoint_uri"]
    execution = settings["PIPELINE_EXECUTION_ARN"]
    response = client.create_model_package(
        ModelPackageGroupName=group["ModelPackageGroupArn"],
        ClientToken=uuid5(NAMESPACE_URL, execution + "/" + settings["MODEL_PACKAGE_GROUP"]).hex,
        ModelApprovalStatus="PendingManualApproval",
        InferenceSpecification={
            "Containers": [{"Image": image, "ModelDataUrl": checkpoint, "Environment": {
                "MODEL_ID": settings["MODEL_ID"], "PYTHONPATH": "/opt/ml/model/code",
                "PYTHONSAFEPATH": "1", "PYTHONUNBUFFERED": "1"}}],
            "SupportedContentTypes": ["application/jsonlines"],
            "SupportedResponseMIMETypes": ["application/jsonlines"],
        },
        ModelMetrics={"ModelQuality": {"Statistics": {
            "ContentType": "application/json", "S3Uri": settings["EVALUATION_URI"]}}},
        MetadataProperties={"CommitId": evaluation["model_source_commit"], "GeneratedBy": execution},
        CustomerMetadataProperties={
            "SourceDirty": evaluation["model_source_dirty"], "ImageDigest": image.split("@")[-1],
            "RunArtifacts": settings["REGISTRATION_OUTPUT_URI"].removesuffix("/reports/registry"),
            "SourceUri": settings["SOURCE_URI"], "DeployMode": settings["DEPLOY_MODE"],
        },
    )
    arn = response["ModelPackageArn"]
    for _ in range(60):
        status = client.describe_model_package(ModelPackageName=arn)["ModelPackageStatus"]
        if status == "Completed":
            break
        if status not in {"Pending", "InProgress"}:
            raise RuntimeError(f"Model package {arn} reached status {status}.")
        time.sleep(2)
    else:
        raise TimeoutError(f"Model package {arn} did not complete within two minutes.")
    record = {"package": arn, "approval": "PendingManualApproval", "image": image,
              "checkpoint": checkpoint, "evaluation": settings["EVALUATION_URI"]}
    bucket, prefix = settings["REGISTRATION_OUTPUT_URI"].removeprefix("s3://").split("/", 1)
    encryption = ({"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": settings["S3_KMS_KEY"]}
                  if settings.get("S3_KMS_KEY") else {})
    s3.put_object(Bucket=bucket, Key=prefix.rstrip("/") + "/package.json",
                  Body=(json.dumps(record, indent=2) + "\n").encode(),
                  ContentType="application/json", **encryption)
    print(json.dumps({"event": "model_registered", **record}), flush=True)
    return record


def validate_approved_model_package(package_arn, group, image_uri, checkpoint_uri, output_dir, *, client=None):
    """Validate the package's group, approval and exact image/checkpoint pair and write an authorization record."""
    print(json.dumps({"event": "event_received", "step": "ValidatePackage",
                      "package": package_arn}), flush=True)

    if not package_arn:
        raise ValueError("ModelPackageArn is required for inference.")
    if client is None:
        import boto3
        client = boto3.client("sagemaker", region_name=package_arn.split(":")[3])
    package = client.describe_model_package(ModelPackageName=package_arn)
    if (package["ModelPackageGroupName"] != group or package["ModelApprovalStatus"] != "Approved"
            or package["ModelPackageStatus"] != "Completed"):
        raise ValueError("The package must belong to this model and be Completed and Approved.")
    containers = package["InferenceSpecification"]["Containers"]
    if len(containers) != 1:
        raise ValueError("This pipeline expects a single-container package.")
    container = containers[0]
    if container["Image"] != image_uri or container["ModelDataUrl"] != checkpoint_uri:
        raise ValueError("Inference image and checkpoint must exactly match the approved package.")
    record = {"approval_required": True, "package": package_arn,
              "image": image_uri, "checkpoint": checkpoint_uri, "approval": "Approved"}

    # Retain the decision beside this execution's other outputs.
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "package.json").write_text(json.dumps(record, indent=2) + "\n")

    print(json.dumps({"event": "success", "step": "ValidatePackage", **record}), flush=True)
    return record


def resolve_latest_approved_model(group, output_dir, *, client=None):
    """Select the highest Completed, Approved package version once for this inference run."""
    if client is None:
        import boto3
        client = boto3.client("sagemaker")
    packages = []
    for page in client.get_paginator("list_model_packages").paginate(
            ModelPackageGroupName=group, ModelApprovalStatus="Approved"):
        packages.extend(p for p in page["ModelPackageSummaryList"]
                        if p["ModelPackageStatus"] == "Completed")
    if not packages:
        raise ValueError(f"No Completed, Approved model exists in {group}. Train and approve a version first.")
    latest = max(packages, key=lambda p: p["ModelPackageVersion"])
    package = client.describe_model_package(ModelPackageName=latest["ModelPackageArn"])
    containers = package["InferenceSpecification"]["Containers"]
    if (package["ModelPackageGroupName"] != group or package["ModelApprovalStatus"] != "Approved"
            or package["ModelPackageStatus"] != "Completed" or len(containers) != 1):
        raise ValueError("The selected package changed or is not a Completed, Approved single-container model.")
    record = {"package": package["ModelPackageArn"], "version": package["ModelPackageVersion"],
              "approval": "Approved", "image": containers[0]["Image"],
              "checkpoint": containers[0]["ModelDataUrl"]}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "package.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"event": "approved_model_selected", **record}), flush=True)
    return record


if __name__ == "__main__":
    import boto3

    evaluation = json.loads(Path("/opt/ml/processing/evaluation/evaluation.json").read_text())
    register_existing_group(evaluation, os.environ,
                            client=boto3.client("sagemaker"), s3=boto3.client("s3"))
