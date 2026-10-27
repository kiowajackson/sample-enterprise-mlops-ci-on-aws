"""Register versions in existing groups; resolve approved versions for inference.

The pipeline-step dispatcher calls these functions for ResolveModel or
ValidatePackage. The registration Processing job invokes this module directly.
It never creates groups and writes its registration receipt directly to S3.
"""

from copy import copy
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import time
from uuid import NAMESPACE_URL, uuid5

from mlops_models.storage import read_model_metadata, s3_location


def register_existing_group(evaluation, settings, *, client, s3):
    """Register a Development version in the supplied group and deployment namespace.

    This notebook delivery has no production-candidate registration mode.
    ProductionCandidate belongs to the future trusted Platform deployment path.
    """
    group = client.describe_model_package_group(ModelPackageGroupName=settings["MODEL_PACKAGE_GROUP"])
    image, checkpoint = settings["IMAGE_URI"], evaluation["checkpoint_uri"]
    execution = settings["PIPELINE_EXECUTION_ARN"]
    metadata_uri = settings["EVALUATION_URI"].rsplit("/", 1)[0] + "/model_metadata.json"
    if len(metadata_uri) > 256:
        raise ValueError("Use a shorter evaluation S3 prefix: Registry metadata URIs must fit 256 characters.")
    metadata_bucket, metadata_key = metadata_uri.removeprefix("s3://").split("/", 1)
    metadata = s3.get_object(Bucket=metadata_bucket, Key=metadata_key)["Body"].read()
    metadata_sha256 = hashlib.sha256(metadata).hexdigest()
    if metadata_sha256 != evaluation["model_metadata_sha256"]:
        raise ValueError("The published model metadata differs from the evaluated model's metadata.")
    source_uri = evaluation.get("model_source_uri", "")
    source_digest = evaluation.get("model_source_digest", "")
    if not source_uri.startswith("s3://") or len(source_digest) != 64:
        raise ValueError("Train a new model with its saved source URI and digest before registering it.")
    scope = {"DeploymentNamespace": settings["DEPLOYMENT_NAMESPACE"],
             "Environment": settings["ENVIRONMENT"], "ModelId": settings["MODEL_ID"]}
    if not all(scope.values()):
        raise ValueError("Registration needs the deployment namespace, environment and ModelId.")
    lifecycle = {"Stage": "Development", "StageStatus": "PendingApproval"}
    response = client.create_model_package(
        ModelPackageGroupName=group["ModelPackageGroupArn"],
        ClientToken=uuid5(NAMESPACE_URL, execution + "/" + settings["MODEL_PACKAGE_GROUP"]).hex,
        ModelApprovalStatus="PendingManualApproval",
        ModelLifeCycle=lifecycle,
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
            "SourceUri": source_uri, "SourceDigest": source_digest,
            "DeployMode": settings["DEPLOY_MODE"], **scope,
            "ModelMetadataUri": metadata_uri, "ModelMetadataSha256": metadata_sha256,
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
              "checkpoint": checkpoint, "evaluation": settings["EVALUATION_URI"],
              "model_metadata": metadata_uri, "model_metadata_sha256": metadata_sha256,
              "lifecycle": lifecycle, "namespace": scope["DeploymentNamespace"],
              "source_uri": source_uri, "source_digest": source_digest}
    bucket, prefix = settings["REGISTRATION_OUTPUT_URI"].removeprefix("s3://").split("/", 1)
    encryption = ({"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": settings["S3_KMS_KEY"]}
                  if settings.get("S3_KMS_KEY") else {})
    s3.put_object(Bucket=bucket, Key=prefix.rstrip("/") + "/package.json",
                  Body=(json.dumps(record, indent=2) + "\n").encode(),
                  ContentType="application/json", **encryption)
    print(json.dumps({"event": "model_registered", **record}), flush=True)
    return record


def development_scope(package, group, namespace, model_id, environment):
    """Check the recorded scope; Platform IAM must separately enforce who may write it."""
    if not all((namespace, model_id, environment)):
        raise ValueError("Supply the deployment namespace, ModelId and environment.")
    properties = package.get("CustomerMetadataProperties", {})
    return (
        package.get("ModelPackageGroupName") == group
        and package.get("ModelPackageStatus") == "Completed"
        and package.get("ModelLifeCycle", {}).get("Stage") == "Development"
        and properties.get("DeploymentNamespace") == namespace
        and properties.get("ModelId") == model_id
        and properties.get("Environment") == environment
    )


def approved_development_package(package, group, namespace, model_id, environment):
    return (development_scope(package, group, namespace, model_id, environment)
            and package.get("ModelApprovalStatus") == "Approved"
            and package.get("ModelLifeCycle", {}).get("StageStatus") == "Approved")


def validate_approved_model_package(package_arn, group, image_uri, checkpoint_uri, output_dir, *,
                                    namespace, model_id, environment, client=None):
    """Validate the package's group, approval and exact image/checkpoint pair and write an authorization record."""
    print(json.dumps({"event": "event_received", "step": "ValidatePackage",
                      "package": package_arn}), flush=True)

    if not package_arn:
        raise ValueError("ModelPackageArn is required for inference.")
    if client is None:
        import boto3
        client = boto3.client("sagemaker", region_name=package_arn.split(":")[3])
    package = client.describe_model_package(ModelPackageName=package_arn)
    if not approved_development_package(package, group, namespace, model_id, environment):
        raise ValueError("The package must be an approved Development model in this deployment namespace.")
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


def resolve_latest_approved_model(group, output_dir, *, namespace, model_id, environment,
                                 package_arn="", client=None, metadata_uri="",
                                 metadata_checkpoint_uri="", s3=None):
    """Resolve an ARN, or latest-approved (also an omitted ARN in older pipelines)."""
    if client is None:
        import boto3
        client = boto3.client("sagemaker")
    # List summaries omit the namespace metadata. Describe candidates before
    # selecting; a newer model from another user's deployment must not win.
    package = None
    if package_arn and package_arn != "latest-approved":
        package = client.describe_model_package(ModelPackageName=package_arn)
    else:
        packages = []
        for page in client.get_paginator("list_model_packages").paginate(
                ModelPackageGroupName=group, ModelApprovalStatus="Approved"):
            packages.extend(p for p in page["ModelPackageSummaryList"]
                            if p["ModelPackageStatus"] == "Completed")
        for candidate in sorted(packages, key=lambda p: p["ModelPackageVersion"], reverse=True):
            described = client.describe_model_package(ModelPackageName=candidate["ModelPackageArn"])
            if approved_development_package(described, group, namespace, model_id, environment):
                package = described
                break
    if package is None or not approved_development_package(package, group, namespace, model_id, environment):
        raise ValueError(
            f"No eligible Development model in {group} for {namespace}/{model_id}/{environment}. "
            "Train in this namespace and approve both the lifecycle stage and model approval. "
            "An explicitly selected version must satisfy the same checks."
        )
    containers = package["InferenceSpecification"]["Containers"]
    if len(containers) != 1:
        raise ValueError("This pipeline expects a single-container Development model.")
    record = {"package": package["ModelPackageArn"], "version": package["ModelPackageVersion"],
              "approval": "Approved", "image": containers[0]["Image"],
              "checkpoint": containers[0]["ModelDataUrl"]}
    properties = package.get("CustomerMetadataProperties", {})
    source_uri, source_digest = properties.get("SourceUri", ""), properties.get("SourceDigest", "")
    if not source_uri.startswith("s3://") or len(source_digest) != 64:
        raise ValueError("The selected model has no saved source binding. Train and approve a new Development version.")
    record.update(namespace=namespace, lifecycle=package["ModelLifeCycle"],
                  source_uri=source_uri, source_digest=source_digest,
                  source_commit=package.get("MetadataProperties", {}).get("CommitId", "unknown"))
    # Always publish these fields so downstream bindings work for older packages.
    # An empty checksum means the loader cannot verify the saved metadata.
    record["model_metadata"] = properties.get("ModelMetadataUri", "")
    record["model_metadata_sha256"] = properties.get("ModelMetadataSha256", "")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if metadata_uri and metadata_uri != "registry":
        if not metadata_checkpoint_uri or metadata_checkpoint_uri == record["checkpoint"]:
            raise ValueError("Metadata overrides require a separate execution output checkpoint URI.")
        s3_location(metadata_checkpoint_uri)
        if s3 is None:
            import boto3
            s3 = boto3.client("s3")
        _, content = read_model_metadata(metadata_uri, s3=s3)
        # Supply one consistent metadata snapshot to every inference stage,
        # including older archived code and native Transform's serving process.
        # The existing Processing uploader publishes this execution-local copy.
        destination = output / "metadata-override/model.tar.gz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        bucket, key = s3_location(record["checkpoint"])
        found = 0
        with s3.get_object(Bucket=bucket, Key=key)["Body"] as stream, \
                tarfile.open(fileobj=stream, mode="r|gz") as original, \
                tarfile.open(destination, "w:gz") as updated:
            for member in original:
                if PurePosixPath(member.name) == PurePosixPath("model_metadata.json"):
                    if not member.isfile():
                        raise ValueError("The registered model metadata must be a regular JSON file.")
                    found += 1
                    member = copy(member)
                    member.pax_headers = {k: v for k, v in member.pax_headers.items() if k != "size"}
                    member.size = len(content)
                    updated.addfile(member, io.BytesIO(content))
                else:
                    updated.addfile(member, original.extractfile(member) if member.isfile() else None)
        if found != 1:
            destination.unlink()
            raise ValueError("The registered model must contain exactly one model_metadata.json.")
        record.update(
            registered_checkpoint=record["checkpoint"],
            registered_model_metadata=record.get("model_metadata"),
            registered_model_metadata_sha256=record.get("model_metadata_sha256"),
            checkpoint=metadata_checkpoint_uri, model_metadata=metadata_uri,
            model_metadata_sha256=hashlib.sha256(content).hexdigest(),
            metadata_override=True)
    (output / "package.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"event": "approved_model_selected", **record}), flush=True)
    return record


def write_registration_record(result, *, s3):
    """Publish the same output description used by other Processing jobs."""
    from tempfile import TemporaryDirectory
    from mlops_models.runtime.load_source import processing_record

    job = json.loads(Path("/opt/ml/config/processingjobconfig.json").read_text())
    source = Path("/opt/ml/processing/source/source.json")
    with TemporaryDirectory() as directory:
        (Path(directory) / "package.json").write_text(json.dumps(result))
        record = processing_record(job, {
            "source": json.loads(source.read_text()),
            "producer_execution_id": os.environ["PIPELINE_EXECUTION_ARN"].rsplit("/", 1)[-1],
            "model_code_source": "pipeline",
        }, Path("/opt/ml/processing/output"), directory)
    record["producer_execution_arn"] = os.environ["PIPELINE_EXECUTION_ARN"]
    bucket, prefix = os.environ["REGISTRATION_OUTPUT_URI"].removeprefix("s3://").split("/", 1)
    encryption = ({"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": os.environ["S3_KMS_KEY"]}
                  if os.getenv("S3_KMS_KEY") else {})
    s3.put_object(Bucket=bucket, Key=prefix.rstrip("/") + "/_execution.json",
                  Body=(json.dumps(record, indent=2) + "\n").encode(),
                  ContentType="application/json", **encryption)


if __name__ == "__main__":
    import boto3

    from mlops_models.pipelines.tracking import job_tracking

    evaluation = json.loads(Path("/opt/ml/processing/evaluation/evaluation.json").read_text())
    output_uri = os.environ["REGISTRATION_OUTPUT_URI"]
    with job_tracking(os.environ["MODEL_ID"], output_uri) as tracking:
        result = register_existing_group(evaluation, os.environ,
                                         client=boto3.client("sagemaker"), s3=boto3.client("s3"))
        write_registration_record(result, s3=boto3.client("s3"))
        if tracking is not None:
            tracking["client"].log_dict(tracking["run_id"], result, "reports/RegisterModel/package.json")
            tracking["client"].set_tag(tracking["run_id"], "step.RegisterModel.status", "Succeeded")
