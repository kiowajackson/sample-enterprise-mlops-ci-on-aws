"""Deploy completion infrastructure before publishing the namespace's pipelines.

Called by notebook 02 or a CI build. Platform can replace this CloudFormation
adapter with its CDK constructs while retaining the handler and event contract.
The ZIP is built separately in the approved image with build.py.
"""

from datetime import datetime, timezone
import io
import json
from pathlib import Path
import re
import time
import zipfile


def deploy_completion(session, namespace, s3_root, code_uri, *, tracking_uri="",
                      tracking_artifact_uris=(), registry_prefix="", kms_key="",
                      permissions_boundary_arn=""):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,45}", namespace):
        raise ValueError("Completion namespace must be at most 46 ordinary name characters.")
    if not code_uri.startswith("s3://") or "/" not in code_uri[5:]:
        raise ValueError("Set completion_code_uri to the immutable Lambda ZIP built by operations/completion/build.py.")
    code_bucket, code_key = code_uri[5:].split("/", 1)
    bucket, prefix = s3_root.removeprefix("s3://").split("/", 1)
    if prefix != namespace:
        raise ValueError("The completion output root must be s3://<bucket>/<namespace>.")
    artifact_resources = []
    for uri in tracking_artifact_uris:
        if not uri.startswith("s3://") or "/" not in uri[5:].rstrip("/"):
            raise ValueError("Supply the S3 artifact location of each selected MLflow experiment.")
        artifact_resources.append("arn:" + session.get_partition_for_region(session.region_name)
                                  + ":s3:::" + uri[5:].rstrip("/") + "/*")
    if tracking_uri and not artifact_resources:
        raise ValueError("Completion needs the selected experiments' S3 artifact locations.")
    response = session.client("s3").get_object(Bucket=code_bucket, Key=code_key)
    with response["Body"] as stream:
        content = stream.read(50 * 1024 * 1024 + 1)
    if len(content) > 50 * 1024 * 1024:
        raise ValueError("Completion ZIP exceeds this deployment's 50 MiB limit.")
    root = Path(__file__).resolve().parents[2]
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        sources = {"handler.py": Path(__file__).with_name("handler.py")}
        for relative in ("__init__.py", "storage.py", "pipelines/__init__.py",
                         "pipelines/inspect.py", "pipelines/tracking.py"):
            sources["mlops_models/" + relative] = root / "mlops-models/src/mlops_models" / relative
        for name, local in sources.items():
            if archive.read(name) != local.read_bytes():
                raise ValueError(f"Completion ZIP contains another revision of {name}; rebuild it from this checkout.")
    cf = session.client("cloudformation")
    name = namespace + "-completion"
    try:
        existing = cf.describe_stacks(StackName=name)["Stacks"][0]
    except cf.exceptions.ClientError as error:
        if "does not exist" not in error.response["Error"]["Message"]:
            raise
        existing = None
    if existing:
        tags = {item["Key"]: item["Value"] for item in existing.get("Tags", [])}
        if tags.get("Deployment") != namespace or tags.get("Component") != "pipeline-completion":
            raise ValueError(f"{name} belongs to another deployment.")
    previous = {item["ParameterKey"]: item["ParameterValue"]
                for item in existing.get("Parameters", [])} if existing else {}
    values = {
        "Namespace": namespace, "CodeBucket": code_bucket, "CodeKey": code_key,
        "DataBucket": bucket, "S3Root": s3_root, "TrackingUri": tracking_uri,
        "TrackingArtifactResources": ",".join(artifact_resources),
        "RegistryPrefix": registry_prefix, "KmsKey": kms_key,
        "PermissionsBoundaryArn": permissions_boundary_arn,
        # Preserve coverage of executions started before a code update.
        "WatchFrom": previous.get("WatchFrom", datetime.now(timezone.utc).isoformat()),
    }
    request = {
        "StackName": name, "TemplateBody": Path(__file__).with_name("template.json").read_text(),
        "Parameters": [{"ParameterKey": key, "ParameterValue": value} for key, value in values.items()],
        "Capabilities": ["CAPABILITY_IAM"],
        "Tags": [{"Key": "Deployment", "Value": namespace},
                 {"Key": "Component", "Value": "pipeline-completion"}],
    }
    try:
        (cf.update_stack if existing else cf.create_stack)(**request)
    except cf.exceptions.ClientError as error:
        if "No updates are to be performed" not in error.response["Error"]["Message"]:
            raise
    deadline = time.monotonic() + 900
    while True:
        stack = cf.describe_stacks(StackName=name)["Stacks"][0]
        status = stack["StackStatus"]
        if status in {"CREATE_COMPLETE", "UPDATE_COMPLETE"}:
            break
        if status not in {"CREATE_IN_PROGRESS", "UPDATE_IN_PROGRESS", "UPDATE_COMPLETE_CLEANUP_IN_PROGRESS"}:
            raise RuntimeError(f"Completion stack {name}: {status}. Inspect its CloudFormation events.")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Completion deployment still running: {name}")
        print(f"Completion infrastructure: {status}", flush=True)
        time.sleep(10)
    outputs = {item["OutputKey"]: item["OutputValue"] for item in stack["Outputs"]}
    for key in ("EventRule", "ReconciliationRule"):
        rule = session.client("events").describe_rule(Name=outputs[key])
        if rule["State"] != "ENABLED":
            raise ValueError(f"Completion rule is not enabled: {outputs[key]}")
    print("Automatic execution completion:", outputs["FunctionArn"])
    return {"stack": name, "code_uri": code_uri, **outputs}
