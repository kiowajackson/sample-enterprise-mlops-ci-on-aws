"""Small, credential-free context summaries for saved deployment notebook output."""

from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys


CONFIG_FIELDS = (
    "namespace", "user_alias", "bucket", "s3_prefix", "role_arn", "image_uri",
    "csv_input", "test_csv_input", "inference_csv_input", "feature_store_config_uri",
)


def context(root, session):
    """Read diagnostic context without making AWS requests or dumping config."""
    root = Path(root)
    details = {
        "repository": str(root), "cwd": str(Path.cwd()),
        "python": sys.version, "executable": sys.executable,
        "region": session.region_name if isinstance(session.region_name, str) else None,
        "config_path": str(root / "notebooks/config.json"),
    }
    try:
        config = json.loads((root / "notebooks/config.json").read_text())
        if not isinstance(config, dict):
            raise ValueError("Expected a JSON object.")
        details["config"] = {key: config[key] for key in CONFIG_FIELDS if key in config}
    except (OSError, ValueError) as error:
        details["config_read_error"] = f"{type(error).__name__}: {error}"
    details["versions"] = {}
    for package in ("boto3", "botocore", "pandas"):
        try:
            details["versions"][package] = version(package)
        except PackageNotFoundError:
            details["versions"][package] = "not installed"
    details["source_sha256"] = {}
    for name in (
        "notebooks/00_deploy_pipelines.ipynb",
        "notebooks/source/setup/deploy_pipelines.py",
        "notebooks/source/setup/deployment_diagnostics.py",
        "notebooks/source/setup/platform_settings.py",
        "notebooks/source/setup/prepare_data.py",
        "notebooks/source/setup/publish_pipelines.py",
        "mlops-models/src/mlops_models/pipelines/source.py",
    ):
        try:
            details["source_sha256"][name] = sha256((root / name).read_bytes()).hexdigest()
        except OSError:
            details["source_sha256"][name] = "unavailable"
    details["loaded_helpers"] = {
        name: str(getattr(sys.modules[name], "__file__", "unknown"))
        for name in ("deploy_pipelines", "platform_settings", "prepare_data", "publish_pipelines")
        if name in sys.modules
    }
    return details


def exception_chain(error):
    """Keep the underlying AWS error when an outer helper wraps it."""
    chain, seen = [], set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        item = {"type": type(error).__name__, "message": str(error)}
        response = getattr(error, "response", None)
        if isinstance(response, dict):
            item["operation"] = getattr(error, "operation_name", None)
            item["aws_error"] = response.get("Error", {}).get("Code")
            metadata = response.get("ResponseMetadata", {})
            item.update({key: metadata[key] for key in
                         ("RequestId", "HTTPStatusCode", "RetryAttempts") if key in metadata})
        chain.append(item)
        error = error.__cause__ or (
            error.__context__ if not error.__suppress_context__ else None
        )
    return chain


def pipeline_call(client, operation, progress, **request):
    """Show the pipeline target and response shape, without printing its definition."""
    target = {key: request[key] for key in
              ("PipelineName", "PipelineVersionId", "ResourceArn", "RoleArn") if key in request}
    label = f"SageMaker {operation} {json.dumps(target, sort_keys=True)}"
    if progress:
        progress("Request: " + label)
    try:
        response = getattr(client, operation)(**request)
    except BaseException as error:
        if progress:
            progress(f"Failed: {label} | {type(error).__name__}")
        raise
    summary = {
        "keys": sorted(response),
        **{key: response[key] for key in ("PipelineArn", "PipelineVersionId") if key in response},
        **{key: response.get("ResponseMetadata", {})[key] for key in
           ("RequestId", "HTTPStatusCode", "RetryAttempts")
           if key in response.get("ResponseMetadata", {})},
    }
    if progress:
        progress("Response: " + label + " | " + json.dumps(summary, sort_keys=True))
    return response
