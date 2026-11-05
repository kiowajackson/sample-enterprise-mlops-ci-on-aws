"""Shared platform configuration stored in an account-local Secrets Manager secret.

Use the caller's existing Boto3 session. No credentials are stored or exported.
Creating the secret and loading a workspace configuration are separate operations.
"""

from pathlib import Path
import json
import os
import re
import tempfile

from botocore.exceptions import ClientError


SECRET_NAME = "wsf-mlops-setup"
REQUIRED_KEYS = (
    "bucket", "s3_prefix", "role_arn", "image_uri", "primary_key",
)
SHARED_KEYS = (
    *REQUIRED_KEYS, "registry_group_prefix", "model_package_group",
    "tracking_uri", "completion_code_uri",
)
DEFAULT_METADATA = "examples/data/feature-metadata.csv"


def _context(session):
    if not session.region_name:
        raise ValueError("Select the workload region in the AWS environment or Boto3 session.")
    account = session.client("sts").get_caller_identity()["Account"]
    return account, session.region_name


def _validate(payload, account, region):
    if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int:
        raise ValueError("The setup secret requires an integer schema_version.")
    if payload["schema_version"] != 1:
        raise ValueError("Unsupported setup secret schema_version; expected 1.")
    if set(payload) != {"schema_version", "account_id", "region", "settings"}:
        raise ValueError("The setup secret requires schema_version, account_id, region and settings.")
    if payload["account_id"] != account or payload["region"] != region:
        raise ValueError("The setup secret does not match the caller's account and region.")
    settings = payload["settings"]
    if not isinstance(settings, dict):
        raise ValueError("The setup secret settings must be a JSON object.")
    unknown = set(settings) - set(SHARED_KEYS)
    if unknown:
        raise ValueError(f"Unsupported shared settings: {', '.join(sorted(unknown))}")
    if any(not isinstance(value, str) for value in settings.values()):
        raise ValueError("Every shared setting must be a string.")
    settings = {key: value.strip() for key, value in settings.items()}
    missing = [key for key in REQUIRED_KEYS if not settings.get(key)]
    if missing:
        raise ValueError(f"Missing shared settings: {', '.join(missing)}")
    if "/" in settings["bucket"] or not settings["bucket"]:
        raise ValueError("bucket must be a bucket name, without s3:// or a folder.")
    prefix = settings["s3_prefix"].strip("/")
    if (not re.fullmatch(r"[A-Za-z0-9_./-]+", prefix)
            or any(part in {"", ".", ".."} for part in prefix.split("/"))):
        raise ValueError("s3_prefix must be a nonempty relative S3 folder path.")
    settings["s3_prefix"] = prefix
    role_parts = settings["role_arn"].split(":", 5)
    if (len(role_parts) != 6 or role_parts[0] != "arn" or role_parts[2] != "iam"
            or role_parts[4] != account or not role_parts[5].startswith("role/")):
        raise ValueError("role_arn must identify a role in the workload account.")
    if not settings.get("registry_group_prefix") and not settings.get("model_package_group"):
        raise ValueError("Supply registry_group_prefix or an existing model_package_group.")
    return {
        "schema_version": 1, "account_id": account, "region": region, "settings": settings,
    }


def _decode(response, account, region):
    if "SecretString" not in response:
        raise ValueError("The setup secret must contain a JSON SecretString.")
    return _validate(json.loads(response["SecretString"]), account, region)


def create_setup_secret(root, session, *, secret_name=SECRET_NAME, kms_key_id=None):
    """Create once from config.json; reuse identical settings, never overwrite."""
    root = Path(root).resolve()
    config = json.loads((root / "notebooks/config.json").read_text())
    account, region = _context(session)
    payload = _validate({
        "schema_version": 1, "account_id": account, "region": region,
        "settings": {key: config[key] for key in SHARED_KEYS if key in config},
    }, account, region)
    client = session.client("secretsmanager")
    try:
        response = client.get_secret_value(SecretId=secret_name)
    except ClientError as error:
        if error.response["Error"]["Code"] != "ResourceNotFoundException":
            raise
        request = {
            "Name": secret_name,
            "Description": "Shared Wealth ML platform settings for notebook setup.",
            "SecretString": json.dumps(payload, sort_keys=True),
        }
        if kms_key_id:
            request["KmsKeyId"] = kms_key_id
        try:
            created = client.create_secret(**request)
        except ClientError as error:
            if error.response["Error"]["Code"] != "ResourceExistsException":
                raise
            # Another administrator created it concurrently. Compare, never update.
            response = client.get_secret_value(SecretId=secret_name)
            status = "reused"
        else:
            response = client.get_secret_value(
                SecretId=created["ARN"], VersionId=created["VersionId"],
            )
            status = "created"
    else:
        status = "reused"
    if _decode(response, account, region) != payload:
        raise ValueError(
            "The existing secret contains different settings. It was not overwritten. "
            "Review it in Secrets Manager and deliberately update it there if needed."
        )
    return {
        "status": status, "secret_arn": response["ARN"], "version_id": response["VersionId"],
        "account_id": account, "region": region,
    }


def read_setup_secret(session, *, secret_name=SECRET_NAME):
    """Retrieve and validate one version without creating or updating any secret."""
    account, region = _context(session)
    response = session.client("secretsmanager").get_secret_value(SecretId=secret_name)
    payload = _decode(response, account, region)
    return payload, {
        "secret_arn": response["ARN"], "version_id": response["VersionId"],
        "account_id": account, "region": region,
    }


def _atomic_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".setup-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(content, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_local_config(root):
    """Distinguish a fresh checkout from malformed or partially populated config."""
    root = Path(root).resolve()
    path = root / "notebooks/config.json"
    template = json.loads((root / "notebooks/config.example.json").read_text())
    original = path.read_bytes() if path.exists() else None
    if original is None:
        return {}, template, original, "missing"
    if not original.strip():
        return {}, template, original, "empty"
    try:
        current = json.loads(original)
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"{path} is malformed; it was not replaced.") from error
    if not isinstance(current, dict):
        raise ValueError(f"{path} must contain a JSON object; it was not replaced.")
    for key in (*SHARED_KEYS, "namespace", "user_alias", "metadata_csv", "csv_input",
                "test_csv_input", "inference_csv_input", "feature_store_config_uri"):
        if key in current and current[key] is not None and not isinstance(current[key], str):
            raise ValueError(f"{path}: {key} must be a string; it was not replaced.")
    if not current:
        state = "empty"
    elif current == template:
        state = "template"
    elif (all(isinstance(current.get(key), str) and current[key].strip() for key in REQUIRED_KEYS)
          and (current.get("registry_group_prefix") or current.get("model_package_group"))):
        state = "configured"
    else:
        state = "partial"
    return current, template, original, state


def load_workspace_settings(root, session, *, secret_name=SECRET_NAME,
                            refresh_shared=False, metadata_csv=DEFAULT_METADATA):
    """Populate config.json while retaining namespace, dataset and personal settings.

    Complete local settings are validated and reused without accessing secrets.
    Incomplete settings or an explicit refresh read the shared secret. Conflicting
    nonempty values stop before writing unless refresh_shared is explicit.
    A refresh changes only keys actually present in the secret.
    The bundled metadata is used only when no metadata path or dataset is set.
    No dataset is generated or downloaded, and no pipeline is deployed.
    """
    root = Path(root).resolve()
    config_path = root / "notebooks/config.json"
    current, template, original, state = read_local_config(root)
    use_local = state == "configured" and not refresh_shared
    if use_local:
        account, region = _context(session)
        payload = _validate({
            "schema_version": 1, "account_id": account, "region": region,
            "settings": {key: current[key] for key in SHARED_KEYS
                         if current.get(key) is not None},
        }, account, region)
        provenance = {"settings_source": "local", "account_id": account, "region": region}
        print("Using complete local config.json; Secrets Manager was not requested.")
    else:
        payload, provenance = read_setup_secret(session, secret_name=secret_name)
        provenance["settings_source"] = "secret"
        print("Loaded shared settings from Secrets Manager.")
    settings = payload["settings"]
    conflicts = [key for key, value in settings.items()
                  if not use_local and state != "template" and current.get(key) is not None
                 and current[key].strip() not in ("", value)]
    if conflicts and not refresh_shared:
        raise ValueError(
            "Local platform settings differ for: " + ", ".join(conflicts)
            + ". No config was written. Review, then set refresh_shared=True to adopt the secret."
        )
    configured = dict(current) if state == "configured" else template | current
    for key in set(template) | set(SHARED_KEYS):
        if key in configured and configured[key] is None:
            configured[key] = template.get(key, "")
    filled_keys = []
    for key, value in settings.items():
        if refresh_shared or state == "template" or not (configured.get(key) or "").strip():
            if configured.get(key) != value:
                filled_keys.append(key)
            configured[key] = value
    dataset_keys = ("csv_input", "test_csv_input", "inference_csv_input", "feature_store_config_uri")
    if not configured.get("metadata_csv") and not any(configured.get(key) for key in dataset_keys):
        metadata = root / metadata_csv
        if not metadata.is_file():
            raise ValueError("Bundled metadata is unavailable; provide metadata_csv or existing dataset inputs.")
        configured["metadata_csv"] = metadata_csv
    # Do not replace edits made while the network request was in flight.
    if (config_path.read_bytes() if config_path.exists() else None) != original:
        raise RuntimeError("config.json changed while loading settings. Rerun to preserve those edits.")
    changed = configured != current or original is None or not original.strip()
    if changed:
        _atomic_json(config_path, configured)
    receipt = {
        **provenance, "config_path": str(config_path), "shared_keys": sorted(settings),
        "existing_config": original is not None, "refreshed_shared": refresh_shared,
        "config_state": state, "config_changed": changed, "filled_keys": filled_keys,
        "data_generated": False, "pipelines_deployed": False,
    }
    _atomic_json(root / ".runs/setup/platform-settings.json", receipt)
    return receipt
