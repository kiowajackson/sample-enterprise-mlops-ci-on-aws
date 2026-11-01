"""Validate deployment identities before building images or contacting AWS."""

import hashlib
from base64 import b32encode
from pathlib import Path
import re

PACKAGE = Path(__file__).resolve().parents[1] / "src/mlops_models"


def validate_namespace(namespace):
    """Take a namespace and return it after checking the shared resource-name budget."""
    if not re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", namespace) or len(namespace) > 20:
        raise ValueError("namespace must use lowercase letters, digits and internal hyphens, at most 20 characters.")
    return namespace


def validate_model_id(model_id):
    """Take a model folder name and return its portable Python, ECR and SageMaker identity."""
    if not re.fullmatch(r"[a-z][a-z0-9]{0,23}", model_id):
        raise ValueError("Model folders must match [a-z][a-z0-9]* and contain at most 24 characters.")
    return model_id


def retired_model_ids(config):
    """Read explicit retired identities whose repositories and registry groups remain managed."""
    retired = config.get("retired_models", [])
    if not isinstance(retired, list) or not all(isinstance(name, str) for name in retired):
        raise ValueError("retired_models must be a list of model IDs.")
    if len(retired) != len(set(retired)):
        raise ValueError("retired_models contains duplicate IDs.")
    return [validate_model_id(name) for name in retired]


def resource_tags(config):
    """Return descriptive ownership tags without allowing the deployment's authorization tag to change."""
    tags = config.get("tags", {})
    if not isinstance(tags, dict) or len(tags) > 40:
        raise ValueError("tags must be a mapping with at most 40 entries.")
    for key, value in tags.items():
        if (not isinstance(key, str) or not key or len(key) > 128 or key.lower().startswith("aws:")
                or key == "Deployment" or not isinstance(value, str) or len(value) > 256):
            raise ValueError("Use string tag keys/values; Deployment and aws: keys are reserved.")
    return dict(tags)


def discover_models(root=None, *, retired=()):
    """Return active model folders while retaining explicit retired identities for resource ownership."""
    folders = sorted((root or PACKAGE / "models").iterdir())
    models = [folder for folder in folders if (folder / "parameters.json").is_file()]
    for folder in models:
        validate_model_id(folder.name)
    identities = {folder.name for folder in models} | {validate_model_id(name) for name in retired}
    prefixes = [hashlib.sha256(name.encode()).hexdigest()[:8] for name in identities]
    if len(prefixes) != len(set(prefixes)):
        raise ValueError("Two model IDs share a shortened job identity; choose a different model ID.")
    if not models and not retired:
        raise ValueError("No parameters.json files were found in the models directory.")
    return [folder for folder in models if folder.name not in retired]


def job_prefix(namespace, model_id, kind):
    """Take deployment and model identities and return a short, stable job-name prefix."""
    validate_namespace(namespace)
    validate_model_id(model_id)
    if kind not in {"training", "testing", "inference"}:
        raise ValueError("Choose training, testing or inference.")
    identity = hashlib.sha256(model_id.encode()).hexdigest()[:8]
    suffix = {"training": "tr", "testing": "te", "inference": "in"}[kind]
    return f"{namespace}--{identity}-{suffix}"


def validate_job_names(definition):
    """Take a rendered definition and reject job prefixes without space for SageMaker's suffix."""
    for step in definition["Steps"]:
        arguments = step.get("Arguments", {})
        for field in ("TrainingJobName", "ProcessingJobName", "TransformJobName", "ModelName"):
            name = arguments.get(field)
            if isinstance(name, str) and (
                not re.fullmatch(r"[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?", name)
                or len(name) > 39
            ):
                raise ValueError(f"{step['Name']}.{field} must reserve 24 of SageMaker's 63 characters: {name}")
        if step["Type"] == "Condition":
            for branch in ("IfSteps", "ElseSteps"):
                validate_job_names({"Steps": arguments.get(branch, [])})


def tuning_prefix(namespace, model_id):
    """Return a six-character tuning prefix derived from this deployment and model."""
    validate_namespace(namespace)
    validate_model_id(model_id)
    return b32encode(hashlib.sha256(f"{namespace}/{model_id}".encode()).digest()).decode()[:6].lower()
