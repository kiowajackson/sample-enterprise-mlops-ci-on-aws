"""Resolve application-owned S3 prefixes and read or publish small metadata objects."""

import hashlib
import json
import os
from pathlib import Path
import re
from urllib.parse import urlparse


def data_prefix(settings):
    """Return a nonempty application root ending in '/', rejecting bucket-root and traversal paths."""
    prefix = (settings.get("prefix") or settings.get("namespace", "")).strip("/")
    if not prefix or any(part in {"", ".", ".."} for part in prefix.split("/")) or not re.fullmatch(r"[A-Za-z0-9_./-]+", prefix):
        raise ValueError("Set a nonempty storage prefix with ordinary path components.")
    return prefix + "/"


def data_key(settings, suffix):
    """Return an object key contained by this application's configured storage prefix."""
    if suffix.startswith("/") or any(part in {".", ".."} for part in suffix.split("/")):
        raise ValueError("Use a relative S3 key without parent traversal.")
    return data_prefix(settings) + suffix


def data_uri(settings, suffix=""):
    """Return an S3 URI under this application's bucket and prefix."""
    return f"s3://{settings['bucket']}/{data_key(settings, suffix)}"


def s3_location(uri):
    """Take an S3 URI and return its bucket and object key."""
    location = urlparse(uri)
    if location.scheme != "s3" or not location.netloc:
        raise ValueError(f"Expected an S3 output location, received {uri}.")
    return location.netloc, location.path.lstrip("/")


def read_model_metadata(uri, *, s3=None):
    """Read one explicit S3 JSON object; malformed or inaccessible input is an error."""
    location = urlparse(uri)
    bucket, key = s3_location(uri)
    if not key or key.endswith("/") or location.query or location.fragment:
        raise ValueError("ModelMetadataUri must name one S3 JSON file, without query or fragment.")
    if s3 is None:
        import boto3
        s3 = boto3.client("s3")
    with s3.get_object(Bucket=bucket, Key=key)["Body"] as stream:
        content = stream.read()
    metadata = json.loads(content)
    if not isinstance(metadata, dict):
        raise ValueError("ModelMetadataUri must contain a JSON object.")
    print(json.dumps({"event": "model_metadata_loaded", "uri": uri,
                      "sha256": hashlib.sha256(content).hexdigest()}), flush=True)
    return metadata, content


def training_metadata(parameter_path):
    """Use the execution URI, or retain source-file behavior for older callers."""
    uri = os.getenv("MLOPS_MODEL_METADATA_URI")
    if uri and uri != "none":
        return read_model_metadata(uri)[0]
    path = Path(parameter_path).with_name("model_metadata.json")
    # New pipelines pass "none" for an initial fit (blank is also accepted). Older/local callers
    # without this environment variable retain their existing source-file input.
    if uri is None and path.exists():
        metadata = json.loads(path.read_text())
        if not isinstance(metadata, dict):
            raise ValueError(f"Training metadata must contain a JSON object: {path}")
        print(f"Training metadata loaded from: {path}", flush=True)
        return metadata
    print("No training metadata supplied; starting without saved settings.", flush=True)
    return {}


def read_existing_json(s3, bucket, key):
    """Read a retained JSON object, returning None only when its key does not exist."""
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
    except s3.exceptions.ClientError as error:
        if error.response["Error"]["Code"] in {"NoSuchKey", "404"}:
            return None
        raise
    with response["Body"] as stream:
        return json.load(stream)


def upload_changed_file(s3, bucket, key, path, content_type):
    """Upload a metadata or log file only when its SHA-256 differs from the retained object."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    sha256 = digest.hexdigest()
    try:
        existing = s3.head_object(Bucket=bucket, Key=key)
    except s3.exceptions.ClientError as error:
        if error.response["Error"]["Code"] not in {"404", "NoSuchKey", "NotFound"}:
            raise
    else:
        if existing.get("Metadata", {}).get("sha256") == sha256:
            return
    s3.upload_file(str(path), bucket, key, ExtraArgs={
        "ContentType": content_type, "Metadata": {"sha256": sha256},
    })
