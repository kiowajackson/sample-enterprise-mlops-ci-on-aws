"""Retain the raw CSV inputs in the job's normal output directory."""

import hashlib
import json
from pathlib import Path
import shutil
from urllib.parse import urlparse

import pandas as pd


def snapshot_input(source, output_dir):
    """Take CSV input and a job output directory; retain the consumed files and return their local path."""
    # Copy the input before any transformation so later source changes cannot alter this run.
    destination = Path(output_dir) / "raw"
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError(f"Use an empty raw output directory: {destination}")
    if isinstance(source, pd.DataFrame):
        source.to_csv(destination / "input.csv", index=False)
    elif str(source).startswith("s3://"):
        download_csv_input(source, destination)
    else:
        source = Path(source)
        files = sorted(source.rglob("*.csv")) if source.is_dir() else [source]
        if source.is_dir() and (source / "provenance.json").is_file():
            files.append(source / "provenance.json")
        for path in files:
            relative = path.relative_to(source) if source.is_dir() else Path(path.name)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)

    if not any(destination.rglob("*.csv")):
        raise FileNotFoundError(f"No CSV input was found at {source}.")
    verify_provenance(destination)
    return destination


def download_csv_input(uri, destination):
    """Take an S3 object or slash-terminated prefix and download its CSV files into a local directory."""
    import boto3

    location = urlparse(uri)
    client = boto3.client("s3")
    prefix = location.path.lstrip("/")
    if prefix.endswith("/"):
        pages = client.get_paginator("list_objects_v2").paginate(Bucket=location.netloc, Prefix=prefix)
        keys = sorted(item["Key"] for page in pages for item in page.get("Contents", [])
                      if item["Key"].endswith(".csv") or item["Key"] == prefix + "provenance.json")
    else:
        keys = [prefix]
    for key in keys:
        relative = key[len(prefix):] if prefix.endswith("/") else Path(key).name
        target = (destination / relative).resolve()
        if not target.is_relative_to(destination.resolve()):
            raise ValueError(f"The S3 key cannot be used as a local input path: {key}")
        target.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(location.netloc, key, str(target))


def verify_provenance(directory):
    """Verify an optional Features provenance record against the retained CSV; return that record or None."""
    path = Path(directory) / "provenance.json"
    if not path.is_file():
        return None
    provenance = json.loads(path.read_text())
    dataset = provenance["dataset"]
    filename = dataset["file"]
    if filename != "data.csv" or sorted(p.relative_to(directory).as_posix() for p in Path(directory).rglob("*.csv")) != [filename]:
        raise ValueError("A Features extraction must contain exactly its declared data.csv and provenance.json.")
    with (Path(directory) / filename).open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != dataset["sha256"]:
        raise ValueError("Input CSV bytes differ from the retained Feature Store extraction; do not train on them.")
    return provenance
