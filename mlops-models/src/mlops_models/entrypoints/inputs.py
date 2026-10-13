"""Retain the raw CSV inputs in the job's normal output directory."""

import hashlib
import json
import os
from pathlib import Path
import shutil
from urllib.parse import urlparse

import pandas as pd


def snapshot_input(source, output_dir):
    """Take CSV input and a job output directory; retain the consumed files and return their local path."""
    # Copy the input before any transformation so later source changes cannot alter this run.
    destination = Path(output_dir) / "raw"
    # Diagnose every configured Processing output before the first real write can fail.
    read_paths = [] if isinstance(source, pd.DataFrame) or str(source).startswith("s3://") else [source]
    write_paths = [Path(output_dir).parent, output_dir, destination]
    settings = None
    job_config = Path("/opt/ml/config/processingjobconfig.json")
    if job_config.is_file():
        try:
            settings = json.loads(job_config.read_text())
            write_paths.extend(item["S3Output"]["LocalPath"]
                               for item in settings.get("ProcessingOutputConfig", {}).get("Outputs", [])
                               if item.get("S3Output", {}).get("LocalPath"))
        except (OSError, ValueError) as error:
            print(json.dumps({"event": "path_check_config_error", "error": str(error)}), flush=True)
    print_path_checks(read_paths, [*write_paths, "/tmp", Path.home()])
    if settings is not None:
        print_s3_write_check(settings, destination)
    destination.mkdir(parents=True, exist_ok=True)
    # The container user must be able to write into the managed output directory.
    permissions = destination.stat()
    print(json.dumps({
        "event": "input_snapshot_destination", "path": str(destination),
        "effective_uid": os.geteuid(), "effective_gid": os.getegid(), "groups": os.getgroups(),
        "owner_uid": permissions.st_uid, "owner_gid": permissions.st_gid,
        "mode": oct(permissions.st_mode & 0o7777),
        "write_access": os.access(destination, os.W_OK, effective_ids=True),
    }), flush=True)
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


def print_path_checks(read_paths=(), write_paths=()):
    """Print mount/ownership evidence and removable write probes; never change permissions."""
    import json
    import os
    from pathlib import Path
    import shutil
    import tempfile

    process = {"event": "path_check_process", "revision": "paths-v3", "uid": os.geteuid(),
               "gid": os.getegid(), "groups": os.getgroups(), "cwd": os.getcwd()}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in {"Umask", "CapEff", "NoNewPrivs"}:
                process[key] = value.strip()
    except OSError as error:
        process["process_details"] = str(error)
    print(json.dumps(process), flush=True)
    mounts = []
    try:
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            fields = line.split()
            separator = fields.index("-")
            mounts.append({"path": fields[4].replace("\\040", " "), "options": fields[5],
                           "filesystem": fields[separator + 1],
                           "super_options": [flag for flag in fields[separator + 3].split(",") if flag in {"ro", "rw"}]})
    except (OSError, ValueError, IndexError) as error:
        print(json.dumps({"event": "path_check_mount_error", "error": str(error)}), flush=True)

    reads = {Path(path).absolute() for path in read_paths}
    writes = {Path(path).absolute() for path in write_paths}
    paths = reads | writes
    for path in list(paths):
        paths.update(path.parents)
    for path in sorted(paths):
        record = {"event": "path_check", "path": str(path), "requested_write": path in writes}
        try:
            record["exists"] = path.exists()
            parent = path
            while not parent.exists() and parent != parent.parent:
                parent = parent.parent
            info = parent.stat()
            record.update(inspected_path=str(parent), owner_uid=info.st_uid, owner_gid=info.st_gid,
                          mode=oct(info.st_mode & 0o7777), symlink=path.is_symlink(),
                          read_access=os.access(path, os.R_OK, effective_ids=True),
                          write_access=os.access(path, os.W_OK, effective_ids=True),
                          search_access=os.access(path, os.X_OK, effective_ids=True))
            candidates = [mount for mount in mounts if parent.resolve().is_relative_to(mount["path"])]
            if candidates:
                record["mount"] = max(candidates, key=lambda mount: len(mount["path"]))
            if parent.is_dir():
                record["free_bytes"] = shutil.disk_usage(parent).free
            if path in reads:
                # A Processing input is usually a directory: verify one real CSV without logging its data.
                sample = next(path.rglob("*.csv"), None) if path.is_dir() else path
                if sample is not None and sample.is_file():
                    record["sample_input_file"] = str(sample)
                    with sample.open("rb") as stream:
                        stream.read(1)
                    record["file_read"] = "ok"
                else:
                    record["file_read"] = "no input CSV found"
            if path in writes:
                record["probe_directory"] = str(parent)
                with tempfile.NamedTemporaryFile(prefix=".mlops-path-check-", dir=parent) as probe:
                    probe.write(b"path check\n")
                    probe.flush()
                with tempfile.TemporaryDirectory(prefix=".mlops-path-check-", dir=parent) as work:
                    child = Path(work) / "check.txt"
                    child.write_text("path check\n")
                    assert child.read_text() == "path check\n"
                record["write_probe"] = "ok" if path.exists() else "parent-only; target does not exist"
        except OSError as error:
            record.update(error=str(error), errno=error.errno)
        print(json.dumps(record), flush=True)



def print_s3_write_check(settings, raw_directory):
    """Probe direct S3 write/read from this job; retain one tiny JSON as evidence."""
    outputs = settings.get("ProcessingOutputConfig", {})
    target = next((item["S3Output"] for item in outputs.get("Outputs", [])
                   if item.get("S3Output", {}).get("LocalPath") == str(raw_directory)), None)
    result = {"event": "s3_write_probe", "revision": "s3-v1", "operation": "setup"}
    if target is None or not isinstance(target.get("S3Uri"), str):
        print(json.dumps({**result, "status": "skipped", "reason": "No resolved S3 output for the raw directory."}), flush=True)
        return
    try:
        import tempfile
        import uuid
        import boto3
        from botocore.config import Config

        nonce = uuid.uuid4().hex
        uri = target["S3Uri"].rstrip("/") + "/_diagnostics/s3-write-" + nonce + ".json"
        location = urlparse(uri)
        if location.scheme != "s3" or not location.netloc:
            raise ValueError("The configured raw output must be an S3 URI.")
        result.update(uri=uri, operation="scratch-file", status="started")
        print(json.dumps(result), flush=True)
        payload = (json.dumps({"probe": "manual-s3-output", "nonce": nonce, "uid": os.geteuid()}) + "\n").encode()
        with tempfile.NamedTemporaryFile(prefix="mlops-s3-probe-", suffix=".json", dir="/tmp") as scratch:
            scratch.write(payload)
            scratch.flush()
            scratch.seek(0)
            client = boto3.client("s3", config=Config(connect_timeout=5, read_timeout=10,
                                                     retries={"max_attempts": 0}))
            encryption = {}
            if outputs.get("KmsKeyId"):
                encryption = {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": outputs["KmsKeyId"]}
            result.update(operation="PutObject", status="started", region=client.meta.region_name)
            print(json.dumps(result), flush=True)
            written = client.put_object(Bucket=location.netloc, Key=location.path.lstrip("/"),
                                        Body=scratch, ContentType="application/json", **encryption)
            result.update(status="succeeded", write_confirmed=True,
                          encryption=written.get("ServerSideEncryption", "not returned"))
            print(json.dumps(result), flush=True)
            result.update(operation="GetObject", status="started")
            print(json.dumps(result), flush=True)
            request = {"Bucket": location.netloc, "Key": location.path.lstrip("/")}
            with client.get_object(**request)["Body"] as stream:
                if stream.read(len(payload) + 1) != payload:
                    raise ValueError("S3 readback differs from the probe bytes.")
            result.update(status="succeeded", read_confirmed=True, bytes_match=True)
            print(json.dumps(result), flush=True)
            print("S3_WRITE_READ_CONFIRMED", uri, flush=True)
    except Exception as error:
        # This diagnostic must not replace Prepare's normal result or hide its original write error.
        result.update(status="failed", error_type=type(error).__name__, error=str(error))
        print(json.dumps(result), flush=True)


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
