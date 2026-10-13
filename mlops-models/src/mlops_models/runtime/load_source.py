"""Load supplied source and run one SageMaker Processing or ordinary Training step.

The publisher copies this file beside mlops_models in S3. SageMaker downloads it
as a source input. Evaluation/inference use the code retained in model.tar.gz.
For application-managed Processing outputs, this script supplies a writable
scratch directory and uploads completed outputs to their configured S3 locations.
Ordinary Training keeps SageMaker's output handling. HPO uses the image entrypoint.
"""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
from urllib.parse import urlparse


def upload_processing_outputs(config, output_dir, scratch):
    """Upload declared Processing outputs; any missing output or upload error fails the job."""
    import boto3

    uploads = []
    for item in config["Outputs"]:
        output = item["S3Output"]
        relative = Path(output["LocalPath"]).relative_to(output_dir)
        directory = scratch / relative
        files = sorted(path for path in directory.rglob("*") if path.is_file())
        if not files:
            raise FileNotFoundError(f"Processing output {item['OutputName']} has no files: {directory}")
        location = urlparse(output["S3Uri"])
        if location.scheme != "s3" or not location.netloc:
            raise ValueError(f"Invalid Processing output URI: {output['S3Uri']}")
        for path in files:
            if path.is_symlink() or not path.resolve().is_relative_to(scratch.resolve()):
                raise ValueError(f"Processing output must be a file inside its scratch directory: {path}")
            key = "/".join(filter(None, [location.path.strip("/"), path.relative_to(directory).as_posix()]))
            uploads.append((path, location.netloc, key))

    encryption = ({"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": config["KmsKeyId"]}
                  if config.get("KmsKeyId") else {})
    s3 = boto3.client("s3")
    for path, bucket, key in uploads:
        print(json.dumps({"event": "processing_output_upload", "source": str(path),
                          "destination": f"s3://{bucket}/{key}"}), flush=True)
        s3.upload_file(str(path), bucket, key, ExtraArgs=encryption)
    print(json.dumps({"event": "processing_outputs_uploaded", "files": len(uploads)}), flush=True)


def print_training_path_checks(output_dir, model_dir):
    """Temporarily diagnose Training write locations without changing their permissions."""
    import pwd
    import shutil

    process = {"event": "training_path_checks", "revision": "training-paths-v1",
               "uid": os.getuid(), "effective_uid": os.geteuid(),
               "gid": os.getgid(), "effective_gid": os.getegid(), "groups": os.getgroups(),
               "cwd": os.getcwd(), "output_dir": str(output_dir), "model_dir": str(model_dir),
               "environment": {key: os.environ[key] for key in (
                   "HOME", "TMPDIR", "TMP", "TEMP", "SM_OUTPUT_DIR", "SM_OUTPUT_DATA_DIR", "SM_MODEL_DIR"
               ) if key in os.environ}}
    home = os.environ.get("HOME")
    try:
        user = pwd.getpwuid(os.geteuid())
        process.update(user=user.pw_name, account_home=user.pw_dir)
        home = user.pw_dir
    except KeyError:
        process["account_home"] = "No passwd entry for the effective UID"
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in {"Umask", "CapEff", "NoNewPrivs"}:
                process[key] = value.strip()
    except OSError as error:
        process["process_details_error"] = str(error)
    mounts = []
    try:
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            fields = line.split()
            separator = fields.index("-")
            mounts.append({"path": fields[4].replace("\\040", " "), "options": fields[5],
                           "filesystem": fields[separator + 1]})
    except (OSError, ValueError, IndexError) as error:
        process["mount_details_error"] = str(error)
    print(json.dumps(process), flush=True)

    paths = [output_dir, Path(output_dir) / "checkpoint", Path(output_dir).parent,
             model_dir, Path(model_dir).parent, "/tmp", "/var/tmp", home,
             *(os.environ.get(key) for key in ("HOME", "TMPDIR", "TMP", "TEMP"))]
    for path in dict.fromkeys(Path(p).absolute() for p in paths if p):
        record = {"event": "training_path_check", "path": str(path)}
        try:
            record["exists"] = path.exists()
            parent = path
            while not parent.exists() and parent != parent.parent:
                parent = parent.parent
            info = parent.stat()
            record.update(inspected_path=str(parent), owner_uid=info.st_uid, owner_gid=info.st_gid,
                          mode=oct(info.st_mode & 0o7777),
                          write_access=os.access(parent, os.W_OK, effective_ids=True),
                          probe_scope="requested path" if path.exists() else "nearest existing parent",
                          free_bytes=shutil.disk_usage(parent).free)
            candidates = [mount for mount in mounts if parent.resolve().is_relative_to(mount["path"])]
            if candidates:
                record["mount"] = max(candidates, key=lambda mount: len(mount["path"]))
            # Test both operations independently; a denied path must not prevent later checks.
            for kind in ("file", "directory"):
                try:
                    if kind == "file":
                        with tempfile.NamedTemporaryFile(prefix=".mlops-training-check-", dir=parent) as probe:
                            probe.write(b"training path check\n")
                            probe.flush()
                            probe.seek(0)
                            matches = probe.read() == b"training path check\n"
                    else:
                        with tempfile.TemporaryDirectory(prefix=".mlops-training-check-", dir=parent) as work:
                            probe = Path(work) / "check.txt"
                            probe.write_bytes(b"training path check\n")
                            matches = probe.read_bytes() == b"training path check\n"
                    record[kind + "_probe"] = {"status": "ok", "readback_matches": matches}
                except OSError as error:
                    record[kind + "_probe"] = {"status": "failed", "errno": error.errno,
                                               "error_type": type(error).__name__, "error": str(error)}
        except OSError as error:
            record.update(error=str(error), errno=error.errno, error_type=type(error).__name__)
        print(json.dumps(record), flush=True)
    print(json.dumps({"event": "training_path_checks_complete", "revision": "training-paths-v1"}), flush=True)


def main():
    """Load source, dispatch one step, then publish its application-managed outputs."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("step")
    parser.add_argument("--model-dir")
    parser.add_argument("--output")
    selected, _ = parser.parse_known_args()
    source = Path(__file__).resolve().parent
    original_directory, original_arguments = Path.cwd(), sys.argv[:]
    original_path = sys.path[:]

    try:
        # Train reads mounted source directly; only Processing needs loader scratch space.
        workspace = (nullcontext() if selected.step == "train" else
                     tempfile.TemporaryDirectory(prefix="mlops-step-", dir="/tmp"))
        with workspace as work:
            checkpoint = None
            if selected.step in {"evaluate", "preprocess", "score", "postprocess"}:
                archive_path = selected.model_dir or "/opt/ml/processing/model/model.tar.gz"
                checkpoint = Path(work) / "model"
                with tarfile.open(archive_path, "r:gz") as archive:
                    archive.extractall(checkpoint, filter="data")
                source = checkpoint / "code"
            if not (source / "mlops_models").is_dir():
                raise ValueError(f"No mlops_models source directory found at {source}.")

            # Choose the model's imports before importing any model package.
            os.chdir(source)
            sys.path.insert(0, str(source))
            identity = source / "source.json"
            print(json.dumps({"event": "source_loaded", "step": selected.step,
                              "source": str(source),
                              "identity": json.loads(identity.read_text()) if identity.exists() else {}}),
                  flush=True)
            if selected.step == "train":
                from mlops_models.runtime.job_paths import job_paths
                _, output_path, model_path = job_paths("train")
                print_training_path_checks(selected.output or output_path, selected.model_dir or model_path)

            # main() is the stable command interface, including for older checkpoints.
            from mlops_models.entrypoints.__main__ import main as dispatch_command
            if checkpoint is not None:
                sys.argv.extend(["--model-dir", str(checkpoint)])

            output_config = None
            if os.getenv("MLOPS_PROCESSING_OUTPUT_UPLOAD") == "application" and selected.step != "train":
                job = json.loads(Path("/opt/ml/config/processingjobconfig.json").read_text())
                output_config = job["ProcessingOutputConfig"]
                from mlops_models.runtime.job_paths import job_paths
                output_dir = Path(selected.output or job_paths(selected.step)[1])
                scratch = Path(work) / "outputs"
                scratch.mkdir()
                # Only output paths change; SageMaker's input mounts stay intact.
                sys.argv.extend(["--output", str(scratch)])
                print(json.dumps({"event": "processing_output_directory", "step": selected.step,
                                  "configured": str(output_dir), "writable": str(scratch)}), flush=True)
            dispatch_command()
            if output_config is not None:
                upload_processing_outputs(output_config, output_dir, scratch)
    finally:
        os.chdir(original_directory)
        sys.argv[:] = original_arguments
        sys.path[:] = original_path


if __name__ == "__main__":
    main()
