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
