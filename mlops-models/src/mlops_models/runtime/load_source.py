"""Load supplied source and work around non-root output-directory permissions.

The current non-root image cannot write to SageMaker's standard output
directories. When application-managed output uploads are enabled, this loader:
- Leaves SageMaker's input downloads and local input paths unchanged.
- Runs the step with writable temporary output directories.
- Uploads completed outputs to their configured S3 locations after the step succeeds.

The publisher ships this file with the source; SageMaker downloads it as an input.
Model artifacts supply the saved model and settings. MLOPS_MODEL_CODE_SOURCE
chooses artifact code or the separately supplied pipeline code for evaluation/
prediction. This workaround covers Processing and ordinary Training; HPO uses
the image entrypoint.

AWS references for the native I/O contract (the permission failure is specific to
the current image/job setup):
https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ProcessingS3Input.html
https://docs.aws.amazon.com/sagemaker/latest/dg/model-train-storage-env-var-summary.html
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


def upload_training_outputs(work):
    """Upload the completed model (including code) and auxiliary reports from Training scratch."""
    import boto3

    model = work / "model"
    if not model.is_dir() or not any(model.iterdir()):
        raise FileNotFoundError(f"Training produced no model files: {model}")
    encryption = ({"ServerSideEncryption": "aws:kms",
                   "SSEKMSKeyId": os.environ["MLOPS_TRAINING_KMS_KEY"]}
                  if os.getenv("MLOPS_TRAINING_KMS_KEY") else {})
    s3 = boto3.client("s3")
    for directory, variable in ((work / "output", "MLOPS_TRAINING_REPORTS_URI"),
                                (model, "MLOPS_TRAINING_MODEL_URI")):
        uri = os.environ[variable]
        location = urlparse(uri)
        if location.scheme != "s3" or not location.netloc or not location.path.strip("/"):
            raise ValueError(f"Invalid Training output URI: {uri}")
        archive_path = work / (directory.name + ".tar.gz")
        with tarfile.open(archive_path, "w:gz") as archive:
            for path in sorted(directory.iterdir()):
                archive.add(path, arcname=path.name)
        print(json.dumps({"event": "training_output_upload", "source": str(archive_path),
                          "destination": uri, "bytes": archive_path.stat().st_size}), flush=True)
        s3.upload_file(str(archive_path), location.netloc, location.path.lstrip("/"),
                       ExtraArgs=encryption)
    print(json.dumps({"event": "training_outputs_uploaded",
                      "model_uri": os.environ["MLOPS_TRAINING_MODEL_URI"],
                      "reports_uri": os.environ["MLOPS_TRAINING_REPORTS_URI"]}), flush=True)


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
    managed_training = selected.step == "train" and bool(
        os.getenv("MLOPS_TRAINING_MODEL_URI") or os.getenv("MLOPS_TRAINING_OUTPUT_UPLOAD") == "application")
    cached_step = bool(os.getenv("MLOPS_STEP_REVISION") or os.getenv("MLOPS_PREPARATION_REVISION"))
    provenance_variables = ("PIPELINE_EXECUTION_ID", "SOURCE_COMMIT", "SOURCE_DIRTY")
    original_provenance = {key: os.environ.get(key) for key in provenance_variables}
    training_variables = ("TMPDIR", "TMP", "TEMP", "SM_MODEL_DIR", "SM_OUTPUT_DATA_DIR", "SM_OUTPUT_DIR",
                          "MLOPS_TRAINING_MODEL_URI", "MLOPS_TRAINING_REPORTS_URI", "MLOPS_TRAINING_KMS_KEY")
    original_environment = {key: os.environ.get(key) for key in training_variables}
    original_tempdir = tempfile.tempdir

    try:
        # This non-root image can use /var/tmp even when Training's /tmp is read-only.
        workspace = (nullcontext() if selected.step == "train" and not managed_training else
                     tempfile.TemporaryDirectory(prefix="mlops-step-",
                                                 dir="/var/tmp" if managed_training else "/tmp"))
        with workspace as work:
            checkpoint = None
            if selected.step in {"get-inference-data", "evaluate", "preprocess", "score", "predict", "postprocess"}:
                archive_path = selected.model_dir or "/opt/ml/processing/model/model.tar.gz"
                checkpoint = Path(work) / "model"
                with tarfile.open(archive_path, "r:gz") as archive:
                    archive.extractall(checkpoint, filter="data")
                code_source = os.getenv("MLOPS_MODEL_CODE_SOURCE", "artifact")
                if code_source not in {"pipeline", "artifact"}:
                    raise ValueError("MLOPS_MODEL_CODE_SOURCE must be pipeline or artifact.")
                if code_source == "artifact":
                    source = checkpoint / "code"
            if not (source / "mlops_models").is_dir():
                raise ValueError(f"No mlops_models source directory found at {source}.")

            if os.getenv("MLOPS_FEATURE_STORE_CONFIG_URI") and not (source / "mlops_models/entrypoints/get_data.py").is_file():
                raise ValueError("This model/source predates the Feature Store reader; train and approve a model with the updated source.")
            # Choose the model's imports before importing any model package.
            os.chdir(source)
            sys.path.insert(0, str(source))
            identity = source / "source.json"
            source_metadata = json.loads(identity.read_text()) if identity.exists() else {}
            processing_job = None
            output_uri = os.getenv("MLOPS_TRAINING_MODEL_URI", "")
            if os.getenv("MLOPS_TRAINING_OUTPUT_UPLOAD") == "application" and selected.step == "train":
                import boto3

                # SageMaker injects TRAINING_JOB_NAME. Read this job's output
                # config instead of including per-execution paths in cache keys.
                job = boto3.client("sagemaker", region_name=os.environ["TRAINING_JOB_ARN"].split(":")[3]).describe_training_job(
                    TrainingJobName=os.environ["TRAINING_JOB_NAME"])
                config = job["OutputDataConfig"]
                output_uri = config["S3OutputPath"].rstrip("/")
                if not output_uri.endswith("/model"):
                    raise ValueError("Training output must end in /runs/<execution-id>/model.")
                run_uri = output_uri.removesuffix("/model")
                os.environ.update(
                    MLOPS_TRAINING_MODEL_URI=output_uri + "/application/model.tar.gz",
                    MLOPS_TRAINING_REPORTS_URI=run_uri + "/reports/training/output.tar.gz",
                    MLOPS_TRAINING_KMS_KEY=config.get("KmsKeyId", ""),
                )
            if cached_step:
                # Cached Processing output references and Training's recorded
                # OutputDataConfig identify the producing execution.
                if selected.step != "train":
                    processing_job = json.loads(Path("/opt/ml/config/processingjobconfig.json").read_text())
                    output_uri = processing_job["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
                elif not managed_training:
                    output_uri = ""
                if output_uri:
                    _, separator, suffix = output_uri.rpartition("/runs/")
                    if not separator or not suffix.split("/")[0]:
                        raise ValueError("Cached steps need an execution-scoped /runs/<id>/ output.")
                    os.environ["PIPELINE_EXECUTION_ID"] = suffix.split("/")[0]
                os.environ.update(SOURCE_COMMIT=source_metadata.get("commit", "uncommitted"),
                                  SOURCE_DIRTY=str(source_metadata.get("dirty", True)).lower())
            print(json.dumps({"event": "source_loaded", "step": selected.step,
                              "source": str(source),
                              "model_code_source": os.getenv("MLOPS_MODEL_CODE_SOURCE", "artifact") if checkpoint else "pipeline",
                              "identity": source_metadata}),
                  flush=True)
            if managed_training:
                work = Path(work)
                for name in ("tmp", "output", "model"):
                    (work / name).mkdir()
                os.environ.update(TMPDIR=str(work / "tmp"), TMP=str(work / "tmp"), TEMP=str(work / "tmp"),
                                  SM_MODEL_DIR=str(work / "model"), SM_OUTPUT_DATA_DIR=str(work / "output"),
                                  SM_OUTPUT_DIR=str(work / "output"))
                tempfile.tempdir = str(work / "tmp")
                sys.argv.extend(["--output", str(work / "output"), "--model-dir", str(work / "model")])
                print(json.dumps({"event": "training_output_directories", "revision": "training-outputs-v1",
                                  "uid": os.geteuid(), "output": str(work / "output"),
                                  "model": str(work / "model"), "temporary": tempfile.gettempdir(),
                                  "model_uri": os.environ["MLOPS_TRAINING_MODEL_URI"]}), flush=True)

            # main() is the stable command interface, including for older checkpoints.
            from mlops_models.entrypoints.__main__ import main as dispatch_command
            if checkpoint is not None:
                sys.argv.extend(["--model-dir", str(checkpoint)])

            output_config = None
            if os.getenv("MLOPS_PROCESSING_OUTPUT_UPLOAD") == "application" and selected.step != "train":
                job = processing_job or json.loads(Path("/opt/ml/config/processingjobconfig.json").read_text())
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
            if managed_training:
                upload_training_outputs(work)
            if output_config is not None:
                upload_processing_outputs(output_config, output_dir, scratch)
    finally:
        os.chdir(original_directory)
        sys.argv[:] = original_arguments
        sys.path[:] = original_path
        if cached_step:
            for key, value in original_provenance.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        if managed_training:
            tempfile.tempdir = original_tempdir
            for key, value in original_environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    main()
