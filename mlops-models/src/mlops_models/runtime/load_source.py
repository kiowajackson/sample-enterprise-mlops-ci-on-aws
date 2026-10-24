"""Wrap a step's entrypoint: prepare input/output paths, run it, and upload outputs.

The current non-root image cannot write to SageMaker's standard output
directories. Its output workaround uses a temporary job folder and direct S3
uploads. If native output writes become available, that workaround can be
removed; source/model loading and step dispatch are separate responsibilities.

AWS references for the native I/O contract (the permission failure is specific to
the current image/job setup):
https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ProcessingS3Input.html
https://docs.aws.amazon.com/sagemaker/latest/dg/model-train-storage-env-var-summary.html
"""

import argparse
from contextlib import nullcontext
import csv
import json
import os
import runpy
from pathlib import Path
import sys
import tarfile
import tempfile
from urllib.parse import urlparse


def prepare_inference_inputs(model_dir, output_dir, model_id):
    """Prepare the saved selection before calling the preprocessing entrypoint."""
    model_dir = Path(model_dir)
    parameters = json.loads((model_dir / "parameters.json").read_text())
    if parameters.get("model_id", model_id) != model_id:
        raise ValueError(f"Artifact belongs to {parameters['model_id']}, not selected model {model_id}.")
    metadata = json.loads((model_dir / "model_metadata.json").read_text())
    mapping = metadata["feature_mapping"]
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("model_metadata.json needs a nonempty feature_mapping.")
    selection_file = Path(output_dir) / "selection" / "selected_features.csv"
    selection_file.parent.mkdir(parents=True, exist_ok=True)
    with selection_file.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["features", "description"])
        writer.writerows(mapping.items())
    return selection_file


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
    # The same record is readable without downloading/unpacking the trained model.
    location = urlparse(os.environ["MLOPS_TRAINING_MODEL_URI"])
    s3.upload_file(str(model / "_execution.json"), location.netloc,
                   location.path.lstrip("/").rsplit("/", 1)[0] + "/_execution.json",
                   ExtraArgs=encryption)
    print(json.dumps({"event": "training_outputs_uploaded",
                      "model_uri": os.environ["MLOPS_TRAINING_MODEL_URI"],
                      "reports_uri": os.environ["MLOPS_TRAINING_REPORTS_URI"]}), flush=True)


def write_execution_record(directory, record):
    """Write a small record of which execution, code and settings produced these files.

    Called after model code returns and before its outputs are uploaded. The
    resulting _execution.json is framework bookkeeping, separate from model
    metadata. It travels with the original output when a later run uses the cache.
    """
    directory = Path(directory)
    # Require real output files first: this record must not hide a missing result.
    if directory.is_dir() and any(path.is_file() and path.name != "_execution.json"
                                  for path in directory.rglob("*")):
        (directory / "_execution.json").write_text(json.dumps(record, indent=2) + "\n")


def main():
    """Configure input/output paths around the selected step's entrypoint.

    1. Read the step request and remember the current process settings.
    2. Locate model/source files and configure writable output directories.
    3. Call the step's entrypoint, which runs the model code.
    4. Record what produced the outputs and upload them to the configured S3 paths.
    5. Restore this process's directory, arguments and environment.
    """
    # 1. Read the job command and retain process state for the final cleanup.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("step")
    parser.add_argument("--model", default=os.getenv("MODEL_ID"))
    parser.add_argument("--model-dir")
    parser.add_argument("--output")
    selected, _ = parser.parse_known_args()
    source = Path(__file__).resolve().parent
    recording_source = source / "mlops_models/runtime/execution.py"
    logging_source = source / "mlops_models/pipelines/tracking.py"
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
    output_config = None

    try:
        # 2. Prepare local input/code and writable output locations.
        # workspace manages this job's temporary folder; it is not a repo or Studio workspace.
        # Prefer the image's owned scratch directory. Managed containers can
        # remove world-write permissions from /tmp and /var/tmp for non-root users.
        scratch_root = os.getenv("TMPDIR") or ("/var/tmp" if managed_training else "/tmp")
        workspace = (nullcontext() if selected.step == "train" and not managed_training else
                     tempfile.TemporaryDirectory(prefix="mlops-step-", dir=scratch_root))
        with workspace as work:
            # Inference/evaluation unpack the supplied model. Choose its archived
            # code or the separately uploaded pipeline code before importing modules.
            checkpoint = None
            if selected.step in {"get-inference-data", "evaluate", "preprocess", "score", "predict", "postprocess", "drift"}:
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

            # Read the configured S3 destinations and identify the producing run.
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
            # Report the code actually loaded, including uncached Postprocess
            # from an older registered release after the pipeline is republished.
            os.environ.update(SOURCE_COMMIT=source_metadata.get("commit", "uncommitted"),
                              SOURCE_DIRTY=str(source_metadata.get("dirty", True)).lower())
            print(json.dumps({"event": "source_loaded", "step": selected.step,
                              "source": str(source),
                              "model_code_source": os.getenv("MLOPS_MODEL_CODE_SOURCE", "artifact") if checkpoint else "pipeline",
                              "identity": source_metadata}),
                  flush=True)

            # Redirect Training's model/report writes into writable scratch folders.
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

            # Processing has its own declared output folders; redirect those writes too.
            output_config = None
            if os.getenv("MLOPS_PROCESSING_OUTPUT_UPLOAD") == "application" and selected.step != "train":
                job = processing_job or json.loads(Path("/opt/ml/config/processingjobconfig.json").read_text())
                output_config = job["ProcessingOutputConfig"]
                output_uri = output_config["Outputs"][0]["S3Output"]["S3Uri"]
                if "/runs/" in output_uri:
                    os.environ["PIPELINE_EXECUTION_ID"] = output_uri.rsplit("/runs/", 1)[1].split("/")[0]
                from mlops_models.runtime.job_paths import job_paths
                output_dir = Path(selected.output or job_paths(selected.step)[1])
                scratch = Path(work) / "outputs"
                scratch.mkdir()
                # Only output paths change; SageMaker's input mounts stay intact.
                sys.argv.extend(["--output", str(scratch)])
                print(json.dumps({"event": "processing_output_directory", "step": selected.step,
                                  "configured": str(output_dir), "writable": str(scratch)}), flush=True)

            # Inference preprocessing also needs the saved model's selected-feature CSV.
            if selected.step == "preprocess":
                from mlops_models.runtime.job_paths import job_paths
                preparation_output = scratch if output_config is not None else Path(
                    selected.output or job_paths(selected.step)[1])
                selection_file = prepare_inference_inputs(checkpoint, preparation_output, selected.model)
                sys.argv.extend(["--selection", str(selection_file)])

            # The published source mount supplies the standalone logger. For
            # model-bound inference that mount is the registered source release.
            logger = runpy.run_path(str(logging_source)) if os.getenv("MLOPS_TRACKING_URI") else None
            connection = logger["job_tracking"](selected.model, output_uri) if logger else nullcontext(None)
            with connection as tracking:
                # 3. RUN THE MODEL STEP through entrypoints.__main__.main().
                # Its selected entrypoint delegates to the use-case/model_base function.
                dispatch_command()

                # 4. Describe which execution/code produced the files just written.
                # write_execution_record() saves this dictionary as _execution.json.
                execution_record = {
                    "step": os.getenv("MLOPS_STEP_NAME", selected.step), "source": source_metadata,
                    "producer_execution_id": os.getenv("PIPELINE_EXECUTION_ID", ""),
                    "step_revision": os.getenv("MLOPS_STEP_REVISION", os.getenv("MLOPS_PREPARATION_REVISION", "")),
                    "model_code_source": os.getenv("MLOPS_MODEL_CODE_SOURCE", "artifact") if checkpoint else "pipeline",
                }
                # Preserve source settings and saved-model settings separately; inference
                # can use newer source with an older registered model.
                if selected.model:
                    parameter_file = source / "mlops_models/models" / selected.model / "parameters.json"
                    if parameter_file.is_file():
                        execution_record["source_parameters"] = json.loads(parameter_file.read_text())
                if checkpoint and (Path(checkpoint) / "parameters.json").is_file():
                    execution_record["checkpoint_parameters"] = json.loads(
                        (Path(checkpoint) / "parameters.json").read_text())
                if checkpoint and (Path(checkpoint) / "_execution.json").is_file():
                    execution_record["model_training"] = json.loads(
                        (Path(checkpoint) / "_execution.json").read_text())
                if output_config is not None:
                    execution_record["outputs"] = {item["OutputName"]: item["S3Output"]["S3Uri"]
                                                   for item in output_config["Outputs"]}

                if tracking is not None:
                    execution_record["mlflow"] = {key: tracking[key] for key in
                        ("run_id", "tracking_uri", "experiment_id", "execution_arn")}
                # Save that record beside each result, then upload the result files.
                if managed_training:
                    execution_record.update(
                        record_version=1,
                        producer_execution_arn=(os.environ["MLOPS_PIPELINE_ARN"] + "/execution/" +
                                                os.environ["PIPELINE_EXECUTION_ID"]
                                                if os.getenv("MLOPS_PIPELINE_ARN") else ""),
                        model_artifact=os.environ["MLOPS_TRAINING_MODEL_URI"],
                        artifacts=[], datasets=[], registry=[],
                        reports={path.name: json.loads(path.read_text())
                                 for path in (work / "output").glob("*.json")
                                 if path.name != "_execution.json"})
                    write_execution_record(work / "output", execution_record)
                    write_execution_record(work / "model", execution_record)
                    upload_training_outputs(work)
                if output_config is not None:
                    recording = runpy.run_path(str(recording_source))
                    execution_record = recording["processing_record"](
                        job, execution_record, output_dir, scratch)
                    for item in output_config["Outputs"]:
                        relative = Path(item["S3Output"]["LocalPath"]).relative_to(output_dir)
                        write_execution_record(scratch / relative, execution_record)
                    upload_processing_outputs(output_config, output_dir, scratch)
                # Log from the job that produced these files; the caller ends the shared run.
                if logger:
                    from mlops_models.runtime.job_paths import job_paths
                    directories = ([work / "output", work / "model"] if managed_training else
                                   [scratch / Path(item["S3Output"]["LocalPath"]).relative_to(output_dir)
                                    for item in output_config["Outputs"]] if output_config is not None else
                                   [Path(selected.output or job_paths(selected.step)[1])])
                    logger["log_job_outputs"](tracking, directories, execution_record)
    finally:
        # 5. Restore process state even if the model step or an upload failed.
        os.chdir(original_directory)
        sys.argv[:] = original_arguments
        sys.path[:] = original_path
        if cached_step or output_config is not None:
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
