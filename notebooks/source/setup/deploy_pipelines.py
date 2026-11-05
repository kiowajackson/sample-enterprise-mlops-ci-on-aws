"""Config -> input data -> publication, shared by the notebook and command line."""

from contextlib import contextmanager, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import argparse
import json
import traceback
import sys
from time import perf_counter

from platform_settings import SECRET_NAME, _atomic_json, load_workspace_settings
from prepare_data import prepare_data
from publish_pipelines import publish_pipelines


CSV_KEYS = ("csv_input", "test_csv_input", "inference_csv_input")


@contextmanager
def deployment_lock(root):
    """One deployment writer per checkout; the OS releases this lock on exit."""
    import fcntl

    path = Path(root) / ".runs/setup/deploy.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another deployment is active in this checkout.") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def data_action(config, options, mode):
    """Decide from configuration, not from whether reading a file happened to fail."""
    if mode not in {"auto", "reuse", "generate"}:
        raise ValueError("data_mode must be auto, reuse or generate.")
    training = options.get("data_source", "csv_direct")
    inference = options.get("inference_data_source", "csv_direct")
    if training not in {"csv_direct", "feature_store"} or inference not in {"csv_direct", "feature_store"}:
        raise ValueError("Choose csv_direct or feature_store for each data source.")
    required = []
    # The existing publisher validates a reference CSV whenever either route uses CSV.
    if "csv_direct" in {training, inference}:
        required.append("csv_input")
    if inference == "csv_direct":
        required.extend(("test_csv_input", "inference_csv_input"))
    feature_store = "feature_store" in {training, inference}
    if feature_store:
        required.append("feature_store_config_uri")
    values = {key: (config.get(key) or "").strip() for key in required}
    supplied = {key for key, value in values.items() if value}
    if mode == "generate":
        if feature_store or config.get("feature_store_config_uri"):
            raise ValueError("Automatic generation supports CSV mode; use notebook 01 for Feature Store preparation.")
        return "generate", required
    if supplied == set(required):
        return "reuse", required
    if supplied:
        missing = sorted(set(required) - supplied)
        raise ValueError(f"Dataset configuration is incomplete. Supply {', '.join(missing)}; existing inputs were preserved.")
    if mode == "reuse" or feature_store or config.get("feature_store_config_uri"):
        raise ValueError("No complete input dataset is configured; no substitute data was generated.")
    return "generate", required


def verify_inputs(root, session, config, keys):
    """Check exact objects/files. Publication subsequently verifies content and hashes."""
    verified = {}
    for key in keys:
        value = (config.get(key) or "").strip()
        if value.startswith("s3://"):
            parts = value[5:].split("/", 1)
            if len(parts) != 2 or not all(parts):
                raise ValueError(f"{key} must name an S3 object, not a bucket.")
            try:
                size = session.client("s3").head_object(Bucket=parts[0], Key=parts[1])["ContentLength"]
            except Exception as error:
                raise RuntimeError(f"Cannot access configured {key}: {value}. No replacement data was generated.") from error
        else:
            path = Path(value).expanduser()
            path = (path if path.is_absolute() else Path(root) / path).resolve()
            if not path.is_file():
                raise ValueError(f"Configured {key} file is missing: {path}. No replacement data was generated.")
            # Opening catches a permissions failure before publication or generation.
            with path.open("rb") as stream:
                stream.read(1)
            size = path.stat().st_size
        if size <= 0:
            raise ValueError(f"Configured {key} is empty: {value}. No replacement data was generated.")
        verified[key] = {"location": value, "bytes": size}
    return verified


def deploy_pipelines(root, session, *, secret_name=SECRET_NAME, refresh_shared=False,
                     data_mode="auto", preparation_options=None, publication_options=None):
    """Fill config gaps, prepare only when needed, then upsert verified pipelines.

    Return the report explicitly. Full logs and success/failure receipts live in a
    unique .runs/setup/deploy-* directory. No training or model approval is started.
    """
    started = perf_counter()
    output = sys.stdout

    def progress(message):
        print(f"[{perf_counter() - started:6.1f}s] {message}", file=output, flush=True)

    root = Path(root).resolve()
    options = dict(publication_options or {})
    with deployment_lock(root):
        label = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:8]
        run = root / ".runs/setup" / ("deploy-" + label)
        run.mkdir()
        report = {"status": "running", "log_path": str(run / "deployment.log"),
                  "report_path": str(run / "report.json")}
        (run / "deployment.log").touch()
        try:
            progress("[1/3] Checking configuration and shared settings...")
            setup = load_workspace_settings(
                root, session, secret_name=secret_name, refresh_shared=refresh_shared,
            )
            report["config"] = setup
            print(f"Config: {setup['config_state']} -> "
                  f"{'updated missing/selected settings' if setup['config_changed'] else 'preserved'}", flush=True)
            config_path = root / "notebooks/config.json"
            config = json.loads(config_path.read_text())
            action, keys = data_action(config, options, data_mode)
            report["data_action"] = action
            with (run / "deployment.log").open("w") as log:
                if action == "generate":
                    progress("[2/3] No dataset selected; generating data..." if data_mode == "auto"
                             else "[2/3] Explicitly regenerating data...")
                    with redirect_stdout(log):
                        report["preparation"] = prepare_data(root, options=preparation_options, progress=progress)
                    config = json.loads(config_path.read_text())
                else:
                    progress("[2/3] Reusing configured data...")
                report["verified_inputs"] = verify_inputs(root, session, config, keys)
                progress("[3/3] Validating and publishing pipelines...")
                with redirect_stdout(log):
                    publication = publish_pipelines(root, session, options=options, setup_receipt=setup, progress=progress)
            report.update({
                "status": "succeeded", "namespace": publication["namespace"],
                "pipelines": publication["pipelines"],
                "publication_path": publication["publication_path"],
                "inputs": {key: publication.get(key + "_uri", "") for key in CSV_KEYS},
                "image": publication["image"],
                "preview": publication.get("data_preview", []),
                "sha256": publication.get("data_sha256", {}),
                "elapsed_seconds": round(perf_counter() - started, 3),
            })
            _atomic_json(run / "report.json", report)
        except Exception as error:
            report.update(status="failed", error_type=type(error).__name__, error=str(error),
                          elapsed_seconds=round(perf_counter() - started, 3))
            with (run / "deployment.log").open("a") as log:
                traceback.print_exc(file=log)
            _atomic_json(run / "report.json", report)
            print("Deployment stopped. Report:", report["report_path"], flush=True)
            print("Detailed log:", report["log_path"], flush=True)
            raise
    progress("Deployment complete.")
    return report


def main():
    parser = argparse.ArgumentParser(description="Prepare missing inputs and deploy Wealth ML pipelines.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--region", default=None)
    parser.add_argument("--secret-name", default=SECRET_NAME)
    parser.add_argument("--data-mode", choices=("auto", "reuse", "generate"), default="auto")
    parser.add_argument("--refresh-shared", action="store_true")
    args = parser.parse_args()
    import boto3
    result = deploy_pipelines(
        args.root, boto3.Session(region_name=args.region), secret_name=args.secret_name,
        refresh_shared=args.refresh_shared, data_mode=args.data_mode,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
