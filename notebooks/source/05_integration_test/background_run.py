"""Run the integration sequence independently of the notebook kernel.

The Studio space must stay running. A file lock prevents duplicate workers and
each launch copies its Python source so later repository edits cannot affect it.
"""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from uuid import uuid4


def _write(path, value):
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def background_status(directory):
    """Read saved status without polling AWS or changing the execution."""
    directory = Path(directory)
    worker = _read(directory / "worker.json")
    active = False
    if directory.is_dir():
        with (directory / "worker.lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                active = True
    status = "Running" if active else worker.get("status", "Not started")
    if not active and status in {"Starting", "Running"}:
        status = "Interrupted"
    return {"worker_status": status, "worker": worker,
            "result": _read(directory / "results.json"),
            "log": str(directory / "worker.log"),
            "state": str(directory / "state.json")}


def launch_background(suite):
    """Launch or resume a prepared suite; return immediately with its receipt."""
    if not suite.prepared:
        raise ValueError("Prepare and inspect the integration matrix before launching.")
    directory = suite.directory
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "worker.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return background_status(directory)
        saved = _read(suite.path)
        if saved.get("status") == "Passed":
            return {"worker_status": "Passed", "result": _read(directory / "results.json"),
                    "log": str(directory / "worker.log"), "state": str(suite.path)}
        if saved.get("settings") != suite.settings:
            raise ValueError("Saved inputs changed. Prepare again before launching.")
        run_id = uuid4().hex
        snapshot = directory / "runner-source" / run_id
        snapshot.mkdir(parents=True)
        helpers = suite.root / "notebooks/source/05_integration_test"
        for name in ("integration_checks.py", "background_run.py"):
            shutil.copy2(helpers / name, snapshot / name)
        shutil.copytree(suite.root / "mlops-models/src/mlops_models",
                        snapshot / "mlops_models",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        request = {
            "root": str(suite.root.resolve()), "publication": suite.publication,
            "options": {
                "run_label": suite.settings["run_label"],
                "model_ids": suite.settings["model_ids"],
                "csv_inputs": suite.settings["csv_inputs"],
                "pipelines": suite.targets, "namespace": suite.namespace,
                "region": suite.region,
                "model_code_source": suite.settings["model_code_source"],
                "poll_seconds": suite.poll_seconds,
                "timeout_minutes": suite.timeout_minutes,
            },
            "directory": str(directory.resolve()), "run_id": run_id,
        }
        request_path = snapshot / "request.json"
        _write(request_path, request)
        _write(directory / "worker.json", {
            "status": "Starting", "run_id": run_id,
            "source": str(snapshot), "started_at": datetime.now(timezone.utc).isoformat(),
        })
        with (directory / "worker.log").open("ab", buffering=0) as log:
            try:
                process = subprocess.Popen(
                    [sys.executable, "-u", str(snapshot / "background_run.py"),
                     str(request_path), str(lock.fileno())],
                    cwd=snapshot, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True,
                    pass_fds=(lock.fileno(),),
                )
            except Exception as error:
                _write(directory / "worker.json", {
                    "status": "Failed", "run_id": run_id, "error": str(error)})
                raise
        # Closing the parent's descriptor leaves the child's lock held. Do not
        # explicitly unlock: that would also release the inherited worker lock.
    receipt = background_status(directory)
    receipt["pid"] = process.pid
    return receipt


def prepare_background(suite):
    """Avoid rewriting saved execution state while a detached worker is active."""
    suite.directory.mkdir(parents=True, exist_ok=True)
    with (suite.directory / "worker.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return None
        return suite.prepare()


def _worker(request_path, lock_fd):
    # The inherited open descriptor owns the lock until this process exits.
    with os.fdopen(lock_fd, "a+"):
        request = _read(Path(request_path))
        directory = Path(request["directory"])
        record = _read(directory / "worker.json")
        record.update(status="Running", pid=os.getpid())
        _write(directory / "worker.json", record)
        try:
            from integration_checks import IntegrationSuite
            suite = IntegrationSuite(request["root"], request["publication"],
                                     **request["options"])
            suite.prepare()
            result = suite.run()
            record.update(status=result["status"])
        except BaseException as error:
            record.update(status="Failed", error=f"{type(error).__name__}: {error}")
            traceback.print_exc()
            raise
        finally:
            record["ended_at"] = datetime.now(timezone.utc).isoformat()
            _write(directory / "worker.json", record)


if __name__ == "__main__":
    _worker(sys.argv[1], int(sys.argv[2]))
