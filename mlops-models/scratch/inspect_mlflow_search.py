# Paste this file into a new cell in notebook 02, after interrupting the stuck cell.
# - Reuse the notebook's tracking_uri and installed Python environment.
# - Search for one experiment without creating experiments, runs or artifacts.
# - Print the worker's Python stack every 20 seconds while it is waiting.
# - Stop the worker after 60 seconds and keep its output visible in the notebook.

import os
import queue
import subprocess
import sys
import threading
import time


_MLFLOW_SEARCH_WORKER = r"""
import faulthandler
import importlib.metadata
import json
import os
import sys
import time
import traceback

request = json.loads(sys.stdin.read())
started = time.monotonic()

def progress(message):
    print(f"[search +{time.monotonic() - started:.1f}s] {message}", flush=True)

faulthandler.enable(file=sys.stderr)
faulthandler.dump_traceback_later(
    request["stack_after"], repeat=True, file=sys.stderr
)
try:
    progress(f"Python: {sys.executable}")
    progress("Import MLflow and SageMaker plugin")
    import mlflow
    import sagemaker_mlflow
    progress(f"MLflow {mlflow.__version__}: {mlflow.__file__}")
    try:
        plugin_version = importlib.metadata.version("sagemaker-mlflow")
    except importlib.metadata.PackageNotFoundError:
        plugin_version = "unknown"
    progress(f"SageMaker plugin {plugin_version}: {sagemaker_mlflow.__file__}")
    for name in (
        "SAGEMAKER_MLFLOW_ASSUME_ROLE_ARN",
        "SAGEMAKER_MLFLOW_CUSTOM_ENDPOINT",
        "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
    ):
        progress(f"{name}: {'set' if os.environ.get(name) else 'not set'}")
    progress("Initialize tracking client")
    client = mlflow.MlflowClient(tracking_uri=request["uri"])
    progress("START mlflow.search_experiments(max_results=1)")
    result = client.search_experiments(max_results=1)
    progress(f"OK search returned {len(result)} experiment(s)")
except Exception:
    traceback.print_exc(file=sys.stdout)
    sys.exit(1)
finally:
    faulthandler.cancel_dump_traceback_later()
"""


def diagnose_mlflow_search(uri, *, timeout_seconds=60, stack_after=20):
    import json
    import math

    if not isinstance(uri, str) or not uri.startswith("arn:"):
        raise ValueError("Use the existing tracking_uri ARN from notebook 02.")
    if not (math.isfinite(timeout_seconds) and math.isfinite(stack_after)
            and 0 < stack_after < timeout_seconds):
        raise ValueError("Require 0 < stack_after < timeout_seconds.")

    environment = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "MLFLOW_HTTP_REQUEST_TIMEOUT": "10",
        "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "0",
    }
    print(
        f"Read-only experiment search. Stack every {stack_after:g}s; "
        f"worker limit {timeout_seconds:g}s. No run/artifact writes.",
        flush=True,
    )
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", _MLFLOW_SEARCH_WORKER],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, env=environment,
    )
    messages = queue.Queue()
    output = []
    timed_out = False

    def read_output():
        try:
            for line in process.stdout:
                messages.put(line)
        finally:
            messages.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    started = time.monotonic()
    next_status = started + 5
    output_closed = False
    try:
        process.stdin.write(json.dumps({"uri": uri, "stack_after": stack_after}))
        process.stdin.close()
        while not output_closed or process.poll() is None:
            now = time.monotonic()
            if now - started >= timeout_seconds:
                timed_out = True
                print("Worker limit reached; stopping the search process.", flush=True)
                break
            try:
                line = messages.get(timeout=min(0.1, timeout_seconds - (now - started)))
            except queue.Empty:
                if now >= next_status:
                    print(f"[notebook +{now - started:.0f}s] Worker still running.", flush=True)
                    next_status = now + 5
                continue
            if line is None:
                output_closed = True
            else:
                output.append(line)
                print(line, end="", flush=True)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        reader.join(timeout=1)
        while not messages.empty():
            line = messages.get_nowait()
            if line is not None:
                output.append(line)
                print(line, end="", flush=True)
        process.stdout.close()
        if not process.stdin.closed:
            process.stdin.close()

    print(
        f"Finished in {time.monotonic() - started:.1f}s; "
        f"timed_out={timed_out}; exit_code={process.returncode}.",
        flush=True,
    )
    return {
        "timed_out": timed_out,
        "exit_code": process.returncode,
        "output": "".join(output),
    }


if __name__ == "__main__":
    mlflow_search_diagnostic = diagnose_mlflow_search(tracking_uri)
