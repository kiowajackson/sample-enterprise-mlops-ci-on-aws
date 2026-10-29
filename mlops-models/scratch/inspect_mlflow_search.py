# Paste this entire file into one notebook cell after interrupting the stuck cell.
# Fill in the tracking server ARN below; no earlier notebook variables are needed.
# This only searches experiments. It does not create runs or write artifacts.
TRACKING_URI = ""  # arn:aws:sagemaker:ca-central-1:123456789012:mlflow-tracking-server/your-server

import importlib.metadata
import os
import re
import sys
import threading
import time
import traceback

TRACKING_URI = TRACKING_URI.strip()
if not re.fullmatch(
    r"arn:[a-z0-9-]+:sagemaker:[a-z0-9-]+:\d{12}:mlflow-tracking-server/[^\s/:]+",
    TRACKING_URI,
):
    raise ValueError(
        "Set TRACKING_URI to the tracking server ARN, for example "
        "arn:aws:sagemaker:ca-central-1:123456789012:mlflow-tracking-server/your-server"
    )

# These settings remain active in this kernel. They limit HTTP waits/retries,
# not the whole operation (for example, AWS credential lookup can wait longer).
os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "10"
os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "0"

print("Importing MLflow and the SageMaker plugin...", flush=True)
import mlflow
import sagemaker_mlflow

print("MLflow:", mlflow.__version__, "|", mlflow.__file__, flush=True)
print("SageMaker plugin:", importlib.metadata.version("sagemaker-mlflow"),
      "|", sagemaker_mlflow.__file__, flush=True)

# The AWS plugin reads this GLOBAL URI for authentication and request headers.
# Passing the ARN only to MlflowClient(tracking_uri=...) is insufficient.
mlflow.set_tracking_uri(TRACKING_URI)
print("Tracking URI:", mlflow.get_tracking_uri(), flush=True)

# If the call waits, print the notebook thread's stack into this cell's output.
# The timer reports where Python is waiting; it does not stop the request.
search_thread = threading.get_ident()
def print_waiting_stack():
    print("\nSearch still waiting after 20 seconds. Python stack:", flush=True)
    traceback.print_stack(sys._current_frames()[search_thread], file=sys.stdout)
    sys.stdout.flush()

stack_timer = threading.Timer(20, print_waiting_stack)
stack_timer.daemon = True
started = time.monotonic()
stack_timer.start()
try:
    print("START mlflow.search_experiments(max_results=1)", flush=True)
    experiments = mlflow.search_experiments(max_results=1)
    print(f"OK: {len(experiments)} experiment(s), {time.monotonic() - started:.1f}s", flush=True)
    for experiment in experiments:
        print(experiment.experiment_id, experiment.name)
finally:
    stack_timer.cancel()
