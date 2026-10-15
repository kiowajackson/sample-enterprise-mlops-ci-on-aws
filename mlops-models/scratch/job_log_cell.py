# Paste this entire file into one notebook cell.
# - Choose Training or Processing and enter the exact job name from the pipeline step.
# - Download every available log page for every stream belonging to that job.
# - Save a complete text file and print the full log when PRINT_FULL_LOG is True.
# - Open the download link to read the complete file if the notebook display truncates it.

JOB_TYPE = "Training"  # "Training" or "Processing" (Evaluate is a Processing job).
JOB_NAME = ""  # Required: exact job name, not the pipeline execution ID.
AWS_REGION = None  # Uses the notebook region; set explicitly if the job ran elsewhere.

PRINT_FULL_LOG = True

from datetime import datetime, timezone
from pathlib import Path
import time

import boto3
from IPython.display import FileLink, display

if JOB_TYPE not in {"Training", "Processing"}:
    raise ValueError('Set JOB_TYPE to "Training" or "Processing".')
job_name = JOB_NAME.strip()
if not job_name:
    raise ValueError("Set JOB_NAME to the exact job name from the pipeline step.")
logs = boto3.client("logs", region_name=AWS_REGION)
log_group = f"/aws/sagemaker/{JOB_TYPE}Jobs"
end_time = int(time.time() * 1000)
streams = [
    stream
    for page in logs.get_paginator("describe_log_streams").paginate(
        logGroupName=log_group, logStreamNamePrefix=job_name + "/"
    )
    for stream in page["logStreams"]
]
if not streams:
    raise ValueError(f"No log streams found for {job_name} in {log_group}, {logs.meta.region_name}.")

log_file = Path(f"{JOB_TYPE.lower()}-{job_name}-full.log")
event_count = 0
with log_file.open("w", encoding="utf-8") as output:
    for stream in streams:
        output.write(f"\n=== {stream['logStreamName']} ===\n")
        request = dict(
            logGroupName=log_group, logStreamName=stream["logStreamName"],
            startFromHead=True, endTime=end_time,
        )
        while True:
            response = logs.get_log_events(**request)
            for event in response["events"]:
                timestamp = datetime.fromtimestamp(event["timestamp"] / 1000, timezone.utc).isoformat()
                output.write(f"{timestamp} {event['message']}\n")
                event_count += 1
            # Empty pages may still have a next page.
            if response["nextForwardToken"] == request.get("nextToken"):
                break
            request["nextToken"] = response["nextForwardToken"]

if PRINT_FULL_LOG:
    print(log_file.read_text(encoding="utf-8"))
print(f"Log snapshot through {datetime.fromtimestamp(end_time / 1000, timezone.utc).isoformat()}")
print(f"Saved {event_count} events from {len(streams)} stream(s): {log_file.resolve()}")
display(FileLink(str(log_file)))
