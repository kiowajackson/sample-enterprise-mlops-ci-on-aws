# Paste this entire file into one notebook cell.
# - Set the failed Training job's exact name below.
# - Read every available log page for every stream belonging to that job.
# - Save the complete log and a separate file containing the directory diagnostics.
# - Display download links; this does not start a job or change the pipeline.

TRAINING_JOB_NAME = ""  # Required: the Training job name from the failed Train step.
AWS_REGION = "ca-central-1"

from datetime import datetime, timezone
from pathlib import Path
import time

import boto3
from IPython.display import FileLink, display

job_name = TRAINING_JOB_NAME.strip()
assert job_name, "Set TRAINING_JOB_NAME to the failed Training job's exact name."
logs = boto3.client("logs", region_name=AWS_REGION)
log_group = "/aws/sagemaker/TrainingJobs"
end_time = int(time.time() * 1000)
streams = [
    stream
    for page in logs.get_paginator("describe_log_streams").paginate(
        logGroupName=log_group, logStreamNamePrefix=job_name + "/"
    )
    for stream in page["logStreams"]
]
assert streams, f"No log streams found for {job_name} in {AWS_REGION}."

full_log = Path(f"{job_name}-full.log")
path_log = Path(f"{job_name}-paths.log")
event_count = 0
path_messages = []
with full_log.open("w", encoding="utf-8") as full, path_log.open("w", encoding="utf-8") as paths:
    for stream in streams:
        heading = f"\n=== {stream['logStreamName']} ===\n"
        full.write(heading)
        paths.write(heading)
        request = dict(
            logGroupName=log_group, logStreamName=stream["logStreamName"],
            startFromHead=True, endTime=end_time,
        )
        while True:
            response = logs.get_log_events(**request)
            for event in response["events"]:
                timestamp = datetime.fromtimestamp(event["timestamp"] / 1000, timezone.utc).isoformat()
                message = event["message"]
                line = f"{timestamp} {message}\n"
                full.write(line)
                event_count += 1
                if any(marker in message for marker in (
                    "TRAIN ENTRY", "training_output_directories",
                    "received_model_dir", "loader_model_dir", "Wrong package destination",
                )) or message.lstrip().startswith(("model_dir ", "package_dir ", "output_directory ")):
                    paths.write(line)
                    path_messages.append(line)
            # An empty page can still have a next page.
            if response["nextForwardToken"] == request.get("nextToken"):
                break
            request["nextToken"] = response["nextForwardToken"]

print(f"Saved {event_count} events from {len(streams)} stream(s): {full_log.resolve()}")
print("Directory diagnostics (first 12 messages; all matches are in the paths file):")
print("".join(path_messages[:12]) if path_messages else "No directory diagnostics were found in these logs.")
display(FileLink(str(full_log)), FileLink(str(path_log)))
