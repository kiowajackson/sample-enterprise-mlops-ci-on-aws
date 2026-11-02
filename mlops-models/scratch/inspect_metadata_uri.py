# November 2, 2026.
# Copy this block into a notebook cell. Use the actual FeatureSelection
# Processing job name, not the pipeline name or execution ID.
REGION = "ca-central-1"
PROCESSING_JOB_NAME = "PASTE_FEATURE_SELECTION_JOB_NAME"

import boto3
from botocore.config import Config

sm = boto3.client(
    "sagemaker",
    region_name=REGION,
    config=Config(connect_timeout=5, read_timeout=5, retries={"total_max_attempts": 1}),
)
job = sm.describe_processing_job(ProcessingJobName=PROCESSING_JOB_NAME)
environment = job.get("Environment", {})
value = environment.get("MLOPS_MODEL_METADATA_URI")

print("Job:", job["ProcessingJobName"])
print("Step:", environment.get("MLOPS_STEP_NAME", "(not recorded)"))
print("MLOPS_MODEL_METADATA_URI:", repr(value))
print("Type:", type(value).__name__)
if value is None:
    print("Missing from this job's environment. Check the deployed FeatureSelection definition.")
elif value.strip().lower() in {"", "none"}:
    print("The job received empty/none text. Check this execution's ModelMetadataUri parameter.")
else:
    print("The job received a value. Compare the lookup inside the running code below.")

# OPTIONAL: copy only these lines into the failing Python function,
# immediately before its existing os.getenv lookup. This must run inside
# that job; running it in your notebook would inspect the notebook instead.
#
# import os
# value = os.getenv("MLOPS_MODEL_METADATA_URI")
# print("Inside job: MLOPS_MODEL_METADATA_URI =", repr(value),
#       "| type =", type(value).__name__, flush=True)
#
# Expected connection:
# execution parameter ModelMetadataUri -> job environment
# MLOPS_MODEL_METADATA_URI -> os.getenv("MLOPS_MODEL_METADATA_URI")
#
# Earlier pipelines/source.py omitted FeatureSelection from the list of steps
# receiving this environment variable. Dev includes the fix since b51fcb6.
# Having the updated file does not update an already-deployed pipeline.
