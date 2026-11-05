# Inspect the stopped FeatureSelection job

Paste these two cells into your notebook in the same AWS account as the execution.
This only reads pipeline and processing-job details; it does not start, stop or change jobs.

## 1. Required input

Paste the **pipeline execution ARN** from SageMaker Studio, not the pipeline ARN or a console URL.

```python
EXECUTION_ARN = ""
```

## 2. Inspect the job

```python
import json
import boto3

assert EXECUTION_ARN.startswith("arn:") and "/execution/" in EXECUTION_ARN, (
    "Set EXECUTION_ARN to the SageMaker pipeline execution ARN."
)
sm = boto3.client("sagemaker", region_name=EXECUTION_ARN.split(":")[3])
execution = sm.describe_pipeline_execution(PipelineExecutionArn=EXECUTION_ARN)
steps = [
    step
    for page in sm.get_paginator("list_pipeline_execution_steps").paginate(
        PipelineExecutionArn=EXECUTION_ARN
    )
    for step in page["PipelineExecutionSteps"]
    if step["StepName"] == "FeatureSelection"
]
assert len(steps) == 1, f"Expected one FeatureSelection step; found {len(steps)}."
step = steps[0]
job_arn = step.get("Metadata", {}).get("ProcessingJob", {}).get("Arn")
assert job_arn, (
    "This step has no processing-job ARN. Step details: " + json.dumps(step, default=str)
)
job = sm.describe_processing_job(ProcessingJobName=job_arn.rsplit("/", 1)[-1])
start, end = job.get("ProcessingStartTime"), job.get("ProcessingEndTime")
report = {
    "execution_arn": EXECUTION_ARN,
    "pipeline_status": execution["PipelineExecutionStatus"],
    "pipeline_failure_reason": execution.get("FailureReason"),
    "step_status": step["StepStatus"],
    "step_failure_reason": step.get("FailureReason"),
    "cache_hit": step.get("CacheHitResult"),
    "job_arn": job_arn,
    "job_status": job["ProcessingJobStatus"],
    "job_failure_reason": job.get("FailureReason"),
    "exit_message": job.get("ExitMessage"),
    "max_runtime_seconds": job.get("StoppingCondition", {}).get("MaxRuntimeInSeconds"),
    "created": job.get("CreationTime"),
    "processing_started": start,
    "processing_ended": end,
    "processing_elapsed_minutes": round((end - start).total_seconds() / 60, 2)
    if start and end else None,
    "compute": job.get("ProcessingResources", {}).get("ClusterConfig"),
}
print(json.dumps(report, indent=2, default=str))
```

Copy the printed JSON back for diagnosis. The key fields are `job_status`,
`job_failure_reason`, `exit_message`, `max_runtime_seconds`, and the processing timestamps.
A 900-second limit in the local template is a possible cause; the actual job details
are needed to confirm why this execution stopped.
