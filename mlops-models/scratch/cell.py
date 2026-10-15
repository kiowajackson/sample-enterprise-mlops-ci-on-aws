"""Paste this entire file into a notebook cell to inspect step caching.

Set the execution ARN from Studio. Optionally add an earlier execution ARN to
compare Processing/Training cache-key fields. Reads SageMaker metadata; starts no jobs.
"""

import json
from pathlib import Path

import boto3

execution_arn = ""
previous_execution_arn = ""  # Optional comparison; use the same pipeline.

assert ":pipeline/" in execution_arn and "/execution/" in execution_arn, (
    "Paste the full pipeline execution ARN from Studio into execution_arn."
)
client = boto3.client("sagemaker", region_name=execution_arn.split(":")[3])


def read_steps(arn):
    steps = {}
    for page in client.get_paginator("list_pipeline_execution_steps").paginate(PipelineExecutionArn=arn):
        steps.update({step["StepName"]: step for step in page["PipelineExecutionSteps"]})
    return steps


def job_keys(step):
    kinds = {
        "ProcessingJob": ("describe_processing_job", "ProcessingJobName",
                          ("AppSpecification", "Environment", "ProcessingInputs")),
        "TrainingJob": ("describe_training_job", "TrainingJobName",
                        ("AlgorithmSpecification", "Environment", "HyperParameters", "InputDataConfig",
                         "CheckpointConfig", "DebugHookConfig", "DebugRuleConfigurations")),
    }
    for kind, (operation, argument, keys) in kinds.items():
        arn = step.get("Metadata", {}).get(kind, {}).get("Arn")
        if arn:
            job = getattr(client, operation)(**{argument: arn.rsplit("/", 1)[-1]})
            return {key: job.get(key, {}) for key in keys}
    return None


def changed_fields(before, after, path=""):
    if isinstance(before, dict) and isinstance(after, dict):
        return [name for key in sorted(before.keys() | after.keys())
                for name in changed_fields(before.get(key), after.get(key), f"{path}.{key}".lstrip("."))]
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        return [name for index, (left, right) in enumerate(zip(before, after))
                for name in changed_fields(left, right, f"{path}[{index}]")]
    return [] if before == after else [path]


steps = read_steps(execution_arn)
print("Execution:", execution_arn)
for name, step in steps.items():
    source = step.get("CacheHitResult", {}).get("SourcePipelineExecutionArn")
    jobs = [value["Arn"] for value in step.get("Metadata", {}).values()
            if isinstance(value, dict) and value.get("Arn")]
    result = "CACHE HIT" if source else "NO CACHE HIT"
    print(f"{name}: {step['StepStatus']} | {result}")
    if source:
        print("  Reused output from:", source)
    for arn in jobs:
        print("  Job/resource:", arn)
    if step.get("FailureReason"):
        print("  Failure:", step["FailureReason"])

comparison = {}
if previous_execution_arn:
    assert execution_arn.rsplit("/execution/", 1)[0] == previous_execution_arn.rsplit("/execution/", 1)[0], (
        "Cache reuse is limited to the same pipeline; choose two executions of that pipeline."
    )
    previous = read_steps(previous_execution_arn)
    print("\nCompare compute cache keys with:", previous_execution_arn)
    for name in ("GetData", "FeatureSelection", "Preprocess", "Train", "Evaluate", "Score", "Postprocess"):
        if name not in steps and name not in previous:
            continue
        before = job_keys(previous.get(name, {}))
        after = job_keys(steps.get(name, {}))
        if before is None or after is None:
            comparison[name] = "No compute job recorded in one or both executions."
        else:
            fields = changed_fields(before, after)
            comparison[name] = fields or "The compared job cache-key fields match."
        print(name, ":", comparison[name])

report = Path.cwd() / f"cache-report-{execution_arn.rsplit('/', 1)[-1]}.json"
report.write_text(json.dumps({
    "execution_arn": execution_arn, "steps": steps,
    "previous_execution_arn": previous_execution_arn, "changed_job_fields": comparison,
}, indent=2, default=str) + "\n")
print("\nSaved:", report.resolve())
print("NO CACHE HIT is expected for failed/pending steps or steps without caching enabled.")
print("Matching successful compute steps can be reused within 24 hours; expired results run again.")
