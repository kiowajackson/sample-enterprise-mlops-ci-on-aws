"""EventBridge/Lambda adapter: finish tracking using the execution's existing job image."""

from copy import deepcopy
import hashlib
import json
import os

import boto3
from botocore.exceptions import ClientError

TERMINAL = {"Succeeded", "Failed", "Stopped"}
PIPELINE_EVENT = "SageMaker Model Building Pipeline Execution Status Change"
PROCESSING_EVENT = "SageMaker Processing Job State Change"
JOB_PREFIX = "mlops-sync-"


def dispatch(event, sm, *, pipeline_arns, job_roles, job_prefix=JOB_PREFIX):
    """Start one tracking-only job per terminal execution state; retry a failed sync once."""
    if event.get("source") != "aws.sagemaker":
        raise ValueError("Supply a SageMaker pipeline/processing event.")
    detail = event["detail"]
    attempt = 0
    replay_id = event.get("replay_id", "")
    if event.get("detail-type") == PIPELINE_EVENT:
        arn = detail["pipelineExecutionArn"]
    elif event.get("detail-type") == PROCESSING_EVENT:
        name = detail["ProcessingJobName"]
        if not name.startswith(job_prefix) or detail.get("ProcessingJobStatus") not in {"Failed", "Stopped"}:
            return {"ignored": "not a failed completion job"}
        previous = sm.describe_processing_job(ProcessingJobName=name)
        environment = previous["Environment"]
        arn = environment["PIPELINE_EXECUTION_ARN"]
        replay_id = environment.get("MLOPS_SYNC_REPLAY_ID", "")
        attempt = int(environment["MLOPS_SYNC_ATTEMPT"]) + 1
        if attempt > 1:
            raise RuntimeError(f"Completion retry failed for {arn}; replay the pipeline event with a new replay_id.")
    else:
        return {"ignored": "unrelated event"}

    pipeline_arn = arn.split("/execution/")[0]
    if pipeline_arn not in pipeline_arns:
        raise ValueError("This pipeline is outside the configured completion scope.")
    # Events may be duplicated or delayed. The native execution is authoritative.
    execution = sm.describe_pipeline_execution(PipelineExecutionArn=arn)
    status = execution["PipelineExecutionStatus"]
    if status not in TERMINAL:
        return {"ignored": "execution is not terminal", "status": status}
    definition = json.loads(sm.describe_pipeline_definition_for_execution(
        PipelineExecutionArn=arn)["PipelineDefinition"])
    observer = next((step for step in definition["Steps"] if step["Name"] == "StartTracking"), None)
    if observer is None:
        return {"ignored": "tracking disabled for this execution"}
    request = deepcopy(observer["Arguments"])
    if request["RoleArn"] not in job_roles:
        raise ValueError("The execution's tracking role is outside the configured completion scope.")
    command = request["AppSpecification"]["ContainerEntrypoint"]
    if command != ["python", "-m", "mlops_models.pipelines.tracking", "--phase", "start"]:
        raise ValueError("The execution does not use the expected tracking entrypoint.")
    command[-1] = "sync"
    # One name per native terminal revision also handles retrying the same pipeline execution.
    revision = f"{arn}/{status}/{execution['LastModifiedTime']}/{replay_id}"
    base = job_prefix + hashlib.sha256(revision.encode()).hexdigest()[:32]
    request["ProcessingJobName"] = name = f"{base}-{attempt}"
    request["Environment"].update(PIPELINE_EXECUTION_ARN=arn, MLOPS_SYNC_ATTEMPT=str(attempt))
    # Preserve a replay identity when a failed Processing job triggers its retry.
    if replay_id:
        request["Environment"]["MLOPS_SYNC_REPLAY_ID"] = str(replay_id)
    request["Tags"] = [{"Key": "Purpose", "Value": "mlops-tracking-completion"}]
    try:
        existing = sm.describe_processing_job(ProcessingJobName=name)
    except ClientError as error:
        if error.response["Error"]["Code"] not in {"ValidationException", "ResourceNotFound"} or (
            error.response["Error"]["Code"] == "ValidationException" and
            "Could not find" not in error.response["Error"]["Message"] and
            "does not exist" not in error.response["Error"]["Message"]
        ):
            raise
    else:
        return {"job_arn": existing["ProcessingJobArn"], "status": existing["ProcessingJobStatus"], "reused": True}
    try:
        result = sm.create_processing_job(**request)
    except ClientError as error:
        if error.response["Error"]["Code"] != "ResourceInUse":
            raise
        result = sm.describe_processing_job(ProcessingJobName=name)
    return {"job_arn": result["ProcessingJobArn"], "execution_arn": arn, "attempt": attempt}


def lambda_handler(event, context):
    """Read the platform's explicit pipeline/job-role allowlists."""
    result = dispatch(
        event, boto3.client("sagemaker"),
        pipeline_arns=json.loads(os.environ["PIPELINE_ARNS"]),
        job_roles=json.loads(os.environ["JOB_ROLE_ARNS"]),
        job_prefix=os.getenv("SYNC_JOB_PREFIX", JOB_PREFIX),
    )
    print(json.dumps(result))
    return result
