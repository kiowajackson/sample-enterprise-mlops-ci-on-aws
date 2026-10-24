"""Finalize native execution reports and MLflow independently of the launcher.

EventBridge invokes this Lambda on terminal pipeline events. A five-minute
reconciliation schedule covers missed events and retries unsuccessful publication.
Only the explicitly configured pipeline ARNs are accepted. No model jobs start here.
"""

from datetime import datetime
import hashlib
import json
import os

import boto3

from mlops_models.pipelines.inspect import collect_execution, write_run_index
from mlops_models.pipelines.tracking import (
    TERMINAL, _lease, _put_json, _read_json, sync_execution,
)


def finalize(session, arn, pipelines, root, watch_from):
    """Use native status/version, and record completion only after every write succeeds."""
    pipeline = arn.rsplit("/execution/", 1)[0]
    if pipeline not in pipelines:
        raise ValueError("Execution is outside this completion service's pipeline scope.")
    sm, s3 = session.client("sagemaker"), session.client("s3")
    state = sm.describe_pipeline_execution(PipelineExecutionArn=arn)
    status = state["PipelineExecutionStatus"]
    if state["CreationTime"] < watch_from or status not in TERMINAL:
        return {"execution_arn": arn, "status": "ignored"}
    stamp = state["LastModifiedTime"].isoformat()
    key = hashlib.sha256(arn.encode()).hexdigest()
    marker_uri = root.rstrip("/") + "/completion/" + key + ".json"
    # Events can be duplicated; reconciliation can overlap event delivery.
    with _lease(s3, marker_uri + ".lock", os.getenv("KMS_KEY", "")):
        marker = _read_json(s3, marker_uri, optional=True)
        if marker and marker.get("modified") == stamp and marker.get("status") == status:
            return marker
        definition = json.loads(sm.describe_pipeline_definition_for_execution(
            PipelineExecutionArn=arn
        )["PipelineDefinition"])
        tracking = json.loads(definition["Metadata"].get("Tracking", "null"))
        if tracking:
            result = sync_execution(session, arn, {
                **tracking, "kind": pipelines[pipeline], "s3_root": root,
                "kms_key": os.getenv("KMS_KEY", ""),
            })
        else:
            report = collect_execution(session, arn)
            result = {"index_uri": write_run_index(
                session, report, kms_key=os.getenv("KMS_KEY", ""))}
        marker = {"execution_arn": arn, "status": status, "modified": stamp, **result}
        # Keep native status separate from the MLflow status returned above.
        marker["status"] = status
        _put_json(s3, marker_uri, marker, os.getenv("KMS_KEY", ""))
        print(json.dumps({"event": "execution_finalized", **marker}), flush=True)
        return marker


def handler(event, context):
    session = boto3.Session()
    pipelines = json.loads(os.environ["PIPELINES"])
    root = os.environ["S3_ROOT"]
    watch_from = datetime.fromisoformat(os.environ["WATCH_FROM"])
    if event.get("source") == "aws.sagemaker":
        if event.get("detail-type") != "SageMaker Model Building Pipeline Execution Status Change":
            raise ValueError("Expected a pipeline execution event.")
        return finalize(session, event["detail"]["pipelineExecutionArn"], pipelines, root, watch_from)
    if event != {"reconcile": True}:
        raise ValueError("Expected a terminal event or the reconciliation schedule.")
    failures, completed = [], 0
    for pipeline in pipelines:
        for page in session.client("sagemaker").get_paginator("list_pipeline_executions").paginate(
                PipelineName=pipeline, CreatedAfter=watch_from):
            for execution in page["PipelineExecutionSummaries"]:
                if execution["PipelineExecutionStatus"] not in TERMINAL:
                    continue
                # Leave enough time to return a clear failure and retry later.
                if context.get_remaining_time_in_millis() < 30000:
                    raise TimeoutError("Reconciliation needs another invocation; completed writes are retained.")
                arn = execution["PipelineExecutionArn"]
                try:
                    finalize(session, arn, pipelines, root, watch_from)
                    completed += 1
                except Exception as error:
                    failures.append(arn)
                    print(json.dumps({"event": "completion_failed", "execution_arn": arn,
                                      "error": str(error)}), flush=True)
    if failures:
        raise RuntimeError(f"Completion failed for {len(failures)} execution(s); see the execution ARNs in logs.")
    return {"checked": completed}
