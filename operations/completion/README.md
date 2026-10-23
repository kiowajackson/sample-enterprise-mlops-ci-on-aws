# Automatic MLflow completion

This optional platform example adds one Lambda function and two EventBridge rules:

```text
Pipeline Succeeded/Failed/Stopped → Lambda → tracking-only Processing job
Failed tracking-only job         → same Lambda → one retry
```

The source uses only boto3 inside Lambda. MLflow runs in the existing approved
tracking image. `DescribePipelineDefinitionForExecution` selects that execution's
tracking source, role and network settings. The callback reads the current native
status before dispatch, so a delayed event during a pipeline retry cannot finish
the active run.

## Prepare the resources for review

Render the template using explicit names from the existing deployment:

```bash
python operations/completion/template.py \
  --name example-completion \
  --pipeline-arn arn:aws:sagemaker:REGION:ACCOUNT:pipeline/NAMESPACE-training \
  --pipeline-arn arn:aws:sagemaker:REGION:ACCOUNT:pipeline/NAMESPACE-inference \
  --job-role-arn arn:aws:iam::ACCOUNT:role/EXISTING-JOB-ROLE \
  --output completion.template.json
```

The renderer creates a file and makes no AWS calls. The template includes a
scoped Lambda role, log group and invocation permissions. The existing job role
needs its normal MLflow/S3/model/job-read access; this template does not edit it.
Use `--failure-destination` with an existing approved SNS topic or SQS queue
to receive exhausted Lambda invocations. No notification topic or queue is created.

The platform team can use its CDK constructs or deploy the rendered example:

```bash
aws cloudformation deploy --template-file completion.template.json \
  --stack-name example-completion --capabilities CAPABILITY_IAM \
  --profile WORKLOAD_PROFILE --region REGION
```

Use a completion name of at most 20 characters. The deployment creates new
resources and can start billable Processing jobs for matching terminal events.
One normal completion costs one small tracking job. The example uses the
pipeline's configured tracking instance and ten-minute runtime limit.

## Verify and replay

1. Publish the reviewed source and pipeline definition with tracking enabled.
2. Run one execution from Studio. After completion, find the extra
   `<completion-name>-sync-*` Processing job.
3. Inspect its logs and the model run's `run.json`. MLflow's `pipeline.status`
   should match SageMaker, with the same run ID as StartTracking.
4. Exercise a failed/stopped execution and a duplicate event. Completion should
   update the same MLflow run, without another training or inference job.

`dispatch` is also callable with an explicit SageMaker client for local/operator
use. To replay via Lambda, provide this event in a JSON file:

```json
{
  "source": "aws.sagemaker",
  "detail-type": "SageMaker Model Building Pipeline Execution Status Change",
  "detail": {
    "pipelineExecutionArn": "arn:aws:sagemaker:REGION:ACCOUNT:pipeline/NAME/execution/ID"
  },
  "replay_id": "operator-retry-1"
}
```

Invoke the installed function using the platform's CLI/profile and that payload.
Change `replay_id` only for a deliberate new attempt. Inspect Lambda logs, failed
Processing jobs and any configured failure destination before replaying.
Source and job-role scope are checked against the configured allowlists.

The automatic path requires tracking in the original execution definition.
For executions with tracking disabled, call `index_execution` explicitly.
CloudWatch/Lambda delivery failures and missed upstream events remain visible
through the platform's normal operations; this example does not install an
additional polling service.

Delete the CloudFormation stack to remove these completion resources. Existing
pipelines, application outputs, MLflow runs and model artifacts are unaffected.
