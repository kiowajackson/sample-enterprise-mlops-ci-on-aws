# Connect the pipeline to an MLflow App

The existing tracking runtime accepts SageMaker MLflow App ARNs. Notebook 02
publishes the App binding into the tracking jobs; changing the configuration
does not update a previously published pipeline.

## Platform setup

Create or identify an approved App and return its ARN, Region, version, service
role and S3 artifact location. Keep automatic SageMaker Model Registry
registration disabled: this pipeline already registers its evaluated model.
Choose any required customer-managed key for App metadata during App creation.
Artifact-bucket encryption is configured separately.

Confirm that the intended users may share the App. Its documented
`CallMlflowAppApi` permission applies to the App resource; separate experiment
names do not establish a permission boundary between teams.

| Identity | Required access |
| --- | --- |
| Notebook and tracking-job roles | `sagemaker:CallMlflowAppApi` on the approved App ARN |
| Users opening the UI | `sagemaker:CreatePresignedMlflowAppUrl` and App API access; `DescribeMlflowApp` for inspection, and `ListMlflowApps` where discovery is required |
| App service role | SageMaker service trust and access to the approved artifact store, including applicable S3 encryption permissions |
| Clients writing or reading artifacts | Direct S3 access to the approved artifact prefix and applicable KMS key |
| Tracking collector | Its existing access to describe pipeline executions/jobs/model packages, list execution steps/parameters, read source reports and write tracking receipts |

The current MLflow client uses the experiment's S3 artifact location directly.
Grant the notebook/job roles the required `s3:GetObject`, `s3:PutObject`,
`s3:AbortMultipartUpload`, scoped `s3:ListBucket` and `s3:GetBucketLocation`
permissions. Permissions on the App service role alone do not authorize the
client's direct S3 requests.

Use the experiment's returned `artifact_location`; do not construct it from an
experiment ID. The path can include workspace components. Platform can precreate
an experiment to scope artifact permissions to that location, or grant the
approved App artifact prefix when clients create experiments themselves.

Keep the existing application receipt/lock access, including deletion of the
temporary writer lock. If a new pipeline or Registry namespace is used, update
any corresponding resource-scoped collector/registration permissions too.

The pipeline orchestration role creates jobs and passes the approved job role.
The tracking jobs call MLflow under their job role. Ordinary callers do not need
App administration or the older Tracking Server's start/stop permissions.

Confirm connectivity from the actual notebook and job network to MLflow,
SageMaker APIs and S3. An MLflow VPC endpoint policy must permit the App ARN and
`sagemaker:CallMlflowAppApi`.

## Configure and publish

1. Install [requirements-tracking.txt](../../requirements-tracking.txt) in the
   notebook and the approved tracking-job image. Validate those client versions
   against the actual App version. A non-root image needs writable scratch,
   including a suitable `TMPDIR`.
2. Replace only `tracking_uri` in the existing `notebooks/config.json` with the
   supplied App ARN:

   ```json
   "tracking_uri": "arn:aws:sagemaker:<region>:<account-id>:mlflow-app/<app-id>"
   ```

3. In Notebook 02's **Defaults** cell, set `tracking_experiment_prefix` to the
   agreed grouping. Set `tracking_image_uri` if tracking uses a separate image;
   leaving it empty uses the model-job image.
4. Choose the intended namespace, Registry grouping and a `deploy_name` label. Source revisions are content-addressed automatically.
   Run Notebook 02. Its preflight checks the notebook caller; publication
   includes the App ARN in `StartTracking` and the final tracking job.
5. Start an execution of the newly published pipeline version and inspect both
   tracking jobs. Their success establishes access under the managed-job role.

The experiment name is `<tracking_experiment_prefix>/<environment>/<ModelId>`.
Training run names begin with `training-` followed by the pipeline execution ID.

## Verify what was captured

In the App UI, open the experiment and inspect the execution's run:

- Parameters: pipeline inputs and saved effective training settings.
- Metrics: the actual evaluation scores plus cached/failed-step counts.
- Tags and inputs: execution/producer/source identity, dataset references,
  Registry version, model image and model archive URI.
- Artifacts: `lineage/execution.json` and the collected report files.

Evaluation reports include `evaluation.json`, `model_metadata.json`,
`parameters.json` and `training_execution.json` when present. The collector also
reads supported inference postprocessing outputs. Eligible CSV/PNG/PDF/TXT
reports are bounded to 10 MiB each.

FeatureSelection CSV/PDF reports and eligible HTML/JSON analysis reports are
collected. A flat `metrics.json` supplies numeric prediction-analysis values.
Raw datasets and model archives remain in S3; the run index links their actual
producer locations. Selecting an App does not move historical tracking records.

After the execution finishes, run Notebook 02's final synchronization cell.
The closing job runs while its own pipeline can still be `Executing`; later
synchronization records the actual terminal state. Failed/stopped pipelines can
skip the closing job. The [optional completion adapter](observability.md#complete-tracking-automatically)
can perform terminal synchronization after Platform deploys and accepts it.
Until then, use the explicit synchronization cell.

## Existing history and rollback

The S3 receipt key includes both the tracking-service ARN and execution ARN.
Changing from a Tracking Server to an App therefore creates separate tracking
records. It does not migrate the earlier service's history.

Keep the earlier pipeline publication and tracking history available while
validating the App. A rollback selects the previously accepted pipeline/config
binding; it does not require deleting the new experiment or model artifacts.

## References

- [App setup](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-app-setup.html)
- [App IAM permissions](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-app-setup-prerequisites-iam.html)
- [App UI URL API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreatePresignedMlflowAppUrl.html)
- [App KMS versus artifact encryption](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-kms.html)
- [MLflow VPC endpoint policy](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-private-link-policy.html)
