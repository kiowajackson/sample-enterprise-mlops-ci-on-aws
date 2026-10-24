# Connect the pipeline to an MLflow App

The existing tracking runtime accepts SageMaker MLflow App ARNs. Notebook 02
publishes the App binding into the existing model jobs; changing the configuration
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
| Notebook and model-job roles | `sagemaker:CallMlflowAppApi` on the approved App ARN |
| Users opening the UI | `sagemaker:CreatePresignedMlflowAppUrl` and App API access; `DescribeMlflowApp` for inspection, and `ListMlflowApps` where discovery is required |
| App service role | SageMaker service trust and access to the approved artifact store, including applicable S3 encryption permissions |
| Clients writing or reading artifacts | Direct S3 access to the approved artifact prefix and applicable KMS key |
| Finalization caller | Its existing access to describe pipeline executions/jobs/model packages, list execution steps/parameters, read source reports and write tracking receipts |

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
The existing model and monitoring jobs call MLflow under their job role. Ordinary callers do not need
App administration or the older Tracking Server's start/stop permissions.

Confirm connectivity from the actual notebook and job network to MLflow,
SageMaker APIs and S3. An MLflow VPC endpoint policy must permit the App ARN and
`sagemaker:CallMlflowAppApi`.

## Configure and publish

1. Have the platform build the shared requirements into the approved model image.
   The client-only MLflow package and AWS plugin connect to the existing App; no
   second image or in-container tracking server is needed.
2. Set tracking_uri in notebooks/config.json to the approved App ARN.
3. Set tracking_experiment_prefix in notebook 02 and republish the pipelines.
4. Run training or inference. Its existing jobs publish directly; no observer job
   appears in the pipeline graph.
5. After completion, run notebook 02's final cell (or invoke the same finalization
   function from Airflow) to synchronize terminal status and cached producers.

The tested client target is the sandbox App's MLflow 3.10.1, using mlflow-skinny
3.10.1 and sagemaker-mlflow 0.5.0. The one-image contract retains the model pins
and explicitly changes requests 2.31.0 to 2.34.2 for Evidently. Apply and verify
this image change in BMO before enabling tracking there.

## Verify what was captured

In the App UI, open the experiment and inspect the execution's run:

- Parameters: pipeline inputs and saved effective training settings.
- Metrics: the actual evaluation scores plus cached/failed-step counts.
- Tags and inputs: execution/producer/source identity, dataset references,
  Registry version, model image and model archive URI.
- Artifacts: `execution.json` and the collected report files.

Evaluation reports include `evaluation.json`, `model_metadata.json`,
`parameters.json` and `training_execution.json` when present. The collector also
reads supported inference postprocessing outputs. Eligible CSV/PNG/PDF/TXT
reports are bounded to 10 MiB each.

FeatureSelection CSV/PDF reports and eligible HTML/JSON analysis reports are
collected. A flat `metrics.json` supplies numeric prediction-analysis values.
Raw datasets and model archives remain in S3; the run index links their actual
producer locations. Selecting an App does not move historical tracking records.

After the execution finishes, run notebook 02's final synchronization cell, or
invoke the same function from Airflow after waiting for the native terminal state.
This also handles cached, failed and stopped executions.

Inference uses the image recorded in the approved Registry package. Before enabling
tracking, verify that package points to the image containing these clients. An
older package retains its older image even after notebook configuration changes.
Train/register a new candidate, or explicitly validate and register the existing
model artifact with the updated image through the agreed approval process.

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
