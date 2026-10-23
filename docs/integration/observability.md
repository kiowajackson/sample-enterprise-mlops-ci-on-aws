# Metrics, reports and execution completion

These helpers extend the existing training and inference pipelines. Model code
still owns preprocessing, fitting, predictions and analysis.

## Publish prediction-analysis metrics

Keep the existing analysis files. Write an optional `metrics.json` directly in
the supplied postprocessing output directory:

```python
import json

metrics = {
    "rmse": float(rmse),
    "mae": float(mae),
    "rows_labeled": int(len(labels)),
    "r2": None,  # Leave unavailable metrics unreported.
}
(output_directory / "metrics.json").write_text(
    json.dumps(metrics, indent=2, allow_nan=False) + "\n"
)
```

The file is a plain dictionary of names and numbers. The collector publishes
`prediction.rmse`, `prediction.mae` and `prediction.rows_labeled` on this
inference execution's MLflow run. Agree the metric definitions, units and
evaluation population with the model owner. Save label/batch context in the
analysis summary or another JSON file alongside the metrics.

There is no dependency on MLflow inside customer analysis. With tracking disabled,
the file is still uploaded to S3. With tracking enabled, the closing tracking job
copies it and logs the numbers. Missing `metrics.json` preserves the existing
behavior. A null metric is unavailable, not zero.

FeatureSelection files are now included. CSV, PNG, PDF, TXT, HTML and JSON reports
up to 10 MiB each are copied into MLflow. Known reports live under `reports/`;
other files retain their subfolders under `outputs/<step>/`. Larger files stay
in S3 and retain their links in the execution inventory. The collector does not
derive scalar metrics from plots or arbitrary tables.

## Find an execution's files

Tracking writes:

```text
<OutputDataRoot>/<ModelId>/runs/<execution-id>/run.json
```

The index lists data, reports, model/Registry references, jobs, source records
and the MLflow run. A cached step links its original producer/output. Failed
steps retain their status and declared output locations; successful artifacts
remain where their jobs wrote them.

To create or refresh this index without MLflow:

```python
from mlops_models.pipelines.tracking import index_execution

index_uri = index_execution(session, execution_arn)
print(index_uri)
```

To refresh both MLflow and the index, use the existing `sync_execution` helper.
A tracking-service failure can be retried without running the model again.

## Complete tracking automatically

An in-pipeline closing job observes the pipeline while it is still running.
Install the optional [completion adapter](../../operations/completion/README.md)
to synchronize after SageMaker reports Succeeded, Failed or Stopped. The adapter
also covers executions started through Studio.

The adapter reuses the tracking image, source, network and role from the definition
for that specific execution. It launches a tracking-only Processing job. It does
not retrain, score, register or approve a model. Duplicate native events reuse the
same job; a failed completion job gets one retry. An operator can replay after
correcting an access or service problem.

Deployment is separate from copying application code. BMO can adapt the included
small resource example to its existing constructs or host the callback through
its own platform. Keep the pipeline/job-role allowlists scoped to this application.

## Try native SageMaker lineage

[Notebook 03](../../notebooks/03_inspect_results.ipynb) inspects an existing
execution and optionally publishes a native graph. It creates native Artifact
records for input/output locations and reuses SageMaker's existing TrialComponent
for each successful job, then associates them with the registered model. Cached
steps use their original job identity. If native job metadata is missing or
ambiguous, publication stops before writing metadata; resolve that condition and
retry. It never creates a second job record. Feature Store group and population references are included when
already present in GetData's provenance.

```python
from mlops_models.pipelines.lineage import publish_lineage, query_lineage

native = publish_lineage(session, execution_arn)
graph = query_lineage(session, native["package_roots"][0])
print(native["receipt_uri"])
```

This is an explicit metadata-publication operation. It reads the existing job
records and locations; no dataset hashing, split reconstruction or model execution
is added. The graph identifies recorded locations and producers. It does not
certify the bytes at a mutable S3 location.

The caller needs the normal execution/job/output read permissions, plus
`sagemaker:ListArtifacts`, `CreateArtifact`, `ListTrialComponents`,
`ListAssociations`, `AddAssociation`, `QueryLineage`, and `AddTags` where required.
Use the platform's approved lineage permission scope. Routine tracking jobs do
not need these additional write permissions unless the platform deliberately
hosts graph publication there.

SageMaker queries one starting entity and up to ten relationship levels at a
time. Query an intermediate job to inspect earlier inputs or a different branch.
The receipt lists all relationships created/reused by this example.

The graph is retained for review. The receipt's `jobs` mapping contains existing
native TrialComponents; these are never owned by this publisher and must not be
deleted during cleanup. `created` records this invocation's added artifacts and
associations, and `owned_artifacts` identifies this example's artifact records.
Preserve reused artifacts and native associations. No model/data resource needs
to be deleted. Older receipts containing custom `actions` require a separate,
scoped migration of their saved associations; normal publication does not remove
existing records.

## Run the small monitoring example

[Notebook 04](../../notebooks/04_monitoring_example.ipynb) compares two supplied
batches with Evidently. Its default data is explicitly synthetic. Install
`requirements-monitoring.txt` in a notebook/report environment; the existing
model image does not need this dependency.

`write_report` writes `report.html`, `evidently.json`, `summary.json` and
`metrics.json`. Pass the numerical/categorical feature names and tune the PSI
threshold and drift share with the model owner. Optional `quality_metrics` are
numbers from their existing prediction analysis. Without labels/analysis,
quality is **unavailable**. A drift flag does not establish poor predictions.

`publish_report` uploads those files to a unique S3 report prefix. It can also:

- Attach the metrics and reports to a caller-supplied MLflow analysis or inference
  run. Use one analysis run per report for straightforward comparison.
- Publish `DriftedFraction` to an approved CloudWatch namespace, dimensioned by
  model and environment.
- Send a breach message to one existing approved SNS topic. The JSON includes the
  exact report URI, model/batch context and metric values.

The notification may be delivered more than once; a receiver can deduplicate on
`summary_uri`. Reuse a destination prefix only when intentionally replacing the
same report. MLflow/network failures are surfaced, and the uploaded S3 files
remain available for replay.

BMO owns the scheduled trigger, reference-data selection, approved thresholds,
recipient/subscriber configuration, alarms, escalation and recovery actions.
The publishing helper creates no scheduler, queue, alarm or response worker.
An explicit `configure_drift_alarm` helper demonstrates one CloudWatch alarm on
`DriftedFraction` with the same SNS destination. Choose either the direct report
message or the alarm as the notification trigger. The alarm example's description
points to the specific report under review; BMO defines ongoing report lookup
when adapting it to a continuously monitored stream.

```python
from mlops_models.monitoring.report import configure_drift_alarm

configure_drift_alarm(
    session.client("cloudwatch"), name="example-batch-drift",
    namespace="Example/Monitoring", model_id="usecase2", environment="dev",
    threshold=0.5, topic_arn=approved_topic_arn, report_uri=receipt["summary_uri"],
)
# Remove the example alarm when review is complete:
# session.client("cloudwatch").delete_alarms(AlarmNames=["example-batch-drift"])
```

It is an Evidently batch report and notification pattern; it does not configure
a native SageMaker Model Monitor schedule.

The publishing caller needs S3 writes to its report prefix and, when enabled,
MLflow run/artifact writes, `cloudwatch:PutMetricData` restricted by namespace,
and `sns:Publish` on the selected topic for direct messages. Installing the optional
alarm additionally needs `cloudwatch:PutMetricAlarm`/`DeleteAlarms` on that example
alarm and a topic policy allowing the CloudWatch alarm to publish. No model's
Registry approval is changed.
