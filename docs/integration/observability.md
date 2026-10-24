# Direct logging, reports and execution status

Model code owns preprocessing, fitting, predictions and analysis. The existing
jobs log their outputs directly to one MLflow run for the pipeline execution.
Training/evaluation share a run; inference gets its own run linked to the model.
Inference monitoring adds its reports and metrics to that same inference run.

## Capture analysis metrics and plots

Write optional `metrics.json` in the supplied analysis output directory:

```python
metrics = {"rmse": float(rmse), "mae": float(mae), "rows_labeled": int(len(labels))}
(output_directory / "metrics.json").write_text(json.dumps(metrics, allow_nan=False))
```

The runtime logs these as `prediction.rmse`, `prediction.mae`, etc. Null values are
unavailable and are omitted. It uploads the existing analysis CSV/PNG/PDF/TXT/HTML/JSON
files up to 10 MiB each, preserving subdirectories under `outputs/<step>/`.
Known JSON reports are under `reports/<step>/`. Larger files and model binaries
stay in their declared S3 outputs. No changes inside customer analysis are required.
Report-producing jobs declare `MLOPS_LOG_ARTIFACTS=1`; new step names do not require
editing a list inside the logger.
An analysis function may also use the runtime's MLFLOW_TRACKING_URI/MLFLOW_RUN_ID;
use explicit client logging and leave final run status to the caller.

## Finalize from the existing orchestrator

The loader derives the execution ID inside the running job. It does not put a
new MLflow run ID into the job definition's Environment, which participates in
SageMaker cache keys. The S3 tracking receipt associates the execution with its run.
A cache hit does not execute model or logging code: finalization records the original
producer and makes its small reports available on the new execution run.

After SageMaker reaches Succeeded, Failed or Stopped, notebook 02 or the Airflow
caller invokes `sync_execution(session, execution_arn, settings)`. This reads native
status, finalizes the same MLflow run and updates:

```text
<OutputDataRoot>/<ModelId>/runs/<execution-id>/run.json
```

Processing inference's Postprocess writes this report after prediction analysis.
When requested, DriftComparison extends it with comparison artifacts. Each producing job writes `_execution.json` describing its actual output
files. The final report assembles those records and native cached-producer links;
adding or renaming a step does not require artifact-discovery rules in `inspect.py`.
Postprocess and DriftComparison remain uncached; upstream compute can reuse successful results.
Before the job finishes uploading, the report says `outputs_prepared` and retains
the native in-progress status. The caller refreshes final status, including upload
failures. For training or an execution that fails before Postprocess, the caller
assembles the available producer records.

For executions started in Studio, run notebook 02's final cell after completion.
Without that caller action, direct metrics/artifacts are visible but MLflow's run
can remain RUNNING. No completion Lambda or tracking-only Processing job is deployed.
The same finalization call handles failures, stops and executions consisting of cache hits.

## Native SageMaker lineage

Notebook 03 contains the optional native graph functions. They link existing data,
source, model and job records without inventing duplicate jobs. Graph publication is
an explicit notebook operation; it is not required for logging or ordinary execution.

## Evidently batch comparison

Notebook 02's `monitoring` setting adds HasReference and a separate DriftComparison
Processing step after Postprocess. Notebook 04 walks through an initial execution
and a second execution with `ReferenceExecutionArn` pointing to the first. An empty
reference skips DriftComparison entirely; Postprocess still writes customer analysis. An explicit reference
must be a successful Processing inference for the same model/environment.
The job reads that execution's `run.json` and its declared input/prediction locations,
including the original producers when outputs were cached. The reference must
have been run with this report-writing pipeline; older executions are not reconstructed
using assumptions about their step names.
It joins inputs/predictions by the configured record key; no model is re-scored.
Current and reference populations need not contain the same record IDs.

Under the current run's `reports/drift/monitoring/` folder, when comparison runs:

- `evidently_data_drift/report.html`: selected feature distributions, when a reference is supplied.
- `evidently_prediction_drift/report.html`: score distributions, when a reference is supplied.
- `evidently_data_quality/report.html`: current and available reference dataset summaries.
- `evidently_model_quality/report.html`: classification or regression quality when labels are available.
- `metrics.json` and `summary.json`: metrics, availability and exact execution/model/S3 references.

MLflow report artifacts are under `outputs/DriftComparison/monitoring/`. Numeric metrics
start with `monitoring.`; reference identities are also tags. Customer analysis
keeps its own files, including its top-level `summary.json` and `metrics.json`.
Postprocess calls customer analysis and execution reporting. The drift entrypoint
dispatches to models/<ModelId>/inference/drift.py, which defaults to
model_base/inference/drift.py. That base owns get_container_input(), comparison/report
calls in compare_drift(), and write_container_output(). It uses monitoring/report.py
as the shared Evidently generator. The main processing function updates run.json.
Drift edits invalidate training so the new model archive contains the updated code. All Processing
inference steps, including DriftComparison, use the selected model's saved source
and pinned image. Train and approve a version containing the drift entrypoint
before using the new graph. The current deployment's Registry resolver still
controls which exact package is eligible.
`load_source.py` prepares paths, loads the selected source, dispatches the entrypoint
and uploads declared outputs; it does not invoke monitoring itself.

Model-quality metrics require actual labels. Without labels the quality status is
unavailable and the data reports still run. Without a reference, drift is not
evaluated, no comparison job starts and no zero/healthy drift metric is emitted. Later outcomes can be joined
to saved predictions and evaluated without scoring again. Model owners choose the
prediction CSV/score column, thresholds, populations and business response.
The report helper retains optional SNS publication for platform use; inference
does not create alert infrastructure. Outcome ingestion, scheduling and alert
routing remain platform integration work.

The job role needs SageMaker execution/parameter/step reads, Processing job and
model-package descriptions, and read access to the selected S3 reference (plus
KMS decrypt where applicable). Existing MLflow permissions are unchanged.

## One image

The root requirements and lock describe one Linux Python 3.12 image containing the
model runtime, mlflow-skinny 3.10.1, sagemaker-mlflow 0.5.0 and Evidently 0.7.20.
The model-library pins are unchanged. Evidently needs requests >=2.32, so the shared
image explicitly updates the supplied requests 2.31.0 pin to 2.34.2. This needs to
be applied through BMO's image process; there is no runtime package installer.
MLflow remains optional: leave tracking_uri empty until its service/access are ready.
