# Enterprise MLOps — training and batch prediction

This repository publishes **one shared training pipeline and one shared inference
pipeline** in SageMaker. Select the model through the **ModelId** execution
parameter. The supplied `usecase1` regression and `usecase2` binary-classification
implementations are replaceable examples.

The current delivery is a generic integration baseline. It demonstrates data and
artifact handoffs, model selection by folder, training and batch prediction using
an existing container. Customer algorithms replace the model code while keeping
the input/output contract. Platform provisioning and production release integration
are separate workstreams.

**Status: October 16, 2026.** The supported default path is
`csv_direct` → training → Model Registry → approved-model Processing batch inference.
Implemented options, validation boundaries and planned work are distinguished below.
This README describes the source in this delivery; a previous model archive may
contain an older copy of that source.

The preprocessing, training and prediction entrypoints follow the same pattern:
resolve input paths, load metadata as `config`, pass the separate `parameter_path`,
then call the model code. The main configuration difference is its source:

| Entry point | `config` comes from | Parameters |
|---|---|---|
| Training preprocessing and Train | `models/<ModelId>/model_metadata.json` in the supplied source; initially `{}` | Separate `models/<ModelId>/parameters.json` path |
| Inference preprocessing and Predict | `model_metadata.json` in the extracted Registry-selected model | Separate `models/<ModelId>/parameters.json` path |

ResolveModel selects the approved version; the loader extracts its archive before
calling an inference entrypoint. The entrypoint reads that local metadata file.
Neither Train nor Predict nests the metadata inside the parameters dictionary.
Their model operations differ, but their input/output boundaries stay consistent.

## Running prediction from a registered model

Notebook 02 publishes one inference pipeline with an **InferenceRuntime** choice.
The default Processing branch runs **ResolveModel → GetData → Preprocess → Score → Postprocess**.
The Transform branch runs **ResolveModel → GetData → PreprocessTransform → CreateModel → Transform → PostprocessTransform**.
ResolveModel and GetData precede the InferenceMode condition; each execution runs
only its chosen branch.
Each step receives local paths to the exact upstream artifacts, including cached
artifacts. Entry points load the saved configuration and call model code; column
normalization, fitted transformations and prediction analysis belong in that code.

For a model whose training already works:

1. Keep the adapted trainer and model parameters. Update the pipeline definition,
   builder, dispatcher, runtime paths and inference entrypoints together. Run All
   in notebook 02 to publish the new graph and source.
2. Adapt `model_base/inference/preprocess.py`. `process_inference` receives the
   raw directory, selected-feature file, output directory, metadata dictionary
   and parameter-file path, in the same order as `process_train`. The example reads
   `raw/data.csv`, keeps the primary key and columns in the saved feature mapping,
    and writes `prepared/data.csv`. The loader creates
    `selection/selected_features.csv` from the registered metadata before calling
    the entrypoint. The entrypoint only loads config and passes paths.
   `get_container_input` returns the raw dataframe and selected-feature dataframe.
   The function calls `write_container_output` without returning anything; the
   loader uploads its written files. It does
   not require a target or rerun feature selection. Apply the same column naming
   and fitted transformations as training wherever the customer code requires them.
3. Adapt `model_base/inference/predict.py`. Its file hook receives
    `user_test_data_path`, `feature_mapping`, `primary_key`, and the keyword
    `output_directory`. The mapping comes from registered metadata; the primary key
    comes from the separate parameters file. `self.checkpoint_path` points to the
    extracted approved model. Set the model code's output root to
    `Path(output_directory)` and retain its existing writes, including subfolders.
    The hook need not return anything. `write_container_output` recursively finds
    every `.csv` file (case-insensitive), prints each path, and returns the output
    directory. It fails if no CSV was written and requires no particular filenames.
    The loader uploads the existing files with their relative paths and bytes
    intact, including any other artifacts under the declared output directory.
    The demo supplies scores and IDs only; customer code supplies explanations.
    The entrypoint, use-case wrapper and base `predict_with_model` all take the
    same five arguments; Processing prediction does not take `predictor_cls`.
4. Adapt `model_base/inference/postprocess.py` and update the small postprocess
   adapter in each model folder. The analysis receives the original raw CSV path,
   the saved selected-feature dataframe, all three prediction dataframes and saved
   configuration. Return the final analysis/leads dataframe. Its writer saves
    `prediction_analysis.csv` and `summary.json`. Additional customer columns remain
    intact; the generic formatter normalizes a `score` column when needed.
    The reader discovers all CSVs recursively and keys the dataframes by relative
    path, preserving subfolders and duplicate basenames. The example analysis uses
    one `messaging_final.csv` at any depth and reports missing or ambiguous matches.
    Replace that example's file choice with the inputs required by customer analysis.
5. Notebook 02 defaults to `model_code_source="pipeline"`,
   `batch_runtime="both"` and `cache_steps=True`. With an existing compatible
   Approved model, start **inference only** with InferenceRuntime=processing, selecting ModelId and the scoring
   CsvInputUri. No retraining is required to test these inference code changes.
   Check the resolved model version, final row identifiers and scores against the
   model owner's original prediction before calling the port accepted.

The archive already holds `parameters.json` and `model_metadata.json`; model loading
uses that exact approved archive. Preprocessing reads its metadata there and receives
the selected source folder's `parameters.json` path separately, matching training.
Prediction follows the same metadata/parameter separation.
With `model_code_source="pipeline"`, keep those source parameters compatible with the
registered model; with `"artifact"`, they come from the archived source. For new registrations, Evaluate
also publishes an identical `model_metadata.json` beside `evaluation.json`.
RegisterModel verifies its checksum and records **ModelMetadataUri** and
**ModelMetadataSha256** in the version's custom metadata. That URI lets an operator
open the JSON directly in S3 without unpacking the model. Older approved versions
remain usable from their archive; run the updated Evaluate/RegisterModel path to
create a new version with the separate metadata reference. Registry stores the
reference, not the JSON document itself.

The example metadata contains an input-to-model mapping and a relative predictor
path:

```json
{
  "feature_mapping": {"input_column": "model_column"},
  "predictor_path": "."
}
```

Names must match exactly, including case. The model location defaults to the
archive root, as when the writer copies `AML.path`; the generic trainer records
`predictor_path="predictor"` for its subdirectory. Source-folder metadata never
substitutes for learned metadata from the approved version. The mapping names
columns; it does not calculate business features or recreate missing fitted state.

In customer `predict_with_model`, replace the local `final_model/` search with the
supplied checkpoint path. ResolveModel already selected the approved Registry
version; selecting the newest local folder would introduce another model choice.
When the training writer receives `AML.path`, it copies that directory's contents
to the package root. In that layout, `checkpoint_path` itself is loadable.
If the predictor is nested, save its exact relative path as `predictor_path`.

Inside `predict_with_model`, the replacement for the directory search is:

```python
from pathlib import Path
from autogluon.tabular import TabularPredictor

model_root = Path(checkpoint_path).resolve()
latest_model_directory = (model_root / config.get("predictor_path", ".")).resolve()
if not latest_model_directory.is_relative_to(model_root):
    raise ValueError("predictor_path must stay inside the registered model archive.")
best_model = TabularPredictor.load(str(latest_model_directory))
```

The saved metadata and predictor directory come from the same registered artifact.
`predictor_cls` was a Python class extension point; it did not select a Registry
version or represent the saved model.

Keep training and inference modules separate for this delivery. The generic
preprocessing modules now have identical bodies except for the function name.
They retain a target and timestamp when those columns exist; scoring data need
not contain a target. The selected-feature file's first column supplies names.
Customer code can be copied into both modules and later consolidated after both
paths pass. Training itself still requires its configured target.

Processing uses a whole-file model hook and loads prediction outputs into memory
for analysis. Size the job for that workload or implement bounded processing in
customer code. Transform now calls the same selected use case's five-argument
`predict_with_model(input_directory, checkpoint_path, output_directory, config, parameter_path)`.
It does not construct a separate ModelPredictor. For each HTTP request it writes
`input/data.csv` in a unique writable temporary directory, calls the model function,
and reads the result from that request's output directory. The function can keep
its existing CSV writes and return nothing. Set its model-loading path to the
supplied checkpoint and its write root to the supplied output directory in both modes.

**TransformOutputFile** selects the CSV returned to SageMaker; its default is
`messaging_final.csv`, found recursively. Supply an exact relative path when a
different file is wanted or several folders contain the same basename. Missing or
ambiguous results fail explicitly; there is no substitution with another CSV.
The response preserves the selected file's columns, requires one result for each
input record, and restores input ID order if the model sorted unique IDs by score.
Other per-request files are temporary and are deleted after the response; Processing
retains the entire output directory in S3. Native Postprocess reads Transform's
JSON-lines outputs and passes the combined result to the analysis hook.

To test both routes at the same time, start two executions of the same inference
pipeline with the same ModelId and CsvInputUri, one with InferenceRuntime=processing
and one with InferenceRuntime=transform. Check both ResolveModel package ARNs:
each execution independently selects the highest Completed, Approved version.
Keep approvals unchanged while launching a comparison. Jobs and S3 output paths
remain separate by execution. Choosing one branch does not execute the other.

Processing uses the notebook's `model_code_source` setting. **Native Transform
uses the code in the approved model archive.** An existing compatible model can
test new Processing code immediately; the new serving adapter must be included
in a training package and approved before that archive can exercise it through
Transform. Updating notebook source does not update an older registered archive.

The image must dispatch its `serve` command to
`python -m mlops_models.entrypoints serve` with `PYTHONPATH=/opt/ml/model/code`.
The existing dispatcher starts Gunicorn on port 8080; the image must include
Flask, Gunicorn and the model dependencies. The model and source remain read-only;
per-request files use `TemporaryDirectory`, respecting the image's writable TMPDIR.
Look for `server_ready` with `adapter=file_prediction_v1` in the Transform logs.
The health check establishes that configuration/imports are ready; model loading
and prediction are exercised by the first invocation. Processing's entrypoint
override does not establish support for the image's native serve startup.

The first file-based adapter may load the model again for each request. Keep the
default single worker and concurrent request while validating correctness.
Prediction functions must avoid global working-directory changes or shared output
filenames outside the supplied request directory. Reusing loaded models and tuning
GPU batching are later optimizations. The existing Transform input formatter
writes one JSON-lines file; multiple-instance distribution needs multiple files.
Some preprocessing/formatting still reads whole dataframes into memory.
Reuse saved fitted transformations in both routes; fitting encoders or scalers
again on individual requests can change predictions relative to the whole-file run.

The final analysis file is currently published to S3. **MLflow artifact/run
logging is planned, not implemented**. Future tracking should retain the model
version, input snapshot and final output URI, with direct artifact upload subject
to the agreed data-access and size policy.

## Updating a trainer that already contains your model code

Keep your adapted training implementation. Align its arguments and output writer
with the entrypoint; the generic trainer remains an integration example.

The use-case wrapper supplies `trainer_cls=UseCaseModelTraining`. Your base
`train_model` must accept `trainer_cls` after `*`, then use `trainer_cls(...)`
where it currently constructs `ModelTraining(...)`. Keep your constructor's
existing arguments. Accepting the keyword alone does not enable the override:
the constructor call must use it too. `ModelTraining` remains defined in the
base file; the base file does not import the use-case module. This is a class
argument, not an instance or a circular import.

1. In your `ModelTraining.evaluate_model`, keep the existing validation-probability
   return and also return the `performance` dictionary from `AML.evaluate(...)`.
   Unpack both at the call site; plotting still receives the probabilities.
2. In `train_model`, accept `parameter_path` immediately after `config`, matching
    the entrypoint and use-case wrapper. Load parameters from that file; keep
    `config` as the decoded model metadata. The example trainer receives both
    dictionaries separately. Read `primary_key`, `target_key` and `task_type`
    from parameters, and learned settings from config.
3. In `mlops-models/src/mlops_models/model_base/training/train.py`, port the
    `write_container_output` and `evaluation_report` functions with their imports.
    Call the writer with `parameters=parameters`, `performance=performance` and
    `data_usage="validation"` alongside the existing checkpoint, learned metadata,
    package directory and output directory. Set `parameters["model_id"]` from
    the selected model folder as in the example. Use the validation label only
    for validation rows. The report adapts real AutoGluon scores for Evaluate;
    the writer performs loss conversion.
4. Keep the working Evaluate and Registry adapters. The writer retains native
    checkpoint files, writes learned `model_metadata.json` and `evaluation.json`,
    and preserves effective `parameters.json` from the checkpoint or supplies it
    separately when absent. Native reports written under `output_directory`
    remain there for the loader to upload. Keep the working checkpoint separate
    from the empty package destination.
5. Rebuild through notebook 02 and start a new execution. The training-source
   change runs Train once to save the report; unchanged upstream steps remain
   eligible for cache reuse. Evaluate should then publish the scores and Registry
   should create a PendingManualApproval version in the configured existing group.
   Review and approve that version before attempting batch inference.

The generic trainer's internal return is now `(checkpoint, metadata, performance)`.
Its metadata identifies `evaluation.usage`; your own train_model orchestration can
instead pass its existing variables directly to the writer as described above.
Printing scores alone does not create evaluation.json. The new Evaluate fails
clearly if the trained archive lacks the report.

## Full Training or Processing job logs

Copy the entire [job_log_cell.py](mlops-models/scratch/job_log_cell.py) into one
notebook cell. Set `JOB_NAME` to the job name shown inside the pipeline step,
then run it. `JOB_TYPE` defaults to Training; choose Processing for Evaluate.
The notebook's AWS region is used unless AWS_REGION is set explicitly.
The cell reads every available page from every job stream, saves a complete
`*-full.log` file and prints it. Use the displayed file link if the notebook UI
truncates its output. A running job can produce more logs after the printed
snapshot time; rerun the cell to refresh the file. This needs CloudWatch Logs
read permissions and does not require changing or restarting the job.

## Navigation for engineers and coding agents

Read the feature table and workflow first, then the contracts and limitations
before changing code. Paths in this README are relative to the repository root.

- [Features and status](#features-and-status)
- [Run the two notebooks](#run-the-two-notebooks)
- [Configuration and model identities](#configuration-and-model-identities)
- [Pipeline and data contracts](#pipeline-and-data-contracts)
- [Execution and code map](#execution-and-code-map)
- [Local paths and S3 artifacts](#local-paths-and-s3-artifacts)
- [Caching and iteration](#caching-and-iteration)
- [Parameters and learned metadata](#parameters-and-learned-metadata)
- [Limitations and platform dependencies](#limitations-and-platform-dependencies)
- [Validation](#validation)
- [Plan and acceptance milestones](#plan-and-acceptance-milestones)
- [Change guidance](#change-guidance)

## Features and status

| Feature | Current behavior | Acceptance boundary |
|---|---|---|
| Shared pipelines | One training/inference pair per namespace; discovered ModelId dropdown | Rename, second model and added-third model exercised in the sandbox |
| Explicit CSV input | Preflight checks the supplied CSV; GetData copies it unchanged | No implicit example-data fallback; model-specific schema validation remains necessary |
| Metadata-based sample generation | Notebook 01 creates synthetic CSV rows and feature mappings | Tests data movement, not realistic distributions or model quality |
| Feature Store | Optional ingestion and an Athena-based pipeline reader | Existing resources, delivery contract and target permissions must be verified |
| Feature selection and preprocessing | Separate Processing steps with CSV/PDF handoffs | Demo algorithms are placeholders for customer code |
| Ordinary training | AutoGluon example saves checkpoint, effective parameters and metadata | Customer training modes and dependencies require porting and acceptance |
| Evaluation | Train saves scores; a separate job publishes the report and checks quality thresholds | Demo scores training rows; customer validation must use the matching checkpoint |
| Model Registry | Required registration after the quality gate; inference resolves a Completed, Approved version | Platform supplies groups/access; approval and promotion are separate operations |
| Batch inference | Uses the image and saved model selected from Registry; default scoring uses Processing | Registry failures stop execution; no direct-S3 model bypass |
| Step caching | Successful preparation, training, evaluation and Processing batch steps eligible for 24-hour reuse | Registry operations remain live; latest Feature Store reads, native HPO and Transform remain uncached |
| Native HPO / Batch Transform | HPO remains optional; Transform and Processing can be selected per inference execution and share the file prediction function | Target container startup, model code, HTTP output and permissions require separate acceptance |
| Initial selection → final training | Metadata is saved with the model | Explicit prior-run metadata input and customer initial/final mode handling are planned |
| CI checks | `.gitlab-ci.yml` declares definition checks and source tests | It does not build containers, provision resources or deploy this delivery |
| Tracking, monitoring and production integration | Requirements and integration boundaries identified | Not implemented by the two notebooks |

## Run the two notebooks

Use a fresh extracted directory and notebook kernel. The notebook environment
needs Python 3.11 or 3.12 and the dependencies in
`requirements-infrastructure.txt`. Install them through the approved dependency
source, for example from the repository root:

```bash
python -m pip install -r requirements-infrastructure.txt
```

The **job image** separately needs the packages imported by the model/runtime
code. Installing notebook dependencies does not update that image. The example
model requirements are under `mlops-models/src/mlops_models/model_base/`.
Training and the Registry lookup job use the supplied image. Supply an image URI
containing its digest when an exact image version is required. The notebook
pins a supplied ECR tag to its digest; a repository-only address selects and reports
its newest tagged image. It does not build an image.
Inference preprocessing, scoring and postprocessing use the image recorded with
the approved Registry version. Retained Python source does not bundle installed
dependencies; the registered image must remain compatible with its saved model.

| Notebook | Required inputs | Result |
|---|---|---|
| [01 — Prepare data](notebooks/01_prepare_data.ipynb) | Explicit metadata CSV; additional bucket/writer role for Feature Store mode | Local sample CSV and mapping; optional ingestion and S3 configuration URI |
| [02 — Deploy pipelines](notebooks/02_deploy_pipelines.ipynb) | Bucket, pipeline role, existing image and explicit CSV or Feature Store configuration | `<namespace>-training` and `<namespace>-inference` saved in Studio |

### Metadata versus pipeline data

`metadata_csv` belongs to notebook 01. It describes columns and reference values;
notebook 01 uses it to create `sample.csv` with actual synthetic rows and a
feature mapping. Notebook 02 does not consume the original metadata file.

| Mode | Notebook 01 | Notebook 02 input | Pipeline GetData |
|---|---|---|---|
| csv_direct | Generate local sample.csv and mappings | csv_input = generated sample.csv; Feature Store URI empty | Notebook 02 uploads the sample, SageMaker mounts it, and GetData copies it unchanged to raw/data.csv |
| feature_store | Generate sample and mappings, then ingest into the store | feature_store_config_uri = saved binding URI; csv_input empty | Query the existing store and write raw/data.csv |

Notebook 01 uses data_source for preparation. Notebook 02 uses data_source for
training and inference_data_source for inference, independently.
CSV mode does not need a Feature Store mapping or ingestion permissions. Both routes produce the flat-file handoff
consumed by FeatureSelection; they do not necessarily select identical columns.
The current Feature Store query selects configured model features; CSV-direct
preserves all supplied columns.

Both notebooks read **notebooks/config.json**. Copy
[config.example.json](notebooks/config.example.json) to that name and fill in
`metadata_csv`, `bucket`, `role_arn`, `image_uri` and `registry_group_prefix` once.
Leave the prefix empty if your groups have none. If config.json is missing,
the first notebook cell creates it from the template and stops with its full path.
Then **Run All in notebook 01, followed by Run All in notebook 02**. With the
default CSV route, no cell edits or copying values between notebooks are needed.

`role_arn` is the pipeline role. In notebook 01's Defaults cell, set
`feature_store_role_arn` for a different Feature Store offline writer. In notebook
02's Defaults cell, set `job_role_arn` for a different job role. Both default to
role_arn. CSV generation alone needs only metadata_csv. Relative metadata_csv and
csv_input paths resolve from the checkout root.

After successful preparation, notebook 01 updates only `csv_input` and
`feature_store_config_uri`. It always retains the generated CSV path relative to
the checkout root. Feature Store mode also saves the published configuration URI;
CSV mode clears that URI. This supports training from the store and inference
from the generated CSV, without changing the shared config between notebooks.
Other settings and additional keys are preserved. Failed preparation leaves the
previous saved handoff unchanged. Notebook 02 reads the file again when started.

The config contains only those five supplied values plus the two automatic data
handoff fields. Mode, namespace, environment, caching, HPO, sample and job options
are declared in each notebook's **Defaults** cell. Keep namespace consistent;
select each pipeline's data source independently in notebook 02. If updating an older expanded config, move
any non-default optional values into those cells; old optional config keys are
not read. A malformed config fails instead of being overwritten. Targets and
model settings remain in each use case's parameters.json.

The populated **config.json is Git-ignored and excluded from ZIP exports**.
Deliveries contain only config.example.json, so unpacking an update does not
replace existing connection values. When moving to a fresh checkout, copy your
config.json separately; copy any referenced local files too, or rerun notebook 01.
Do not populate the distributable example file with your environment values.

### 1. Prepare data

For `data_source="csv_direct"`, run notebook 01 with the intended metadata CSV.
All CSV-mode cells can run without AWS access. The generated sample.csv path is
saved in config.json for notebook 02. To use an existing flat CSV instead, set
csv_input in config.json and run notebook 02 directly; another sample is unnecessary.

Metadata fields are `column_name`, `column_map`, `table_nm` and `ref_val1` through
`ref_val5`. `column_name` supplies CSV headers; `column_map` preserves the source
mapping. Generated rows cycle through reference values, with synthetic identifiers
and the configured event time. Shared identifier/time columns are included once.
Other duplicate feature names need disambiguation. Labels are not invented.

To deliberately use the supplied example, set
`metadata_csv="examples/data/feature-metadata.csv"`. No missing path silently
substitutes it.

For `data_source="feature_store"`, notebook 01 additionally ingests into existing
groups and saves `feature_store_config_uri` in config.json. Missing-group creation is opt-in
and requires the corresponding permissions. It creates no roles, bucket, Athena
workgroup or tracking server. See [Feature Store details](mlops-features/README.md).

### 2. Publish the shared pipelines

Notebook 02 reads the required inputs from config.json. The default namespace
`wsf-dev` names this deployment **Wealth ML SageMaker Foundation (WSF)**. It
publishes `wsf-dev-training` and `wsf-dev-inference`, with source, uploaded CSVs and
run outputs under `s3://<bucket>/wsf-dev/`. The existing `source-dev` pipelines and
objects are not updated. Namespace ownership checks still apply if `wsf-dev`
already exists. Registry group names are **`<prefix><ModelId>-sm-<environment>`**. The platform
must create these groups before publication. Notebook 02 checks each group using
the notebook caller; a missing group or denied read stops before source upload.
Set `registry_group_prefix` in config.json to the exact prefix, including any
trailing hyphen. It is not shortened, lowercased or given an extra separator.
The model folder name and environment supply the rest: a prefix of `team-` gives
`team-usecase1-sm-dev`. The same name is used by preflight, training registration
and inference lookup. Change the prefix in config.json and environment in notebook
02's Defaults cell; republish to use another environment's existing groups.
Empty prefix keeps `usecase1-sm-dev`.

For either pipeline using CSV mode, supply `csv_input` as a local file or exact
S3 CSV URI. Also retain `feature_store_config_uri` when the other pipeline uses
Feature Store. Local relative paths resolve against this
checkout's root, even when it is a renamed subfolder inside another repository.
Use a fresh kernel for the new checkout; the notebooks stop if cached imports
point at another checkout. Keep the old notebooks on their own kernels.
Targets come from each use case's `parameters.json` field `target_key`. The
notebook shows target presence but does not require one CSV to contain every
use case's target. Select the intended ModelId and its CSV at execution time.
The demo names `target_value` and `churned_flag` are example target columns; replace
them with your own. They are not pipeline requirements. The notebook verifies
file access, headers, rows and uploaded bytes, without inventing labels or
renaming columns. Model-specific target handling belongs in the model code.
These options are declared in notebook 02's **Defaults** cell:

```python
data_source = "csv_direct"             # Training; later switch to feature_store
inference_data_source = "csv_direct"   # Scoring snapshot
environment = "dev"
batch_runtime = "both"  # InferenceRuntime defaults to processing at execution time.
model_code_source = "pipeline"
include_hpo = False
cache_steps = True
model_ids = None
```

Run all cells. They inspect the actual CSV/header/model targets, compile model Python
files for syntax, verify the existing Registry groups, upload source and any local
CSV, verify uploaded CSV bytes, render definitions, publish and read back the saved
definitions. Group reads do not prove pipeline registration or job permissions.
The preflight does not execute customer algorithms, validate every feature
requirement, or prove that the job role has the notebook role's access.

**Publication does not execute training.** The notebook saves definitions and
defaults. It prints pipeline names/versions, source/data locations and a local
`.runs/source-deploy/<namespace>/publication.json` receipt.

### 3. Execute training and prediction

1. In Studio, open the printed training pipeline and create an execution using
   the published version. Select **ModelId** and verify **CsvInputUri**. HPO is hidden
   by default; UseHPO only appears if include_hpo was enabled when publishing.
2. Wait for Evaluate, QualityGate and **RegisterModel** to succeed. Its
   registration job role registers a **PendingManualApproval** version in
   **`<prefix><ModelId>-sm-<environment>`**. Registration cannot be switched off.
   RegisterModel is a Processing step calling the Registry API. It requires an
   existing group and never creates one; this avoids the native RegisterModel
   step's attempt to create a group. Registration and approval
   lookup always run, so their job startup time remains even when compute is cached.
3. Review that version in **Models → Registered models** and approve it using an
   authorized operator role. Start inference with the **same ModelId** and set
   **CsvInputUri** to the scoring CSV. ResolveModel selects the highest numeric
   Completed, Approved version and records its package ARN, image and artifact URI
   in `package.json`. Scoring data needs input columns and identifiers, not a target.
4. Inspect the actual Postprocess output URI for `prediction_analysis.csv` and
   `summary.json`. Score publishes all three intermediate prediction CSVs. The
   demo final analysis has identifier/prediction; binary classification also has
   probability. Customer analysis can add the final lead fields.

Studio execution overrides can select a different CSV from the publication
default. Check the execution's effective parameters when diagnosing input data.
Notebook 02 verifies the supplied file without running model algorithms.
An unlabeled scoring CSV can be selected at inference execution.

Registry access is required. A missing group, denied API call, or absence of a
Completed, Approved version fails visibly. Inference has no ModelDataUri or image
override: SageMaker downloads the artifact whose S3 URI is recorded in Registry.
Approval lookup runs on every inference execution, even when compute is cached.
Changing approval after resolution does not cancel an already running inference.

The old `registry_mode` and `inference_model_uri` builder arguments have been
removed; callers using them must be updated. The notebook no longer has a
RegisterModel toggle or direct-artifact input. Existing pipelines must be
republished to receive this contract. Creating or naming a production group does
not implement promotion.

## Configuration and model identities

| Setting or location | Responsibility |
|---|---|
| Notebook 02 `namespace` | Shared pipeline names and deployment S3 prefix |
| `models/<ModelId>/` | Runtime model identity, selected folder and Registry naming input |
| `parameters.json` | Model settings and algorithm parameters |
| `model_metadata.json` | Learned model information; initially `{}` |
| Notebook `environment` | Registry suffix, such as `dev` or `prod` |
| `role_arn` / optional `job_role_arn` | Pipeline execution role / job role; empty job role uses the pipeline role |
| Notebook infrastructure settings | Network, encryption and related job configuration |
| Execution parameters | ModelId, input locations, instance choices and supported runtime switches |

Registry group names default to **`<prefix><ModelId>-sm-<environment>`**, for example
`usecase1-sm-dev`. Platform provisioning owns their creation.

To add or rename a use case, copy or rename a complete directory under
`mlops-models/src/mlops_models/models/`, edit its parameters and any overrides,
then rerun notebook 02 with `model_ids=None` from a fresh kernel. Model IDs must
match `[a-z][a-z0-9]{0,23}`. The publisher discovers folders containing
`parameters.json` and updates the **same two pipelines** and dropdowns.

Renaming changes the model's S3 prefix and Registry group. It does not migrate
old artifacts or delete old resources. The old archive retains its original
model identity. Use a newly trained artifact for a newly named model.

Notebook 02 defaults to `include_hpo=False`: it publishes a direct Train step and
omits TrainingMode, Tune, the tuned evaluation/registration branch and tuning
inputs. Ordinary publication does not require a `tuning` block. To restore HPO,
set `include_hpo=True` and republish. Selected models then require compatible
`tuning.parameters` and `tuning.fixed`; other common tuning defaults come from the
first selected model or execution parameters. This switch preserves the HPO code;
it does not establish container compatibility or execute tuning.

## Pipeline and data contracts

```text
CSV or Feature Store
  -> GetData -> raw CSV on S3
  -> FeatureSelection -> selected-feature CSV + PDF on S3
  -> Preprocess(raw CSV, selected features) -> prepared CSVs on S3
  -> Train(prepared CSVs, selected features) -> model archive + reports on S3
  -> Evaluate -> QualityGate -> Registry version (PendingManualApproval)

Approve a Registry version
  -> ResolveModel -> exact approved image + model archive
  + scoring CSV or Feature Store
  -> inference GetData -> raw CSV
  -> Preprocess(raw CSV, saved metadata) -> prepared CSV + saved feature list
  -> Score -> predictions.csv + messaging.csv + messaging_final.csv
  -> Postprocess(raw CSV, saved features, prediction files)
     -> prediction_analysis.csv + summary.json
```

Each step consumes a completed artifact from the previous producer. Feature
selection does not call GetData; preprocessing does not rerun feature selection.
Job definitions pass S3 locations to SageMaker, which downloads the inputs into
the container. The model functions read those local inputs.

| Step / model hook | Input | Output contract |
|---|---|---|
| GetData shared entrypoint | Explicit CSV or Feature Store binding | `raw/data.csv`, `raw/provenance.json` |
| `run_feature_selection` | Raw CSV directory and settings | `selection/selected_features.csv`, `selection/random_forest_plots.pdf` |
| `process_train` | Raw directory, selected-feature file, output directory, metadata dictionary and parameter-file path | `prepared/` containing one or more CSVs; demo writes `data.csv` |
| `train_model` | Prepared directory, selected-feature file, package/work paths, metadata dictionary and parameter-file path | Saved model directory with separate parameters/metadata, evaluation.json and runtime code |
| Evaluation hook | Saved artifact containing evaluation.json and model_metadata.json; current quality thresholds | Published evaluation.json and an exact copy of model_metadata.json |
| `process_inference` | Raw directory, selected-feature file, output directory, registered metadata dictionary and parameter-file path | `prepared/data.csv`; loader also writes `selection/selected_features.csv` |
| `predict_with_model` | Prepared directory, extracted checkpoint, writable output, registered metadata dictionary and parameter-file path | All CSV files under the output directory, including subfolders; existing files are retained |
| `postprocess_predictions` | Raw directory, selected-feature file, prediction directory and saved settings | `prediction_analysis.csv`, `summary.json`, effective `parameters.json` |

Both preprocessing functions receive
`(data_directory, selection_file, output_directory, config, parameter_path)`.
Training loads `models/<ModelId>/model_metadata.json` as `config`; inference loads
the approved model's extracted `model_metadata.json`. Both pass the selected source
folder's `models/<ModelId>/parameters.json` separately as a Path. They do not merge
the documents. The use-case wrappers re-export the base function and need no change.

Both functions call `get_container_input`, which returns two dataframes: the data
and the selected-feature table. They process those dataframes and call
`write_container_output` without returning anything. The loader uploads files under the supplied
output directory according to the pipeline's output definitions. The entrypoints
do not unpack an output path or row count. Keep writes under the supplied directory.
There is no batching, generator or `shard_rows` setting in preprocessing.

Prediction follows the same pattern. The wrapper reads the prepared CSV path,
feature mapping and primary key, supplies a writable output directory to the model
code, then finds and prints all CSV files recursively under the output directory.
Set the model's output root to that supplied directory. Code that appends `output/`
or other subfolders can retain them: the collector and loader preserve relative
paths. The downstream analysis reader now discovers and prints every CSV
recursively, with relative paths as dictionary keys such as
`output/messaging_final.csv`. No flattening or CSV rewriting happens during
publication; the model's analysis chooses which input files it consumes.

Processing and native Transform both call the selected use case's
`predict_with_model`. Its wrapper imports only that function from the base module;
there is no required `predict_model` helper or predictor-class export.
The base example constructs its own predictor; customer code can replace that
section after the three inputs are resolved. Override `predict_with_model` in a
use-case folder when that model needs different prediction logic.

Native Batch Transform's existing HTTP server expects JSON lines. Only that
optional route invokes `runtime/transform_input.py` after preprocessing has
written its CSV. This adapter formats one file under `prepared/shards/`; it does
not select features or divide the dataframe into batches. Processing inference
uses the CSV directly and does not invoke this adapter.

Training's demo selection CSV has a **`feature`** column. Inference reconstructs
its selection from saved feature_mapping and writes **`features,description`**.
If customer code changes either schema, update its producer and consumers together.
Identifiers are read as strings to preserve leading zeroes. Feature names,
case, target, timestamp and expected types must match the model implementation.

CSV GetData copies every column and the input bytes unchanged; selection belongs
to the next step. The current Feature Store reader instead requests the fields
listed in the model's `features` settings, joins groups by identifier/event time,
and writes the resulting flat snapshot. Configure the complete candidate feature
set before expecting feature selection to consider it. Inference Feature Store
reads use the approved model's feature mapping and do not request the target.
The normal scoring route is CSV: upstream data preparation computes business
features before delivering the scoring file. Passing through Feature Store is
optional for that file; it is not an extra feature-computation stage.

Feature Store reads use this application's configuration and completed-delivery
receipt contract, not just a group name. Empty **FeatureStoreDeliveryId** selects
the most recently published completion receipt; it does not mean the largest
business event date. Input visibility is assumed at job execution. Ingestion
does not rerun for every pipeline execution.

## Execution and code map

The call chain for a source-deployed job is:

```text
notebook publication / pipelines/source.py
  -> saved SageMaker job definition
  -> runtime/load_source.py <step>
  -> entrypoints/__main__.py: main -> dispatch_pipeline_step
  -> entrypoints/<step>.py: execute
  -> models/<ModelId>/<training-or-inference>/<module>.py
  -> inherited or overridden model_base implementation
  -> local output writer
  -> load_source.py uploads completed outputs
```

SageMaker starts the command recorded in the job definition; an entrypoint then
calls the model function. Model files do not need an independently executed
`if __name__ == "__main__"` block.

| Read or change here | Responsibility |
|---|---|
| [Publication builder](mlops-models/src/mlops_models/pipelines/source.py) | Shared ModelId pipelines, source uploads, required Registry and image/job bindings |
| [Training handoffs](mlops-models/src/mlops_models/pipelines/training_files.py) | GetData/selection/preprocess graph and native input channels |
| [Definition loader](mlops-models/mlops_skeleton/pipelines.py) | Render underlying model templates/settings |
| [Dispatcher](mlops-models/src/mlops_models/entrypoints/__main__.py) | Parse step/model/path arguments and select the execute function |
| [Runtime loader](mlops-models/src/mlops_models/runtime/load_source.py) | Select supplied or retained source, writable outputs, archive extraction and S3 upload |
| [Job paths](mlops-models/src/mlops_models/runtime/job_paths.py) | Default container mount paths |
| [Model base](mlops-models/src/mlops_models/model_base) | Shared, replaceable selection/preprocessing/training/prediction implementations |
| [Use-case folders](mlops-models/src/mlops_models/models) | Per-model settings and function/class overrides |
| [Registry operations](mlops-models/src/mlops_models/registry/model_package.py) | Registration into an existing group; approved package resolution and validation |
| [Inspection helpers](mlops-models/src/mlops_models/pipelines/inspect.py) | Existing execution/job/log/artifact inspection with explicit collection limits |
| [Feature package](mlops-features/README.md) | Sample generation, ingestion, delivery receipts and optional platform setup |
| [Release exporter](release/README.md) | Portable source inventory and notebook-output clearing |

`model_base/*/definition.json` files are templates, not the final published
graph. The publication builder transforms them into the shared pipelines.
Inspect the rendered definitions or publication receipt when tracing a run.

Use-case modules inherit corresponding base functions/classes. For example,
the use-case training wrapper passes its `UseCaseModelTraining`, derived from
`ModelTraining`, into the base
`train_model` function. Model-base code does not select another use case.
Entry points load parameters from the selected model folder; the process
working directory is not the parameter directory.

## Local paths and S3 artifacts

For a Processing Score error about a missing primary key, paste the entire
`mlops-models/scratch/trace_inference_inputs.py` file into a notebook cell and set
the failed Score and producing Preprocess job names. It verifies the producer
link, compares both jobs' primary-key settings and traces original/raw/prepared
CSV headers. It saves the deployed preprocessing/prediction files for inspection.
Use the original producer when preprocessing was cached. This read-only
diagnostic supports pipeline source mode; it does not inspect archived code.
If the prepared CSV contains the key, inspect the actual file/key used inside
the customer predictor and any column selection, casing or index conversion
before scoring. Keep IDs separately while selecting model inputs; they are
needed when building prediction outputs, even when they are not model features.

The non-root compatibility workaround is local filesystem handling, not an IAM
permission grant. It preserves native input downloads and provides writable
scratch for outputs. The job role still needs permission and network access to
read/write the configured S3 objects. With `cache_steps=True`, application-managed
Training also needs `sagemaker:DescribeTrainingJob` on its own job and network
access to the regional SageMaker API. The loader uses
the injected `TRAINING_JOB_NAME` and `TRAINING_JOB_ARN` to read that job's output
configuration. It does not create resources or grant itself permissions.

| Job input | Default local location |
|---|---|
| CSV input for training or inference GetData | `/opt/ml/processing/input/` |
| Raw input for selection / preprocessing / inference analysis | `/opt/ml/processing/raw/data.csv` |
| Selected features for training preprocessing / inference analysis | `/opt/ml/processing/selection/selected_features.csv` |
| Prepared inference CSV | `/opt/ml/processing/prepared/data.csv` |
| Prepared training files | `/opt/ml/input/data/prepared/` |
| Selected features for Train | `/opt/ml/input/data/selection/selected_features.csv` |
| Source for Processing / Train | `/opt/ml/processing/source/` / `/opt/ml/input/data/source/` |
| Model archive for evaluation/inference | `/opt/ml/processing/model/model.tar.gz`, then extracted to scratch |

Model functions receive these resolved paths and should use them rather than
reconstructing absolute paths. Processing scratch is under `/tmp/mlops-step-*`;
ordinary Training scratch is under `/var/tmp/mlops-step-*`. Job-specific directory
names are temporary and must not be stored as reusable model locations.

For Train, **output_directory** is the writable working area for fitting,
logs and reports. **model_dir** is the initially empty final package destination.
The writer receives a saved checkpoint **directory path**, not an in-memory
predictor. Do not reassign model_dir to the working checkpoint directory.

On successful completion, the loader uploads declared Processing outputs and
archives the Training model and report directories. A missing declared output or
upload failure fails the job. Application uploads are not an atomic transaction;
inspect job success before consuming a result. Scratch is not durable storage.

The loader does not intercept hard-coded writes such as `./output` or
`/opt/ml/model` inside customer code. Pass the writable directory through to
export/save methods. Enabling hard network isolation disables the application
upload path; do not use it as a drop-in setting for this workaround.

Outputs are grouped under:

```text
s3://<bucket>/<namespace>/models/<ModelId>/runs/<execution-id>/
  data/raw/data.csv
  data/raw/provenance.json
  data/selection/selected_features.csv
  data/selection/random_forest_plots.pdf
  data/prepared/data.csv
  model/application/model.tar.gz
  reports/training/output.tar.gz
  reports/evaluation/evaluation.json
  reports/evaluation/model_metadata.json
```

Inference uses its own execution directory: raw snapshot under `data/raw/`,
prepared CSV and optional shards under `data/modeling/`, saved feature list under
`data/selection/`, the three scoring CSVs under `predictions/raw/`, and final analysis under
`predictions/prediction_analysis.csv`. Cache hits retain the original producer
locations rather than copying artifacts to the current run. Only the relevant step artifacts appear
in each run. The model archive contains the demo's `predictor/`, effective
`parameters.json`, `model_metadata.json`, `selected_features.csv`, `evaluation.json`
and retained runtime source under `code/`. Evaluation rows are not packaged.

Train calculates scores once. Its writer packages a JSON report containing
`model_id`, `task`, `data_usage`, `metric_source`, `raw_metrics` and `metrics`.
`raw_metrics` preserves AutoGluon's higher-is-better scores. `metrics` converts
losses to their original scale through AutoGluon's metric definitions and uses
`rmse`, `mse` and `mae` for those three names. A negative R² remains negative.
The writer accepts the raw dictionary returned by `TabularPredictor.evaluate`;
it does not parse logs, text files or already converted metrics. Unknown metrics,
empty results and nonfinite scores fail before the model is packaged.

Evaluate reads the trained artifact. The loader unpacks it into a temporary
`model/` input directory and writes reports to a separate output directory.
The entrypoint reads the selected use-case folder's current `parameters.json`
for task and quality thresholds. The inherited base evaluator reads exactly
`model/evaluation.json`, verifies model/task identity and publishes those saved
scores. It does not load a predictor, read a dataset or require parameters inside
the archive. The entrypoint adds checkpoint/source provenance and the quality
decision, publishes the archive's exact metadata JSON and records its checksum.
Registration attaches the report and the metadata reference to the same model version.

Model parameters and learned metadata are still retained for inference; they
are not substitutes for the evaluation report. Metrics must describe the exact
checkpoint being packaged. If the model is refitted after validation, identify
that change and evaluate the final artifact before using its scores for approval.
The demo's prepared-input score is a smoke check, not held-out model quality.
To inspect a failed evaluation without running it again, paste the full
[scratch/trace_evaluation_inputs.py](mlops-models/scratch/trace_evaluation_inputs.py)
into a notebook cell and set `EVALUATION_JOB_NAME`. It inspects the job's exact
archive and saves a file manifest/report locally without extracting files,
executing the model or substituting another configuration.

To inspect a failed training job's actual input references, paste the complete
[scratch/trace_training_inputs.py](mlops-models/scratch/trace_training_inputs.py)
file into a notebook cell and set TRAINING_JOB_NAME and FEATURE_SELECTION_JOB_NAME.
Use the processing job that produced training's selection CSV; for a cache hit,
use the original job. The cell verifies that its output matches training's input.
It reads metadata/sample paths from this checkout's `notebooks/config.json`.
It counts metadata definitions, duplicate names, mapped features, local sample
columns, uploaded CSV columns, GetData's raw columns, selected predictors and
prepared columns. A metadata hash checks the local generation record; header
comparisons show where names/order/case change. Current local files are compared
with recorded job inputs; they are not assumed to be the same generation.
It also reads that selector job's parameters and
saves its supplied base/use-case scripts, the selection CSV and a report locally.
An explicit `features` list can restrict the placeholder selector's candidates
even when `feature_selection.top_k` is 50. Custom selectors control their own
selection rules. No data rows are printed or source executed.
A cached selection can reference an older producing execution;
the diagnostic uses the job's recorded URI, not a guessed current-execution path.

The demo's selection CSV contains predictor names and excludes the target. Its
cap is feature_selection.top_k (50 by default); fewer candidates or a replacement
selector can produce fewer rows. Preprocessing must retain the target separately
in the prepared data. Train receives the selection path resolved by the dispatcher
and fails if that file is absent; it does not try another selection file. The
diagnostic reports target case differences and missing selected columns without
changing data or inferring the cause from the number of selected features.

Evaluation uses the **uploaded evaluation code and current quality thresholds**
with the checkpoint's saved scores. Report/threshold changes can be checked without
refitting a model that already contains evaluation.json. Older artifacts without
that file require one new Train run; there is no re-scoring fallback. Inference
has an explicit `model_code_source` choice in notebook 02's Defaults cell:

- `artifact`: use the runtime source packaged with the approved model.
  Changing uploaded model code does not change what that archive executes.
  Batch Transform always uses this mode, including its branch when batch_runtime="both".
- `pipeline` (notebook default): use uploaded preprocessing, prediction and postprocessing code with
  the Registry-selected saved model. This supports debugging Processing steps
  independently; Registry approval is still required.

Artifact mode requires bundled source compatible with the published stage
interfaces; retrain to refresh that source when changing those interfaces.
Neither choice modifies the checkpoint or its learned settings. Compatible model
code and fitted preprocessing state remain the model owner's responsibility.
Evaluation records the model's saved source commit separately from the evaluation
source commit. Registration uses the model's saved commit, including after a cache hit.

## Caching and iteration

Notebook 02 defaults to `cache_steps=True`. It sets
`CacheConfig={"Enabled": True, "ExpireAfter": "PT24H"}` on these compute steps:

| Pipeline | Cached steps |
| --- | --- |
| Training with a fixed CSV | GetData, FeatureSelection, Preprocess, Train, Evaluate |
| Processing batch inference with a fixed CSV | GetData, Preprocess, Score, Postprocess |

Start another execution normally. Each matching **successful step** can reuse
its output within 24 hours in the same pipeline, even if a later step failed.
Failed or still-running steps are not reusable. The first execution after this
upgrade establishes the new cache entries.

Each step has its own source fingerprint. With the same inputs/model/image:

| Edit | Behavior |
| --- | --- |
| Nothing changed | Reuse all eligible successful compute steps |
| Training preprocessing | Reuse GetData/FeatureSelection; run Preprocess and downstream |
| Training implementation | Reuse preparation; run Train and Evaluate |
| Evaluation only | Reuse preparation and Train; run Evaluate |
| Batch prediction in pipeline-code mode | Reuse inference GetData/Preprocess; run Score and Postprocess |
| Batch postprocessing in pipeline-code mode | Reuse inference GetData/Preprocess/Score; run Postprocess |
| CSV contents, model artifact or earlier step output | Rerun its consumers and downstream steps |

Rerun notebook 02 after source/input edits to upload and republish, then start the
pipeline. Do not rerun notebook 01 unless new sample data is intended. The existing
source filenames/prefix and pipeline namespace must remain stable for reuse.
A new image, input URI, source layout or other native cache-key value can require
fresh jobs. Shared helpers are conservatively included in all fingerprints;
model settings from all discovered folders can invalidate model-processing steps.
Training's fingerprint includes inference code retained in its model package;
changing that code invalidates a future training run so new packages carry it.

Use new S3 keys for changed scoring data or models. SageMaker compares input
locations; it does not discover changed bytes behind an overridden execution URI.
Notebook 02 fingerprints its supplied CSV when rebuilding. It cannot fingerprint
a different mutable input that is supplied later through Studio.

Cached outputs remain under the execution that produced them. Training artifacts
are resolved from `Steps.Train.OutputDataConfig.S3OutputPath`, evaluation reports
from the actual Evaluate output, and batch outputs from their producing step.
The current execution can therefore have no new objects for cached steps.
Retain the producing execution's artifacts while they remain in use.

Paste [scratch/cell.py](mlops-models/scratch/cell.py) into a notebook cell, set
`execution_arn`, and run it to see each actual cache hit and source execution.
An optional `previous_execution_arn` compares Processing/Training cache-key fields.
The cell starts no jobs and saves records locally. Scratch is not uploaded as
pipeline source. `CacheHitResult.SourcePipelineExecutionArn` is the authoritative
reuse record; an old job's duration is not a new run's duration.

Exceptions are deliberate:

- Registry approval lookup/validation runs each time so approval changes are observed.
- Registration creates a version when requested; condition/quality gates run normally.
- Latest-delivery Feature Store reads are uncached. Their new output URI can also
  cause downstream steps to rerun even when the resulting CSV is identical.
- Native HPO remains opt-in and uncached in this delivery.
- Native Batch Transform remains uncached: this builder creates a new ModelName
  each execution, and ModelName is part of Transform's cache key. Reuse there
  needs a stable existing SageMaker Model resource; Processing is the tested
  path for independently cached batch steps.

SageMaker documents [step caching](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-caching.html),
[cache-key attributes](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-default-keys.html)
and [injected training job identity](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-running-container.html).

Direct-deploy source prefixes permit iterative uploads; avoid changing them while
jobs are active. The `ci-deploy` source identity path supports a clean explicit
commit. Connecting this to the receiving release process remains separate work.

## Parameters and learned metadata

Each model folder currently contains **both** JSON files. There is no separate
`model_config.json` dependency in this delivery.

| File | Current contract |
|---|---|
| `parameters.json` | `target_key`, `primary_key`, `task_type`; additional fields are passed unchanged to the model code |
| `model_metadata.json` | Initially `{}`; decoded directly as `config` for training; learned metadata comes from the Registry-selected artifact for inference |

The three shared names mean:

- `target_key`: the actual training target column name.
- `primary_key`: the record identifier column name.
- `task_type`: `classification` or `regression`.

There are no parallel `label`, `identifier` or `task` settings to maintain.
The included classification demo expects binary 0/1 targets. Customer model code
owns other classification conventions and any additional parameters, including
weights. The demo can run with these three keys alone: it derives candidate
features from the CSV and uses small training defaults. The examples also include
optional feature lists, forest/CPU settings and tuning settings for demonstration.
`num_cpus` controls the demo fit, not the SageMaker instance type. Missing
parameter/metadata files fail rather than loading another model's settings.
Newly trained archives retain this parameter schema and matching runtime code;
keep older artifacts with their original code.

The demo trainer saves effective parameters and newly generated metadata with
the checkpoint, including the selected feature mapping and relative predictor
path consumed by inference. It does **not** consume prior metadata to implement customer
initial-selection/final-training choices. Its generated fields such as
`selected_model` and input row count are not a complete `best_estimator` /
`best_config` contract. The same metadata is written as `training.json` inside
the reports archive. Training does not overwrite the source model folder.

The next planned contract is:

```text
parameters + prepared data + selected features + no learned metadata
  -> initial model selection / validation
  -> saved model + model_metadata.json
  -> explicit selection of that prior result
  -> final training, or HPO with a validation objective
  -> new model artifact + updated metadata
  -> prediction from the exact artifact
```

Prior-run metadata selection and initial/final execution-mode inputs are not
implemented yet. The first generic run keeps `{}` in the metadata file; customer
code must define what it produces and requires. A final all-data fit alone does
not define an HPO comparison: trials still need a model-owner-approved validation
objective. Final fitting can follow selection of a configuration.

Preserve selected features, feature mappings and fitted encoders/scalers with
the appropriate model. JSON metadata does not replace fitted preprocessing
objects. Inference must reuse the training transformations instead of fitting
incompatible encodings on a scoring batch.

## Limitations and platform dependencies

- **Model quality:** the demo trainer evaluates its prepared training rows and labels
  the result `prepared_input_smoke_check`. Default examples have no quality
  thresholds configured. Model owners must supply independent evaluation,
  acceptance thresholds and initial/final training semantics.
- **Tasks/frameworks:** examples implement regression and binary classification
  using AutoGluon. Other task types/frameworks require an adapter; dependency
  declarations alone do not establish support.
- **Scale:** several preparation, Feature Store join and training operations
  load a full dataframe in memory. Small/wide synthetic checks do not establish
  production volume, performance or cost.
- **Feature Store:** the reader assumes a completed, aligned delivery is visible
  offline. Independent table refreshes and historical point-in-time joins need
  an agreed data contract. See the feature README for keys, labels and receipts.
- **Native HPO:** hidden by default with include_hpo=False. Enable it only when
  ready to test image startup, per-trial configuration, validation metric emission
  and model collection together.
  Ordinary Training's loader override does not establish tuning compatibility.
- **Serving:** Processing and Transform share the file prediction function.
  Transform requires the updated adapter inside the approved archive and a
  compatible image serve startup. Per-request model loading remains an overhead;
  only the explicitly selected result CSV is returned by HTTP. Real-time endpoint
  deployment is not provided.
- **Registry:** both pipelines require precreated groups and the relevant role
  permissions. Inference requires approval of the actual candidate; it stops on
  missing approval or Registry errors.
  Environment naming does not provide cross-environment promotion or rollback.
- **Container paths:** writable-output compatibility is specific to the image/job
  setup. Sandbox root-user success does not prove non-root target compatibility.
  Both actual filesystem probes and target job execution remain relevant.
- **Platform services:** these notebooks do not create IAM roles, tracking
  servers or a production deployment system. Notebook 01 can create Feature
  Groups only through its explicit opt-in route.
- **Source selection:** artifact mode executes retained model code. Pipeline mode
  supports independently edited Processing adapters; it does not rewrite the
  saved model, update fitted preprocessing, or establish production approval.

Keep role responsibilities explicit:

| Identity / owner | Required responsibility |
|---|---|
| Notebook operator | Read data and Registry groups; upload source/CSV; publish pipelines and pass the approved pipeline role. Approval/deletion requires separate operator permissions |
| Pipeline role | Orchestrate jobs and pass approved job roles |
| Job role | Read inputs/artifacts and write outputs; describe existing Registry groups, create/describe model versions for RegisterModel, and list/describe versions for ResolveModel; query Feature Store when enabled |
| Feature Store writer role | Service-side offline storage/catalog/encryption access required by the platform |
| Platform provisioning owner | Create approved groups, tracking/storage/network resources and scoped permissions |

These are responsibilities, not a requirement for five newly created roles.
Notebook access does not prove job access. Hard network isolation and private VPC
networking are different choices; jobs still require the service connectivity
needed by their selected mode. Platform owners bind exact resources and roles.

## Validation

The shared file-based serving adapter passed **161 source tests** and notebook
build/publication checks for Processing, Transform and the combined runtime
dropdown with simulated AWS clients. Four underlying model definitions also pass
the repository's definition checks. Branch checks verify one shared pipeline pair,
valid references within each branch, a fresh Registry lookup, and preserved
Processing compute caching.

Real local Gunicorn tests used existing fitted AutoGluon regression and binary
classification models with the updated code copied into their local archives.
Each task scored 37 rows through four HTTP requests with two workers, matched
direct file prediction IDs/scores, and completed native-format Postprocess.
Model directories were read-only and unchanged; the test ran without root.
Additional tests exercise customer-style file functions returning None, nested
results, explicit output-file selection, input-order restoration, concurrent
request isolation, invalid inputs and temporary-directory cleanup.
No managed jobs were launched. These checks do not establish the receiving
image's serve startup, model code's batch independence, GPU performance or
managed Transform permissions. The local test reused fitted weights; it did not
change or re-register any remote model.

The recursive postprocessing reader passed **152 source tests**. The prediction
and analysis path was exercised for regression and classification with outputs
under `native/output/`. A custom analysis consumed an arbitrarily named nested
CSV without any of the three example prediction filenames. Tests preserve string
IDs and duplicate basenames, allow the example analysis to use its combined result
without the other two files, and reject missing or ambiguous analysis inputs.
AWS calls were simulated; no new managed jobs ran.

The recursive prediction-output update passed **150 source tests**. Checks cover
CSV files at several directory levels, uppercase extensions, identical basenames
in different subfolders, printed paths, unchanged bytes through simulated S3
uploads, and failure when no CSV exists. Calls through the selected use case
require no predictor-class argument. The default prediction/analysis flow remains
covered; custom analysis must be adapted to its own output layout.

The earlier metadata/parameter separation and file-output update passed **148 source
tests**. Two real local AutoGluon fits, one regression and one binary
classification, completed training and produced separate parameters and metadata.
Each then completed loader → preprocessing → prediction → postprocessing using
both pipeline and archived source. All 37 scoring identifiers and direct scores
matched after the same CSV serialization; model archives were unchanged by
inference. The checks also verify that a prediction hook can return nothing and
that its CSV formatting and additional files remain intact. The example predictor
allows mapped inputs discarded during fitting while requiring all features used
by the fitted model. AWS calls were simulated; these checks do not establish
receiving-container or customer-algorithm acceptance.

The earlier preprocessing-interface correction passed **144 source tests**. A customer
function with the same five arguments as training returned nothing, wrote its
prepared CSV, and the loader collected that file plus the saved selection.
Checks also cover separate metadata/parameters, model identity, missing registered
metadata, unchanged model artifacts, identical training/inference CSV outputs and
Transform-only format conversion. AWS uploads were
simulated; this correction has not yet been run in the receiving container.

The explicit inference-stage update passed **138 source tests**, **26 simulated-AWS
notebook scenarios**, and definition checks. Coverage includes the
separate data-source settings, cached producer references, all three prediction
outputs, rich customer columns, and metadata checksum verification before Registry
registration. Real local AutoGluon regression and binary classification each
completed GetData → Preprocess → Score → Postprocess using both archive layouts.
All 37 row identifiers matched and scores matched direct prediction within CSV
floating-point precision; the registered model files were unchanged. AWS uploads
and Registry calls were simulated. This does not establish customer container or
customer analysis acceptance.

The earlier inference metadata/file-handoff update passed **132 source tests** and real
local AutoGluon regression and binary-classification checks. Each task was scored
from both a nested demo checkpoint and an archive-root checkpoint with no demo
features setting. All 37 scoring rows retained their identifiers and matched
direct predictor scores; metadata traveled through the loader-created archive,
and extracted model files were unchanged. S3 uploads were simulated. The receiving
container and customer prediction/preprocessing logic still require acceptance.


The saved-training-metrics update passed **120 source tests** and real local
AutoGluon 1.4 training for both supplied tasks (200 synthetic rows each). Both
runs packaged the actual performance dictionary, exercised the loader's model
archive/report upload paths, ran Evaluate without loading a predictor, and
produced the expected Registry request. S3 and Registry calls were simulated;
these checks do not establish customer IAM, container or Registry acceptance.
Missing reports, wrong model/task, invalid metrics and failing thresholds were
also checked. The scores are training-data smoke checks, not quality claims.

The minimal-config update passed 92 source tests and 25 simulated-AWS notebook
cases. The config carries supplied inputs and data handoff fields; optional
settings come from the Defaults cells, even if old optional keys remain in the file.
Both notebooks executed in sequence from config.json with no copied values;
CSV and Feature Store handoffs preserved other settings, and populated config.json
was excluded from export. Prefix/environment bindings matched across Registry
preflight, ordinary/HPO registration and inference lookup. Changing the prefix
left training compute definitions unchanged. These are local/configuration checks;
this increment did not launch new managed jobs.

The required-Registry update passed 83 source tests and 23 simulated-AWS
notebook cases. Managed sandbox acceptance verified:

- Registration creates pending versions in an existing group without granting
  group creation or group tagging to the registration job.
- Repeated training reuses all five compute steps while creating a new pending
  Registry version. Registration itself is uncached.
- Inference selects the approved version even when a newer version is pending,
  uses its exact image/checkpoint, and produces 100 matching prediction IDs.
- Repeated inference refreshes approval and reuses all three compute steps.
  With approval revoked, the next execution fails at ResolveModel before scoring,
  even though earlier prediction outputs are cached.
- Missing approval stops inference. Isolated rehearsal pipelines, versions/group,
  S3 objects, job log streams and temporary role grants were removed after evidence
  was captured.

These checks used generic code, synthetic CSV data and a root-running sandbox
image. They do not establish customer model, target non-root, Feature Store,
native HPO or Batch Transform acceptance. The final candidate runtime was tested
before its delivery commit; retained evidence records that candidate separately.

The preceding cache update passed 77 source tests, 20 simulated-AWS notebook scenarios
and 11 focused cache/diagnostic checks. A managed sandbox rehearsal verified:

- All five training compute steps and all three Processing batch steps reuse
  successful outputs after a deliberately failed downstream step.
- An unchanged training repeat took 8 seconds, compared with about 12 minutes
  for the first execution; the unchanged batch repeat took 7 seconds.
- Editing evaluation reruns Evaluate against the original cached model.
  Editing batch postprocessing reuses Preprocess and Score.
- Original model/report S3 references, source identities and all 100 prediction
  row identifiers match the expected artifacts.

That earlier cache rehearsal used CSV input, the since-removed direct-artifact route
and a root-running sandbox image.
Logs confirm writable scratch paths remained `/var/tmp/mlops-step-*` for Training
and `/tmp/mlops-step-*` for Processing. This does not establish target non-root
acceptance or native HPO/Transform acceptance. Cached application-managed
Training requires the read-only job permission described above.

The parameter/input/HPO follow-up passed 70 source checks and 20 notebook
scenarios using simulated AWS. Two complete local model flows passed with custom
targets, only the three shared parameter keys plus an unchanged custom setting,
and unlabeled inference inputs. HPO-disabled and HPO-enabled definitions both
passed reference/dependency checks.

The preceding October 15 generic baseline had:

- The earlier baseline passed 68 source tests and 20 notebook scenarios, including explicit-input failures
  and CSV preparation without AWS access.
- Local AutoGluon training/evaluation and Processing/HTTP prediction checks.
- Six successful managed sandbox executions: training and direct-artifact inference
  for a renamed model, the second model and an added third model.
- Artifact/parameter/source checks and 23 matching prediction identifiers per
  inference execution, with the same shared pipeline pair retained across
  rename/add publication.

Those results apply to generic code and test data. They do not establish customer
model correctness, every training mode, production scale, target non-root behavior
or live acceptance of every optional platform integration.

For local source checks, use an environment with the declared infrastructure,
test/build and model dependencies installed. The existing CI uses the lock files
in `release/` and `model_base/`. From the repository root:

```bash
PYTHONPATH="$PWD/mlops-models:$PWD/mlops-models/src" python tests/checks.py
PYTHONPATH="$PWD/mlops-models:$PWD/mlops-models/src" python -m unittest discover -s tests -v
```

Definition checks render underlying templates; they do not publish pipelines.
For an adapter change, also exercise the affected read → algorithm → write
contract and inspect actual artifacts. For managed acceptance, record the
pipeline version, effective parameters, source/image identity, execution IDs,
logs, output locations and prediction row identities.

## Plan and acceptance milestones

Work proceeds in two parallel tracks. Dates and owners are agreed with the
receiving team; the sequence below is not a delivery-date commitment.

| Stage | Workshop / feature track | Production integration / CI/CD track |
|---|---|---|
| Baseline review | Confirm supported features, limitations and source; export the reviewed package | Confirm repository/folder, reviewers and source-transfer route |
| Customer adoption | Commit the generic baseline, then separately port code/parameters and rename model folders | Confirm approved image, dependency source, runtime roles and service bindings |
| First model acceptance | Train, register a pending version, approve it, and run inference from that Registry version | Verify exact group names and role permissions; establish Feature Store/tracking access and agreed container changes |
| Metadata reuse | Explicit prior-result input, final training and compatible prediction; repeat for the second model | Exercise native HPO startup and artifact collection |
| Workshop | Demonstrate supported modes, collect usability/model feedback and agree remaining acceptance | Decide provisioning, build/release, promotion, manual steps and operating ownership |
| Handoff | Close agreed tests, publish operating guidance and demonstrate use by another engineer | Prove the agreed monitoring/release/recovery scope and record remaining gates |

A remaining functional extension is prior-run metadata input for final training. Likely edits are
`pipelines/source.py` and `training_files.py` for execution inputs,
`entrypoints/train.py` for thin dispatch, and the model training adapter for
initial/final behavior. Notebook settings, tests and this README must follow the
implemented contract. Keep new training-only inputs from unnecessarily
invalidating preparation caching.

Tracking/lineage, native HPO, monitoring, alerts, approval/promotion and recovery
remain explicit backlog items. Do not mark them complete because the generic
training/inference demonstration works, or silently remove them when separating
production integration from workshop preparation.

Production work should distinguish shared-resource provisioning, code/image
release and model-candidate promotion. For each manual step, record the owner,
inputs, retained version/decision, verification and recovery procedure.
Integrate with the receiving organization's delivery process; replacing that
process is not an assumed scope of this repository.

Possible follow-on work includes prediction-consumer integration, additional
framework/container adapters and agent-assisted model development. A first agent
workflow could scaffold a model, run existing checks and a bounded experiment,
then prepare a reviewable change with artifact evidence. These are proposed
extensions, not delivered features or autonomous production approval.

## Change guidance

Preserve the following when extending this baseline:

1. Put shared model behavior in `model_base/`; keep use-case functions/classes
   as inherited defaults or explicit overrides. Model identity comes from the
   selected folder, not a hard-coded usecase1 import.
2. Keep each model script's **get_container_input → model code →
   write_container_output** structure. Replace demo internals with customer code;
   avoid refactoring the whole algorithm merely to fit orchestration.
3. Keep pipeline steps independent through artifacts. Do not make feature
   selection import/call GetData or Train import/call a preparation step.
   Evaluate consumes scores saved by Train; batch inference owns model loading and prediction.
4. Preserve the current callable signatures unless all entrypoints, use-case
   wrappers and callers are changed together. Parameters after `*`, such as
   `trainer_cls`, are keyword-only.
5. Read parameters through the selected model folder and pass resolved paths.
   Do not use the process working directory as the configuration location.
6. Change CSV headings, selected features, checkpoint layout and metadata
   producers/readers together. Keep model_dir separate from fitting outputs.
7. Preserve explicit CSV input and approved Registry model selection and fail visible required-input
   checks. Do not introduce automatic example-data or Registry-error fallbacks.
8. Validate the changed behavior, refresh status/limitations here and retain the
   baseline-versus-adapter diff for review.

Optional diagnostics and older notebooks are in `scratch/` in the development
checkout. The exporter excludes them, generated work, Git history and notebook
outputs; the delivery has only the two active notebooks. See
[source export](release/README.md) before creating a package. The exporter
includes current working files, so retain the actual source revision and archive
checksum with acceptance evidence.
