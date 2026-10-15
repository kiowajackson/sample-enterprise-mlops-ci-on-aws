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

**Status: October 15, 2026.** The supported default path is
`csv_direct` → ordinary training → `offline_registry` → Processing batch inference.
Implemented options, validation boundaries and planned work are distinguished below.
This README describes the source in this delivery; a previous model archive may
contain an older copy of that source.

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
| Evaluation | Separate job reloads the model and produces metrics and a quality decision | Demo uses training rows for a smoke check, not independent quality validation |
| Development inference | `offline_registry` takes an exact S3 model archive; default scoring uses Processing | Bypasses Registry lookup explicitly; does not establish model approval |
| Model Registry | Optional registration and resolution of a Completed, Approved version | Platform supplies groups/access; approval and promotion are separate operations |
| Preparation caching | Successful CSV GetData, FeatureSelection and Preprocess steps eligible for 24-hour reuse | Train and downstream steps run; Feature Store caching is disabled |
| Native HPO / Batch Transform | Definition and runtime code paths exist | Target container startup, output handling and serving require separate acceptance |
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
Both default pipelines use the supplied existing image. Supply an image URI
containing its digest when an exact image version is required. The notebook
pins a supplied ECR tag to its digest; a repository-only address selects and reports
its newest tagged image. It does not build an image.
Offline inference uses that supplied image with the selected artifact. Retained
Python source does not bundle its installed dependencies; the image must remain
compatible with the saved model.

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
| feature_store | Generate sample and mappings, then ingest into the store | feature_store_config_uri = printed binding URI; csv_input empty | Query the existing store and write raw/data.csv |

Select the route explicitly through data_source. CSV mode does not need a Feature
Store mapping or ingestion permissions. Both routes produce the flat-file handoff
consumed by FeatureSelection; they do not necessarily select identical columns.
The current Feature Store query selects configured model features; CSV-direct
preserves all supplied columns.

Input cells contain only values you supply. Namespace, data_source and other
presets are grouped below in a separate Defaults cell. In notebook 02, the AWS
values and data-input block are separate cells so each can be pasted as a whole.
Notebook 01 prints the complete two-line block for notebook 02's data-input cell.
Running Defaults does not assign those input values again.

### 1. Prepare data

For `data_source="csv_direct"`, run notebook 01 with the intended metadata CSV.
All CSV-mode cells can run without AWS access. Copy its printed `sample.csv` path
into notebook 02. An existing flat CSV can also be supplied directly to notebook
02; generating another sample is unnecessary.

Metadata fields are `column_name`, `column_map`, `table_nm` and `ref_val1` through
`ref_val5`. `column_name` supplies CSV headers; `column_map` preserves the source
mapping. Generated rows cycle through reference values, with synthetic identifiers
and the configured event time. Shared identifier/time columns are included once.
Other duplicate feature names need disambiguation. Labels are not invented.

To deliberately use the supplied example, set
`metadata_csv="examples/data/feature-metadata.csv"`. No missing path silently
substitutes it.

For `data_source="feature_store"`, notebook 01 additionally ingests into existing
groups and prints `feature_store_config_uri`. Missing-group creation is opt-in
and requires the corresponding permissions. It creates no roles, bucket, Athena
workgroup or tracking server. See [Feature Store details](mlops-features/README.md).

### 2. Publish the shared pipelines

In notebook 02, set the required inputs at the top. The default namespace
`wsf-dev` names this deployment **Wealth ML SageMaker Foundation (WSF)**. It
publishes `wsf-dev-training` and `wsf-dev-inference`, with source, uploaded CSVs and
run outputs under `s3://<bucket>/wsf-dev/`. The existing `source-dev` pipelines and
objects are not updated. Namespace ownership checks still apply if `wsf-dev`
already exists. Registry group names remain model/environment based; offline
Registry mode does not write them.

For CSV mode, supply `csv_input` as a local file or exact S3 CSV URI and leave
`feature_store_config_uri` empty. Local relative paths resolve against this
checkout's root, even when it is a renamed subfolder inside another repository.
Use a fresh kernel for the new checkout; the notebooks stop if cached imports
point at another checkout. Keep the old notebooks on their own kernels.
Targets come from each selected model's `parameters.json` (`label` in this
example), not an additional notebook setting. The shared default CSV must contain
those targets with the exact spelling. The notebook prints the model-to-target
mapping and stops before publication if any are missing. Update the generic
parameters to match your use cases when porting; labels are not generated by this
check. The defaults are usecase1 → target_value and usecase2 → churned_flag.
Keep the initial options:

```python
data_source = "csv_direct"
registry_mode = "offline_registry"
batch_runtime = "processing"
cache_preparation = True
model_ids = None
```

Run all cells. They inspect the actual CSV/header/model targets, compile model Python
files for syntax, upload source and any local CSV, verify uploaded CSV bytes,
render definitions, publish and read back the saved definitions.
The preflight does not execute customer algorithms, validate every feature
requirement, or prove that the job role has the notebook role's access.

**Publication does not execute training.** The notebook saves definitions and
defaults. It prints pipeline names/versions, source/data locations and a local
`.runs/source-deploy/<namespace>/publication.json` receipt.

### 3. Execute training and prediction

1. In Studio, open the printed training pipeline and create an execution using
   the published version. Select **ModelId**, verify **CsvInputUri**, and keep
   **UseHPO=false** and **RegisterModel=false** for the initial development run.
2. After Train completes, obtain its exact artifact URI:
   `s3://<bucket>/<namespace>/models/<ModelId>/runs/<training-execution-id>/model/application/model.tar.gz`.
3. Start the inference pipeline with the **same ModelId** and that archive as
   **ModelDataUri**. Set **CsvInputUri** to the scoring CSV. It needs the saved
   model's input columns and identifiers, but does not need a training target.
4. Inspect `predictions/predictions.csv` under the inference execution's S3
   directory. Regression produces identifier/prediction; binary classification
   adds probability and applies the saved threshold.

Studio execution overrides can select a different CSV from the publication
default. Check the execution's effective parameters when diagnosing input data.
Notebook 02's CSV preflight is for publication/training; select an unlabeled
scoring CSV at inference execution.

`offline_registry` requires an exact artifact URI, either at execution or as an
explicit publication default. It neither scans S3 for the newest model nor
recovers silently from a Registry authorization error.

When Registry groups and permissions are ready, set
`registry_mode="model_registry"` and republish. Registration defaults on in that
mode. Training registers a pending candidate after evaluation; inference resolves
an eligible Completed, Approved version. Creating or naming a production group
does not implement promotion.

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

Registry group names default to **`<ModelId>-sm-<environment>`**, for example
`usecase1-sm-dev`. Platform provisioning owns their creation.

To add or rename a use case, copy or rename a complete directory under
`mlops-models/src/mlops_models/models/`, edit its parameters and any overrides,
then rerun notebook 02 with `model_ids=None` from a fresh kernel. Model IDs must
match `[a-z][a-z0-9]{0,23}`. The publisher discovers folders containing
`parameters.json` and updates the **same two pipelines** and dropdowns.

Renaming changes the model's S3 prefix and Registry group. It does not migrate
old artifacts or delete old resources. The old archive retains its original
model identity. Use a newly trained artifact for a newly named model.

The shared definition currently requires matching `tuning.parameters` and
`tuning.fixed` across all selected folders, even when UseHPO is false. Ordinary
algorithm settings and tasks can differ. Other common tuning defaults come from
the first selected model or the execution parameters.

## Pipeline and data contracts

```text
CSV or Feature Store
  -> GetData -> raw CSV on S3
  -> FeatureSelection -> selected-feature CSV + PDF on S3
  -> Preprocess(raw CSV, selected features) -> prepared CSVs on S3
  -> Train(prepared CSVs, selected features) -> model archive + reports on S3
  -> Evaluate -> optional Registry registration

Exact S3 model archive OR approved Registry model
  + scoring CSV or Feature Store
  -> inference Preprocess -> JSON-lines shards
  -> Score -> score files
  -> Postprocess -> predictions.csv + summary.json
```

Each step consumes a completed artifact from the previous producer. Feature
selection does not call GetData; preprocessing does not rerun feature selection.
Job definitions pass S3 locations to SageMaker, which downloads the inputs into
the container. The model functions read those local inputs.

| Step / model hook | Input | Output contract |
|---|---|---|
| GetData shared entrypoint | Explicit CSV or Feature Store binding | `raw/data.csv`, `raw/provenance.json` |
| `run_feature_selection` | Raw CSV directory and settings | `selection/selected_features.csv`, `selection/random_forest_plots.pdf` |
| `process_train` | Raw directory, selected-feature file and settings | `prepared/` containing one or more CSVs; demo writes `data.csv` |
| `train_model` | Prepared directory, selected-feature file, package/work paths and settings | Saved model directory with metadata and runtime code |
| Evaluation hook | Saved artifact and evaluation data | `evaluation.json`, metrics and model-specific reports |
| `process_inference` | Scoring snapshot and saved model settings | `shards/part-*.jsonl` |
| `predict_with_model` | Shards, extracted checkpoint and writable output | Identified score records in `*.out` files |
| `postprocess_predictions` | Score records and saved settings | `predictions.csv`, `summary.json`, effective `parameters.json` |

The demo selection CSV has a **`feature`** column. If customer code exports
`features` or another heading, update both its producer and all consumers.
Identifiers are read as strings to preserve leading zeroes. Feature names,
case, target, timestamp and expected types must match the model implementation.

CSV GetData copies every column and the input bytes unchanged; selection belongs
to the next step. The current Feature Store reader instead requests the fields
listed in the model's `features` settings, joins groups by identifier/event time,
and writes the resulting flat snapshot. Configure the complete candidate feature
set before expecting feature selection to consider it.

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
| [Publication builder](mlops-models/src/mlops_models/pipelines/source.py) | Shared ModelId pipelines, source uploads, data/Registry mode and image/job bindings |
| [Training handoffs](mlops-models/src/mlops_models/pipelines/training_files.py) | GetData/selection/preprocess graph, native input channels and preparation cache |
| [Definition loader](mlops-models/mlops_skeleton/pipelines.py) | Render underlying model templates/settings |
| [Dispatcher](mlops-models/src/mlops_models/entrypoints/__main__.py) | Parse step/model/path arguments and select the execute function |
| [Runtime loader](mlops-models/src/mlops_models/runtime/load_source.py) | Select supplied or retained source, writable outputs, archive extraction and S3 upload |
| [Job paths](mlops-models/src/mlops_models/runtime/job_paths.py) | Default container mount paths |
| [Model base](mlops-models/src/mlops_models/model_base) | Shared, replaceable selection/preprocessing/training/prediction implementations |
| [Use-case folders](mlops-models/src/mlops_models/models) | Per-model settings and function/class overrides |
| [Registry resolver](mlops-models/src/mlops_models/registry/model_package.py) | Approved package resolution and validation |
| [Inspection helpers](mlops-models/src/mlops_models/pipelines/inspect.py) | Existing execution/job/log/artifact inspection with explicit collection limits |
| [Feature package](mlops-features/README.md) | Sample generation, ingestion, delivery receipts and optional platform setup |
| [Release exporter](release/README.md) | Portable source inventory and notebook-output clearing |

`model_base/*/definition.json` files are templates, not the final published
graph. The publication builder transforms them into the shared pipelines.
Inspect the rendered definitions or publication receipt when tracing a run.

Use-case modules inherit corresponding base functions/classes. For example,
the use-case training wrapper passes its `UseCaseModelTrainer` into the base
`train_model` function. Model-base code does not select another use case.
Entry points load parameters from the selected model folder; the process
working directory is not the parameter directory.

## Local paths and S3 artifacts

The non-root compatibility workaround is local filesystem handling, not an IAM
permission grant. It preserves native input downloads and provides writable
scratch for outputs. The job role still needs permission and network access to
read/write the configured S3 objects.

| Job input | Default local location |
|---|---|
| CSV input for GetData / inference preprocessing | `/opt/ml/processing/input/` |
| Raw input for selection / training preprocessing | `/opt/ml/processing/raw/data.csv` |
| Selected features for training preprocessing | `/opt/ml/processing/selection/selected_features.csv` |
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
```

Inference uses its own execution directory, including its raw snapshot, shards,
scores and `predictions/predictions.csv`. Only the relevant step artifacts appear
in each run. The model archive contains the demo's `predictor/`, effective
`parameters.json`, `model_metadata.json`, `selected_features.csv`, evaluation
rows and retained runtime source under `code/`.

Evaluation and inference import that **retained model source**. Uploading a new
source folder alone does not change an existing archive's model logic or saved
settings. Keep the artifact, its model ID, code and preprocessing state together.

## Caching and iteration

Notebook 02 defaults to `cache_preparation=True`. In CSV mode it sets
`CacheConfig={"Enabled": True, "ExpireAfter": "PT24H"}` on GetData,
FeatureSelection and Preprocess. Start another execution normally; matching
successful preparation steps can reuse their outputs. Failed steps are not reused.

The preparation fingerprint includes the CSV identity and runtime package
files/settings, excluding existing `train.py` files in training/entrypoint
directories. Editing only those Train files preserves preparation eligibility.
Changes to CSV, preparation source, parameters, metadata or other package files
can invalidate it. Matching job settings and inputs are also required.

After editing source, rerun notebook 02 to upload it and republish before starting
the next execution. If input bytes change, republish to recompute the fingerprint;
do not mutate objects behind a previously published input URI and assume the old
cache signature detects that change. An explicit new input object is preferable.

Cache hits reference the producer's earlier S3 output locations. The current
execution may therefore have no new `data/` objects for reused steps. Downstream
jobs follow the recorded step outputs; retain those objects while they are needed.
Train, evaluation, registration and inference are not cached by this option.
Feature Store mode leaves preparation caching disabled.

Direct-deploy source prefixes permit iterative uploads. A changed set of source
filenames selects a new prefix to avoid stale renamed/deleted files; unchanged
filenames can be overwritten. This is not an immutable release system. Avoid
overwriting a source prefix still needed by an active job. The `ci-deploy` source
identity path supports a clean explicit commit; connecting it to the receiving
CI/CD process remains work to complete.

## Parameters and learned metadata

Each model folder currently contains **both** JSON files. There is no separate
`model_config.json` dependency in this delivery.

| File | Current contract |
|---|---|
| `parameters.json` | `task`, identifier/target/time columns, candidate features, selection/training settings, algorithm and tuning settings |
| `model_metadata.json` | Initially `{}`; loaded by the training entrypoint into `config["model_metadata"]` |

Existing demo parameter keys are retained; they do not prescribe the customer's
schema. `num_cpus`, for example, controls the demo fit's CPU use, not the
SageMaker instance type. Missing parameter/metadata files currently fail instead
of falling back to another model or default values.

The demo trainer saves effective parameters and newly generated metadata with
the checkpoint. It does **not** consume prior metadata to implement customer
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

- **Model quality:** the demo evaluator uses prepared training rows and labels
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
- **Native HPO:** keep UseHPO false until image startup, per-trial configuration,
  validation metric emission and model collection are exercised together.
  Ordinary Training's loader override does not establish tuning compatibility.
- **Serving:** default Processing inference needs no persistent inference
  endpoint. Batch Transform's serving path is separate; real-time endpoint
  deployment is not provided.
- **Registry:** offline mode is a development route. Approved-model inference
  requires precreated groups, permissions and approval of the actual candidate.
  Environment naming does not provide cross-environment promotion or rollback.
- **Container paths:** writable-output compatibility is specific to the image/job
  setup. Sandbox root-user success does not prove non-root target compatibility.
  Both actual filesystem probes and target job execution remain relevant.
- **Platform services:** these notebooks do not create IAM roles, tracking
  servers or a production deployment system. Notebook 01 can create Feature
  Groups only through its explicit opt-in route.
- **Source retention:** model archives carry runtime code. An archive made before
  an adapter/reader change needs a compatible replacement before it can use it.

Keep role responsibilities explicit:

| Identity / owner | Required responsibility |
|---|---|
| Notebook operator | Read supplied data; upload source/CSV; publish pipelines and pass the approved pipeline role |
| Pipeline role | Orchestrate supported jobs and pass approved job roles |
| Job role | Read job inputs and artifacts; write execution outputs; query Feature Store services or Registry when those modes are enabled |
| Feature Store writer role | Service-side offline storage/catalog/encryption access required by the platform |
| Platform provisioning owner | Create approved groups, tracking/storage/network resources and scoped permissions |

These are responsibilities, not a requirement for five newly created roles.
Notebook access does not prove job access. Hard network isolation and private VPC
networking are different choices; jobs still require the service connectivity
needed by their selected mode. Platform owners bind exact resources and roles.

## Validation

The October 15 generic baseline has:

- 68 source tests and 20 notebook scenarios, including explicit-input failures
  and CSV preparation without AWS access.
- Local AutoGluon training/evaluation and Processing/HTTP prediction checks.
- Six successful managed sandbox executions: training and offline inference
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
| First model acceptance | Initial training without learned metadata; inspect generated metadata and predict from the exact model | Establish Registry/Feature Store/tracking access and agreed container changes |
| Metadata reuse | Explicit prior-result input, final training and compatible prediction; repeat for the second model | Exercise native HPO startup and artifact collection |
| Workshop | Demonstrate supported modes, collect usability/model feedback and agree remaining acceptance | Decide provisioning, build/release, promotion, manual steps and operating ownership |
| Handoff | Close agreed tests, publish operating guidance and demonstrate use by another engineer | Prove the agreed monitoring/release/recovery scope and record remaining gates |

The first functional extension is the metadata handoff. Likely edits are
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
   Evaluation reusing the model prediction function is intentional.
4. Preserve the current callable signatures unless all entrypoints, use-case
   wrappers and callers are changed together. Parameters after `*`, such as
   `trainer_cls`, are keyword-only.
5. Read parameters through the selected model folder and pass resolved paths.
   Do not use the process working directory as the configuration location.
6. Change CSV headings, selected features, checkpoint layout and metadata
   producers/readers together. Keep model_dir separate from fitting outputs.
7. Preserve explicit CSV/artifact selection and fail visible required-input
   checks. Do not introduce automatic example-data or Registry-error fallbacks.
8. Validate the changed behavior, refresh status/limitations here and retain the
   baseline-versus-adapter diff for review.

Optional diagnostics and older notebooks are in `scratch/` in the development
checkout. The exporter excludes them, generated work, Git history and notebook
outputs; the delivery has only the two active notebooks. See
[source export](release/README.md) before creating a package. The exporter
includes current working files, so retain the actual source revision and archive
checksum with acceptance evidence.
