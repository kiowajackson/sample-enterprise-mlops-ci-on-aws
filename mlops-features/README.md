# Feature Store setup, ingestion and pipeline reads

Features owns sample generation, ingestion and verification of offline deliveries.
It publishes storage configuration and completed-delivery receipts. Models reads
that contract using its own Feature Store reader; it does not import this package.
Scheduling and upstream business feature calculations remain outside this application.

## Normal workflow

1. Run [04A](../examples/notebooks/04A_metadata_to_feature_store.ipynb) with a bucket
   and exact offline-writer role ARN. Leave `metadata_csv` empty for the included
   schema matching both example models, or supply a metadata CSV.
2. Review the generated mapping, create/reuse the offline groups, and ingest the
   sample. The notebook prints `feature_store_config_uri`. Use its final readback
   to confirm the initial delivery is visible before starting the model pipeline.
3. Paste that URI into [03Fv2](../examples/notebooks/03Fv2_run_from_source.ipynb),
   publish both pipelines, then start them from Studio.

Each execution resolves one completed ingestion, queries the model's configured
fields through Athena, joins them by identifier/event time, and retains:

```text
<OutputDataRoot>/<ModelId>/runs/<execution-id>/data/raw/
  data.csv
  provenance.json
```

Training's GetData step publishes the snapshot to S3. FeatureSelection consumes
it and publishes its selection; Preprocess consumes both outputs and publishes
prepared CSVs. Train receives the prepared CSVs and selected-feature CSV as separate
input channels. Batch Preprocess creates scoring shards from its own snapshot.
Later stages consume those outputs without querying the store
again. A notebook does not need to stay open or export data before another run.
The existing source loader uploads these outputs, including in non-root images
that require the application's writable scratch directory.

`FeatureStoreDeliveryId` can select a recorded delivery ID. Empty selects the
most recently published completion receipt under the configured delivery prefix;
it does not mean the greatest business event date. Completion is recorded only
after all mapped groups accept their rows. The model job assumes those rows are
already visible offline and executes one query per required group. It does not
poll ingestion status or compare returned counts against the receipt. Initial
readback belongs to setup, not every training run. Subsequent ingestion with
the same schema needs no pipeline republish.

## Sample schema and labels

The metadata columns are `column_name`, `column_map`, `table_nm`, and `ref_val1`
through `ref_val5`; optional `type`/`length` values are retained as metadata.
`column_name` becomes the CSV field; `column_map` preserves the source mapping.
Rows cycle through the five reference columns, with unique synthetic IDs and a
fixed event date. Text, leading zeros, empty fields and literal `NA` survive.
These samples test data movement, not realistic relationships or model quality.

Storage suggestions come from the reference values. Review `type_overrides`
before creating groups; numeric type codes in supplied metadata are not assumed
to have a particular meaning. One source table becomes one group, split above
2,497 mapped fields to leave space for the two keys and delivery ID. Source and
column mappings are retained. Existing groups are reused only with matching
schema, keys, writer role and storage.

The included `examples/data/feature-metadata.csv` contains both example targets.
Your metadata may contain features only. The generator does not invent missing
labels. Training requires either the configured target in the store or
`UniverseDataRoot` containing `<ModelId>/universe.csv` with exactly ID/date/label.
For scoring, that CSV needs ID/date only; an optional label is ignored. Empty
`UniverseDataRoot` on inference scores all rows in the selected delivery.

Population joins require unique, matching ID/date pairs. Synthetic IDs will not
match a real customer population. Port model config feature names, identifier,
event time and label to the delivered schema before using it.

## Code responsibilities

| File | Responsibility |
|---|---|
| `mlops_features/metadata.py` | Generate the sample and table/field mapping |
| `mlops_features/feature_mapping.py` | Validate mapping and group definitions |
| `mlops_features/data.py` | Ingest deliveries; query offline groups and join by keys |
| `data.py` | Command-line entrypoint for the same ingestion/query code |

Models' GetData entrypoint defines its Feature Store query and input/output functions
locally. Batch preprocessing contains its own copy. They consume this application's
published configuration and receipts without importing its Python code. The Models publisher uploads only its own
package. Training retains that package in the model archive, so batch preprocessing
uses the reader and model hooks saved with the approved model. Train and
approve an updated model before enabling this path for inference; older archives
may not contain the reader. No runtime code is generated by a notebook.

## Permissions and network access

04A's caller needs Feature Store create/describe/PutRecord, PassRole and S3 access.
The supplied offline writer role needs SageMaker trust, offline S3 writes and
catalog/KMS permissions required by the platform. The notebook creates no IAM
roles, buckets, workgroups or permission grants.

Prepare/Preprocess run as the **job role**. They need:

- `sagemaker:DescribeFeatureGroup` for the configured groups.
- Athena `StartQueryExecution`, `GetQueryExecution`, `StopQueryExecution` on the workgroup.
- Glue database/table/partition read permissions, plus Lake Formation access when enforced.
- S3 list/read for configuration, completed receipts, offline data, query results and optional populations; write for query results and execution outputs.
- KMS access for the corresponding encrypted objects, when configured.

The role does not need ingestion or group-creation permissions to read features.
Notebook access does not prove job-role access. The existing `primary` Athena
workgroup is the default; use the platform workgroup if required. Jobs need
network access to these service endpoints. Hard network isolation is rejected for
Feature Store mode; private VPC routing must provide the required connectivity.

## Reproducibility and limits

A delivery carries its source/mapping metadata and completion cutoff. The model
reader uses that delivery and cutoff to select the latest available nondeleted
row per ID/date and joins feature groups by those keys. It assumes the groups
have been populated with the intended aligned delivery.

Each model run retains its raw CSV plus a short provenance file containing the
binding, receipt, query IDs/SQL/parameters, model ID, selected features, row count
and optional population URI. The model runtime no longer calculates CSV
checksums or scans for ingestion completeness. Feature ingestion and its optional
readback retain their own validation. Live MLflow publication is separate.

The current contract is one complete aligned delivery, possibly containing
multiple dates. Independently refreshed tables and historical point-in-time
selection need an agreed business contract. The joined dataframe is held in
memory; small wide-data checks do not establish production row-scale capacity.

## Optional platform/CDK path

The existing CDK app remains available. The notebook path does not require it.
From a CDK-provisioned deployment, the CLI still supports:

```bash
python mlops-features/data.py --outputs .runs/features-outputs.json ingest \
  --input examples/data/curated_fake.csv --receipt .runs/ingestion.json

python mlops-features/data.py --outputs .runs/features-outputs.json read \
  --receipt .runs/ingestion.json --output .runs/extracted-features
```

Use the mapping that matches the deployed bindings. The standalone `read` command
can still publish an extraction; the shared pipeline path performs its own read.
Retry an interrupted ingestion with the same delivery ID and unchanged source;
changed bytes or definitions require a new ID. Keep the prior mapping, receipts
and snapshots needed by registered models when migrating or retiring groups.

### Shared identifiers in supplied metadata

Set `record_id` and `event_time` in 04A to the shared key names. Source metadata
may repeat these columns for every table. The generator includes each key once
in the flat sample and excludes those rows from the ordinary feature mapping.
Each Feature Store group still receives both keys. Dummy IDs are unique and
event times use the configured fixed date; reference ID/date values are not reused.
Other duplicate feature names remain ambiguous and must have distinct column names.
