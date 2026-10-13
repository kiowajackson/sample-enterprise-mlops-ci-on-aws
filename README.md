# Debugging replacements for the reviewed handoff

The full source matching the handoff ZIP lives in GitLab, on branch
`feature/reviewed-training-inference-handoff` targeting `handoff/adoption`.
This GitHub `dev` branch contains only replacement files for that ZIP.
The previous contents of `dev` are preserved on `dev_old`.

Baseline: `models-features-single-container-2026-10-13-training-inference-review.zip`

SHA-256: `973bc50ad256bf1de2de4f711fe0ef32544b17071eb6b3abb09a11d8a14c4de0`

## Current fix: repeated shared keys in metadata

The sample CSV contains the configured record ID and event time once. Every
Feature Store group receives those same keys, even when the metadata lists them
for each source table. Other duplicate feature names still produce an error.

To fix the current notebook error:

1. Copy `mlops-features/mlops_features/metadata.py` to the same relative path in
   the extracted full checkout.
2. In 04A, set `record_id` to the shared identifier in your schema (for example,
   `ocif_id`) and keep `event_time` aligned with the intended timestamp field.
3. Restart the notebook kernel so it reloads the module, then rerun 04A.

The updated 04A notebook changes explanatory text only. You can retain the notebook
and inputs you already filled in. The Features README and regression test document
and verify the fix. This root README describes the patch branch; it is not a
replacement for the full checkout's README.

The generator uses unique synthetic IDs rather than repeating reference IDs.
Model configuration must use the same identifier and feature names as the data.

Validation: 51 tests passed. Actual 04A cells with simulated AWS and local SQL
produced 100 unique IDs, one shared-key column, and the same key in all three
feature groups, with exact readback. Target-account validation is still required.
