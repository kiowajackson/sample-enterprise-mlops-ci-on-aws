# Shared notebook config and Registry prefix handoff

Source commit: `00011f94548bcd29e29f42d0ac301eef441e2b3f`. Complete archive: `wsf-full-handoff-2026-10-15T195327Z-password-protected.zip`. Password: `pass`.

Copy notebooks/config.example.json to notebooks/config.json and fill in your values once. Both notebooks read the same file. Notebook 01 saves the generated csv_input or feature_store_config_uri; notebook 02 picks it up without any cell edits. Existing config.json is preserved, Git-ignored and excluded from ZIP exports. New optional settings use the template defaults.

Set registry_group_prefix including its trailing separator. The exact group name is <prefix><ModelId>-sm-<environment>; preflight, registration and inference use the same name. Keep lowercase model folders and the existing groups. Re-run notebook 02 to republish.

## All files changed since the previous T190829Z ZIP

- `.gitignore`
- `README.md`
- `mlops-models/src/mlops_models/pipelines/source.py`
- `notebooks/01_prepare_data.ipynb`
- `notebooks/02_deploy_pipelines.ipynb`
- `notebooks/config.example.json`
- `release/export.py`
- `tests/test_model_handoff.py`
- `tests/test_notebook_config.py`

The archive is the full project, including both notebooks. Preserve customer model ports and your populated config.json when updating. Model algorithm files are unchanged. 92 source tests and 25 simulated-AWS notebook cases passed; this increment did not launch managed jobs.
