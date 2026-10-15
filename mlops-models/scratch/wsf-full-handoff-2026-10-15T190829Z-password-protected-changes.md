# Required Registry handoff

Source commit: `17a39de55e1fe7e6b44318e5fd0e1274ee87c3b3`. Complete archive: `wsf-full-handoff-2026-10-15T190829Z-password-protected.zip`. Password: `pass`.

Use the complete archive in a fresh folder. Preserve customer model code and configuration in the receiving repository; this is the generic baseline. The supplied notebooks contain blank connection inputs and no captured outputs.

Training must register into an existing group; inference resolves an approved version. There is no direct-artifact Registry bypass. RegisterModel uses a Processing job to call the API without creating groups. Its job role needs DescribeModelPackageGroup, CreateModelPackage and DescribeModelPackage; ResolveModel also needs ListModelPackages. Compute caching remains enabled, while registration and approval lookup always run.

Before notebook 02, ensure model folder names and the environment match the existing `<ModelId>-sm-<environment>` groups. Preserve and repaste your input cell, then republish both pipelines.

## Files changed in this Registry increment

- `README.md`
- `mlops-models/mlops_skeleton/pipelines.py`
- `mlops-models/scratch/registry_access.py`
- `mlops-models/src/mlops_models/entrypoints/__main__.py`
- `mlops-models/src/mlops_models/model_base/inference/definition.json`
- `mlops-models/src/mlops_models/model_base/training/definition.json`
- `mlops-models/src/mlops_models/pipelines/source.py`
- `mlops-models/src/mlops_models/registry/model_package.py`
- `notebooks/02_deploy_pipelines.ipynb`
- `tests/test_model_handoff.py`
- `tests/test_pipeline_cache.py`
- `tests/test_source_deployment.py`

## Differences from the previous full ZIP on dev (T145812Z)

- added: `mlops-models/scratch/cell.py`
- added: `mlops-models/scratch/registry_access.py`
- added: `tests/test_pipeline_cache.py`
- modified: `README.md`
- modified: `mlops-models/mlops_skeleton/pipelines.py`
- modified: `mlops-models/src/mlops_models/entrypoints/__main__.py`
- modified: `mlops-models/src/mlops_models/entrypoints/evaluate.py`
- modified: `mlops-models/src/mlops_models/model_base/inference/definition.json`
- modified: `mlops-models/src/mlops_models/model_base/training/definition.json`
- modified: `mlops-models/src/mlops_models/pipelines/source.py`
- modified: `mlops-models/src/mlops_models/pipelines/training_files.py`
- modified: `mlops-models/src/mlops_models/registry/model_package.py`
- modified: `mlops-models/src/mlops_models/runtime/load_source.py`
- modified: `notebooks/02_deploy_pipelines.ipynb`
- modified: `tests/test_model_handoff.py`
- modified: `tests/test_preparation_cache.py`
- modified: `tests/test_source_deployment.py`

The second list also includes the earlier cache improvements and diagnostics. The shared model training, feature-selection, preprocessing and prediction Python files are unchanged by this Registry increment.
