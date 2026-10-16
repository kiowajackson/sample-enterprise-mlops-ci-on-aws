# Paste immediately before shap.TreeExplainer(trained_model, ...).
# trained_model is the existing xgboost.core.Booster variable at that call site.
# Run in the failing job's environment, not an unrelated notebook kernel.
# This prints dependency versions and selected model configuration fields only.
# It does not modify the model, install packages, or run SHAP.
#
# Matching upstream issue and fix:
# https://github.com/shap/shap/issues/4202
# https://github.com/shap/shap/releases/tag/v0.50.0
# XGBoost format change:
# https://github.com/dmlc/xgboost/blob/v3.1.0/doc/changes/v3.1.0.rst

import json
import sys
from importlib.metadata import PackageNotFoundError, version

_shap_check_versions = {}
for _shap_check_package in ("shap", "xgboost", "autogluon.tabular", "numpy"):
    try:
        _shap_check_versions[_shap_check_package] = version(_shap_check_package)
    except PackageNotFoundError:
        _shap_check_versions[_shap_check_package] = "distribution not found"

_shap_check_config = json.loads(trained_model.save_config())
_shap_check_learner = _shap_check_config["learner"]
_shap_check_parameters = _shap_check_learner["learner_model_param"]
_shap_check_base = _shap_check_parameters["base_score"]
print("SHAP diagnostic Python:", sys.version, flush=True)
print("SHAP diagnostic packages:", json.dumps(_shap_check_versions), flush=True)
print("SHAP diagnostic model type:", type(trained_model), flush=True)
print("SHAP diagnostic base_score:", repr(_shap_check_base),
      "| type:", type(_shap_check_base).__name__, flush=True)
print("SHAP diagnostic objective:", _shap_check_learner["objective"]["name"],
      "| num_class:", _shap_check_parameters.get("num_class"),
      "| num_target:", _shap_check_parameters.get("num_target"), flush=True)
