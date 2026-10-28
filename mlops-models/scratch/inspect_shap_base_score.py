"""Paste immediately before shap.TreeExplainer(trained_model, ...).

Run in the same process and scope where trained_model is already loaded.
Print library versions and the XGBoost base_score exactly as stored.
This only inspects the model; it does not change settings or run explanations.
"""

import json
import shap
import xgboost

print("SHAP version:", shap.__version__, flush=True)
print("SHAP file:", shap.__file__, flush=True)
print("XGBoost version:", xgboost.__version__, flush=True)
print("XGBoost file:", xgboost.__file__, flush=True)
print("Model class:", type(trained_model), flush=True)

if isinstance(trained_model, xgboost.Booster):
    if callable(getattr(trained_model, "save_config", None)):
        _shap_booster_config = json.loads(trained_model.save_config())
        _shap_base_score = _shap_booster_config["learner"]["learner_model_param"]["base_score"]
        print("base_score:", repr(_shap_base_score), flush=True)
        print("base_score type:", type(_shap_base_score).__name__, flush=True)
    else:
        print("This XGBoost Booster does not expose save_config().", flush=True)
else:
    print("Skipping XGBoost config: trained_model is not an xgboost.Booster.", flush=True)
