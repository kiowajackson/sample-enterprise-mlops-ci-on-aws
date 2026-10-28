# TEMPORARY workaround for SHAP 0.49.1 with XGBoost's vector base_score format.
# Paste this entire block immediately before the existing TreeExplainer call.
# Keep the existing TreeExplainer and shap_values calls below it.
# Only an actual XGBoost Booster activates this fix; other models are skipped.
#
# This changes SHAP's in-memory decoder only. It does not change the Booster,
# model files, installed packages, or input data. No root access is needed.
# Supports single-output models only; never discards a multi-output intercept.
# Remove this block once the platform supplies a compatible, tested image.
# Upstream fix: https://github.com/shap/shap/pull/4187 (released in 0.50.0).

def enable_shap_xgboost_scalar_base_score_workaround(model):
    from xgboost import Booster

    if not isinstance(model, Booster):
        return

    import json
    import math
    from importlib.metadata import version

    import shap.explainers._tree as shap_tree

    installed = version("shap")
    if installed != "0.49.1":
        raise RuntimeError(
            f"This temporary workaround is only for SHAP 0.49.1; found {installed}. "
            "Remove it and validate the updated container's explanation support."
        )

    original_decoder = shap_tree.decode_ubjson_buffer
    if getattr(original_decoder, "_wsf_scalar_base_score_workaround", False):
        return

    def decode_with_scalar_base_score(buffer):
        decoded = original_decoder(buffer)
        parameters = decoded.get("learner", {}).get("learner_model_param", {})
        raw = parameters.get("base_score")
        if not isinstance(raw, str) or not raw.lstrip().startswith("["):
            return decoded

        values = json.loads(raw)
        if (
            not isinstance(values, list)
            or len(values) != 1
            or int(parameters.get("num_target", 1)) != 1
            or int(parameters.get("num_class", 0)) > 1
        ):
            raise ValueError(
                "The temporary SHAP workaround supports only a single-output "
                "XGBoost model with one intercept. Use a compatible SHAP image "
                "for multiclass or multi-output explanations."
            )
        value = values[0]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("XGBoost base_score must contain one finite number.")

        # Only the newly decoded dictionary changes. SHAP keeps its existing
        # objective/link conversion; the original Booster remains untouched.
        parameters["base_score"] = str(float(value))
        return decoded

    decode_with_scalar_base_score._wsf_scalar_base_score_workaround = True
    shap_tree.decode_ubjson_buffer = decode_with_scalar_base_score
    print("Enabled temporary SHAP 0.49.1/XGBoost scalar-intercept workaround.", flush=True)


enable_shap_xgboost_scalar_base_score_workaround(trained_model)
