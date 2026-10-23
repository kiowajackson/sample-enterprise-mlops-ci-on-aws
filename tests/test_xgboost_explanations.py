"""Validate public XGBoost contributions with the pinned model runtime."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
import shap
import xgboost


class XGBoostExplanations(unittest.TestCase):
    def test_saved_single_output_models_explain_the_same_scoring_margin(self):
        rng = np.random.default_rng(17)
        frame = pd.DataFrame(rng.normal(size=(96, 7)).astype("float32"),
                             columns=[f"feature_{i}" for i in range(7)])
        for objective in ("binary:logistic", "reg:squarederror"):
            with self.subTest(objective=objective), TemporaryDirectory() as folder:
                target = ((frame.iloc[:, 0] + 0.4 * frame.iloc[:, 1] > 0.2).astype("float32")
                          if objective == "binary:logistic"
                          else 2.0 + frame.iloc[:, 0] - 0.5 * frame.iloc[:, 2])
                matrix = xgboost.DMatrix(frame, label=target)
                model = xgboost.train(
                    {"objective": objective, "max_depth": 3, "seed": 17, "nthread": 1},
                    matrix, num_boost_round=12,
                )
                path = Path(folder) / "model.json"
                model.save_model(path)
                model = xgboost.Booster(model_file=path)
                before = bytes(model.save_raw(raw_format="ubj"))
                # Both ordinary scoring and a selected iteration range must agree.
                for iterations in ((0, 0), (0, 5)):
                    predictions = model.predict(matrix, iteration_range=iterations)
                    margin = model.predict(matrix, output_margin=True, iteration_range=iterations)
                    for approximate in (False, True):
                        contributions = model.predict(
                            matrix, pred_contribs=True, approx_contribs=approximate,
                            iteration_range=iterations,
                        )
                        self.assertEqual(contributions.shape, (len(frame), frame.shape[1] + 1))
                        np.testing.assert_allclose(contributions.sum(axis=1), margin, atol=2e-6)
                        explanation = shap.Explanation(
                            values=contributions[:, :-1], base_values=contributions[:, -1],
                            data=frame.to_numpy(), feature_names=list(frame.columns),
                        )
                        np.testing.assert_allclose(
                            explanation.values.sum(axis=1) + explanation.base_values, margin, atol=2e-6)
                    np.testing.assert_array_equal(
                        model.predict(matrix, iteration_range=iterations), predictions)
                self.assertEqual(bytes(model.save_raw(raw_format="ubj")), before)
                config = json.loads(model.save_config())
                self.assertTrue(config["learner"]["learner_model_param"]["base_score"].startswith("["))
