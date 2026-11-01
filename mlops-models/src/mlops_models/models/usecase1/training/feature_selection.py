"""Example fit override; the shared selection function stays in the base."""

from mlops_models.model_base.training.feature_selection import FeatureSelector as BaseSelector


class FeatureSelector(BaseSelector):
    def fit(self, *args, **kwargs):
        print("usecase1: running custom feature selection", flush=True)
        return super().fit(*args, **kwargs)
