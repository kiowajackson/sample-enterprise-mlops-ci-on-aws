"""Examples of overriding the training step and its trainer class."""

from mlops_models.frameworks.autogluon import gluonTraining
from mlops_models.model_base.training.train import train_model as base_train_model


class ModelTraining(gluonTraining):
    def train_model(self, *args, **kwargs):
        print("usecase1: running the custom trainer", flush=True)
        return super().train_model(*args, **kwargs)


def train_model(*args, **kwargs):
    print("usecase1: running the custom training step", flush=True)
    return base_train_model(*args, **kwargs)
