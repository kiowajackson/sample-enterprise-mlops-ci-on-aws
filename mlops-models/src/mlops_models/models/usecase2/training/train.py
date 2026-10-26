"""Inherit the training function and class, or replace them for this model."""

from mlops_models.model_base.training.train import ModelTraining, train_model as base_train_model


class UseCaseModelTraining(ModelTraining):
    """Override train_model to save the model/reports and return checkpoint and metadata."""


def train_model(data_directory, selection_file, model_dir, output_directory, parameter_path,
                *, hyperparameters=None):
    return base_train_model(
        data_directory, selection_file, model_dir, output_directory, parameter_path,
        trainer_cls=UseCaseModelTraining, hyperparameters=hyperparameters,
    )
