"""Inherit the training function and class, or replace them for this model."""

from mlops_models.model_base.training.train import ModelTrainer, train_model as base_train_model


class UseCaseModelTrainer(ModelTrainer):
    """Override train_model to return this model's saved checkpoint, metadata and performance."""


def train_model(data_directory, selection_file, model_dir, output_directory, config, parameter_path,
                *, hyperparameters=None):
    return base_train_model(
        data_directory, selection_file, model_dir, output_directory, config, parameter_path,
        trainer_cls=UseCaseModelTrainer, hyperparameters=hyperparameters,
    )
