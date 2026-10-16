"""Inherit the model loader/predictor, or replace them with this model's code."""

from mlops_models.model_base.inference.predict import ModelPredictor, predict_model
from mlops_models.model_base.inference.predict import predict_with_model as base_predict_with_model


class UseCaseModelPredictor(ModelPredictor):
    """Override loading or prediction for native Transform and local evaluation."""


def use_case_predict_model(df, checkpoint_path, *, output_dir=None):
    return predict_model(df, checkpoint_path, predictor_cls=UseCaseModelPredictor,
                         output_dir=output_dir)


def predict_with_model(input_directory, checkpoint_path, output_directory, config, parameter_path):
    """Inherit shared Processing prediction; override this function for this use case."""
    return base_predict_with_model(input_directory, checkpoint_path, output_directory, config, parameter_path)
