"""Inherit file prediction for Processing and Transform, or override it here."""

from mlops_models.model_base.inference.predict import predict_with_model as base_predict_with_model


def predict_with_model(input_directory, checkpoint_path, output_directory, config, parameter_path):
    """Forward the prepared data, registered model and separate settings."""
    return base_predict_with_model(input_directory, checkpoint_path, output_directory, config, parameter_path)
