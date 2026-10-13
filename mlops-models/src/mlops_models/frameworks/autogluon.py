"""Fit, save, load and predict using AutoGluon Tabular."""

from .inputs import PreparedTrainingData
from .results import FitResult


class Framework:
    def fit(self, data: PreparedTrainingData, config, checkpoint_path) -> FitResult:
        """Take prepared train/validation data and settings; return the fitted AutoGluon model and its name.

        - Combine transformed features and labels into AutoGluon dataframes.
        - Fit the configured random forest using the supplied validation rows.
        - Return FitResult without adding bagging, stacking or a final refit.
        """
        from autogluon.features.generators import IdentityFeatureGenerator
        from autogluon.tabular import TabularPredictor

        # AutoGluon expects labels in the same dataframe as the transformed features.
        train_frame, validation_frame = data.X_train.copy(), data.X_validation.copy()
        train_frame[config["label"]] = data.y_train
        validation_frame[config["label"]] = data.y_validation
        parameters = config["parameters"]["autogluon"][config["task"]]
        classification = config["task"] == "binary_classification"
        predictor = TabularPredictor(
            label=config["label"], problem_type="binary" if classification else "regression",
            eval_metric="roc_auc" if classification else "root_mean_squared_error",
            path=str(checkpoint_path / "predictor"), verbosity=2,
            # AutoGluon 1.2 requires the positive-class setting through its learner.
            learner_kwargs={"cache_data": False, **({"positive_class": 1} if classification else {})},
        )

        # Fit the configured random forest without adding bagging or stacked models.
        predictor.fit(
            train_data=train_frame, tuning_data=validation_frame,
            time_limit=config["time_limit_seconds"],
            hyperparameters={"RF": {**parameters, "random_state": config["seed"]}},
            num_cpus=config["num_cpus"], num_gpus=0,
            num_bag_folds=0, num_stack_levels=0, fit_weighted_ensemble=False, refit_full=False,
            # Supported by both AutoGluon 1.2 and 1.4; fail if the configured RF cannot fit.
            feature_generator=IdentityFeatureGenerator(), raise_on_no_models_fitted=True,
        )
        return FitResult(predictor, predictor.model_best)

    def save(self, result, checkpoint_path):
        """Save the fitted AutoGluon predictor at its existing checkpoint location."""
        result.predictor.save()

    def load(self, checkpoint_path, config):
        """Take a checkpoint directory and return its saved AutoGluon predictor."""
        from autogluon.tabular import TabularPredictor

        return TabularPredictor.load(str(checkpoint_path / "predictor"))

    def predict(self, predictor, values, config):
        """Take a fitted predictor and feature rows; return probabilities or regression values."""
        if config["task"] == "regression":
            return predictor.predict(values).to_numpy()
        return predictor.predict_proba(values, as_multiclass=True)[1].to_numpy()
