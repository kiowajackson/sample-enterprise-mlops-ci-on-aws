"""Named trainer/predictor pairs; import only the selected framework."""

from importlib import import_module


FRAMEWORKS = {
    "autogluon": ("mlops_models.frameworks.autogluon", "gluonTraining", "gluonPredictor"),
}


def get_framework(name="autogluon"):
    """Return the configured trainer and predictor classes."""
    if name not in FRAMEWORKS:
        raise ValueError(f"Unknown framework {name!r}. Available: {', '.join(sorted(FRAMEWORKS))}")
    module_name, trainer, predictor = FRAMEWORKS[name]
    module = import_module(module_name)
    return getattr(module, trainer), getattr(module, predictor)
