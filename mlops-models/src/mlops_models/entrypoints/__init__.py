"""Select named functions and model classes for the existing entrypoints."""

from importlib import import_module
from pathlib import Path


_BASE_DEFAULT = object()


def load_step(model_id, module_name, name, *, default=_BASE_DEFAULT):
    """Use a named use-case override; otherwise use the supplied default or base."""
    models = Path(__file__).resolve().parents[1] / "models"
    if not model_id.isidentifier() or not (models / model_id / "parameters.json").is_file():
        raise ValueError(f"Unknown model: {model_id!r}; expected models/<model>/parameters.json.")
    prefix = f"mlops_models.models.{model_id}"
    module_path = f"{prefix}.{module_name}"
    try:
        module = import_module(module_path)
    except ModuleNotFoundError as error:
        # An optional module may be absent; a dependency broken inside it is an error.
        if error.name not in {prefix, f"{prefix}.{module_name.split('.')[0]}", module_path}:
            raise
    else:
        if hasattr(module, name):
            return getattr(module, name)
    # A class override can default to None: the base then selects the configured framework.
    if default is not _BASE_DEFAULT:
        return default
    return getattr(import_module(f"mlops_models.model_base.{module_name}"), name)
