"""Paste immediately before the failing trained_model.save_config() call.

Run in the same process and scope where trained_model is already loaded.
This prints the actual model class, library and available methods.
"""

import importlib

print("Model class:", type(trained_model), flush=True)
_model_library = importlib.import_module(type(trained_model).__module__.split(".")[0])
print("Library version:", getattr(_model_library, "__version__", "unknown"), flush=True)
print("Library file:", getattr(_model_library, "__file__", "unknown"), flush=True)
print("Has save_config:", callable(getattr(trained_model, "save_config", None)), flush=True)
print("Has save_model:", callable(getattr(trained_model, "save_model", None)), flush=True)
