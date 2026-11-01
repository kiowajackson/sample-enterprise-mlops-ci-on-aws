# Paste immediately before: from mlops_models.frameworks import get_framework
import importlib
import sys
from pathlib import Path

print("Executing file:", globals().get("__file__", "<notebook cell>"), flush=True)
print("Python search path:", sys.path, flush=True)
for name in ("mlops_models", "mlops_models.frameworks"):
    print(f"\n{name} — already imported: {name in sys.modules}", flush=True)
    package = importlib.import_module(name)
    print("  Loaded file:", getattr(package, "__file__", None), flush=True)
    print("  Import origin:", getattr(package.__spec__, "origin", None), flush=True)
    print("  Import loader:", type(package.__loader__).__name__, flush=True)
    for directory in getattr(package, "__path__", []):
        folder = Path(directory)
        print("  Package directory:", folder, flush=True)
        print("  Exact __init__.py exists:", (folder / "__init__.py").is_file(), flush=True)
        if name == "mlops_models.frameworks":
            print("  Filenames:", sorted(p.name for p in folder.iterdir())
                  if folder.is_dir() else "<directory missing>", flush=True)
    if name == "mlops_models.frameworks":
        print("  get_framework is callable:",
              callable(getattr(package, "get_framework", None)), flush=True)
