# Copy these blocks into your existing train_model function at the marked places.
# They execute directly: no helper definitions, imports from scratch, or extra calls.

# BLOCK 1: First lines inside train_model, before training or path reassignment.
import os
from pathlib import Path

package_dir = Path(model_dir)
print("TRAIN ENTRY", {
    "source": __file__,
    "received_model_dir": str(model_dir),
    "loader_model_dir": os.environ.get("SM_MODEL_DIR"),
    "output_directory": str(output_directory),
}, flush=True)
# END BLOCK 1


# BLOCK 2: Immediately before your existing write_container_output call.
# Keep package_dir captured by Block 1; do not assign it again here.
import os
from pathlib import Path

for debug_name, debug_value in {
    "model_dir": model_dir,
    "package_dir": package_dir,
    "output_directory": output_directory,
}.items():
    debug_path = Path(debug_value)
    print(debug_name, str(debug_path),
          sorted(p.name for p in debug_path.iterdir())[:20]
          if debug_path.is_dir() else "<not a directory>",
          flush=True)

debug_expected_package_dir = Path(os.environ["SM_MODEL_DIR"])
assert Path(package_dir).resolve() == debug_expected_package_dir.resolve(), (
    f"Wrong package destination: {package_dir}; loader expects {debug_expected_package_dir}"
)
# END BLOCK 2
