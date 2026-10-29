"""Debug local post-processing using paths prepared by notebook 03."""

import json
import os
from pathlib import Path
import sys

# Paste the paths.json location printed by notebook 03, then use Run/Debug Python File.
INPUTS_FILE = ""


def main():
    if not INPUTS_FILE:
        raise ValueError("Set INPUTS_FILE to the paths.json prepared by notebook 03.")
    inputs = json.loads(Path(INPUTS_FILE).expanduser().read_text())
    root = next(p for p in Path(__file__).resolve().parents
                if (p / "mlops-models/src/mlops_models").is_dir())
    sys.path[:0] = [str(root / "mlops-models/src"), str(root / "mlops-models")]
    os.environ.update({
        "MODEL_ID": inputs["model_id"],
        "MLOPS_MODEL_DIR": inputs["model_directory"],
        "SM_MODEL_DIR": inputs["model_directory"],
        "SM_OUTPUT_DATA_DIR": inputs["output_directory"],
        "SM_OUTPUT_DIR": inputs["output_directory"],
        "MLOPS_ANALYSIS_MODE": inputs["analysis_mode"],
        "MLOPS_PREDICTION_FORMAT": "csv",
        "MLOPS_TRACKING_URI": "",
        "MLOPS_WRITE_RUN_REPORT": "0",
        "MLOPS_LOG_ARTIFACTS": "0",
    })
    from mlops_models.model_base.inference.postprocess import postprocess_predictions

    # Set a breakpoint here, then step into the local model code.
    postprocess_predictions(
        Path(inputs["data_directory"]),
        Path(inputs["selection_file"]),
        Path(inputs["prediction_directory"]),
        Path(inputs["output_directory"]),
        Path(inputs["model_directory"]),
        analysis_mode=inputs["analysis_mode"],
    )


if __name__ == "__main__":
    main()
