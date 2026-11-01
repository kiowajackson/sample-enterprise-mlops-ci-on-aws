"""Call the selected use case's batch prediction function."""

import json
from pathlib import Path

from mlops_models.entrypoints import load_step


def execute(model_id, input_path, model_dir, output_path):
    parameter_path = Path(__file__).parents[1] / "models" / model_id / "parameters.json"
    config_path = Path(model_dir) / "model_metadata.json"
    config = json.loads(config_path.read_text())
    predict_with_model = load_step(model_id, "inference.predict", "predict_with_model")
    predictor = load_step(model_id, "inference.predict", "ModelPredictor", default=None)
    return predict_with_model(
        Path(input_path), Path(model_dir), Path(output_path), config, parameter_path,
        predictor_cls=predictor,
    )
