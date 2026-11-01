"""Load the selected use case's settings and call its selection function."""

from pathlib import Path
import json

from mlops_models.entrypoints import load_step


def execute(model_id, input_path, output_dir):
    config_path = Path(__file__).parents[1] / "models" / model_id / "parameters.json"
    config = json.loads(config_path.read_text())
    run = load_step(model_id, "training.feature_selection", "run_feature_selection")
    selector = load_step(model_id, "training.feature_selection", "FeatureSelector")
    return run(Path(input_path), Path(output_dir), config, selector_cls=selector)
