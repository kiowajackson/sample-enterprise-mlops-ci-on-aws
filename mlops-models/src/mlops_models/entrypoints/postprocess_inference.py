"""Run the selected model's inference analysis from explicit mounted inputs."""

from pathlib import Path

from mlops_models.entrypoints import load_step


def execute(model_id, input_path, output_dir, *, model_dir, data_directory, selection_file,
            chunk_rows=10000):
    prediction_inference = load_step(model_id, "inference.postprocess_inference", "prediction_inference")
    return prediction_inference(
        Path(data_directory), Path(selection_file), Path(input_path), Path(output_dir),
        Path(model_dir), chunk_rows=chunk_rows,
    )
