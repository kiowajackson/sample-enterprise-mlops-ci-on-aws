"""Pipeline-step dispatcher: route one container command to one model action.

runtime/load_source.py calls this after loading supplied code. Compatible
images can also run `python -m mlops_models.entrypoints <step>` directly.
This module does not run the entire pipeline or download source.
"""

import argparse
import os
from pathlib import Path


def main():
    """Read a container command and call the selected step without entering its main guard."""
    # Read the selected step and its job paths.
    args = parse_job_arguments()

    # Run the step's production function.
    _ = dispatch_pipeline_step(args)


def parse_job_arguments():
    """Read CLI arguments and return the selected model, step and SageMaker job paths."""
    # Read the explicit command and any path overrides.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=[
        "get-data", "get-inference-data", "select-features", "preprocess-training", "train", "evaluate", "validate-package", "preprocess", "predict", "postprocess", "drift", "serve", "score", "resolve-model",
    ])
    parser.add_argument("--model", default=os.getenv("MODEL_ID"), required=not os.getenv("MODEL_ID"))
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--selection")
    parser.add_argument("--data", default="/opt/ml/processing/raw")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--chunk-rows", type=int, default=10000)
    parser.add_argument("--analysis-mode", choices=["prediction_inference", "prediction_test"],
                        default=os.getenv("MLOPS_ANALYSIS_MODE", "prediction_inference"))
    parser.add_argument("--model-package", default="")
    parser.add_argument("--image-uri", default="")
    parser.add_argument("--checkpoint-uri", default="")
    args = parser.parse_args()

    # Resolve standard paths centrally; direct Python/local callers may override them.
    from ..runtime.job_paths import job_paths, selected_features_path
    input_path, output_path, model_path = job_paths(args.step)
    args.input = args.input or input_path
    args.output = args.output or output_path
    args.model_dir = args.model_dir or model_path
    args.selection = args.selection or selected_features_path(args.step, args.input)
    return args


def dispatch_pipeline_step(args):
    """Take job arguments and call the selected step's execute function."""
    if args.step == "resolve-model":
        from ..registry.model_package import resolve_latest_approved_model
        return resolve_latest_approved_model(
            os.environ["MODEL_PACKAGE_GROUP"], args.output,
            namespace=os.environ["DEPLOYMENT_NAMESPACE"], model_id=args.model,
            environment=os.environ["ENVIRONMENT"], package_arn=args.model_package)
    if args.step == "score":
        from . import predict
        return predict.execute(args.model, args.input, args.model_dir, args.output)

    # Authorize the exact package before executing caller-selected inference images.
    if args.step == "validate-package":
        from ..registry.model_package import validate_approved_model_package
        return validate_approved_model_package(
            args.model_package, os.environ["MODEL_PACKAGE_GROUP"], args.image_uri,
            args.checkpoint_uri, args.output, namespace=os.environ["DEPLOYMENT_NAMESPACE"],
            model_id=args.model, environment=os.environ["ENVIRONMENT"],
        )

    # Run the training-pipeline operations.
    if args.step == "get-data":
        from . import get_data
        return get_data.execute(args.model, args.input, args.output)
    if args.step == "get-inference-data":
        from . import get_data
        return get_data.execute(args.model, args.input, args.output, model_dir=args.model_dir)
    if args.step == "select-features":
        from . import feature_selection
        return feature_selection.execute(args.model, args.input, args.output)
    if args.step == "preprocess-training":
        from . import preprocess_training
        return preprocess_training.execute(args.model, args.input, args.selection, args.output)
    if args.step == "train":
        from . import train
        hyperparameters = None
        if os.getenv("TUNING_OBJECTIVE"):
            import json
            hyperparameters = json.loads(Path("/opt/ml/input/config/hyperparameters.json").read_text())
        return train.execute(args.model, args.input, args.model_dir, args.output,
                             selection_path=args.selection, hyperparameters=hyperparameters)
    if args.step == "evaluate":
        from . import evaluate
        return evaluate.execute(args.model, args.model_dir, args.output,
                                input_path=None if os.getenv("MLOPS_EVALUATION_INPUT") == "model-artifact" else args.input)

    # Run the prediction-pipeline operations.
    if args.step == "preprocess":
        from . import preprocess_inference
        result = preprocess_inference.execute(args.model, args.input, args.selection, args.output,
                                             model_dir=args.model_dir, analysis_mode=args.analysis_mode)
        if os.getenv("MLOPS_PREPARE_TRANSFORM_INPUT") == "1":
            from ..runtime.transform_input import prepare_transform_input
            parameter_path = Path(__file__).parents[1] / "models" / args.model / "parameters.json"
            prepare_transform_input(Path(args.output) / "prepared", parameter_path)
        return result
    if args.step == "predict":
        from . import predict
        return predict.execute(args.model, args.input, args.model_dir, args.output)
    if args.step == "postprocess":
        from . import postprocess
        return postprocess.execute(args.model, args.input, args.output,
                                   model_dir=args.model_dir, chunk_rows=args.chunk_rows,
                                   data_directory=args.data, selection_file=args.selection,
                                   analysis_mode=args.analysis_mode)
    if args.step == "drift":
        from . import drift
        return drift.execute(args.model, args.input, args.output, model_dir=args.model_dir,
                             data_directory=args.data, selection_file=args.selection)
    return start_batch_server(args.model, args.model_dir)


def start_batch_server(model_id, model_dir):
    """Take a model and checkpoint path; start Gunicorn using the serving function."""
    application = f"mlops_models.entrypoints.serve:execute({model_id!r}, {model_dir!r})"
    # Gunicorn adds its starting directory to sys.path before reading configuration.
    os.chdir(Path(__file__).resolve().parents[2])
    os.execvp("gunicorn", [
        "gunicorn", "--bind", "0.0.0.0:8080", "--workers", os.getenv("MODEL_SERVER_WORKERS", "1"),
        "--timeout", "300",
        "--access-logfile", "-", "--error-logfile", "-", application,
    ])


if __name__ == "__main__":
    main()
