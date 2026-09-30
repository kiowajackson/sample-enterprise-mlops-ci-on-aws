# Enterprise MLOps CI on AWS

Sample enterprise MLOps and CI/CD workflows on AWS.

## Managed access check

[00D_check_managed_pipeline.ipynb](examples/notebooks/00D_check_managed_pipeline.ipynb) checks whether the current AWS role can create, update and execute a SageMaker pipeline using a standard AWS container. It runs independently with boto3; no application checkout, Docker, CDK or SageMaker SDK is required.

It starts one small, billable Processing job using synthetic values and displays a single access summary. S3 output is optional. An optional final cell removes the probe pipeline and is off by default. Keep execution outputs and account-specific settings in your environment.

## Development notebook

[03B_model_development.ipynb](examples/notebooks/03B_model_development.ipynb) runs Prepare, Train, Evaluate and Predict directly in Python.

Download it into `examples/notebooks/` in your existing application checkout and open it with a fresh kernel. It needs the application's `mlops-models/` source, model configuration and `examples/data/` CSVs. This branch contains notebooks, not the full application source.

The development notebook uses the current kernel's modeling packages and does not need Docker, the SageMaker SDK, CDK or AWS credentials for its local CSV example. Outputs are cleared before publication.
