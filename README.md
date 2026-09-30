# Enterprise MLOps CI on AWS

Sample enterprise MLOps and CI/CD workflows on AWS.

## Development notebook

[03B_model_development.ipynb](examples/notebooks/03B_model_development.ipynb) runs Prepare, Train, Evaluate and Predict directly in Python.

Download it into `examples/notebooks/` in your existing application checkout and open it with a fresh kernel. It needs the application's `mlops-models/` source, model configuration and `examples/data/` CSVs. This branch contains the notebook only; it is not a standalone application installation.

The notebook uses the current kernel's modeling packages and does not need Docker, the SageMaker SDK, CDK or AWS credentials for its local CSV example. Outputs are cleared before publication.
