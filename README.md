# Enterprise MLOps CI on AWS

Sample enterprise MLOps and CI/CD workflows on AWS.

## Managed access check

[00D_check_managed_pipeline.ipynb](examples/notebooks/00D_check_managed_pipeline.ipynb) shows one explicit boto3 command per check, followed by its response or exact error. S3, existing SageMaker resources, ECR metadata, CloudWatch and CDK bootstrap reads are checked independently. A failed pipeline creation does not stop those checks. No application checkout, Docker, CDK or SageMaker SDK is required.

Start with the read-only checks. Configure the input bucket/key to test direct S3 access even if listing buckets is denied. S3 writes and pipeline create/start each have explicit switches, off by default. The managed probe uses separate pipeline and job roles and starts one small, billable Processing job with synthetic values. Separate cleanup cells remove only the probe pipeline and its scratch object. Results remain below each command and are also saved locally. Keep outputs and account-specific settings in your environment; the published notebook has cleared outputs.

## Development notebook

[03B_model_development.ipynb](examples/notebooks/03B_model_development.ipynb) walks through the training and batch entry points directly in Python. Prepare exposes feature selection and the splits; Train shows transformed X/y and the checkpoint; Evaluate shows metrics and held-out predictions. It then runs batch preprocessing, the real serving handler through an in-process request client, and postprocessing. Direct Predict remains a separate comparison.

Download it into `examples/notebooks/` in your existing application checkout and open it with a fresh kernel. It needs the application's `mlops-models/` source, model configuration and `examples/data/` CSVs. This branch contains notebooks, not the full application source.

The development notebook uses the current kernel's modeling packages plus Flask, an existing application dependency. It does not need Docker, the SageMaker SDK, CDK or AWS credentials for its local CSV example. Package validation uses explicit local-development mode; registry approval, registration, CreateModel and managed jobs are not performed. Outputs are cleared before publication.
