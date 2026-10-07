# Enterprise MLOps CI on AWS

Sample enterprise MLOps and CI/CD workflows on AWS.

## Quick access check

[00E_check_access_defaults.ipynb](examples/notebooks/00E_check_access_defaults.ipynb) reproduces access errors with the current notebook role and an automatically selected AWS image. There are no required settings. CreatePipeline, UpdatePipeline and StartPipelineExecution come first, with the actual boto3 calls visible and the original exceptions printed. Starting the probe launches one small, billable Processing job.

For S3, supply a bucket or leave it blank to use the first listed bucket; the notebook selects a file and reads up to 1 KiB without printing its contents. Pipeline failure does not stop these checks. Later cells inspect execution, job status and logs. The final cleanup cell is optional. No reporting helpers, custom error framework or application checkout are required. Outputs are cleared before publication.

[00D_check_managed_pipeline.ipynb](examples/notebooks/00D_check_managed_pipeline.ipynb) remains available for explicitly chosen resources and more extensive checks, including separate pipeline/job roles and S3 writes.

## Development notebook

[03B_model_development.ipynb](examples/notebooks/03B_model_development.ipynb) walks through the training and batch entry points directly in Python. Prepare exposes feature selection and the splits; Train shows transformed X/y and the checkpoint; Evaluate shows metrics and held-out predictions. It then runs batch preprocessing, the real serving handler through an in-process request client, and postprocessing. Direct Predict remains a separate comparison.

Download it into `examples/notebooks/` in your existing application checkout and open it with a fresh kernel. It needs the application's `mlops-models/` source, model configuration and `examples/data/` CSVs. This branch contains notebooks, not the full application source.

The development notebook uses the current kernel's modeling packages plus Flask, an existing application dependency. It does not need Docker, the SageMaker SDK, CDK or AWS credentials for its local CSV example. Package validation uses explicit local-development mode; registry approval, registration, CreateModel and managed jobs are not performed. Outputs are cleared before publication.


## Remote build and managed execution

[03E_build_and_run_managed.ipynb](examples/notebooks/03E_build_and_run_managed.ipynb)
packages the selected model with the application's existing Dockerfile and locked
dependencies, uploads the source ZIP, and asks an approved CodeBuild project to build
and push the image. It then creates or updates a separate development pipeline and
runs Prepare → Train → Evaluate, with visible boto3 calls and log/output inspection.
The notebook itself needs neither Docker nor CDK.

Download it into `examples/notebooks/` in the existing application checkout.
Provide an approved S3 prefix, an existing build-only CodeBuild project, a pipeline
role and a separate job role. Supply an approved base-image digest when required.
The platform team provisions the builder and roles; ECR repository creation is
included when permitted. Builds and managed jobs incur normal AWS charges.
The optional final cell deletes only the development pipeline. Outputs are cleared.
