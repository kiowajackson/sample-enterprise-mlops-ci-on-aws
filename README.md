# Enterprise MLOps CI on AWS

Sample enterprise MLOps and CI/CD workflows on AWS.

## Run code in existing managed images

[00G_check_managed_images.ipynb](examples/notebooks/00G_check_managed_images.ipynb)
compares an AWS-provided scikit-learn image with a platform image discovered by a
repository-name fragment (`model-build` by default). Paste the full working IAM
role ARN, including its path; the notebook never guesses it or calls IAM.
Supply a separate job role when available. Enable either test independently.

Each test creates a small, billable Processing pipeline, runs a visible Python
command, waits for completion and prints actual job details and CloudWatch output.
No local Docker, image build/copy, CDK, input data or S3 write is required. The
initial code prints Python/module availability and a calculation; an alternative
small scikit-learn fit/predict command is included. Longer source folders require
an approved S3 input location. Runtime and orchestration roles still need their
respective permissions; technical success does not establish image-use approval.

The notebook also inspects Studio's Docker proxy/settings without starting a
daemon or changing the domain. Cleanup of its own pipelines is explicit and
optional. It requires only boto3 and Python's standard library. Outputs are cleared.

## Deployment diagnostics

[00F_deployment_diagnostics.ipynb](examples/notebooks/00F_deployment_diagnostics.ipynb)
runs with no required settings and makes no cloud changes by default. Individual
cells show the actual AWS/Docker commands, original errors and independent results.
It checks identity, Python packages, Docker, Cloud9, S3, ECR, existing roles/pipelines,
build projects and standard CDK bootstrap visibility. The last cell prints one summary.

Separate switches test pipeline create/update/start, S3 write/read/delete, ECR
repository creation, and Docker pull/build/push using an approved base/repository.
The pipeline probe launches no jobs and needs no container. It does not establish
runtime job permissions. Cleanup cells remove only this kernel's probe resources.
Run it in Studio and in the proposed builder to compare actual capabilities.
The notebook stands alone; no application checkout, CDK or SageMaker SDK is required.
Outputs are cleared. Older notebooks remain available below.

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
