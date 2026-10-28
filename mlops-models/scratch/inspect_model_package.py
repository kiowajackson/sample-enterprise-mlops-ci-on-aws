"""Paste this whole cell into a notebook and set the two identifiers below.

- Read the expected group, namespace, model and environment from the ResolveModel job.
- Describe the exact model package you want to check against that job.
- Print all eight eligibility checks, with actual and expected values for each.
- Read only: no approvals, package updates, application imports or job starts.
"""

RESOLVE_JOB = ""  # Failed ResolveModel Processing job name or ARN.
MODEL_PACKAGE_ARN = ""  # Exact Registry version you expected it to accept.

import argparse

import boto3
from botocore.config import Config

if not RESOLVE_JOB or ":model-package/" not in MODEL_PACKAGE_ARN:
    raise ValueError("Set RESOLVE_JOB and MODEL_PACKAGE_ARN at the top of this cell.")

region = MODEL_PACKAGE_ARN.split(":")[3]
sm = boto3.client(
    "sagemaker", region_name=region,
    config=Config(connect_timeout=5, read_timeout=15, retries={"total_max_attempts": 2}),
)
if RESOLVE_JOB.startswith("arn:") and RESOLVE_JOB.split(":")[3:5] != MODEL_PACKAGE_ARN.split(":")[3:5]:
    raise ValueError("The ResolveModel job and model package must be in the same region/account.")

print("Reading ResolveModel job:", RESOLVE_JOB, flush=True)
job = sm.describe_processing_job(ProcessingJobName=RESOLVE_JOB.rsplit("/", 1)[-1])
env = job.get("Environment", {})
app = job.get("AppSpecification", {})

# Respect --model if it overrides MODEL_ID, just as the application entrypoint does.
parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
parser.add_argument("--model", default=env.get("MODEL_ID"))
parser.add_argument("--model-package", default="")
arguments, _ = parser.parse_known_args(
    app.get("ContainerEntrypoint", []) + app.get("ContainerArguments", [])
)
group = env.get("MODEL_PACKAGE_GROUP")
namespace = env.get("DEPLOYMENT_NAMESPACE")
model_id = arguments.model
environment = env.get("ENVIRONMENT")

missing = [name for name, value in (
    ("MODEL_PACKAGE_GROUP", group), ("DEPLOYMENT_NAMESPACE", namespace),
    ("--model / MODEL_ID", model_id), ("ENVIRONMENT", environment),
) if not value]
if missing:
    raise ValueError(
        "Cannot read the expected values from this job: " + ", ".join(missing)
        + ". Use the failed ResolveModel Processing job, not Score or a pipeline ARN."
    )

print("Reading model package:", MODEL_PACKAGE_ARN, flush=True)
package = sm.describe_model_package(ModelPackageName=MODEL_PACKAGE_ARN)
properties = package.get("CustomerMetadataProperties", {})
lifecycle = package.get("ModelLifeCycle", {})

# Each row is one exact comparison from development_scope / approved_development_package.
# Evaluate every row independently: a failed check must not hide the remaining checks.
checks = [
    ("ModelPackageGroupName", package.get("ModelPackageGroupName"), group),
    ("ModelPackageStatus", package.get("ModelPackageStatus"), "Completed"),
    ("ModelLifeCycle.Stage", lifecycle.get("Stage"), "Development"),
    ("DeploymentNamespace", properties.get("DeploymentNamespace"), namespace),
    ("ModelId", properties.get("ModelId"), model_id),
    ("Environment", properties.get("Environment"), environment),
    ("ModelApprovalStatus", package.get("ModelApprovalStatus"), "Approved"),
    ("ModelLifeCycle.StageStatus", lifecycle.get("StageStatus"), "Approved"),
]
failed = [name for name, actual, expected in checks if actual != expected]

print("\nELIGIBILITY:", "FAIL — " + ", ".join(failed) if failed else "PASS — all 8 checks match")
print("Job requested:", arguments.model_package or "latest-approved (argument omitted)")
if arguments.model_package not in ("", "latest-approved", MODEL_PACKAGE_ARN):
    print("NOTE: This job explicitly requested a different package from the one being inspected.")
print("Expected values come from the job above. Actual values are the Registry values NOW.")
print("Quotes expose case/whitespace differences; None means the field is missing.\n")
for name, actual, expected in checks:
    print(f"{'PASS' if actual == expected else 'FAIL'}  {name}")
    print(f"      actual:   {actual!r}")
    print(f"      expected: {expected!r}")

# This explains eligibility for this package. It does not change the selection policy
# or check the later container/source bindings. Registry values may have changed since the job.
