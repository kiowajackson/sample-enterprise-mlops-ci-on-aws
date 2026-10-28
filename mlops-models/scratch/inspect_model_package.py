"""Paste this whole cell into your notebook and set MODEL_PACKAGE_ARN.

- Describe that exact Registry version using the notebook's AWS credentials.
- Print package readiness, both approval fields, and the recorded model scope.
- Compare the scope with the namespace/model/environment in the ResolveModel error.
- Read only: this cell does not approve models, modify packages, or start jobs.
"""

import json

import boto3

MODEL_PACKAGE_ARN = ""

if not MODEL_PACKAGE_ARN:
    raise ValueError("Set MODEL_PACKAGE_ARN to the Registry model package ARN.")

sm = boto3.client("sagemaker", region_name=MODEL_PACKAGE_ARN.split(":")[3])
print("Inspecting:", MODEL_PACKAGE_ARN, flush=True)
package = sm.describe_model_package(ModelPackageName=MODEL_PACKAGE_ARN)

details = {
    key: package.get(key)
    for key in (
        "ModelPackageArn",
        "ModelPackageGroupName",
        "ModelPackageStatus",
        "ModelApprovalStatus",
        "ModelLifeCycle",
    )
}
details["Scope"] = {
    key: package.get("CustomerMetadataProperties", {}).get(key)
    for key in ("DeploymentNamespace", "ModelId", "Environment")
}
print(json.dumps(details, indent=2))

# Completed is SageMaker's package readiness state, not another manual approval.
# Selecting an explicit ARN still requires the resolver's approval and scope checks.
