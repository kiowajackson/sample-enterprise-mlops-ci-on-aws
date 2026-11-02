# November 2, 2026. Copy into a notebook cell; running it requests deletion.
TRACKING_SERVER_ARN = ""  # Paste the full ARN of the tracking server to delete.

import boto3

arn = TRACKING_SERVER_ARN.strip()
parts = arn.split(":", 5)
if (len(parts) != 6 or parts[0] != "arn" or parts[2] != "sagemaker"
        or not parts[5].startswith("mlflow-tracking-server/")):
    raise ValueError("Paste a full SageMaker MLflow tracking-server ARN above.")

region = parts[3]
server_name = parts[5].split("/", 1)[1]
sm = boto3.client("sagemaker", region_name=region)
server = sm.describe_mlflow_tracking_server(TrackingServerName=server_name)
if server["TrackingServerArn"] != arn:
    raise ValueError("The server returned by AWS does not match the requested ARN.")

response = sm.delete_mlflow_tracking_server(TrackingServerName=server_name)
print("Deletion requested:", response["TrackingServerArn"])

# Optional: run these lines in a later cell to check progress.
# Once deleted, describe reports that the resource was not found.
# server = sm.describe_mlflow_tracking_server(TrackingServerName=server_name)
# print(server["TrackingServerStatus"])
