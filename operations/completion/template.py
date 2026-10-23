"""Render the small completion example for deployment by the platform owner."""

import argparse
import json
from pathlib import Path
import re


def template(name, pipeline_arns, job_roles, *, failure_destination=""):
    """One Lambda and two event rules; model execution remains in the existing pipelines."""
    if not pipeline_arns or not job_roles:
        raise ValueError("Supply explicit pipeline and job-role ARNs.")
    if not re.fullmatch(r"[A-Za-z0-9-]{1,20}", name):
        raise ValueError("Use a completion name of at most 20 letters, numbers and hyphens.")
    partition, _, region, account = pipeline_arns[0].split(":")[1:5]
    job_prefix = name + "-sync-"
    function_arn = f"arn:{partition}:lambda:{region}:{account}:function:{name}"
    job_arn = f"arn:{partition}:sagemaker:{region}:{account}:processing-job/{job_prefix}*"
    log_arn = f"arn:{partition}:logs:{region}:{account}:log-group:/aws/lambda/{name}:*"
    resources = {
        "Role": {"Type": "AWS::IAM::Role", "Properties": {
            "AssumeRolePolicyDocument": {"Version": "2012-10-17", "Statement": [{
                "Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}]},
            "Policies": [{"PolicyName": "completion", "PolicyDocument": {
                "Version": "2012-10-17", "Statement": [
                    {"Effect": "Allow", "Action": ["sagemaker:DescribePipelineExecution",
                     "sagemaker:DescribePipelineDefinitionForExecution"],
                     "Resource": pipeline_arns + [arn + "/execution/*" for arn in pipeline_arns]},
                    {"Effect": "Allow", "Action": ["sagemaker:CreateProcessingJob",
                     "sagemaker:DescribeProcessingJob", "sagemaker:AddTags"], "Resource": job_arn},
                    {"Effect": "Allow", "Action": "iam:PassRole", "Resource": job_roles,
                     "Condition": {"StringEquals": {"iam:PassedToService": "sagemaker.amazonaws.com"}}},
                    {"Effect": "Allow", "Action": ["logs:CreateLogStream", "logs:PutLogEvents"], "Resource": log_arn},
                ]}}],
        }},
        "Logs": {"Type": "AWS::Logs::LogGroup", "Properties": {
            "LogGroupName": "/aws/lambda/" + name, "RetentionInDays": 14}},
        "Function": {"Type": "AWS::Lambda::Function", "DependsOn": "Logs", "Properties": {
            "FunctionName": name, "Runtime": "python3.12", "Handler": "index.lambda_handler",
            "Role": {"Fn::GetAtt": ["Role", "Arn"]}, "Timeout": 60, "MemorySize": 256,
            "Code": {"ZipFile": Path(__file__).with_name("handler.py").read_text()},
            "Environment": {"Variables": {"PIPELINE_ARNS": json.dumps(pipeline_arns),
                                           "JOB_ROLE_ARNS": json.dumps(job_roles),
                                           "SYNC_JOB_PREFIX": job_prefix}},
        }},
    }
    patterns = {
        "Pipeline": {"source": ["aws.sagemaker"],
                     "detail-type": ["SageMaker Model Building Pipeline Execution Status Change"],
                     "detail": {"pipelineArn": pipeline_arns,
                                "currentPipelineExecutionStatus": ["Succeeded", "Failed", "Stopped"]}},
        "SyncFailure": {"source": ["aws.sagemaker"],
                        "detail-type": ["SageMaker Processing Job State Change"],
                        "detail": {"ProcessingJobName": [{"prefix": job_prefix}],
                                   "ProcessingJobStatus": ["Failed", "Stopped"]}},
    }
    for label, pattern in patterns.items():
        resources[label + "Rule"] = {"Type": "AWS::Events::Rule", "DependsOn": "Function", "Properties": {
            "EventPattern": pattern, "State": "ENABLED", "Targets": [{
                "Id": "completion", "Arn": function_arn,
                "RetryPolicy": {"MaximumRetryAttempts": 3, "MaximumEventAgeInSeconds": 3600},
            }],
        }}
        resources[label + "Permission"] = {"Type": "AWS::Lambda::Permission", "Properties": {
            "Action": "lambda:InvokeFunction", "FunctionName": {"Ref": "Function"},
            "Principal": "events.amazonaws.com", "SourceArn": {"Fn::GetAtt": [label + "Rule", "Arn"]},
        }}
    if failure_destination:
        resources["AsyncFailures"] = {"Type": "AWS::Lambda::EventInvokeConfig", "Properties": {
            "FunctionName": {"Ref": "Function"}, "Qualifier": "$LATEST",
            "MaximumRetryAttempts": 2, "MaximumEventAgeInSeconds": 3600,
            "DestinationConfig": {"OnFailure": {"Destination": failure_destination}},
        }}
        service = failure_destination.split(":")[2]
        if service not in {"sns", "sqs"}:
            raise ValueError("Use an existing approved SNS topic or SQS queue for completion failures.")
        resources["Role"]["Properties"]["Policies"][0]["PolicyDocument"]["Statement"].append({
            "Effect": "Allow", "Action": "sns:Publish" if service == "sns" else "sqs:SendMessage",
            "Resource": failure_destination,
        })
    return {"AWSTemplateFormatVersion": "2010-09-09", "Resources": resources,
            "Outputs": {"FunctionArn": {"Value": function_arn}}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--pipeline-arn", action="append", required=True)
    parser.add_argument("--job-role-arn", action="append", required=True)
    parser.add_argument("--failure-destination", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(template(args.name, args.pipeline_arn, args.job_role_arn,
                                              failure_destination=args.failure_destination), indent=2) + "\n")
