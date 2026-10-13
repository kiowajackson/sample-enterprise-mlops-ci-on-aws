# Read the latest training execution; reuse sm, session and namespace from 03Fv2.
execution = sm.list_pipeline_executions(
    PipelineName=f"{namespace}-training", SortBy="CreationTime",
    SortOrder="Descending", MaxResults=1,
)["PipelineExecutionSummaries"][0]
execution_arn = execution["PipelineExecutionArn"]
print("Execution:", execution_arn)

steps = sm.get_paginator("list_pipeline_execution_steps").paginate(PipelineExecutionArn=execution_arn)
train_step = next(step for page in steps for step in page["PipelineExecutionSteps"]
                  if step["StepName"] == "Train")
print("Step failure:", train_step.get("FailureReason", "Not returned"))
job_name = train_step["Metadata"]["TrainingJob"]["Arn"].rsplit("/", 1)[1]
job = sm.describe_training_job(TrainingJobName=job_name)
print("Training job:", job_name, "|", job["TrainingJobStatus"])
print("Job failure:", job.get("FailureReason", "Not returned"))

logs = session.client("logs")
streams = logs.get_paginator("describe_log_streams").paginate(
    logGroupName="/aws/sagemaker/TrainingJobs", logStreamNamePrefix=job_name + "/",
)
for page in streams:
    for stream in page["logStreams"]:
        request = dict(logGroupName="/aws/sagemaker/TrainingJobs",
                       logStreamName=stream["logStreamName"], startFromHead=True)
        while True:
            response = logs.get_log_events(**request)
            for event in response["events"]:
                print(event["message"])
            if response["nextForwardToken"] == request.get("nextToken"):
                break
            request["nextToken"] = response["nextForwardToken"]
