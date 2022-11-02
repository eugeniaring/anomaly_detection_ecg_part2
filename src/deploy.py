import mlflow 
import os 
import mlflow.sagemaker
from create_data import read_yaml
params = read_yaml('hyperparams.yaml')

DAGSHUB_USER_NAME = params['MLFLOW_TRACKING_USERNAME']
DAGSHUB_REPO_NAME = params['DAGSHUB_REPO_NAME']
DAGSHUB_TOKEN = params['MLFLOW_TRACKING_PASSWORD']

import os

# TODO: Explain that it's recommended to define this in the code because it's project specific
os.environ['MLFLOW_TRACKING_URI']=f"https://dagshub.com/{DAGSHUB_USER_NAME}/{DAGSHUB_REPO_NAME}.mlflow"

# Recommended to define as environment variables
os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USER_NAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN

# PARSE YOUR MLFLOW INFO

# image build and push by mlflow
image_url = '849711432510.dkr.ecr.us-east-1.amazonaws.com/mlflow-pyfunc:1.28.0'

# the model uri
model_uri = 'runs:/8474a07e554e4207b9dd99c60627d6dd/models'

# your region
region = "us-east-1"
# the arn of the sagemaker deployment role
arn = "arn:aws:iam::849711432510:role/awssagemakerdeployment"

#arn = "arn:aws:iam::849711432510:role/service-role/AmazonSageMaker-ExecutionRole-20221101T215612"

# deploy to the endpoint
mlflow.sagemaker.deploy(app_name="ifTest",
    mode='create',
    model_uri=model_uri,
    image_url=image_url,
    execution_role_arn=arn,
    instance_type="ml.t2.medium",
    instance_count=1,
    region_name=region
)

