import pandas as pd
import numpy as np
import json
import boto3

from create_data import read_dataset

global app_name
global region
app_name = 'ifTest'
region = "us-east-1"

def query_endpoint(app_name, input_json):
    """ Invoke the SageMaker endpoint and send the 
    input request to be processed 
    """
    client = boto3.session.Session().client('sagemaker-runtime', region)

    response = client.invoke_endpoint(
        EndpointName = app_name,
        Body = input_json,
        ContentType = 'application/json; format=pandas-split',
        )

    preds = response['Body'].read().decode('ascii')
    preds = np.array(json.loads(preds))
    preds[preds == 1] = 0
    preds[preds == -1] = 1 
    print('Received response: {}'.format(preds))
    return preds

def sample_random_ecg_data_patient(path='ecg_data/test.csv'):
    test_df = read_dataset(path,'test')
    test_df = test_df[test_df.patient_id==14]
    expected_output = test_df.label.values
    test_df = test_df[['heart_rate','hr_diff','peak_label']]
    return test_df,expected_output


# Convert input into json for SageMaker endpoint
input_data,expected_output = sample_random_ecg_data_patient()
input_data = input_data.to_json(orient="split")

print('Expected_output: ',expected_output)
# compare ground truth with predictions
predictions = query_endpoint(app_name=app_name, input_json=input_data)