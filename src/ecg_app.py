import pandas as pd
import numpy as np
import json
import boto3
import os

import streamlit as st
import requests
import zipfile

from create_data import read_dataset,read_yaml
from visualize import visualize_predict_vs_gt

if not os.path.exists('test_eval'):
    os.mkdir('test_eval')

global app_name
global region
app_name = 'ifTest'
region = "us-east-1"

model_name = 'if'

import subprocess

#subprocess.call("aws configure import --csv credentials.csv", shell=True)

st.markdown('# **Web App to detect Anomalies from ECG signals**')

file_csv = st.sidebar.file_uploader("Choose CSV file to evaluate model",type=["csv"])
params = read_yaml('src/hyperparams.yaml')
button = st.sidebar.button('Check Anomalies!')

def query_endpoint(app_name,params, input_json):
    """ Invoke the SageMaker endpoint and send the 
    input request to be processed 
    """
    client = boto3.session.Session(aws_access_key_id=params['aws_access_key_id'], aws_secret_access_key=params['aws_secret_access_key'], region_name=params['region_name']).client('sagemaker-runtime', region)

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

def filter_df(df,patient_id):
    subset_df = df[df.patient_id==patient_id]
    label = subset_df.label.values
    subset_df = subset_df[['heart_rate','hr_diff','peak_label']]
    return subset_df,label

def save_zip(l_patient_ids):
    list_files = ["test_eval/gt_vs_pred_{}_if.html".format(patient_id) for patient_id in l_patient_ids]
    with zipfile.ZipFile('final.zip', 'w') as zipF:
      for file in list_files:
         zipF.write(file, compress_type=zipfile.ZIP_DEFLATED)
      zipF.close()

if file_csv is not None and button is not None:  
    patients_df = read_dataset(file_csv,'test')
    l_ids = list(patients_df.patient_id.unique())
    rows = []
    for idx in l_ids:
        input_data,expected_output =  filter_df(patients_df,idx)
        input_data = input_data.to_json(orient="split")
        predictions = query_endpoint(app_name=app_name,params=params, input_json=input_data)
        # print(f"prediction: {pred}")
        # print(f"Expected output: {expected_output.astype(int)}")
        rows.extend(predictions)
    patients_df['test_preds']=rows

    patient_id = st.selectbox('Patient id',tuple(l_ids))
    fig = visualize_predict_vs_gt(patients_df,patient_id,'if')
    st.plotly_chart(fig)
    l_ids_copy = l_ids.copy()
    l_ids_copy.remove(patient_id)

    for idx in l_ids_copy:
        fig = visualize_predict_vs_gt(patients_df,idx,'if')

    save_zip(l_ids)
    with open("final.zip", "rb") as zip_download:
        btn = st.download_button(
            label="Download",
            data=zip_download,
            file_name="final.zip",
            mime="application/zip"
        )




