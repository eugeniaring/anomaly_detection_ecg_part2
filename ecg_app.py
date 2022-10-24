import pandas as pd
import numpy as np
import os

import streamlit as st
import requests
import zipfile

#from servicerequest import make_request_to_bento_service
from src.create_data import read_dataset
from src.visualize import visualize_predict_vs_gt

if not os.path.exists('test_eval'):
    os.mkdir('test_eval')

SERVICE_URL = "https://ke496mufza.execute-api.us-west-1.amazonaws.com/predict"
model_name = 'if'

st.markdown('# **Web App to detect Anomalies from ECG signals**')
#bar = st.progress(0)

file_csv = st.sidebar.file_uploader("Choose CSV file to evaluate model",type=["csv"])

button = st.sidebar.button('Check Anomalies!')


def filter_df(df,patient_id):
    subset_df = df[df.patient_id==patient_id]
    return subset_df,subset_df.label.values

def transform_string(s):
    s = s.replace('[','').replace(']','').replace(',','')
    l = s.split(" ")
    return np.array(list(map(int, l)))

def map_label(y):
    y[y == 1] = 0
    y[y == -1] = 1
    return y   

def make_request_to_bento_service(
    service_url: str, df: pd.DataFrame
) -> np.array:
    serialized_input_data = df.to_json()
    response = requests.post(
        service_url,
        data=serialized_input_data,
        headers={"content-type": "application/json"}
    )
    y = transform_string(response.text)
    y = map_label(y)
    return y

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
        patient_df,expected_output = filter_df(patients_df,idx)
        pred = make_request_to_bento_service(SERVICE_URL, patient_df)
        # print(f"prediction: {pred}")
        # print(f"Expected output: {expected_output.astype(int)}")
        rows.extend(pred)
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




