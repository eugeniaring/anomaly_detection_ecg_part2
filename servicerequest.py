from typing import Tuple
import numpy as np
import pandas as pd
import requests
from src.create_data import read_dataset

#SERVICE_URL = "http://0.0.0.0:3000/predict"

# aws endpoint
SERVICE_URL = "https://ke496mufza.execute-api.us-west-1.amazonaws.com/predict"

def sample_random_ecg_data_patient() -> Tuple[np.ndarray, np.ndarray]:
    test_df = read_dataset('ecg_data/test.csv','test')
    test_df = test_df[test_df.patient_id==14]
    expected_output = test_df.label.values
    return test_df,expected_output


def make_request_to_bento_service(
    service_url: str, df: pd.DataFrame
) -> np.array:
    serialized_input_data = df.to_json()
    response = requests.post(
        service_url,
        data=serialized_input_data,
        headers={"content-type": "application/json"}
    )
    response = response.text.replace('[','').replace(']','').replace(',','')
    print(response)
    response = response.split(" ")
    y = np.array(list(map(int, response)))
    #y = [int(r) for r in response]
    y[y == 1] = 0
    y[y == -1] = 1    
    return y


def main():
    input_data,expected_output  = sample_random_ecg_data_patient()
    prediction = make_request_to_bento_service(SERVICE_URL, input_data)

    print(f"prediction: {prediction}")
    print(f"Expected output: {expected_output.astype(int)}")

if __name__ == "__main__":
    main()