import numpy as np
import pandas as pd
import bentoml
from bentoml.io import NumpyNdarray, PandasDataFrame


BENTO_MODEL_TAG = "if:latest"

ad_runner = bentoml.sklearn.get(BENTO_MODEL_TAG).to_runner()
service = bentoml.Service("if_model_ad", runners=[ad_runner])

@service.api(input=PandasDataFrame(), output=NumpyNdarray())
def predict(df: pd.DataFrame) -> np.ndarray:
    X_test = df[['heart_rate','hr_diff','peak_label']].values
    test_pred = ad_runner.predict.run(X_test)
    return np.array(test_pred)