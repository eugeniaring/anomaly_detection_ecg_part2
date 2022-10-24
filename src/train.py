import os
import mlflow
from sklearn.ensemble import IsolationForest

from create_data import read_dataset,count_anomalies,create_X_y,read_yaml
from mlflow_log import dagshub_log

from ad_if import evaluate_test_if

if __name__ == "__main__":

    params = read_yaml('src/hyperparams.yaml')
    mlflow.set_tracking_uri(params['mlflow_url'])
    os.environ['MLFLOW_TRACKING_USERNAME'] = params['MLFLOW_TRACKING_USERNAME']
    os.environ['MLFLOW_TRACKING_PASSWORD'] = params['MLFLOW_TRACKING_PASSWORD']
    train_df = read_dataset('ecg_data/train.csv')
    test_df = read_dataset('ecg_data/test.csv','test')
    count_anomalies(test_df)

    X_train, y_train = create_X_y(train_df)
    X_test, y_test = create_X_y(test_df)
    mlflow.tensorflow.autolog()
    with mlflow.start_run():
        if params['model_name']=='autoencoder':
            from ad_ae import *
            from autoencoder import *
            Autoencoder,eval_meas = train_eval_model(X_train,X_test,y_test,params,test_df)
            print(eval_meas)
            dagshub_log(params,eval_meas)
        elif params['model_name']=='if':
            from ad_if import *
            iforest = IsolationForest(n_estimators=100,contamination=params['contamination'],random_state=123)
            eval_meas = evaluate_test_if(iforest,X_train,y_train,X_test, y_test,test_df,params)
            print(eval_meas)
            dagshub_log(params,eval_meas)
            ##{'recall': 0.8888888888888888, 'precision': 0.2926829268292683, 'f1': 0.4403669724770642} with contamination = 0.01
            ##{'recall': 0.8888888888888888, 'precision': 0.4067796610169492, 'f1': 0.5581395348837209} with contamination = 0.008
            ##{'recall': 0.8888888888888888, 'precision': 0.5853658536585366, 'f1': 0.7058823529411764} with contamination = 0.005
            ##{'recall': 0.8518518518518519, 'precision': 0.6388888888888888, 'f1': 0.7301587301587301} with contamination = 0.004
        else:
            print('wrong model name: write autoencoder or if')
    mlflow.end_run()