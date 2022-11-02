import os
import mlflow
from sklearn.ensemble import IsolationForest

from create_data import read_dataset,count_anomalies,create_X_y,read_yaml
from mlflow_log import dagshub_log, get_experiment_id

from ad_if import evaluate_test_if


MODELS_DIR = "models"

# conda_env = {
#     'channels': ['conda-forge'],
#     'dependencies': [
#         'python=3.9',
#         'pip',
#         'scikit_learn==1.1.2',
#         'dagshub==0.1.8',
#         'numpy==1.22.4',
#         'keras==2.9.0',
#         'pandas==1.4.2',
#         'plotly==5.10.0',
#         'PyYAML==6.0',
#         'requests==2.28.0'
#         ],
#     'pip': [
#         'mlflow==1.28.0'
#     ],
#     'name': 'mlflow-env'
# }

if __name__ == "__main__":

    params = read_yaml('src/hyperparams2.yaml')
    mlflow.set_tracking_uri(params['mlflow_url'])
    os.environ['MLFLOW_TRACKING_USERNAME'] = params['MLFLOW_TRACKING_USERNAME']
    os.environ['MLFLOW_TRACKING_PASSWORD'] = params['MLFLOW_TRACKING_PASSWORD']
    train_df = read_dataset('ecg_data/train.csv')
    test_df = read_dataset('ecg_data/test.csv','test')
    count_anomalies(test_df)

    X_train, y_train = create_X_y(train_df)
    X_test, y_test = create_X_y(test_df)
    exp_id = get_experiment_id("anomaly_detection_ecg_part2")
    mlflow.tensorflow.autolog()
    with mlflow.start_run():
        if params['model_name']=='autoencoder':
            from ad_ae import *
            from autoencoder import *
            Autoencoder,eval_meas = train_eval_model(X_train,X_test,y_test,params,test_df)
            #Autoencoder.save(MODELS_DIR)
            print(eval_meas)
            dagshub_log(params,eval_meas)
            # Log the sklearn model and register as version 1
            #mlflow.keras.log_model(Autoencoder, artifact_path=MODELS_DIR,registered_model_name=params['mlflow_model_name'])
            mlflow.keras.log_model(Autoencoder, artifact_path=MODELS_DIR,registered_model_name=params['mlflow_model_name'],conda_env=conda_env)

        elif params['model_name']=='if':
            from ad_if import *
            iforest = IsolationForest(n_estimators=100,contamination=params['contamination'],random_state=123)
            eval_meas = evaluate_test_if(iforest,X_train,y_train,X_test, y_test,test_df,params)
            print(eval_meas)
            #iforest.save(MODELS_DIR)
            dagshub_log(params,eval_meas)
            
            # Log the sklearn model and register as version 1
            mlflow.sklearn.log_model(iforest,artifact_path=MODELS_DIR,registered_model_name=params['mlflow_model_name'])
            #conda_env=conda_env
        else:
            print('wrong model name: write autoencoder or if')
    mlflow.end_run()



##{'recall': 0.8888888888888888, 'precision': 0.2926829268292683, 'f1': 0.4403669724770642} with contamination = 0.01
##{'recall': 0.8888888888888888, 'precision': 0.4067796610169492, 'f1': 0.5581395348837209} with contamination = 0.008
##{'recall': 0.8888888888888888, 'precision': 0.5853658536585366, 'f1': 0.7058823529411764} with contamination = 0.005
##{'recall': 0.8518518518518519, 'precision': 0.6388888888888888, 'f1': 0.7301587301587301} with contamination = 0.004