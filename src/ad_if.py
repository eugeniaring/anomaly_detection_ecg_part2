import logging
import bentoml
from pathlib import Path
import os
import joblib

from sklearn.metrics import (recall_score, f1_score,precision_score)
from mlflow_log import log_metrics_plot

def load_model_and_save_it_to_bento(model_file,filename='isolation_forest.sav'):
    """Loads the sklearn model from local PC and saves it to BentoML."""
    if not os.path.exists('if'):
       os.mkdir('if')
    model = joblib.load('if/'+filename)
    bento_model = bentoml.sklearn.save_model("if", model)
    print(f"Bento model tag = {bento_model.tag}")


def save_model(model,model_name,filename='isolation_forest.sav'):
    joblib.dump(model, filename)
    load_model_and_save_it_to_bento(Path(model_name))
    logging.log(logging.INFO, "Done!")


def train_eval_if(iforest,train_X,train_y,test_X):
    #iforest = IsolationForest(n_estimators=100,contamination=0.05)
    iforest.fit(train_X,train_y)
    test_preds = iforest.predict(test_X)
    test_scores = iforest.decision_function(test_X)
    #bentoml.sklearn.save_model("if", iforest)
    save_model(iforest,'if')
    return test_preds, test_scores

def map_pred_values(y):
    #if returns -1 if the item is anomalous, 0 otherwise
    y[y == 1] = 0
    y[y == -1] = 1
    return y


def evaluate_test_if(model, train_X,train_y,test_X,test_labels,test_df,params):

    preds, _ = train_eval_if(model,train_X,train_y,test_X)
    preds = preds.astype(int)
    preds = map_pred_values(preds)
    test_df['test_preds'] = preds

    prec = precision_score(test_labels, preds)
    recall = recall_score(test_labels, preds)
    f1 = f1_score(test_labels, preds) 

    logging.log(logging.INFO,
                    f"{model.__class__.__name__} model Precision test: {prec}")
    logging.log(logging.INFO,
                    f"{model.__class__.__name__} model Recall test: {recall}")
    logging.log(logging.INFO,
                    f"{model.__class__.__name__} model f1-score test: {f1}")

    eval_meas = {'recall':recall,"precision":prec,"f1":f1} 
    log_metrics_plot(test_df,params,eval_meas)

    return eval_meas