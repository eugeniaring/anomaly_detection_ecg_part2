import numpy as np
import keras
import logging
from pathlib import Path

from sklearn.metrics import (recall_score, f1_score,precision_score)
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from autoencoder import define_ae
import bentoml
from visualize import *
from mlflow_log import log_metrics_plot


def load_model_and_save_it_to_bento(model_file):
    """Loads the keras model from local PC and saves it to BentoML."""
    model = keras.models.load_model(model_file)
    bento_model = bentoml.keras.save_model("keras_model", model)
    print(f"Bento model tag = {bento_model.tag}")

def save_model(model,model_name):
    model.save(Path(model_name))
    load_model_and_save_it_to_bento(Path(model_name))
    logging.log(logging.INFO, "Done!")

def calculate_loss(reconstructions,data,type_loss):
    if type_loss=="mse":
        train_loss = np.mean((reconstructions - data)**2, axis=1)
        return train_loss
    elif type_loss=="mae":
        train_loss = np.mean(np.abs(reconstructions - data), axis=1)
        return train_loss 

def obtain_threshold(model,normal_train_data,params,thresh_q=0.95):
    reconstructions = model.predict(normal_train_data)
    train_loss = calculate_loss(reconstructions, normal_train_data,params['type_loss'])
    threshold = np.quantile(train_loss,thresh_q)
    #save_pickle(threshold,'threshold.pickle')
    return reconstructions,train_loss,threshold

def predict(model, data, threshold,params):
    reconstructions = model(data)
    loss = calculate_loss(reconstructions, data,params['type_loss'])
    return loss > threshold,reconstructions

def evaluate_test(model, test_data,test_labels, threshold,params,test_df):

    preds, _ = predict(model, test_data, threshold,params)
    test_df['test_preds'] = preds
    preds = preds.astype(int)

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

def train_eval_model(X_train,X_test,y_test,params,test_df):
    es = EarlyStopping(monitor='val_loss',mode='min', patience=params['patience'])
    Autoencoder = define_ae((X_train.shape[1],))
    Autoencoder.compile(optimizer=params['optimizer'], loss=params['type_loss'])

    history = Autoencoder.fit(X_train, X_train, 
        epochs=params['n_epochs'], 
        batch_size=params['batch_size'],
        validation_data=(X_test, X_test),
        shuffle=True,
        callbacks=[es])

    visualize_loss(history)
    save_model(Autoencoder,'Autoencoder')
    reconstructions = Autoencoder.predict(X_test)

    reconstructions,train_loss,threshold = obtain_threshold(Autoencoder,X_train,params)
    eval_meas = evaluate_test(Autoencoder, X_test,y_test, threshold,params,test_df)
    return Autoencoder,eval_meas