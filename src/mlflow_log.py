from dagshub import dagshub_logger
import mlflow
import logging
from visualize import visualize_predict_vs_gt

def dagshub_log(params,eval_meas):
        with dagshub_logger() as logger:
            logger.log_hyperparams({"model_name": params['model_name'],'type_loss':params['type_loss']
            })
            logger.log_metrics(recall=eval_meas['recall'], precision=eval_meas['precision'], f1_score=eval_meas['f1'])

        # logging the scores with mlflow
        mlflow.log_params({"model_name": params['model_name'],'type_loss':params['type_loss']})

        mlflow.log_metrics({
            "test_set_recall": eval_meas['recall'],
            "test_set_precision": eval_meas['precision'],
            "test_set_f1_score": eval_meas['f1'],
        })

        logging.log(logging.INFO, "Saving autoencoder...")   

def log_metrics_plot(df,params,eval_meas):
    dagshub_log(params,eval_meas)
    for idx in df.patient_id.unique():
        _ = visualize_predict_vs_gt(df,patient_id=idx,model_name=params['model_name'])
        mlflow.log_artifact('test_eval/gt_vs_pred_{}.html'.format(idx))