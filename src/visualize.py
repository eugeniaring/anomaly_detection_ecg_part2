import matplotlib.pyplot as plt
import plotly.graph_objects as go

def visualize_loss(history):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.savefig('test_eval/loss.jpg')
    plt.close()

def visualize_predict_vs_gt(df,patient_id,model_name,show=False,path='test_eval'):
    fig = go.Figure()
    
    ## part1: ecg signal
    df_patient = df[df.patient_id==patient_id]

    fig.add_trace(go.Scatter(
        x=df_patient['timestamp'],
        y=df_patient['heart_rate'],
        name="ECG Signal"       # this sets its legend entry
    ))
    
    ## part 2 ground truth
    if 1 in df['label'].values:
      df_patient_gt = df_patient[df_patient.label==1]

      fig.add_trace(go.Scatter(
          x=df_patient_gt['timestamp'],
          y=df_patient_gt['heart_rate'],
          mode="markers",
          marker = dict(size = 15, color = 'red', symbol = 'cross'),
          name="Ground Truth"
      ))
    ## part 3 prediction
    df_patient_pred = df_patient[df_patient.test_preds==1]

    fig.add_trace(go.Scatter(
        x=df_patient_pred['timestamp'],
        y=df_patient_pred['heart_rate'],
        mode="markers",
        name="Prediction"
    )) 

    ### merge all
    fig.update_layout(
      title={
          'text': "Ground Truth vs Prediction in Test Set",
          'y':0.9,
          'x':0.5,
          'xanchor': 'center',
          'yanchor': 'top'},
      xaxis_title="Timestamp in seconds",
      yaxis_title="Heart Rate",
      #legend_title="Legend Title"
  )
    if show==False:
      fig.write_html("{}/gt_vs_pred_{}_{}.html".format(path,patient_id,model_name))
      return fig
    else:
      fig.show()



# def visualize_ecg_patient(df,patient_id=0,show=False,path='ecg_data'):
#     df_patient = df[df.patient_id==patient_id]
#     plt.plot(df_patient['timestamp'], df_patient['heart_rate'])
#     if 1 in df_patient['label'].values:
#         plt.plot(df_patient[df_patient.label==1]['timestamp'], df_patient[df_patient.label==1]['heart_rate'], 'rx')
#     plt.title('ECG Signals of Patient {}'.format(patient_id))
#     plt.xlabel('Time in seconds')
#     plt.ylabel('Heart rate')
#     if show==False:
#       plt.savefig(f'{path}/example_{patient_id}.png')
#     else:
#       plt.show()  
#     plt.close()    

# def visualize_ecg_predicted(df,patient_id,show=False,path='ecg_data/test_eval'):
#     df_patient = df[df.patient_id==patient_id]
#     plt.plot(df_patient['timestamp'], df_patient['heart_rate'])
#     if 1 in df_patient['test_preds'].values:
#         plt.plot(df_patient[df_patient.test_preds==1]['timestamp'], df_patient[df_patient.test_preds==1]['heart_rate'], 'rx')
#     plt.title('ECG Signals of Patient {}'.format(patient_id))
#     plt.xlabel('Time in seconds')
#     plt.ylabel('Heart rate')
#     if show==False:
#       plt.savefig(f'{path}/example_{patient_id}_pred.png')
#     else:
#       plt.show()  
#     plt.close()    
