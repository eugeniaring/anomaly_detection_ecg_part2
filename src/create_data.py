import pandas as pd
import os
import yaml


def read_dataset(path='ecg_data/train.csv',mode='train'):
    df = pd.read_csv(path,index_col=0)
    df.dropna(inplace=True)
    return df

def count_anomalies(test_df):
    count_anomalies = (test_df.cycle_id==1111).sum()
    perc_anomalies = round(count_anomalies/len(test_df),4)
    print('Percentage of anomalies in test set: {}/{}={}\n'.format(count_anomalies,len(test_df),perc_anomalies))

def create_X_y(df):
    return df[['heart_rate','hr_diff','peak_label']].values,df['label'].values


def read_yaml(namefile):
    f = open(namefile,'rb')
    diz = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    return diz

def join_path(path,file):
    return os.path.join(path,file)

def normalize_data(X):
    min_val,max_val = X.min(),X.max()
    return (X - min_val) / (max_val - min_val)

class ECG_data():
    def __init__(self,path):
        self.path = path
        self.train_df = pd.DataFrame()
        self.test_df = pd.DataFrame()
        self.l_files = os.listdir(self.path)

    def read_data(self,path_f,idx):
        df = pd.read_csv(path_f,header=None)
        df.columns = ['cycle_id','timestamp','time_gap','heart_rate']
        df['patient_id'] = idx
        df['heart_rate'] = df['heart_rate'].interpolate(method='polynomial', order=2)
        df['hr_diff'] = df['heart_rate'].diff()
        df['hr_diff'] = df['hr_diff'].apply(lambda x: abs(x))

        ## create peak label to identify peaks
        df['hr_diff_4'] = df['heart_rate'].diff(4)
        df['hr_diff_4'] = df['hr_diff_4'].apply(lambda x: abs(x))
        peak_thresh = df['hr_diff_4'].mean()+df['hr_diff_4'].std()
        df['peak_label'] = df['hr_diff_4'].apply(lambda x: 1 if x>peak_thresh else 0)
        
        #df['hr_diff'] = df['hr_diff'].fillna(0)
        df.dropna(subset='hr_diff',inplace=True)
        df[['heart_rate','hr_diff']] = normalize_data(df[['heart_rate','hr_diff']].values)
        return df

    def concat_data(self,df1,df2):
        return pd.concat([df1,df2])

    def export_csv(self,path,df,filename):
        df.to_csv(join_path(path,filename))

    def create_train_test(self,train_path,train_namefile='train.csv',test_namefile='test.csv'):
        for idx, f in enumerate(self.l_files):
            path_f = join_path(self.path,f)
            if idx not in [22,27,39]:
              df = self.read_data(path_f,idx)
            count_anomalies = (df.cycle_id==1111).sum()
            if count_anomalies==0:
                df['label'] = 0
                self.train_df = self.concat_data(self.train_df,df)
            else:
                df['label'] = df['cycle_id'].apply(lambda x: 0 if x!=1111 else 1)
                self.test_df = self.concat_data(self.test_df,df)

        self.export_csv(train_path,self.train_df,train_namefile)
        self.export_csv(train_path,self.test_df,test_namefile)


if __name__ == "__main__":
    ecg = ECG_data('ecg_data/all_data')
    ecg.create_train_test('ecg_data')

