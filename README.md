# Anomaly Detection project in ECG signals - Part 2
The goal of this project is to detect irregular heart rhythms from ECG signals. Since irregular heart rhythms usually constitute a minority compared to normal ones, we are solving an anomaly detection problem. It's not only anomaly detection, but it's also peak detection. So, we need to identify peaks and establish if these peaks are anomalous.

The article with the explanations is An End to End Anomaly Detection App for ECG signals with DagsHub, SageMaker, and Streamlit.

## Tools used in the project

* [DVC](https://dvc.org/)
* [DagsHub](https://dagshub.com/)
* [MLflow](https://mlflow.org/)
* [AWS Lambda](https://aws.amazon.com/lambda/?nc1=h_ls)
* [Streamlit](https://streamlit.io/)

## Project Structure

* ```src```: contains the scripts to train and track experiments of the ml model
* ```ecg_data```: contains the data

## Set up the environment

1. Clone the repository

```
git clone --branch branch_2 https://dagshub.com/eugenia.anello/anomaly_detection_ecg_part2.git
````

2. Create a virtual environment in Python

Windows commands

```
py -m venv venv
echo venv >> .gitignore
venv/Scripts/activate 
````

Linux/Ubuntu commands
```
python3 -m venv venv
source venv/bin/activate
```
3. Install the requirements

```
pip install -r requirements.txt
````

4. Pull the data

```
dvc pull
```

## Set up to obtain data pipeline
Install and initialize dvc
```
pip install dvc
dvc init
```
Add DagsHub DVC remote
```
dvc remote add origin your_repository_url.dvc
```
Set up credentials
```
dvc remote add origin your_repository_url.dvc
```

```
dvc remote modify origin --local auth basic 
dvc remote modify origin --local user your_username 
dvc remote modify origin --local password your_token
```

add data you want to store in DagsHub
```
dvc add ecg_data/all_data
```
Add and push the untracked and modified files using Git tracking
```
git add .dvc .dvcignore .gitignore
git commit -m "Initialize DVC"
git push
```

After creating dvc.yaml, run the command
```
dvc repro
dvc push
```
Push changes in GitHub
```
git add
git commit
```
## Split data into training and test set

```
python src/create_data.py
```

## Train and evaluate model 
```
python src/train.py
```
