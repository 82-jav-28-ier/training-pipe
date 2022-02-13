from os import getenv, path

import boto3
import pandas as pd
from joblib import dump
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, recall_score, 
    plot_confusion_matrix, precision_score, plot_roc_curve
)

from sklearn.ensemble import RandomForestClassifier

print('Downloading file')
LOCAL_BASE_PATH = getenv('BASE_PATH')
BUCKET_STORAGE = getenv('BUCKET_STORAGE')
REMOTE_TRAINED_PATH = getenv('REMOTE_TRAINED_PATH')
REMOTE_MODEL_PATH = getenv('REMOTE_MODEL_PATH') 

LOCAL_DATA_PATH = path.join(LOCAL_BASE_PATH,
                      'trained_model.parquet')
LOCAL_MODEL_PATH = path.join(LOCAL_BASE_PATH,
                      'model_risk.joblib')

s3 = boto3.resource('s3')
s3.Object(BUCKET_STORAGE, REMOTE_TRAINED_PATH).download_file(LOCAL_DATA_PATH)
print('File downloaded')


df = pd.read_parquet(LOCAL_DATA_PATH)
cust_df = df.copy()
cust_df.fillna(0, inplace=True)
# Split target and features
Y = cust_df['status']
cust_df.drop(['status'], axis=1, inplace=True)
# remove id from features
X = cust_df.drop(['id'], axis=1)

# Using Synthetic Minority Over-Sampling Technique(SMOTE) to overcome sample imbalance problem.
Y = Y.astype('int')
X_balance, Y_balance = SMOTE().fit_resample(X, Y)
X_balance = pd.DataFrame(X_balance, columns=X.columns)

# split train and test data
X_train, X_test, y_train, y_test = train_test_split(X_balance,Y_balance, 
                                                    stratify=Y_balance, test_size=0.3,
                                                    random_state = 123)

# train model
model = RandomForestClassifier(n_estimators=5)

model.fit(X_train, y_train)
y_predict = model.predict(X_test)

print('Accuracy Score is {:.5}'.format(accuracy_score(y_test, y_predict)))
print('Precision Score is {:.5}'.format(precision_score(y_test, y_predict)))
print('Recall Score is {:.5}'.format(precision_score(y_test, y_predict)))
print(pd.DataFrame(confusion_matrix(y_test,y_predict)))

# save model on disk
dump(model, LOCAL_MODEL_PATH) 

# store on s3
s3.meta.client.upload_file(LOCAL_MODEL_PATH, BUCKET_STORAGE, REMOTE_MODEL_PATH)