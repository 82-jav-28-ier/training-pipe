#!/usr/bin/env python
import signal
import sys
from os import getenv, path


import pandas as pd
from joblib import dump
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,
                             confusion_matrix,
                             precision_score)

from sklearn.ensemble import RandomForestClassifier


input_dir = "/opt/ml/input"
model_dir = "/opt/ml/model"
output_dir = "/opt/ml/output"

# we're arbitrarily going to iterate through 5 epochs here, a real algorithm
# may choose to determine the number of epochs based on a more realistic
# convergence criteria
num_epochs = 5
channel_name = "training"
terminated = False


def main():
    # trapping signals and responding to them appropriately is required by
    # SageMaker spec
    trap_signal()

    # writing to a failure file is also part of the spec
    failure_file = output_dir + "/failure"
    LOCAL_DATA_PATH = path.join(input_dir, 
                                channel_name,
                                'trained_model.parquet')
    LOCAL_MODEL_PATH = path.join(model_dir,
                                 'model_risk.joblib')
    try:
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

    except Exception as e:
        print("Failed to train: %s" % (sys.exc_info()[0]))
        print(f'Error is: {e}')
        touch(failure_file)
        raise




def touch(fname):
    open(fname, "wa").close()


def on_terminate(signum, frame):
    print("caught termination signal, exiting gracefully...")
    global terminated
    terminated = True


def trap_signal():
    signal.signal(signal.SIGTERM, on_terminate)
    signal.signal(signal.SIGINT, on_terminate)


if __name__ == "__main__":
    main()