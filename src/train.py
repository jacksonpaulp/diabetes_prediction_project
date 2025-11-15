import os

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, recall_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import pickle


FILENAME = os.path.join('data', 'diabetes_012_health_indicators_BRFSS2015.csv')
print(FILENAME)
MODELNAME = os.path.join('models', 'model.bin')

def load_data(filename):
    data = pd.read_csv(filename)
    data.columns = data.columns.str.lower()

    target_column = 'diabetes_012'

    data[target_column] = data[target_column].replace({1:0})
    data[target_column] = data[target_column].replace({2:1})

    return data

def train_model(data):

    target_column = 'diabetes_012'
    X = data.drop(columns=[target_column])
    y = data[target_column]

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.2,
        max_depth=3,
        subsample=0.7,
        colsample_bytree=1.0,
        min_child_weight=1,
        scale_pos_weight=6.176998974431517,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )

    model.fit(X, y)

    return model

def save_model(model, filename=MODELNAME):
    with open(filename, 'wb') as f_out:
        pickle.dump(model, f_out)


if __name__ == '__main__':
    data = load_data(FILENAME)
    model = train_model(data)
    save_model(model)

    print("Model training and saving completed.")


