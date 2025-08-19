import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import json

test_data = pd.read_csv(os.path.join('data', 'processed', 'test_processed.csv'))
X_test = test_data.iloc[:, 0:-1].values
y_test = test_data.iloc[:, -1].values

model = pickle.load(open('random_forest_model.pkl', 'rb'))
y_pred = model.predict(X_test)

metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, average='weighted'),
    'recall': recall_score(y_test, y_pred, average='weighted'),
    'f1_score': f1_score(y_test, y_pred, average='weighted')
}

with open('model_evaluation_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)
