import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import json

def get_data(filepath: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        test_data = pd.read_csv(filepath)
        X_test = test_data.iloc[:, 0:-1].values
        y_test = test_data.iloc[:, -1].values
        return X_test, y_test
    except Exception as e:
        raise Exception(f'Error: ', e)

def write_metrics(X_test: pd.DataFrame, y_test: pd.DataFrame) -> None:
    try:
        model = pickle.load(open('models/random_forest_model.pkl', 'rb'))
        y_pred = model.predict(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        with open('reports/model_evaluation_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
    except Exception as e:
        raise Exception(f'Error: ', e)


def main():
    DATA_PATH = os.path.join('data', 'processed', 'test_processed.csv')
    try:
        X_test, y_test = get_data(DATA_PATH)
        write_metrics(X_test, y_test)
    except Exception as e:
        raise Exception(f'Error: ', e)
    
if __name__ == '__main__':
    main()