import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import json
import yaml
from dvclive import Live

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
        params = yaml.safe_load(open('params.yaml', 'r'))
        test_size = params['data-collection']['test_size']
        n_estimators = params['model-building']['n_estimators']

        model = pickle.load(open('models/random_forest_model.pkl', 'rb'))
        y_pred = model.predict(X_test)

        with Live(save_dvc_exp= True) as live:
            live.log_metric('accuracy', accuracy_score(y_test, y_pred))
            live.log_metric('precision', precision_score(y_test, y_pred))
            live.log_metric('f1-score', f1_score(y_test, y_pred))
            live.log_metric('recall_score', recall_score(y_test, y_pred))

            live.log_param("test_size", test_size)
            live.log_param("n_estimators",n_estimators)

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