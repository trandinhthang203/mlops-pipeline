import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml

def get_train_data(filepath: str) -> tuple[pd.DataFrame]:
    try:
        df = pd.read_csv(filepath)
        X_train = df.drop(columns=['Potability'], axis=1)
        y_train = df['Potability']
        return X_train, y_train
    except Exception as e:
        raise Exception(f'Error: ', e)


def get_n_estimators(filepath: str) -> int:
    try:
        return yaml.safe_load(open(filepath))['model-building']['n_estimators']
    except Exception as e:
        raise Exception(f'Error: ', e)
    

def training_model(n_estimators: int, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
    try:
        clf = RandomForestClassifier(n_estimators=n_estimators)
        clf.fit(X_train, y_train)
        pickle.dump(clf, open('models/random_forest_model.pkl', 'wb'))
    except Exception as e:
        raise Exception(f'Error: ', e)
    

def main():
    try:
        DATA_PATH = os.path.join('data', 'processed', 'train_processed.csv')
        X_train, y_train = get_train_data(DATA_PATH)
        n_estimators = get_n_estimators('params.yaml')
        training_model(n_estimators, X_train, y_train)
    except Exception as e:
        raise Exception(f'Error: ', e)
    
    
if __name__ == '__main__':
    main()