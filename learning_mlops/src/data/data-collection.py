import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml


def load_params(filepath : str) -> float:
    try:
        test_size = yaml.safe_load(open(filepath, 'r'))['data-collection']['test_size']
        return test_size
    except Exception as e:
        raise Exception(f'Error: ', e)

def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f'Error: ', e)

def split_data(data: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
        return train_data, test_data
    except Exception as e:
        raise Exception(f'Error: ', e)

def save_data(df: pd.DataFrame, filepath: str) -> None:
    try:
        df.to_csv(filepath, index=False)
    except Exception as e:
        raise Exception(f'Error: ', e)

def main():
    DATA_PATH = os.path.join('data', 'external', 'water_potability.csv')
    RAW_DATA_PATH = os.path.join('data', 'raw')
    test_size = load_params('params.yaml')

    try:
        data = load_data(DATA_PATH)
        train_data, test_data = split_data(data, test_size)

        os.makedirs(RAW_DATA_PATH, exist_ok=True)
        save_data(train_data, os.path.join(RAW_DATA_PATH, 'train.csv'))
        save_data(test_data, os.path.join(RAW_DATA_PATH, 'test.csv'))
    except Exception as e:
        raise Exception(f'Error: ', e)
    
if __name__ == '__main__':
    main()