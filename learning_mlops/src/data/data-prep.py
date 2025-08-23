import pandas as pd
import os

def get_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f'Error: ', e)

def fill_missing_with_mean(df):
    try:
        for columns in df.columns:
            if df[columns].isnull().any():
                medien_value = df[columns].median()
                df[columns].fillna(medien_value, inplace=True)
        return df
    except Exception as e:
        raise Exception(f'Error: ', e)

def save_data_processed(df: pd.DataFrame, filepath: str) -> None:
    try:
        df.to_csv(filepath, index=False)
    except Exception as e:
        raise Exception(f'Error: ', e)
    
def main():
    TRAIN_RAW_PATH = os.path.join('data', 'raw', 'train.csv')
    TEST_RAW_PATH = os.path.join('data', 'raw', 'test.csv')
    DATA_PATH = os.path.join('data', 'processed')
    os.makedirs(DATA_PATH, exist_ok=True)

    try:
        train_data = get_data(TRAIN_RAW_PATH)
        test_data = get_data(TEST_RAW_PATH)

        train_processed = fill_missing_with_mean(train_data)
        test_processed = fill_missing_with_mean(test_data)

        save_data_processed(train_processed, os.path.join(DATA_PATH, 'train_processed.csv'))
        save_data_processed(test_processed, os.path.join(DATA_PATH, 'test_processed.csv'))
    except Exception as e:
        raise Exception(f'Error: ', e)

if __name__ == '__main__':
    main()