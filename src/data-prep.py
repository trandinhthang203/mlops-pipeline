import pandas as pd
import os
import numpy as np


train_data = pd.read_csv(os.path.join('data', 'raw', 'train.csv'))
test_data = pd.read_csv(os.path.join('data', 'raw', 'test.csv'))

def fill_missing_with_mean(df):
    for columns in df.columns:
        if df[columns].isnull().any():
            medien_value = df[columns].median()
            df[columns].fillna(medien_value, inplace=True)
    return df


train_processed = fill_missing_with_mean(train_data)
test_processed = fill_missing_with_mean(test_data)

DATA_PATH = os.path.join('data', 'processed')
os.makedirs(DATA_PATH, exist_ok=True)

train_processed.to_csv(os.path.join(DATA_PATH, 'train_processed.csv'), index=False)
test_processed.to_csv(os.path.join(DATA_PATH, 'test_processed.csv'), index=False)