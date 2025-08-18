import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join('src', 'data', 'water_potability.csv')

data = pd.read_csv(DATA_PATH)

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

RAW_DATA_PATH = os.path.join('data', 'raw')
os.makedirs(RAW_DATA_PATH, exist_ok=True)

train_data.to_csv(os.path.join(RAW_DATA_PATH, 'train.csv'), index=False)
test_data.to_csv(os.path.join(RAW_DATA_PATH, 'test.csv'), index=False)