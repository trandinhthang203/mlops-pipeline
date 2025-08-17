import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join('data', 'water_potability.csv')
data = pd.read_csv(DATA_PATH)

print(data.head())

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

train_data.to_csv(os.path.join('data', 'raw', 'train.csv'), index=False)
test_data.to_csv(os.path.join('data', 'raw', 'test.csv'), index=False)