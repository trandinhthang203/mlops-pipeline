import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

train_data = pd.read_csv(os.path.join('data', 'processed', 'train_processed.csv'))
test_data = pd.read_csv(os.path.join('data', 'processed', 'test_processed.csv'))

X_train = train_data.iloc[:,0:-1].values
y_train = train_data.iloc[:,-1].values

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

pickle.dump(clf, open('random_forest_model.pkl', 'wb'))

