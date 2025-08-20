from fastapi import FastAPI
import pickle
import pandas as pd
from data_model import Water


app = FastAPI()

with open("D:/ml-pipeline/random_forest_model.pkl", "rb") as F:
    model = pickle.load(F)

@app.get('/')
def index():
    return 'Welcome'


@app.post('/predict')
def predict(water: Water):
    sample = pd.DataFrame({
        'ph': [water.ph],
        'Hardness': [water.Hardness],
        'Solids': [water.Solids],
        'Chloramines': [water.Chloramines],
        'Sulfate': [water.Sulfate],
        'Conductivity': [water.Conductivity],
        'Organic_carbon': [water.Organic_carbon],
        'Trihalomethanes': [water.Trihalomethanes],
        'Turbidity': [water.Turbidity]
    })

    predict_sample = model.predict(sample)
    if predict_sample == 1:
        return 'consumable'
    else:
        return 'not consumable'