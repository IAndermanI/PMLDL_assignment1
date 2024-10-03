import pickle
from pydantic import BaseModel
from fastapi import FastAPI
import numpy as np

with open("/app/models/model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

class PredictionRequest(BaseModel):
    features: list

@app.post("/predict")
def predict(request: PredictionRequest):
    input_features = np.array(request.features).reshape(1, -1)
    prediction = model.predict(input_features)[0]
    return {"prediction": prediction}
