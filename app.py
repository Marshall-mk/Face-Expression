from fastapi import FastAPI
from inference import FEPredictor
app = FastAPI(title="MLOps Basics App-S1")
predictor = FEPredictor("./models/1/emotionModel.hdf5")

@app.get("/")
async def home():
    return "<h2>This is a sample CV Project</h2>"

@app.get("/predict")
async def get_prediction(img_path: str):
    return predictor.infer(img_path)