from fastapi import FastAPI, UploadFile, File
from app.classify import predict
import shutil

app = FastAPI()

@app.post("/predict")
async def get_prediction(file: UploadFile = File(...)):
    with open("temp.jpg", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    result = predict("temp.jpg")
    return {"class": result}
