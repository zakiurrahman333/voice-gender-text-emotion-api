from fastapi import FastAPI, Form, UploadFile, File
from emotion.main import predict_emotion_from_text  # ✅ Import correct function
from gender.api.main import predict_gender
import shutil
import os

app = FastAPI()

@app.post("/emotion-predict")
async def emotion_predict(text: str = Form(...)):  # ✅ Accept text via form field
    emotion, confidence, *error = predict_emotion_from_text(text)
    if error:
        return {
            "status": "error",
            "message": f"Error processing text: {error[0]}"
        }
    return {
        "status": "success",
        "emotion": emotion,
        "confidence": confidence
    }

@app.post("/gender-predict")
async def gender_predict(file: UploadFile = File(...)):  # ✅ Accept audio file
    with open("temp.wav", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    result = predict_gender("temp.wav")
    os.remove("temp.wav")
    return {
        "status": "success",
        "gender": result
    }

# used this on browser  http://localhost:9000/docs#/default/gender_predict_gender_predict_post 