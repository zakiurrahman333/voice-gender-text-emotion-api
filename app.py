from fastapi import FastAPI, Form, UploadFile, File
from emotion.main import predict_emotion_from_text
from gender.api.utils import predict_gender  # ✅ Use utils, not main
import shutil
import os

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Welcome to the Combined Emotion and Gender Prediction API"}

@app.post("/emotion-predict")
async def emotion_predict(text: str = Form(...)):
    emotion, confidence, *error = predict_emotion_from_text(text)
    if error:
        return {"status": "error", "message": f"Error processing text: {error[0]}"}
    return {"status": "success", "emotion": emotion, "confidence": confidence}

@app.post("/gender-predict")
async def gender_predict(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        gender = predict_gender(temp_path)
    except Exception as e:
        os.remove(temp_path)
        return {"status": "error", "message": str(e)}
    os.remove(temp_path)
    return {"status": "success", "gender": gender}



# used this on browser  http://localhost:9000/docs#/default/gender_predict_gender_predict_post 