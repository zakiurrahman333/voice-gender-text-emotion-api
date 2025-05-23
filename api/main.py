from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
from api.utils import predict_gender
import uvicorn
from api.main import app  # or just `from main import app` if no folder
app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        gender = predict_gender(temp_path)
    except Exception as e:
        os.remove(temp_path)
        return JSONResponse(status_code=500, content={"error": str(e)})

    os.remove(temp_path)
    return {"predicted_gender": gender}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
