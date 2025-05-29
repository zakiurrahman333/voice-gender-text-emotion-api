import shutil
import os
from .utils import predict_gender
# Optionally keep this as a helper function, but do NOT run as FastAPI app
def handle_gender_prediction(file: UploadFile):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        gender = predict_gender(temp_path)
    except Exception as e:
        os.remove(temp_path)
        raise RuntimeError(f"Prediction failed: {e}")

    os.remove(temp_path)
    return gender

