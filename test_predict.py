import requests

url = "http://127.0.0.1:10000/predict/"
file_path = r"C:\Users\Lenovo\Downloads\voice_gender2\dataset\male\0_50_0.wav"

with open(file_path, "rb") as f:
    files = {"file": (file_path, f, "audio/wav")}
    response = requests.post(url, files=files)

print("Server response:", response.json())
