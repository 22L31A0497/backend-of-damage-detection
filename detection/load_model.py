import os
import requests
from ultralytics import YOLO

MODEL_PATH = "/tmp/best.pt"  # stored in temp folder at runtime

def download_model():
    # 🔹 Replace with your own file ID from Google Drive
    file_id = "1jeL98r_B4H7JJpnU5geZKTjp5AhdbAOm"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading YOLO model from Google Drive...")
        r = requests.get(url)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
    else:
        print("✅ Model already downloaded.")
    return MODEL_PATH

def load_model():
    path = download_model()
    return YOLO(path)
