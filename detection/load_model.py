import os
import gdown
from ultralytics import YOLO

MODEL_PATH = "/tmp/best.pt"
FILE_ID = "1jeL98r_B4H7JJpnU5geZKTjp5AhdbAOm"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading YOLO model from Google Drive with gdown...")
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    else:
        print("✅ Model already downloaded.")
    return MODEL_PATH

def load_model():
    path = download_model()
    return YOLO(path)
