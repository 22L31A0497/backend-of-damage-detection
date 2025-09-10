import os
import requests
from ultralytics import YOLO

MODEL_PATH = "/tmp/best.pt"  # stored in temp folder at runtime

def download_file_from_google_drive(file_id, dest_path):
    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={"id": file_id}, stream=True)
    
    # If Google Drive requires confirmation (large files)
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            response = session.get(URL, params={"id": file_id, "confirm": value}, stream=True)
            break

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)


def download_model():
    file_id = "1jeL98r_B4H7JJpnU5geZKTjp5AhdbAOm"  # your file ID
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading YOLO model from Google Drive...")
        download_file_from_google_drive(file_id, MODEL_PATH)
    else:
        print("✅ Model already downloaded.")
    return MODEL_PATH


def load_model():
    path = download_model()
    return YOLO(path)
