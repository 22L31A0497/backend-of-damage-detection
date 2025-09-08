# detection/apps.py
from django.apps import AppConfig
import os

class DetectionConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'detection'
    yolo_model = None

    def ready(self):
        # runs once when Django starts
        try:
            from ultralytics import YOLO
            model_path = os.path.join(os.path.dirname(__file__), "best.pt")
            # load model and store on config object
            self.yolo_model = YOLO(model_path)
            print("YOLO model loaded from:", model_path)
        except Exception as e:
            # print error to console - fix by checking model path / deps
            print("Error loading YOLO model in DetectionConfig.ready():", e)
