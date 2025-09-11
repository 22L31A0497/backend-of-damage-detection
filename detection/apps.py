# detection/apps.py
from django.apps import AppConfig

class DetectionConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'detection'
    yolo_model = None

    def ready(self):
        try:
            from .load_model import load_model
            DetectionConfig.yolo_model = load_model()   # store at class level
            print("✅ YOLO model loaded once in AppConfig")
        except Exception as e:
            print("❌ Error loading YOLO model:", e)
