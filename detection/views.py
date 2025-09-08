import os
import cv2
import tempfile
from django.core.files.base import ContentFile
from django.apps import apps
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from .models import DetectionResult


# ✅ Define YOLO model loader (lazy load, only once)
def get_model():
    cfg = apps.get_app_config('detection')
    if getattr(cfg, 'yolo_model', None) is None:
        from ultralytics import YOLO
        model_path = os.path.join(os.path.dirname(__file__), "best.pt")
        cfg.yolo_model = YOLO(model_path)
    return cfg.yolo_model


class DamageDetectView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        if 'file' not in request.FILES:
            return Response({"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)

        uploaded = request.FILES['file']
        model = get_model()  # ✅ now defined
        suffix = os.path.splitext(uploaded.name)[1] or ".jpg"

        # Save temp file for YOLO
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            for chunk in uploaded.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        try:
            results = model.predict(source=tmp_path, conf=0.25, verbose=False)
            img = cv2.imread(tmp_path)
            detections = []

            for r in results:
                for box in r.boxes:
                    xyxy = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = model.names[cls] if hasattr(model, "names") else str(cls)

                    detections.append({
                        "class": label,
                        "confidence": conf,
                        "bbox": [float(x) for x in xyxy]
                    })

                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"{label} {conf:.2f}", (x1, max(15, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Save annotated image in memory
            _, img_encoded = cv2.imencode('.jpg', img)
            annotated_file = ContentFile(img_encoded.tobytes(), name=f"annotated_{uploaded.name}")

            # Save record in DB
            record = DetectionResult.objects.create(
                image=uploaded,
                annotated_image=annotated_file,
                detections_json=detections
            )

            return Response({
                "id": record.id,
                "detections": detections,
                "original_image_url": request.build_absolute_uri(record.image.url),
                "annotated_image_url": request.build_absolute_uri(record.annotated_image.url),
            })

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        finally:
            try:
                os.remove(tmp_path)
            except:
                pass
