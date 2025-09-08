import os
import cv2
import tempfile
import numpy as np
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


# Define colors for each class (BGR format)
CLASS_COLORS = {
    "rach": (0, 0, 255),       # Red
    "vo_kinh": (0, 255, 0),    # Green
    "mop_lom": (255, 0, 0),    # Blue
    "be_den": (0, 255, 255),   # Yellow
    "tray_son": (255, 0, 255), # Magenta
    "mat_bo_phan": (255, 255, 0), # Cyan
    "thung": (128, 128, 128)   # Gray
}


class DamageDetectView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        if 'file' not in request.FILES:
            return Response({"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)

        uploaded = request.FILES['file']
        model = get_model()
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
                # ✅ If segmentation masks exist, resize and overlay them
                if hasattr(r, 'masks') and r.masks is not None:
                    masks = r.masks.data.cpu().numpy()  # (num_masks, mask_h, mask_w)
                    for idx, mask in enumerate(masks):
                        # Resize mask to match original image size
                        mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
                        cls_idx = int(r.boxes[idx].cls[0])
                        label = model.names[cls_idx]
                        color = np.array(CLASS_COLORS.get(label, (0, 255, 0)), dtype=np.uint8)
                        # Overlay mask on image
                        img[mask_resized > 0.5] = img[mask_resized > 0.5] * 0.5 + color * 0.5

                # Process bounding boxes
                for box in r.boxes:
                    xyxy = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_idx = int(box.cls[0])
                    label = model.names[cls_idx]
                    color = CLASS_COLORS.get(label, (0, 255, 0))  # Default green

                    detections.append({
                        "class": label,
                        "confidence": conf,
                        "bbox": [float(x) for x in xyxy]
                    })

                    # Draw bounding box with class-specific color
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, f"{label} {conf:.2f}", (x1, max(15, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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
