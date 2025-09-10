from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
import os
import cv2
import tempfile
import numpy as np
from django.core.files.base import ContentFile
from .models import DetectionResult
from detection.load_model import load_model  # ✅ Import load_model

# ✅ Load YOLO model once
model = load_model()

# ✅ Define colors for each class
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
            total_mask_pixels, total_pixels = 0, img.shape[0] * img.shape[1]
            max_conf = 0.0

            for r in results:
                if hasattr(r, 'masks') and r.masks is not None:
                    masks = r.masks.data.cpu().numpy()  # (num_masks, mask_h, mask_w)

                    for idx, mask in enumerate(masks):
                        # Resize mask to original image size
                        mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))

                        # Count pixels in this mask
                        mask_pixels = np.sum(mask_resized > 0.5)
                        total_mask_pixels += mask_pixels

                        # Get class info
                        cls_idx = int(r.boxes[idx].cls[0])
                        label = model.names[cls_idx]
                        conf = float(r.boxes[idx].conf[0])
                        max_conf = max(max_conf, conf)

                        # ✅ Use tuple for drawing, numpy array for blending
                        color_tuple = tuple(CLASS_COLORS.get(label, (0, 255, 0)))
                        color_array = np.array(color_tuple, dtype=np.uint8)

                        # Overlay mask
                        img[mask_resized > 0.5] = (
                            img[mask_resized > 0.5] * 0.5 + color_array * 0.5
                        )

                        # Per-object damage percentage
                        obj_damage = (mask_pixels / total_pixels * 100) if total_pixels > 0 else 0

                        # Add detection entry
                        detections.append({
                            "class": label,
                            "confidence": round(conf * 100, 2),
                            "damagePercentage": round(obj_damage, 2),
                            "bbox": [float(x) for x in r.boxes[idx].xyxy[0].tolist()]
                        })

                        # Draw box + label
                        x1, y1, x2, y2 = map(int, r.boxes[idx].xyxy[0].tolist())
                        cv2.rectangle(img, (x1, y1), (x2, y2), color_tuple, 2)
                        cv2.putText(
                            img,
                            f"{label} {conf:.2f}",
                            (x1, max(15, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color_tuple,
                            2
                        )

            # ✅ Overall damage percentage
            overall_damage = (total_mask_pixels / total_pixels * 100) if total_pixels > 0 else 0

            # Save annotated image in memory
            _, img_encoded = cv2.imencode('.jpg', img)
            annotated_file = ContentFile(img_encoded.tobytes(), name=f"annotated_{uploaded.name}")

            # Save record in DB
            record = DetectionResult.objects.create(
                image=uploaded,
                annotated_image=annotated_file,
                detections_json=detections
            )

            # ✅ Final response
            return Response({
                "id": record.id,
                "overallDamagePercentage": round(overall_damage, 2),
                "overallConfidence": round(max_conf * 100, 2),
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
