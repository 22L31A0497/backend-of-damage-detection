from django.db import models

class DetectionResult(models.Model):
    image = models.ImageField(upload_to="detections/")  # original uploaded image
    annotated_image = models.ImageField(upload_to="detections/annotated/")  # annotated output
    detections_json = models.JSONField()  # store detections as JSON
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Detection {self.id} - {self.created_at}"
