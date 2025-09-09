from django.urls import path
from .views import DamageDetectView

# urlpatterns = [
#     path("detect/", DamageDetectView.as_view(), name="damage-detect"),
# ]
urlpatterns = [
    path('analyze/', DamageDetectView.as_view(), name='damage_detect'),
]