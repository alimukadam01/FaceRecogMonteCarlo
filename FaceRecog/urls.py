from django.urls import path
from rest_framework.routers import DefaultRouter
from .views import CaptureFaceView, PredictFace, StudentViewSet

router = DefaultRouter()
router.register('students', StudentViewSet)

urlpatterns = [
    path('facecapture/', CaptureFaceView.as_view(), name='face-capture'),
    path('predict/', PredictFace.as_view(), name='predict'),
] + router.urls