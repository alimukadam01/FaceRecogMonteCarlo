from datetime import datetime

from rest_framework import views
from rest_framework.views import APIView
from rest_framework.viewsets import ModelViewSet
from rest_framework.response import Response
from rest_framework.decorators import action
from rest_framework import status

from FaceRecog.serializers import StudentSerializer, AttendanceSerializer

from .config.main import capture_face_and_build_data
from .config.model_train import train_and_evaluate_model
from .config.model_predict import main as predict_face

from .models import Student, Attendance

# Create your views here.

class StudentViewSet(ModelViewSet):
    queryset = Student.objects.all()
    
    def get_serializer_class(self):
        
        if self.action == 'attendance':
            return AttendanceSerializer
        return StudentSerializer

    @action(methods=['GET'], detail=True)
    def attendance(self, request, pk=None):
        queryset = Attendance.objects.filter(student_id = pk)
        serializer = AttendanceSerializer(queryset, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    

class CaptureFaceView(APIView):

    def post(self, request):
        method = request.method

        if method == 'POST':
            roll = request.data.get("roll_no")
            
            if not roll:
                return Response({
                    "detail": "Please enter Student Roll Number"
                }, status=status.HTTP_400_BAD_REQUEST)
            
            student = Student.objects.get(roll=roll)
            if not student:
                return Response({
                    "detail": "No student found with the provided Roll No. Please make sure you're registered."
                }, status=status.HTTP_400_BAD_REQUEST)

            try:
                capture_face_and_build_data(student.roll)
            except Exception as error:
                print(error)
                return Response({
                    "detail": "error capturing face."
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            student.face_capture_status = True
            student.save()
            
            try:
                train_and_evaluate_model()
            except Exception as error:
                print(error)
                return Response({
                    "detail": "error training model."
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            return Response({
                "detail": "Face Registered Succesfully."
            }, status=status.HTTP_200_OK)
        

class PredictFace(APIView):
    def post(self, request):

        if request.method == 'POST':

            roll = predict_face()

            student = Student.objects.get(roll = roll)

            if not Attendance.objects.filter(student = student, timestamp__date = datetime.now().date()).exists():
                attendance = Attendance.objects.create(student=student)
                serializer = AttendanceSerializer(attendance)
                return Response({
                    "detail": {
                        "name": attendance.student.name,
                        "date": attendance.timestamp,
                        "status": "Attendance marked successfully"
                    }
                }, status=status.HTTP_201_CREATED)
            return Response({'detail': 'Attendance already marked.'}, status=status.HTTP_400_BAD_REQUEST)

