from django.conf import settings
from rest_framework import serializers
from .models import Student, Attendance

class StudentSerializer(serializers.ModelSerializer):

    face_capture_status = serializers.BooleanField(read_only=True)
    ONR = serializers.SerializerMethodField()
    RLR = serializers.SerializerMethodField()
    TVR = serializers.SerializerMethodField()
    captured_images = serializers.SerializerMethodField()
    ON_grid = serializers.SerializerMethodField()
    RL_grid = serializers.SerializerMethodField()
    testing_graph = serializers.SerializerMethodField()
    training_graph = serializers.SerializerMethodField()

    base_path = f"{settings.BASE_URL}media/Dataset/"

    def get_ONR(self, obj,):
        if obj.face_capture_status:
            return f"{self.base_path}graph/{obj.roll}/occlusion_noise_ratio.jpg"
        return None
    
    def get_RLR(self, obj):
        if obj.face_capture_status:
            return f"{self.base_path}graph/{obj.roll}/rotation_lighting_ratio.jpg"
        return None
    
    def get_TVR(self, obj):
        if obj.face_capture_status:
            return f"{self.base_path}graph/{obj.roll}/train_valid_ratio.jpg"
        return None
    
    def get_captured_images(self, obj):
        if obj.face_capture_status:
            return f"{self.base_path}grid/{obj.roll}/CapturedImages.jpg"
        return None
    
    def get_ON_grid(self, obj):
        if obj.face_capture_status:
            return f"{self.base_path}grid/{obj.roll}/OcclusionAndNoise.jpg"
        return None

    def get_RL_grid(self, obj):
        if obj.face_capture_status:
            return f"{self.base_path}grid/{obj.roll}/RotationAndLighting.jpg"
        return None
    
    def get_testing_graph(self, obj):
        if obj.face_capture_status:
            return f"{self.base_path}grid/{obj.roll}/TestingSet.jpg"
        return None

    def get_training_graph(self, obj):
        if obj.face_capture_status:
            return f"{self.base_path}grid/{obj.roll}/TrainingSet.jpg"
        return None

    class Meta:
        model = Student
        fields = [
            'id', 'name', 'roll', 'face_capture_status', 'ONR', 'RLR', 'TVR', 'captured_images', 
            'ON_grid', 'RL_grid', 'testing_graph', 'training_graph'
        ]


class AttendanceSerializer(serializers.ModelSerializer):
    student = StudentSerializer()
    timestamp = serializers.DateTimeField()

    class Meta:
        model = Attendance
        fields = ['id', 'student', 'timestamp',]

