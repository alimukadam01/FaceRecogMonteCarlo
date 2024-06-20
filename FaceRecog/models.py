from django.db import models

# Create your models here.

class Student(models.Model):
    name = models.CharField(max_length=256)
    roll = models.CharField(max_length=256)
    face_capture_status = models.BooleanField(default=False)

class Attendance(models.Model):
    student = models.ForeignKey(Student, on_delete = models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add = True)
    
    