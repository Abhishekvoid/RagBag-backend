import uuid
from django.db import models
from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin
from django.utils import timezone
from django.core.validators import EmailValidator
from .manager import CustomUserManager


def user_document_path(instance, filename):
 
    if hasattr(instance, 'chapter') and instance.chapter and hasattr(instance.chapter, 'subject') and instance.chapter.subject:
     
        return f'{instance.user.id}/{instance.chapter.subject.id}/{instance.chapter.id}/{filename}'
    else:
    
        return f'{instance.user.id}/standalone/{filename}'

class CustomUserModel(AbstractBaseUser, PermissionsMixin):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    email = models.EmailField(unique=True, validators=[EmailValidator()], db_index=True)
    name = models.CharField(max_length=100)
    is_staff = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    date_joined = models.DateTimeField(default=timezone.now)
    last_login = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['name']

    objects = CustomUserManager()

    class Meta:
        db_table = 'auth_user'
        verbose_name = 'user'
        verbose_name_plural = 'users'

    def __str__(self):
        return self.email

class Subject(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(CustomUserModel, on_delete=models.CASCADE, related_name='subjects')
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} ({self.user.email})"

class Chapter(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(CustomUserModel, on_delete=models.CASCADE, related_name='chapters')
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE, related_name='chapters', null=True, blank=True)
    name = models.CharField(max_length=100)
    order = models.PositiveIntegerField(default=1)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        
        if self.subject:
            return f"{self.subject.name} - {self.name}"
        return f"Standalone Chapter - {self.name}"

class Document(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    chapter = models.ForeignKey(Chapter,on_delete=models.SET_NULL, related_name='documents', null=True, blank=True)
    user = models.ForeignKey(CustomUserModel, on_delete=models.CASCADE, related_name='documents')
    title = models.CharField(max_length=100)
    file = models.FileField(upload_to=user_document_path)
    file_type = models.CharField(max_length=10, blank=True)
    size_bytes = models.PositiveIntegerField(null=True, blank=True)
    
    
    extracted_text = models.TextField(blank=True)

    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('PROCESSING', 'Processing'),
        ('COMPLETED', 'Completed'),
        ('FAILED', 'Failed'),
    ]
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.title} ({self.file_type})"

    def save(self, *args, **kwargs):
      
        if self.file and not self.pk:
            self.size_bytes = self.file.size
            ext = self.file.name.split(".")[-1].lower()
            self.file_type = ext
        super().save(*args, **kwargs)

class ChatSession(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(CustomUserModel, on_delete=models.CASCADE, related_name='chat_sessions')
    subject = models.ForeignKey(Subject, null=True, blank=True, on_delete=models.SET_NULL, related_name='chat_sessions')
    chapter = models.ForeignKey(Chapter, null=True, blank=True, on_delete=models.SET_NULL, related_name='chat_sessions')
    title = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    context_snapshot = models.TextField(blank=True)

    def __str__(self):
        return f"Session {self.id} by {self.user.email}"

class ChatMessage(models.Model):

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    sender = models.CharField(max_length=16, choices=[('user', 'User'), ('ai', 'AI')])
    text = models.TextField()
    tokens = models.PositiveIntegerField(null=True, blank=True)
    citations = models.JSONField(null=True, blank=True)
    error = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.sender}: {self.text[:30]}"