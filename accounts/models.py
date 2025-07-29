from django.db import models
from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin
from django.utils import timezone
from .manager import CustomUserManager
from django.core.validators import EmailValidator
import uuid


def user_document_path(instance, filename):
    
    return f'docs/{instance.chapter.subject.user.id}/{instance.chapter.subject.id}/{instance.chapter.id}/{filename}'

class CustomUserModel(AbstractBaseUser, PermissionsMixin):

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    email = models.EmailField(
        unique=True,
        validators=[EmailValidator()],
        db_index=True
        )
    is_staff = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    date_joined= models.DateTimeField(default=timezone.now)
    last_login = models.DateTimeField(null=True, blank=True)


    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []

    objects = CustomUserManager()

    class Meta:
        db_table = 'auth_user'
        verbose_name = 'user'
        verbose_name_plural = 'users'
        indexes = [
            models.Index(fields=['email']),
            models.Index(fields=['is_active'])
        ]

    def __str__(self):
        return self.email 


    @property
    def is_authenticated_user(self):
        return self.is_active and not self.is_anonymous


class Subject(models.Model):
    user = models.ForeignKey(CustomUserModel, on_delete=models.CASCADE, related_name='subjects')
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


    def __str__(self):
        return f"{self.name} ({self.user.email})" 
    
class Chapter(models.Model):
    subject = models.ForeignKey(
        Subject, 
        on_delete=models.CASCADE,
        related_name='chapters'
    )

    name = models.CharField(max_length=100)
    order = models.PositiveIntegerField(default=1)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now = True)

    def __str__(self):
        return f"{self.subject.name} - {self.name}"
    
class Document(models.Model):
    chapter = models.ForeignKey(Chapter, on_delete=models.CASCADE, related_name='documents')
    user = models.ForeignKey(CustomUserModel, on_delete=models.CASCADE, related_name='documents')

    title = models.CharField(max_length=100)
    file = models.FileField(upload_to= user_document_path)
    file_type = models.CharField(max_length=10)
    size_bytes = models.PositiveIntegerField()
    extracted_text =  models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


    def __str__(self):
        return f"{self.title} ({self.file_type})"
    
    def save(self, *args, **kwargs):
        if self.file and not self.size_bytes:
            self.size_bytes = self.file.size
        if self.file and not self.file_type:
            ext = self.file.name.split(".")[-1].lower()
            self.file_type = ext
        super().save(*args, **kwargs)

class ChatSession(models.Model):

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey("CustomUserModel", on_delete=models.CASCADE, related_name='chat_session')
    subject = models.ForeignKey("Subject", null=True, blank=True, on_delete=models.SET_NULL, related_name='chat_session')
    chapter = models.ForeignKey("Chapter", null=True, blank=True, on_delete=models.SET_NULL, related_name='chat_sessions')
    title = models.CharField(max_length=255, blank=True, help_text="Optional title for this chat session.")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    context_snapshot = models.TextField(blank=True, help_text="serialized context for responsibilties / forking later.")

    def __str__(self):
        return f"sesssion {self.id} by {self.user.email} ({self.title or 'untitled'})"
    

class ChatMessage(models.Model):

    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='message')
    sender = models.CharField(max_length=16, choices=[('user', 'User'), ('ai', 'AI')])
    text = models.TextField()

    # for LLM usage / pricing
    tokens = models.PositiveIntegerField(null=True, blank=True)

    # RAG supporting docs
    citations = models.JSONField(null=True, blank=True)

    # failed LLM calls 
    error =models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.sender} ({self.created_at}): {self.text[:30]}"
    
    