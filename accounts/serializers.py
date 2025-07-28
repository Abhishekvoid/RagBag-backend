from djoser.serializers import UserCreateSerializer as BaseUserCreateSerializer
from django.contrib.auth import get_user_model
from rest_framework import serializers
from .models import Document, ChatMessage, ChatSession

User = get_user_model()

ALLOWED_EXTENSIONS = ["pdf", "doc", "docx", "ppt", "pptx", "jpg", "jpeg", "png", "gif"]
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB

class RegisterSerializers(BaseUserCreateSerializer):
    password1 = serializers.CharField(write_only=True)
    password2 = serializers.CharField(write_only=True)

    class Meta(BaseUserCreateSerializer.Meta):
        model = User
        fields = ['email', 'password1', 'password2']  
    def validate(self, attrs):
        pw1 = attrs.get('password1')
        pw2 = attrs.get('password2')

        if pw1 != pw2:
            raise serializers.ValidationError("Password should match!")
        
        attrs['password'] = pw1
        attrs.pop('password1')
        attrs.pop('password2')
        return super().validate(attrs)


class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = [
            "id", "chapter", "user", "title", "file", "file_type", "size_bytes", "created_at", "updated_at"
        ]
        read_only_fields = ("file_type", "size_bytes", "created_at", "updated_at")

    def validate_file(self, file):
        ext = file.name.split(".")[-1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise serializers.ValidationError("Only PDF, Word, PPT, and image files allowed.")
        if file.size > MAX_UPLOAD_SIZE:
            raise serializers.ValidationError("File exceeds maximum allowed size (50MB).")
        return file


class ChatSessionSerializer(serializers.ModelSerializer):

    class Meta:
        model = ChatSession
        fields = ['id', 'user', 'subject', 'chapter', 'title', 'created_at', 'updated_at', 'context_snapshot']

        ready_only_fields = ('id', 'user', 'created_at', 'updated_at', 'context_snapshot')


class ChatMessageSerializer(serializers.ModelSerializer):

    class Meta:
        model =  ChatMessage

        fields = ['id', 'session', 'sender', 'text', 'created_at', 'citations', 'tokens', 'error']
        read_only_fields = ['id', 'created_at', 'citations', 'tokens', 'error']