from djoser.serializers import UserCreateSerializer as BaseUserCreateSerializer
from django.contrib.auth import get_user_model
from rest_framework import serializers
from .models import Document, ChatMessage, ChatSession, Chapter, Subject
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

# ------------- subject ------------
class SubjectSerializer(serializers.ModelSerializer):
    class Meta:
        model  = Subject
        fields = ['user', 'name', 'description', 'created_at', 'updated_at']
        
        ready_only_fieelds = ['user', 'name', 'created_at', 'updated_at']


        def validate_name(self, value):
            if not value.strip():
                raise serializers.ValidationError("subject name can't be blank")
            return value
        
        def create(self, validated_data):

            user= self.context['request'].user
            validated_data['user'] = user
            return super().create(validated_data)
        
# ------------ chapter -----------------

class ChapterSerializer(serializers.ModelSerializer):
    class Meta:
        model = Chapter
        fields = ['id', 'subject', 'name', 'order', 'created_at', 'updated_at']
        read_only_fields = ('created_at', 'updated_at')

    def validate_name(self, value):
        if not value.strip():
            raise serializers.ValidationError("Chapter name cannot be blank.")
        return value

    def validate_order(self, value):
        if value < 1:
            raise serializers.ValidationError("Order must be a positive integer.")
        return value

    def validate(self, data):
        # Optional: if subject is set, you can validate the subject belongs to user here
        request = self.context.get('request')
        subject = data.get('subject')
        if subject and subject.user != request.user:
            raise serializers.ValidationError("Subject does not belong to the authenticated user.")
        return data

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