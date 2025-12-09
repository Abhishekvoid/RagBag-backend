from djoser.serializers import UserCreateSerializer as BaseUserCreateSerializer
from django.contrib.auth import get_user_model
from rest_framework import serializers
from .models import Document, ChatMessage, ChatSession, Chapter, Subject, GenerateQuestion, GenerateFlashCards
User = get_user_model()
import logging

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = ["pdf", "doc", "docx", "ppt", "pptx", "jpg", "jpeg", "png", "gif"]
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB

class RegisterSerializers(BaseUserCreateSerializer):
    password1 = serializers.CharField(write_only=True)
    password2 = serializers.CharField(write_only=True)
    name = serializers.CharField(required=True, max_length=100)

    class Meta(BaseUserCreateSerializer.Meta):
        model = User
        fields = ['email', 'name', 'password1', 'password2']  
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
            "id", "chapter", "user", "title", "file", "file_type", 
            "size_bytes", "status", "error_message", "created_at", "updated_at"
        ]
        # Make 'title' and 'chapter' optional in the API
        read_only_fields = ("user", "file_type", "size_bytes","status", "error_message", "created_at", "updated_at")
        extra_kwargs = {
            'title': {'required': False},
           
        }

    def validate_file(self, file):
        ext = file.name.split(".")[-1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise serializers.ValidationError("Only PDF, Word, PPT, and image files allowed.")
        if file.size > MAX_UPLOAD_SIZE:
            raise serializers.ValidationError("File exceeds maximum allowed size (50MB).")
        return file

    def create(self, validated_data):
        logger.info("DocumentSerializer.create called.")
        chapter_id = validated_data.pop('chapter_id', None)

        validated_data['user'] = self.context['request'].user
        file = validated_data.get('file')
        if not file:
            raise serializers.ValidationError("File required.")

        
        if file:
            validated_data['size_bytes'] = file.size
            validated_data['file_type'] = file.name.split(".")[-1].lower()
            if 'title' not in validated_data:
                validated_data['title'] = file.name.rsplit('.', 1)[0]

        chapter_instance = None
        if chapter_id:
            try:
                chapter_instance = Chapter.objects.get(id=chapter_id, user=self.context['request'].user)
                validated_data['chapter'] = chapter_instance
                logger.info(f"  - Found existing chapter: {chapter_instance.id} - {chapter_instance.name}")
            except Chapter.DoesNotExist:
                logger.error(f"  - Chapter with id {chapter_id} not found for user {self.context['request'].user.id}.")
                raise serializers.ValidationError({"chapter_id":"Chapter not found or does not belong to user."})
        else:
            logger.info("  - No chapter_id provided. Document will be standalone.")
        
        return super().create(validated_data)
# ------------ chapter -----------------

class ChapterWriteSerializer(serializers.ModelSerializer):
    class Meta:
        model = Chapter
        # The user will provide the parent subject's UUID, the name, and the order.
        fields = ['subject', 'name', 'order']

    extra_kwargs = {
        'subject': {'required': False, 'allow_null': True}
    }
    def validate_subject(self, value):
        """
        Check that the subject is owned by the user making the request.
        """
        if value is None:
            return value
        
        request = self.context.get('request')
        if value.user != request.user:
            raise serializers.ValidationError("You do not have permission to add a chapter to this subject.")
        return value

class ChapterReadSerializer(serializers.ModelSerializer):
    documents = DocumentSerializer(many=True, read_only=True)
    class Meta:
        model = Chapter
        fields = ['id', 'subject', 'name', 'order', 'created_at', 'updated_at', 'documents']

# ------------- subject ------------
class SubjectReadSerializer(serializers.ModelSerializer):
    # This is the key: It uses the ChapterReadSerializer to nest the chapter data.
    chapters = ChapterReadSerializer(many=True, read_only=True)

    class Meta:
        model = Subject
        fields = ['id', 'user', 'name', 'description', 'created_at', 'updated_at', 'chapters']

class SubjectWriteSerializer(serializers.ModelSerializer):
    class Meta:
        model = Subject
        fields = ['name', 'description']

    
    def validate_name(self, value):
        if not value.strip():
            raise serializers.ValidationError("Subject name can't be blank.")
        return value



class ChatSessionSerializer(serializers.ModelSerializer):

    class Meta:
        model = ChatSession
        fields = ['id', 'user', 'subject', 'chapter', 'title', 'created_at', 'updated_at', 'context_snapshot']

        read_only_fields = ('id', 'user', 'created_at', 'updated_at', 'context_snapshot')


class ChatMessageSerializer(serializers.ModelSerializer):

    class Meta:
        model =  ChatMessage

        fields = ['id', 'session', 'sender', 'text', 'created_at', 'citations', 'tokens', 'error']
        read_only_fields = ['id', 'created_at', 'citations', 'tokens', 'error']


class RAGChatMessageSerializer(serializers.Serializer):

    chapter = serializers.UUIDField()
    text = serializers.CharField()

# ------------ generateQuestions

class GeneratedQuestionsSerializer(serializers.ModelSerializer):
    class Meta:
        model = GenerateQuestion
        fields = ['id', 'question_text', 'answer_text', 'created_at']


class GeneratedFlashCardsSerializer(serializers.ModelSerializer):
    
    class Meta:
        model = GenerateFlashCards
        fields = ['id', 'chapter', 'flashcard_front', 'flashcard_back', 'known', 'need_review', 'created_at']