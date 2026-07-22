import uuid
import os
from django.db import models
from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin
from django.utils import timezone
from django.core.validators import EmailValidator
from .manager import CustomUserManager
from django.conf import settings
from django.utils.text import slugify


def user_document_path(instance, filename):

    base, ext = os.path.splitext(filename)
    safe_name = slugify(base)[:50]  # Limit to 50 chars and remove spaces/symbols
    unique_suffix = uuid.uuid4().hex[:8]
    filename = f"{safe_name}_{unique_suffix}{ext}"
 
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
    name = models.CharField(max_length=255)
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
    file = models.FileField(upload_to=user_document_path, max_length=500)
    file_type = models.CharField(max_length=10, blank=True)
    size_bytes = models.PositiveIntegerField(null=True, blank=True)
    
    
    extracted_text = models.TextField(blank=True)

    # --- NEW: Define status choices as constants ---
    STATUS_PENDING = 'PENDING'
    STATUS_PROCESSING = 'PROCESSING'
    STATUS_COMPLETED = 'COMPLETED'
    STATUS_FAILED = 'FAILED'
    
    STATUS_CHOICES = [
        (STATUS_PENDING, 'Pending'),
        (STATUS_PROCESSING, 'Processing'),
        (STATUS_COMPLETED, 'Completed'),
        (STATUS_FAILED, 'Failed'),
    ]
    # --- CHANGED: Update the status field to use the constants and default to PENDING ---
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default=STATUS_PENDING
    )
    
    # --- NEW: Add a field to store error details for debugging ---
    error_message = models.TextField(blank=True, null=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.title} ({self.file_type})"


class DocumentPage(models.Model):
    """One page of a document's canonical, reader-facing text.

    ``reconstructed_md`` is the clean markdown shown in the reader and used for
    RAG/flashcards/questions. ``image_url`` is the rendered original page, kept
    as a verification layer for the AI's ``[?word]`` uncertainty markers.
    """

    SOURCE_LAYER = 'layer'       # born-digital: used the PDF's own text layer
    SOURCE_VISION = 'vision'     # reconstructed by the vision model
    SOURCE_FALLBACK = 'fallback' # vision unavailable/failed: tesseract or raw layer
    SOURCE_CHOICES = [
        (SOURCE_LAYER, 'Text layer'),
        (SOURCE_VISION, 'Vision'),
        (SOURCE_FALLBACK, 'Fallback'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='pages')
    page_number = models.PositiveIntegerField()          # 1-indexed
    image_url = models.TextField(blank=True)             # S3 url of the rendered original page
    reconstructed_md = models.TextField(blank=True)
    text_source = models.CharField(max_length=10, choices=SOURCE_CHOICES, default=SOURCE_LAYER)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('document', 'page_number')
        ordering = ['page_number']

    def __str__(self):
        return f"{self.document_id} p.{self.page_number} ({self.text_source})"


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
    suggestions = models.JSONField(null=True, blank=True)
    error = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.sender}: {self.text[:30]}"
    

# --------------- Question generation

class GenerateQuestion(models.Model):
    id =models.UUIDField(primary_key=True, default=uuid.uuid4  , editable= False)
    chapter = models.ForeignKey(Chapter, on_delete=models.CASCADE, related_name='generated_questions')
    question_text = models.TextField()
    answer_text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)


    def __str__ (self):
        return self.question_text[:50]

class GenerateFlashCards(models.Model):

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='flash_card')
    chapter = models.ForeignKey(Chapter, on_delete=models.CASCADE, related_name='flash_card')
    flashcard_front = models.TextField()
    flashcard_back = models.TextField()
    known = models.BooleanField(default=False)
    need_review = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"flashcard for chapter{self.chapter.name} (User: {self.user.email})"


# --------------- Co-reading notes

class Note(models.Model):
    """A note or highlight in the co-reading workspace.

    Highlight/note/ai/chat notes anchor to a character range in a document's
    cleaned reader text (``anchor_start``/``anchor_end`` into that text, with
    ``quoted_text`` kept as a fuzzy-match fallback). A ``scratch`` note is the
    single freeform pad per chapter and has no document/anchor.
    """

    KIND_HIGHLIGHT = 'highlight'   # bare highlight, no body
    KIND_NOTE = 'note'             # user-written note on a passage
    KIND_AI = 'ai'                 # AI-generated (Explain / synthesis)
    KIND_CHAT = 'chat'             # a chat answer saved as a note
    KIND_SCRATCH = 'scratch'       # the one freeform pad per chapter

    KIND_CHOICES = [
        (KIND_HIGHLIGHT, 'Highlight'),
        (KIND_NOTE, 'Note'),
        (KIND_AI, 'AI'),
        (KIND_CHAT, 'Chat'),
        (KIND_SCRATCH, 'Scratch'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='notes')
    chapter = models.ForeignKey(Chapter, on_delete=models.CASCADE, related_name='notes')
    # CASCADE + nullable: deleting a document removes its anchored notes; a
    # scratch note has no document.
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='notes', null=True, blank=True)

    kind = models.CharField(max_length=16, choices=KIND_CHOICES, default=KIND_NOTE)
    anchor_start = models.PositiveIntegerField(null=True, blank=True)
    anchor_end = models.PositiveIntegerField(null=True, blank=True)
    quoted_text = models.TextField(blank=True)
    body = models.TextField(blank=True)
    color = models.CharField(max_length=20, default='vermillion')

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['document_id', 'anchor_start', 'created_at']

    def __str__(self):
        return f"{self.kind} note ({self.user.email}) — {self.quoted_text[:30] or self.body[:30]}"