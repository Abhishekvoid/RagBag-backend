from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import CustomUserModel, Subject, Chapter, Document, ChatMessage, ChatSession

class CustomUserAdmin(BaseUserAdmin):
    model = CustomUserModel
    list_display = ['email', 'name', 'is_staff', 'is_active']
    list_filter = ['is_staff', 'is_active']
    fieldsets = (
        (None, {'fields': ('email', 'name', 'password')}),
        ('Permissions', {'fields': ('is_staff', 'is_active', 'is_superuser', 'groups', 'user_permissions')}),
        ('Dates', {'fields': ('last_login', 'date_joined')}),
    )
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('email', 'password1', 'password2', 'is_staff', 'is_active')}
        ),
    )
    search_fields = ('email', 'name')
    ordering = ('email',)

admin.site.register(CustomUserModel, CustomUserAdmin)


@admin.register(Subject)
class SubjectAdmin(admin.ModelAdmin):
    list_display = ['name', 'user', 'created_at', 'updated_at']
    search_fields = ['name', 'user__email']
    list_filter = ['user']
    ordering = ['user', 'name']

@admin.register(Chapter)
class ChapterAdmin(admin.ModelAdmin):
    list_display = ['name', 'subject', 'order', 'created_at', 'updated_at']
    search_fields = ['name', 'subject__name']
    list_filter = ['subject']
    ordering = ['subject', 'order']


class DocumentAdmin(admin.ModelAdmin):
    # Add 'status' to list_display and list_filter
    list_display = ["title", "user", "chapter", "status", "file_type", "size_bytes", "created_at"]
    search_fields = ["title", "chapter__name", "user__email"]
    list_filter = ["status", "file_type", "user", "chapter__subject"] # Add status here too
    ordering = ["-created_at"]


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):

    list_display = ['id', 'session', 'get_user_email', 'sender', 'text_preview', 'created_at', 'tokens']
    search_fields = ['id', 'session__id', 'text', 'session__user__email']
    list_filter = ['session', 'sender', 'created_at']
    ordering = ['-created_at']
    raw_id_fields = ['session']

    def text_preview(self, obj):
        return obj.text[:40] + ('...' if len(obj.text) > 40 else '')
    text_preview.short_description = "Text"


    def get_user_email(self, obj):
        return obj.session.user.email if obj.session and obj.session.user else None
    get_user_email.short_description = "User Email"


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'subject', 'chapter', 'title', 'created_at', 'updated_at']
    search_fields = ['id', 'title', 'user__email', 'subject__name', 'chapter__name']
    list_filter = ['user', 'subject', 'chapter']
    ordering = ['-created_at']
    raw_id_fields = ['user', 'subject', 'chapter']