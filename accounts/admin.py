from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import CustomUserModel, Subject, Chapter, Document

class CustomUserAdmin(BaseUserAdmin):
    model = CustomUserModel
    list_display = ['email', 'is_staff', 'is_active']
    list_filter = ['is_staff', 'is_active']
    fieldsets = (
        (None, {'fields': ('email', 'password')}),
        ('Permissions', {'fields': ('is_staff', 'is_active', 'is_superuser', 'groups', 'user_permissions')}),
        ('Dates', {'fields': ('last_login', 'date_joined')}),
    )
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('email', 'password1', 'password2', 'is_staff', 'is_active')}
        ),
    )
    search_fields = ('email',)
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


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ["title", "user", "chapter", "file_type", "size_bytes", "created_at"]
    search_fields = ["title", "chapter__name", "user__email"]
    list_filter = ["file_type", "user", "chapter__subject"]
    ordering = ["-created_at"]