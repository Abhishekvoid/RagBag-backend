from django.db import models
from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin
from django.utils import timezone
from .manager import CustomUserManager
from django.core.validators import EmailValidator
import uuid


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