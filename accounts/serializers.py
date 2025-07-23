from djoser.serializers import UserCreateSerializer as BaseUserCreateSerializer
from django.contrib.auth import get_user_model
from rest_framework import serializers

User = get_user_model()

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
