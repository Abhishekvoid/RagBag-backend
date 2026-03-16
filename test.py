import os
import django
from io import BytesIO

# Configure Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')  # Your project name
django.setup()

from storages.backends.s3boto3 import S3Boto3Storage
storage = S3Boto3Storage()
print(storage.bucket_name)  # 'media'
print(storage.connection.meta.client.meta.endpoint_url) 