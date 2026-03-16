import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
import django
django.setup()

from qdrant_client import QdrantClient

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# Correct way to check collections
collections = client.get_collections()
print("✅ Qdrant connected!")
print("Collections count:", len(collections.collections))  # ← .collections
print("Collections:", [c.name for c in collections.collections])
