import os
from qdrant_client import QdrantClient

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = "studywise_documents"

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    port=6333,
    https=True,
    timeout=30
)

print(f"Attempting to delete Qdrant collection: '{QDRANT_COLLECTION_NAME}'...")

try:
    client.delete_collection(collection_name=QDRANT_COLLECTION_NAME)
    print("✅ Collection deleted successfully.")
except Exception as e:
    print(f"ℹ️ Could not delete collection (it may not exist): {e}")