from qdrant_client import QdrantClient
import os

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "studywise_documents"

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

print("Deleting old collection...")
try:
    client.delete_collection(collection_name=COLLECTION_NAME)
    print("✅ Deleted")
except Exception as e:
    print("⚠️ Already deleted or not exists:", e)

print("Creating new collection with correct dimension...")

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "size": 384,  # 🔥 THIS IS THE FIX
        "distance": "Cosine"
    }
)

print("✅ Collection recreated with 384 dims")