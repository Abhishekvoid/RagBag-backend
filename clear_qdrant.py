from qdrant_client import QdrantClient

# Make sure this matches your Qdrant URL and collection name
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION_NAME = "studywise_documents"

client = QdrantClient(QDRANT_URL)

print(f"Attempting to delete Qdrant collection: '{QDRANT_COLLECTION_NAME}'...")

try:
    client.delete_collection(collection_name=QDRANT_COLLECTION_NAME)
    print("✅ Collection deleted successfully.")
except Exception as e:
    # This will likely happen if the collection doesn't exist, which is fine.
    print(f"ℹ️ Could not delete collection (it may not exist): {e}")