from qdrant_client import QdrantClient

client = QdrantClient(
    url="https://34dd525e-ad21-460f-8409-18af905d9875.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.jJSWB-0-riiK_Pz0pdGisjHN1mgypC_vQe3BBKUrk2M"
)

# List collections
collections = client.get_collections()
print(collections)