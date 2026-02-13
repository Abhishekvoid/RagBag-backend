import torch
from sentence_transformers import SentenceTransformer
from typing import List

class BGEEmbedder:
    _model = None

    def __init__(self, device: str | None = None):
        if BGEEmbedder._model is None:
            device = device or ("cuda" if torch.cuda.is_available() else "cpu")

            BGEEmbedder._model = SentenceTransformer(
                "BAAI/bge-base-en-v1.5",
                device=device
            )

            # Production-safe settings
            BGEEmbedder._model.eval()

        self.model = BGEEmbedder._model

    def embed(self, texts: List[str]) -> list[list[float]]:
        """
        Generate normalized embeddings.
        Stable, deterministic, cosine-ready.
        """
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                batch_size=32,              # safe for 8GB VRAM
                normalize_embeddings=True,  # IMPORTANT
                show_progress_bar=False
            )

        return embeddings.tolist()