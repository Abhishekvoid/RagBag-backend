# from sentence_transformers import CrossEncoder
# import torch

# device = "cuda" if torch.cuda.is_available() else "cpu"

# print(f"Loading CrossEncoder reranker on {device}...")

# reranker = CrossEncoder(
#     "cross-encoder/ms-marco-MiniLM-L-6-v2",
#     device=device
# )

# # WARMUP
# print("Warming up reranker...")
# reranker.predict([["hello", "hello world"]], show_progress_bar=False)

# print("Reranker ready")

import os

reranker = None

USE_LOCAL_RERANKER = os.getenv("USE_LOCAL_RERANKER", "False") == "True"

if USE_LOCAL_RERANKER:
    from sentence_transformers import CrossEncoder
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading CrossEncoder reranker on {device}...")

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device=device
    )

    print("Warming up reranker...")
    reranker.predict([["hello", "hello world"]], show_progress_bar=False)

    print("Reranker ready")
else:
    print("Reranker disabled (production mode)")