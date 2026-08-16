#!/usr/bin/env python
"""Vector-compatibility check: local TEI vs the managed provider.

    python scripts/compare_embeddings.py

Embeds one fixed sentence through both, and reports cosine similarity. This is
the test that decides whether the existing Pinecone index can be reused: two
services can both return 384 floats and still be pooling differently, which the
dimension check cannot detect but which makes the vectors incompatible.

    >= 0.99   PASS   same vector space, reuse the index
    0.95-0.99 STOP   investigate pooling / normalization
    <  0.95   STOP   do NOT reuse existing vectors

Reads EMBEDDING_* from the environment (or .env). Prints no vectors and no
credentials — only dimensions, L2 norms, the similarity, and the verdict.

Pooling is reported alongside, because it is the usual cause of a near-miss
score: bge-small-en-v1.5 is a CLS-pooled model, but Cloudflare Workers AI
defaults to MEAN pooling, which yields 384 perfectly-shaped incompatible floats.

TEI side: set TEI_COMPARE_URL (default http://127.0.0.1:8080/embed) with a
local `docker compose --profile full up embedder` running.
"""

import json
import math
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from utils import embedding as emb  # noqa: E402

SENTENCE = "The quick brown fox jumps over the lazy dog."
TEI_COMPARE_URL = os.getenv("TEI_COMPARE_URL", "http://127.0.0.1:8080/embed")
TEI_INFO_URL = TEI_COMPARE_URL.rsplit("/", 1)[0] + "/info"

PASS_THRESHOLD = 0.99
INVESTIGATE_THRESHOLD = 0.95


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def tei_vector():
    req = urllib.request.Request(
        TEI_COMPARE_URL,
        data=json.dumps({"inputs": [SENTENCE]}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)[0]


def tei_pooling():
    """What TEI actually pooled with — read from the server, not assumed."""
    try:
        with urllib.request.urlopen(TEI_INFO_URL, timeout=15) as r:
            info = json.load(r)
    except Exception:
        return "unknown"
    model_type = info.get("model_type") or {}
    embedding = model_type.get("embedding") if isinstance(model_type, dict) else None
    if isinstance(embedding, dict) and embedding.get("pooling"):
        return embedding["pooling"]
    return info.get("pooling") or "unknown"


def provider_vector():
    import asyncio

    client = emb.EmbeddingClient()

    async def run():
        try:
            return await client.embed_texts([SENTENCE])
        finally:
            await client.close()

    return asyncio.run(run())[0]


def main() -> int:
    try:
        a = tei_vector()
    except Exception as exc:
        print(f"FAIL: could not reach local TEI at {TEI_COMPARE_URL}")
        print(f"      {type(exc).__name__}")
        print("      Start it with: docker compose --profile full up -d embedder")
        return 2

    try:
        b = provider_vector()
    except Exception as exc:
        print(f"FAIL: provider request failed: {type(exc).__name__}: {str(exc)[:160]}")
        return 2

    sim = cosine(a, b)

    print(f"Sentence:            {SENTENCE!r}")
    print(f"TEI dimension:       {len(a)}")
    print(f"Provider dimension:  {len(b)}")
    print(f"TEI norm:            {norm(a):.6f}")
    print(f"Provider norm:       {norm(b):.6f}")
    print(f"Provider:            {emb.EMBEDDING_PROVIDER}")
    print(f"TEI pooling:         {tei_pooling()}")
    print(f"Provider pooling:    {emb.EMBEDDING_POOLING} (sent explicitly)")
    print(f"Cosine similarity:   {sim:.6f}")
    print()

    if len(a) != len(b):
        print("Pass/Fail: FAIL — dimensions differ; vectors are not comparable.")
        return 1

    if sim >= PASS_THRESHOLD:
        print("Pass/Fail: PASS — same vector space. Existing Pinecone index is reusable.")
        return 0

    if sim >= INVESTIGATE_THRESHOLD:
        print("Pass/Fail: STOP — investigate pooling / normalization differences.")
        print("           Do not reindex or switch providers on this result.")
        return 1

    print("Pass/Fail: STOP — vectors are NOT compatible.")
    print("           Do NOT reuse existing Pinecone vectors with this provider.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
