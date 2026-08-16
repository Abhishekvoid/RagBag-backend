#!/usr/bin/env python
"""One-command smoke test against the REAL embedding provider.

    python scripts/smoke_embedding.py

Reads EMBEDDING_* from the environment (or .env) and sends exactly one short
sentence, then checks the vector width. Nothing else — this is a wiring check,
not a benchmark.

The key is read from the environment and never printed, never written anywhere,
and never passed on the command line (argv is visible to other processes and
lands in shell history). Only its length and last 4 characters are shown, which
is enough to tell two keys apart without disclosing either.

This is deliberately NOT part of `manage.py test`: the suite must never depend
on a third party being up, or spend anyone's quota.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from utils import embedding as emb  # noqa: E402


def redacted(value: str) -> str:
    if not value:
        return "(not set)"
    return f"(set, {len(value)} chars, ends …{value[-4:]})"


def main() -> int:
    print("Embedding provider smoke test")
    print(f"  provider : {emb.EMBEDDING_PROVIDER}")
    print(f"  url      : {emb.EMBEDDING_URL}")
    print(f"  model    : {emb.EMBEDDING_MODEL}")
    print(f"  api key  : {redacted(emb.EMBEDDING_API_KEY)}")
    print(f"  pooling  : {emb.EMBEDDING_POOLING}")
    print(f"  expects  : {emb.EXPECTED_DIM} dimensions")
    print()

    if not emb.EMBEDDING_URL:
        print("FAIL: EMBEDDING_URL is not set.")
        return 2

    if emb.EMBEDDING_PROVIDER != "tei" and not emb.EMBEDDING_API_KEY:
        print(f"FAIL: provider '{emb.EMBEDDING_PROVIDER}' needs EMBEDDING_API_KEY.")
        return 2

    import asyncio

    client = emb.EmbeddingClient()
    sentence = "The mitochondria is the powerhouse of the cell."

    async def run():
        try:
            vectors = await client.embed_texts([sentence])
        finally:
            await client.close()
        return vectors

    try:
        vectors = asyncio.run(run())
    except emb.EmbeddingDimensionMismatch as exc:
        print(f"FAIL: {exc}")
        print("      The provider is serving a different model. Do NOT reindex")
        print("      against this endpoint — the vectors are incompatible.")
        return 1
    except Exception as exc:
        # Message only; provider errors do not echo credentials, but the class
        # name alone is enough to diagnose auth vs network vs shape.
        print(f"FAIL: {type(exc).__name__}: {str(exc)[:200]}")
        return 1

    if len(vectors) != 1:
        print(f"FAIL: expected 1 vector, got {len(vectors)}")
        return 1

    import math

    dim = len(vectors[0])
    l2 = math.sqrt(sum(v * v for v in vectors[0]))
    print(f"OK: received 1 vector of {dim} dimensions")
    # The norm is enough to tell a normalized vector from a raw one without
    # disclosing the vector itself.
    print(f"    L2 norm: {l2:.6f}")

    if dim != emb.EXPECTED_DIM:
        print(f"FAIL: expected {emb.EXPECTED_DIM}, got {dim}")
        return 1

    print()
    print("PASS — provider wiring is correct and dimensions match the index.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
