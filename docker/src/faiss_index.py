#!/usr/bin/env python3
"""
Build a FAISS flat inner‑product index from pre‑computed embeddings.

Loads float32 embedding vectors from /data/embeddings/embeddings.npy,
L2‑normalizes them so that inner product equals cosine similarity,
creates a FAISS IndexFlatIP, adds all vectors, and persists the index to
/data/faiss/faiss.index.

Requires config.py for EMBEDDING_DIR and INDEX_DIR.
"""

print("FAISS indexing starting...\n")
import numpy as np
import faiss
from pathlib import Path
from config import EMBEDDING_DIR, INDEX_DIR

INDEX_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# LOAD EMBEDDINGS
# =============================================================================
embeddings = np.load(EMBEDDING_DIR / "embeddings.npy").astype(np.float32)
dimension = embeddings.shape[1]
print(f"...found {embeddings.shape[0]} vectors of dimension {dimension}...")

# =============================================================================
# L2 NORMALISATION FOR COSINE SIMILARITY
# =============================================================================
faiss.normalize_L2(embeddings)

# =============================================================================
# CREATE INDEX
# =============================================================================
index = faiss.IndexFlatIP(dimension)  # Inner Product = Cosine nach Normalisierung
index.add(embeddings)
print(f"...created FAISS-index for {index.ntotal} vectors...")

# =============================================================================
# SAVE INDEX
# =============================================================================
faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
print(f"...saved index to: {INDEX_DIR / 'faiss.index'}...")
print("\n...FAISS indexing finished")
print("")
print("")