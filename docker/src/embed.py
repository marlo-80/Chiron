#!/usr/bin/env python3
"""
Compute PubMedBERT embeddings for all text chunks and persist them.

Loads the chunk DataFrame from /data/chunks/chunks.pkl, encodes each chunk
with the neuml/pubmedbert-base-embeddings SentenceTransformer model (GPU
accelerated when available), saves the resulting float32 array to
/data/embeddings/embeddings.npy, and copies the chunk metadata to
/data/embeddings/metadaten.pkl.

Requires config.py for EMBEDDING_DIR, EMBED_BATCH_SIZE, and CHUNKS_DIR.
"""

import pickle
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path
from config import EMBEDDING_DIR, EMBED_BATCH_SIZE, CHUNKS_DIR

CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# GPU CHECK
# =============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"CUDA ready: {torch.cuda.is_available()}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Device: {device}")

# =============================================================================
# LOAD DATA
# =============================================================================
with open(CHUNKS_DIR / "chunks.pkl", "rb") as f:
    df_chunks = pd.read_pickle(f)

chunks = df_chunks["chunk_text"].tolist()
print(f"Chunks loaded: {len(chunks)}")

# =============================================================================
# LOAD MODEL
# =============================================================================
embed_model = SentenceTransformer('neuml/pubmedbert-base-embeddings', device=device)
print(f"Model ready on device: {embed_model.device}")

# =============================================================================
# EMBEDDING CALCULATIONS
# =============================================================================
chunk_embeddings = embed_model.encode(
    chunks,
    batch_size=EMBED_BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True
)

# =============================================================================
# SAVE EMBEDDINGS
# =============================================================================
np.save(EMBEDDING_DIR / "embeddings.npy", chunk_embeddings)

# Metadaten unverändert durchreichen
with open(CHUNKS_DIR / "metadaten.pkl", "rb") as f:
    metadaten = pickle.load(f)
with open(EMBEDDING_DIR / "metadaten.pkl", "wb") as f:
    pickle.dump(metadaten, f)

print(f"Embeddings saved: {chunk_embeddings.shape[0]} vectors with {chunk_embeddings.shape[1]} dimensions each.")
print(f"  → {EMBEDDING_DIR / 'embeddings.npy'}")
print(f"  → {EMBEDDING_DIR / 'metadaten.pkl'}")