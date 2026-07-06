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
print("Embedding starting...\n")
import sys
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
all_chunks = df_chunks["chunk_text"].tolist()
print(f"Chunks available: {len(all_chunks)}")

with open(CHUNKS_DIR / "metadaten.pkl", "rb") as f:
    all_metadata = pickle.load(f)

# =============================================================================
# DETERMINE NEW CHUNKS
# =============================================================================
existing_embeddings = None
existing_metadata = []
if (EMBEDDING_DIR / "embeddings.npy").exists() and (EMBEDDING_DIR / "metadaten.pkl").exists():
    existing_embeddings = np.load(EMBEDDING_DIR / "embeddings.npy")
    with open(EMBEDDING_DIR / "metadaten.pkl", "rb") as f:
        existing_metadata = pickle.load(f)
    already_embedded = len(existing_metadata)
    print(f"Chunks already embedded: {already_embedded}")
else:
    already_embedded = 0

new_chunks = all_chunks[already_embedded:]
new_metadata = all_metadata[already_embedded:]

if not new_chunks:
    print("\n...all chunks already embedded")
    print("")
    print("")
    exit(0)

print(f"New chunks to embed: {len(new_chunks)}")

# =============================================================================
# LOAD MODEL AND EMBED ONLY NEW CHUNKS
# =============================================================================
embed_model = SentenceTransformer('neuml/pubmedbert-base-embeddings', device=device)
# embed_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
print(f"Model loaded on: {embed_model.device}")

new_embeddings = embed_model.encode(
    new_chunks,
    batch_size=EMBED_BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True
)

# =============================================================================
# APPEND TO EXISTING DATA
# =============================================================================
if existing_embeddings is not None:
    all_embeddings = np.vstack([existing_embeddings, new_embeddings])
    all_metadata = existing_metadata + new_metadata
    print(f"New chunks embedded:{len(new_embeddings)}")

else:
    all_embeddings = new_embeddings
    all_metadata = new_metadata


# =============================================================================
# SAVE EMBEDDINGS AMD METADATA
# =============================================================================
np.save(EMBEDDING_DIR / "embeddings.npy", all_embeddings)
with open(EMBEDDING_DIR / "metadaten.pkl", "wb") as f:
    pickle.dump(all_metadata, f)

print(f"\nEmbeddings saved: {all_embeddings.shape[0]} vectors total with {all_embeddings.shape[1]} dimensions each.")
print(f"  → {EMBEDDING_DIR / 'embeddings.npy'}")
print(f"  → {EMBEDDING_DIR / 'metadaten.pkl'}")
print("\n...embedding finished")
print("")
print("")