#!/usr/bin/env python3
"""
retrieval.py – Retrieval mit FAISS und PubMedBERT-Embeddings
Input:  /data/4_embeddings/embeddings.npy
        /data/4_embeddings/metadaten.pkl
        /data/5_index/faiss.index
        /data/3_chunks/chunks.pkl
"""

import numpy as np
import pickle
import faiss
import torch
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_DIR, INDEX_DIR, CHUNKS_DIR, RETRIEVAL_K

# ── Gerät wählen ────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Einmalig laden ──────────────────────────────────────────────
_embed_model = SentenceTransformer("neuml/pubmedbert-base-embeddings", device=device)

_embeddings = np.load(EMBEDDING_DIR / "embeddings.npy").astype(np.float32)
faiss.normalize_L2(_embeddings)

with open(EMBEDDING_DIR / "metadaten.pkl", "rb") as f:
    _metadaten = pickle.load(f)

_index = faiss.read_index(str(INDEX_DIR / "faiss.index"))

_chunks_df = pd.read_pickle(CHUNKS_DIR / "chunks.pkl")

print(f"Retrieval bereit: {_index.ntotal} Chunks, Dimension {_embeddings.shape[1]}, Gerät: {device}")

# ── Retrieval-Funktion ───────────────────────────────────────────
def retrieve_with_metadata(query, k=RETRIEVAL_K):
    """
    Sucht top-k Chunks via Cosine-Ähnlichkeit (FAISS Inner Product).
    Gibt eine Liste von Tupeln zurück:
    [(metadaten_dict, chunk_text, score), ...]
    """
    k = min(k, _index.ntotal)
    query_emb = _embed_model.encode([query], show_progress_bar=False)[0].astype(np.float32)
    query_emb = query_emb / np.linalg.norm(query_emb)

    scores, indices = _index.search(query_emb.reshape(1, -1), k)

    results = []
    for j, idx in enumerate(indices[0]):
        meta = _metadaten[idx]
        text = _chunks_df.iloc[idx]["chunk_text"]
        score = float(scores[0][j])
        results.append((meta, text, score))

    return results