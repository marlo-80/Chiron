#!/usr/bin/env python3
"""
retrieval.py – FAISS-based retrieval with PubMedBERT embeddings.
Loads embeddings, metadata, FAISS index, and chunk texts once,
then exposes the `retrieve_with_metadata` function for the rest
of the pipeline.
"""

import pickle
import numpy as np
import faiss
import torch
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_DIR, INDEX_DIR, CHUNKS_DIR, RETRIEVAL_K

# =============================================================================
# INITIALISATION (runs once at import time)
# =============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

# Embedding model – same PubMedBERT as used during ingestion
_embed_model = SentenceTransformer("neuml/pubmedbert-base-embeddings", device=device)
#_embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# Load pre-computed embeddings and normalise them for cosine similarity
_embeddings = np.load(EMBEDDING_DIR / "embeddings.npy").astype(np.float32)
faiss.normalize_L2(_embeddings)

# Load metadata (pmcid, title, section, etc.) and chunk texts
with open(EMBEDDING_DIR / "metadaten.pkl", "rb") as f:
    _metadaten = pickle.load(f)

_index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
_chunks_df = pd.read_pickle(CHUNKS_DIR / "chunks.pkl")

print(f"Retrieval ready: {_index.ntotal} chunks, dimension {_embeddings.shape[1]}, device {device}")

# =============================================================================
# RETRIEVAL FUNCTION
# =============================================================================
def retrieve_with_metadata(query, k=RETRIEVAL_K):
    """
    Retrieve the top-k most similar chunks for a given query.

    Parameters
    ----------
    query : str
        The search query (e.g. a medical question).
    k : int, optional
        Number of chunks to return (default from config).

    Returns
    -------
    list of tuple
        Each tuple contains (metadata_dict, chunk_text, similarity_score).
        The list is ordered by descending similarity.
    """
    k = min(k, _index.ntotal)

    # Embed the query and normalise
    query_emb = _embed_model.encode([query], show_progress_bar=False)[0].astype(np.float32)
    query_emb = query_emb / np.linalg.norm(query_emb)

    # Search the FAISS index
    scores, indices = _index.search(query_emb.reshape(1, -1), k)

    results = []
    for j, idx in enumerate(indices[0]):
        meta = _metadaten[idx]
        text = _chunks_df.iloc[idx]["chunk_text"]
        score = float(scores[0][j])
        results.append((meta, text, score))

    return results