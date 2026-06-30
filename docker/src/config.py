#!/usr/bin/env python3
"""
Central configuration for the PubMed RAG pipeline.

Defines all shared paths (DTD, chunks, embeddings, FAISS index, evaluation,
golden data, database) and processing parameters (chunk size, overlap, embedding
batch size, retrieval k).  Most values can be overridden via environment
variables for flexible deployment.

Constants imported by all pipeline scripts.
"""

import os
from pathlib import Path


# =============================================================================
# DIRECTORIES
# =============================================================================
DTD_DIR       = Path(os.environ.get("DTD_DIR", "/dtd"))
CHUNKS_DIR    = Path("/data/chunks")
EMBEDDING_DIR = Path("/data/embeddings")
INDEX_DIR     = Path("/data/faiss")
EVAL_DIR      = Path("/data/evaluation")
GOLDEN_DIR    = Path("/data/golden_data")
DB_PATH       = Path("/data/database/database.db")


# =============================================================================
# CHUNKING
# =============================================================================
SEKTIONEN     = ['abstract', 'introduction', 'methods', 'results', 'discussion', 'conclusion']
CHUNK_SIZE    = int(os.environ.get("CHUNK_SIZE", 10))
OVERLAP       = int(os.environ.get("OVERLAP", 3))

# =============================================================================
# EMBEDDING
# =============================================================================
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", 32))

# =============================================================================
# RETRIEVAL
# =============================================================================
RETRIEVAL_K   = int(os.environ.get("RETRIEVAL_K", 15))