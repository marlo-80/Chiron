#!/bin/bash
set -e

echo "========================================="
echo "CREATION OF CHIRON DATABASE"
echo "========================================="

echo "[1/4] Fetching Data..."
python /app/src/fetch.py

echo "[2/4] Chunking..."
python /app/src/chunk.py

echo "[3/4] Embedding..."
python /app/src/embed.py

echo "[4/4] FAISS-Index..."
python /app/src/faiss_index.py

echo "========================================="
echo "CHIRON READY"
echo "========================================="