#!/bin/bash
set -e

echo "========================================="
echo "PubMed RAG Pipeline"
echo "========================================="

# Step 1: Extraction only if input contains archives
if ls /data/input/*.tar.gz >/dev/null 2>&1; then
    echo "[1/6] Extraktion..."
    python extract_pmc.py
    echo "[2/6] Golden Papers mergen..."
    python merge_golden.py
else
    echo "Keine .tar.gz-Archive in /data/input. Pipeline abgebrochen."
    exit 1
fi    

echo "[3/6] Chunking..."
python chunk.py

echo "[4/6] Embedding..."
python embed.py

echo "[5/6] FAISS-Index..."
python faiss_index.py

echo "[6/6] Evaluation..."
python evaluation.py

echo "========================================="
echo "Pipeline abgeschlossen."
echo "========================================="