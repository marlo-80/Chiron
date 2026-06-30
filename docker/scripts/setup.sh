#!/bin/bash
set -e

echo "========================================="
echo "CREATION OF CHIRON DATABASE"
echo "========================================="

echo "[1/6] Fetching Data..."
python /app/src/fetch.py

# Prüfen, ob die goldenen Test‑Papers eingespielt werden sollen
FLAG_FILE="/data/database/include_golden.flag"
if [ -f "$FLAG_FILE" ] && [ "$(cat $FLAG_FILE)" = "1" ]; then
    echo "[2/6] Merging golden papers into database..."
    python /app/src/merge_golden.py
else
    echo "[2/6] No merging of golden papers!"
fi

echo "[3/6] Chunking..."
python /app/src/chunk.py

echo "[4/6] Embedding..."
python /app/src/embed.py

echo "[5/6] FAISS-Index..."
python /app/src/faiss_index.py


if [ -f "$FLAG_FILE" ] && [ "$(cat $FLAG_FILE)" = "1" ]; then
    echo "[6/6] Evaluating database with golden questions..."
    python /app/src/evaluation.py
else
    echo "[6/6] No evaluation possible"
fi

echo "========================================="
echo "CHIRON READY"
echo "========================================="