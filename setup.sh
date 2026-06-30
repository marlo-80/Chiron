#!/bin/bash
set -e

COMPOSE_FILE="docker/compose.yml"
SERVICE="chiron"
EXEC="docker compose -f $COMPOSE_FILE exec $SERVICE"

# Stelle sicher, dass der Container läuft
docker compose -f $COMPOSE_FILE up -d $SERVICE

echo "========================================="
echo "CREATION OF CHIRON DATABASE"
echo "========================================="

echo "[1/6] Fetching Data..."
$EXEC python /app/src/fetch.py

# Golden‑Paper‑Flag im Container prüfen
FLAG_FILE="/data/database/include_golden.flag"
GOLDEN_FLAG=$($EXEC bash -c "if [ -f $FLAG_FILE ] && [ \"\$(cat $FLAG_FILE)\" = '1' ]; then echo 1; else echo 0; fi")

if [ "$GOLDEN_FLAG" = "1" ]; then
    echo "[2/6] Merging golden papers into database..."
    $EXEC python /app/src/merge_golden.py
else
    echo "[2/6] No merging of golden papers!"
fi

echo "[3/6] Chunking..."
$EXEC python /app/src/chunk.py

echo "[4/6] Embedding..."
$EXEC python /app/src/embed.py

echo "[5/6] FAISS-Index..."
$EXEC python /app/src/faiss_index.py

if [ "$GOLDEN_FLAG" = "1" ]; then
    echo "[6/6] Evaluating database with golden questions..."
    $EXEC python /app/src/evaluation.py
else
    echo "[6/6] No evaluation possible"
fi

echo "========================================="
echo "CHIRON READY"
echo "========================================="