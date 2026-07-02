#!/bin/bash
set -e

# =============================================================================
# PRE-FLIGHT CHECKS (all run inside the Chiron container)
# =============================================================================

echo "Running pre-flight checks inside the Chiron container..."

# 1. Required directories
for d in /data/database /data/chunks /data/embeddings /data/faiss /data/evaluation /data/golden_data /dtd; do
    if ! docker compose -f docker/compose.yml exec -T chiron test -d "$d"; then
        echo "ERROR: $d not found inside the Chiron container. Check your Docker volume mounts."
        exit 1
    fi
done

# 2. Disk space (at least 5 GB free on /data/database)
AVAIL_KB=$(docker compose -f docker/compose.yml exec -T chiron df /data/database | awk 'NR==2 {print $4}')
AVAIL_GB=$((AVAIL_KB / 1024 / 1024))
if [ "$AVAIL_GB" -lt 5 ]; then
    echo "WARNING: Only ${AVAIL_GB} GB free on /data/database – you may run out of space."
fi

# 3. GPU availability (inside the container)
if ! docker compose -f docker/compose.yml exec -T chiron python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "WARNING: GPU not available – embedding and LLM will run on CPU, which may be very slow."
fi

# 4. Ollama server reachable (from inside Chiron's network)
if ! docker compose -f docker/compose.yml exec -T chiron curl -s --max-time 5 http://ollama:11434/api/tags > /dev/null; then
    echo "WARNING: Ollama server is not reachable. Start it if you plan to use evaluation."
else
    # Optional: check that the required model exists
    MODEL_NAME="llama3-gradient:8b"
    if ! docker compose -f docker/compose.yml exec -T chiron curl -s http://ollama:11434/api/tags | grep -q "$MODEL_NAME"; then
        echo "WARNING: Model '$MODEL_NAME' not found in Ollama. Pull it if you plan to use evaluation."
    fi
fi

# 5. Configuration file present inside the container
if ! docker compose -f docker/compose.yml exec -T chiron test -f /app/db_config.yml; then
    echo "ERROR: db_config.yml not found inside the container. Place it in docker/ and rebuild."
    exit 1
fi

echo "All pre-flight checks passed."


# =============================================================================
# PIPELINE EXECUTION (inside Chiron container)
# =============================================================================

echo "========================================="
echo "CREATION OF CHIRON DATABASE"
echo "========================================="

echo "[1/6] Fetching Data..."
docker compose -f docker/compose.yml exec chiron python /app/src/fetch.py

# Ask user about golden paper test
FLAG_FILE="./data/database/include_golden.flag"
if [ -f "$FLAG_FILE" ] && [ "$(cat $FLAG_FILE)" = "1" ]; then
    echo "[2/6] Merging golden papers into database..."
    docker compose -f docker/compose.yml exec -T chiron python /app/src/merge_golden.py
else
    echo "[2/6] No merging of golden papers!"
fi

echo "[3/6] Chunking..."
docker compose -f docker/compose.yml exec -T chiron python /app/src/chunk.py

echo "[4/6] Embedding..."
docker compose -f docker/compose.yml exec -T chiron python /app/src/embed.py

echo "[5/6] FAISS-Index..."
docker compose -f docker/compose.yml exec -T chiron python /app/src/faiss_index.py

if [ -f "$FLAG_FILE" ] && [ "$(cat $FLAG_FILE)" = "1" ]; then
    echo "[6/6] Evaluating database with golden questions..."
    docker compose -f docker/compose.yml exec -T chiron python /app/src/evaluation.py
else
    echo "[6/6] No evaluation possible"
fi

echo "========================================="
echo "CHIRON READY"
echo "========================================="