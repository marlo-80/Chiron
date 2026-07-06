#!/bin/bash
set -e

echo "=============================================================================="
echo "                                 CHIRON SETUP                                 "
echo "=============================================================================="


# =============================================================================
# PRE-FLIGHT CHECKS (all run inside the Chiron container)
# =============================================================================

echo "Setup checks starting..."

# 1. Required directories

for d in /data/chunks /data/embeddings /data/faiss /data/evaluation; do
    if ! docker compose -f docker/compose.yml exec -T chiron test -d "$d"; then
        echo "Creating folder: $d"
        docker compose -f docker/compose.yml exec -T chiron mkdir -p "$d" || {
            echo "Error during creation of $d"
            exit 1
        }
        echo "$d Folder created!"
    else
        echo "."
    fi
done


for d in /data/database /data/chunks /data/embeddings /data/faiss /data/evaluation /data/golden_data /dtd; do
    if ! docker compose -f docker/compose.yml exec -T chiron test -d "$d"; then
        echo "ERROR: $d not found inside the Chiron container. Check your Docker volume mounts."
        exit 1
    else
        echo "."
    fi    
done

# 2. Disk space (at least 5 GB free on /data/database)
AVAIL_KB=$(docker compose -f docker/compose.yml exec -T chiron df /data/database | awk 'NR==2 {print $4}')
AVAIL_GB=$((AVAIL_KB / 1024 / 1024))
if [ "$AVAIL_GB" -lt 5 ]; then
    echo "WARNING: Only ${AVAIL_GB} GB free on /data/database – you may run out of space."
else
    echo "."
fi

# 3. GPU availability (inside the container)
if ! docker compose -f docker/compose.yml exec -T chiron python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "WARNING: GPU not available – embedding and LLM will run on CPU, which may be very slow."
else
    echo "."    
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
    echo "."
fi

# 5. Configuration file present inside the container
if ! docker compose -f docker/compose.yml exec -T chiron test -f /app/db_config.yml; then
    echo "ERROR: db_config.yml not found inside the container. Place it in docker/ and rebuild."
    exit 1
else
    echo "."     
fi

echo "...setup checks finished."
echo ""

# =============================================================================
# PIPELINE EXECUTION (inside Chiron container)
# =============================================================================


echo "=============================================================================="
echo "                               [1/6] DOWNLOAD                                 "
echo "=============================================================================="
docker compose -f docker/compose.yml exec chiron python /app/src/fetch.py

echo "=============================================================================="
echo "                     [2/6] MERGE OF EVALUATION PAPERS                         "
echo "=============================================================================="
FLAG_FILE="./data/database/include_golden.flag"
if [ -f "$FLAG_FILE" ] && [ "$(cat $FLAG_FILE)" = "1" ]; then
    docker compose -f docker/compose.yml exec -T chiron python /app/src/merge_golden.py

else
    echo "No merging of golden papers!"
fi

echo "=============================================================================="
echo "                               [3/6] CHUNKING                                 "
echo "=============================================================================="
docker compose -f docker/compose.yml exec -T chiron python /app/src/chunk.py

echo "=============================================================================="
echo "                              [4/6] EMBEDDING                                 "
echo "=============================================================================="
docker compose -f docker/compose.yml exec -T chiron python /app/src/embed.py

echo "=============================================================================="
echo "                             [5/6] FAISS INDEX                                "
echo "=============================================================================="
docker compose -f docker/compose.yml exec -T chiron python /app/src/faiss_index.py

echo "=============================================================================="
echo "                              [6/6] EVALUATION                                "
echo "=============================================================================="
if [ -f "$FLAG_FILE" ] && [ "$(cat $FLAG_FILE)" = "1" ]; then
    docker compose -f docker/compose.yml exec -T chiron python /app/src/evaluation.py
else
    echo "No evaluation possible"
fi

echo "=============================================================================="
echo "                                  CHIRON READY                                "
echo "=============================================================================="