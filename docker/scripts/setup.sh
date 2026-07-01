#!/bin/bash
set -e

#=========================================
# CHECK FOR REQUIRED DIRECTORIES
#=========================================
REQUIRED_DIRS="/data/database /data/chunks /data/embeddings /data/faiss /data/evaluation /data/golden_data /dtd"
for d in $REQUIRED_DIRS; do
    if [ ! -d "$d" ]; then
        echo "ERROR: Required directory $d is missing – check your Docker volume mounts."
        exit 1
    fi
done

#=========================================
# TEST DISK SPACE (on the volume mounted at /data/database)
#=========================================
AVAIL_KB=$(df -k /data/database | awk 'NR==2 {print $4}')
AVAIL_GB=$((AVAIL_KB / 1024 / 1024))
if [ "$AVAIL_GB" -lt 5 ]; then
    echo "WARNING: Only ${AVAIL_GB} GB free on /data/database – you may run out of space."
fi

#=========================================
# TEST GPU AVAILABILITY (does not crash on absence)
#=========================================
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "WARNING: GPU not available – embedding and LLM will run on CPU, which may be very slow."
fi

#=========================================
# TEST LLM AVAILABILITY (always checked if evaluation could be wanted)
#=========================================
if ! curl -s --max-time 5 http://ollama:11434/api/tags > /dev/null; then
    echo "WARNING: Ollama server is not reachable. Start it if you plan to use evaluation."
else
    # Optional: check that the specific model exists
    MODEL_NAME="cniongolo/biomistral"
    if ! curl -s http://ollama:11434/api/tags | grep -q "$MODEL_NAME"; then
        echo "WARNING: Model '$MODEL_NAME' not found in Ollama. Pull it if you plan to use evaluation."
    fi
fi

#=========================================
# TEST DB_CONFIG.YML AVAILABILITY
#=========================================
if [ ! -f /app/db_config.yml ]; then
    echo "ERROR: db_config.yml not found. Place your configuration file in the docker/ directory."
    exit 1
fi


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