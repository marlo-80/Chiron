#!/bin/bash
# reset.sh – löscht alle Zwischenergebnisse und die SQLite‑Datenbank,
#            behält aber die golden‑paper‑Flag‑Datei.
# Auszuführen im Projekt‑Root (dort, wo docker/compose.yml liegt).

set -e

echo "This will delete all chunks, embeddings, FAISS index, evaluation"
echo "results, AND the SQLite article database."
echo "The golden‑paper flag (include_golden.flag) will be kept."
read -p "Continue? (y/N) " -r REPLY
if [[ ! "$REPLY" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo "Cleaning up data directories..."

# Chunks
rm -rf data/chunks/*
echo "  ✓ data/chunks/ emptied"

# Embeddings
rm -rf data/embeddings/*
echo "  ✓ data/embeddings/ emptied"

# FAISS index
rm -rf data/faiss/*
echo "  ✓ data/faiss/ emptied"

# Evaluation results
rm -rf data/evaluation/*
echo "  ✓ data/evaluation/ emptied"

# SQLite database – remove only the database and error CSV, keep the flag
rm -f data/database/database.db
rm -f data/database/extraction_errors_database.csv
echo "SQLite database and error log removed (flag preserved)"

echo ""
echo "Reset complete. You can now run ./setup.sh to rebuild the RAG database."