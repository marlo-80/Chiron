#!/bin/bash
set -e

echo "========================================="
echo "PubMed RAG EVALUATION"
echo "========================================="

echo "[1/1] Starting Evaluation..."
python evaluation.py

echo "========================================="
echo "EVALUATION FINISHED"
echo "========================================="