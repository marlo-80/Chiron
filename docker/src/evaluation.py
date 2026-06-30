#!/usr/bin/env python3
"""
Evaluate the RAG pipeline against 69 golden annotated questions.

For each question:
  - Retrieve the top‑3 chunks from the FAISS index.
  - Prompt the configured Ollama model (llama3‑gradient:8b) to answer
    with a structured JSON object containing a "decision" field
    ("yes", "no", or "maybe").
  - Compare the decision to the ground truth and compute accuracy,
    per‑class precision/recall/F1, confusion matrix, and retrieval
    metrics (Hit‑Rate@15, MRR).

Results are written to /data/evaluation/ as CSV and JSON files.

Requires:
    retrieval.py   – retrieve_with_metadata()
    config.py      – GOLDEN_DIR, EVAL_DIR
    golden69_questions.pkl in the golden data directory.
"""

import time, json, pickle
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from retrieval import retrieve_with_metadata
from config import GOLDEN_DIR, EVAL_DIR
import json

EVAL_DIR.mkdir(parents=True, exist_ok=True)
OLLAMA_CHAT_URL = "http://ollama:11434/api/chat"

# =============================================================================
# MODEL SETTINGS
# =============================================================================
#MODEL_NAME = "qwen2.5:7b-instruct"
MODEL_NAME = "llama3-gradient:8b"
# MODEL_NAME = "llama3:8b"

OLLAMA_GENERATE_URL = "http://ollama:11434/api/generate"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def generate_answer(query: str, chunks: list) -> str:
    kontext = " ".join(chunks[:3]).strip()
    
    # Universal promt for Qwen and Llama
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You are a precise medical referee. Answer the user's question based ONLY on the provided context.\n"
        "You must respond ONLY with a JSON object following this exact schema:\n"
        "{\n"
        "  \"decision\": \"yes\" or \"no\" or \"maybe\"\n"
        "}\n"
        "Rules:\n"
        "- 'yes': the context explicitly and directly confirms the question.\n"
        "- 'no': the context explicitly denies or contradicts the question.\n"
        "- 'maybe': the context is insufficient, missing direct proof, or ambiguous.<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"Context:\n{kontext}\n\n"
        f"Question: {query}<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": {
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": ["yes", "no", "maybe"]
                }
            },
            "required": ["decision"]
        },
        "options": {
            "temperature": 0.0,
            "num_predict": 15
        }
    }
    
    resp = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["response"].strip()


def parse_answer(text: str) -> str:
    text = text.strip()
    if not text:
        return "unknown"
    try:
        data = json.loads(text)
        decision = data.get("decision", "unknown").lower().strip()
        if decision in ("yes", "no", "maybe"):
            return decision
    except Exception:
        for w in ("yes", "no", "maybe"):
            if w in text.lower():
                return w
    return "unknown"



# =============================================================================
# QUESTION RETRIEVAL
# =============================================================================
df_questions = pd.read_pickle(GOLDEN_DIR / "golden69_questions.pkl")
has_retrieval_gt = 'Accession ID' in df_questions.columns

true_labels, pred_labels, hit_rates, mrr_scores, latencies, detailed_results = [], [], [], [], [], []

for idx, row in tqdm(df_questions.iterrows(), total=len(df_questions), desc="Evaluating"):
    start = time.time()
    question = row['question']
    true = row['final_decision'].strip().lower()

    results = retrieve_with_metadata(question, k=3)
    chunks_ret = [r[1] for r in results]
    retrieved_pmcids = [r[0].get('pmcid') for r in results]

    if has_retrieval_gt:
        true_accid = row.get('Accession ID', None)
        if true_accid is not None:
            hit = 1 if true_accid in retrieved_pmcids else 0
            hit_rates.append(hit)
            mrr_scores.append(1/(retrieved_pmcids.index(true_accid)+1) if hit else 0)
        else:
            hit_rates.append(0); mrr_scores.append(0)

    answer = generate_answer(question, chunks_ret)
    pred = parse_answer(answer)

    detail = {
        'question': question, 'true_label': true, 'predicted_label': pred,
        'is_correct': true == pred, 'latency_seconds': time.time() - start,
        'num_chunks_retrieved': len(chunks_ret), 'retrieved_pmcids': retrieved_pmcids,
        'llm_raw_answer': answer, 'parsed_answer': pred
    }
    if has_retrieval_gt and true_accid is not None:
        detail['true_pmcid'] = true_accid
        detail['hit'] = hit
    detailed_results.append(detail)

    true_labels.append(true); pred_labels.append(pred); latencies.append(detail['latency_seconds'])


# =============================================================================
# RESULT CALCULATION
# =============================================================================
valid = [i for i, p in enumerate(pred_labels) if p in ['yes', 'no', 'maybe']]
true_valid = [true_labels[i] for i in valid]
pred_valid = [pred_labels[i] for i in valid]
labels = ['yes', 'no', 'maybe']

accuracy = accuracy_score(true_valid, pred_valid)
p, r, f1, _ = precision_recall_fscore_support(true_valid, pred_valid, labels=labels, average=None, zero_division=0)
mp, mr, mf1, _ = precision_recall_fscore_support(true_valid, pred_valid, average='macro', zero_division=0)
cm = confusion_matrix(true_valid, pred_valid, labels=labels)


# =============================================================================
# PRINTING
# =============================================================================
print("\n" + "="*60)
print("Results:")
print("="*60)
print(f"Questions: {len(df_questions)}, Unknown: {pred_labels.count('unknown')}, True: {len(true_valid)}")
print(f"Accuracy: {accuracy:.4f}")
for i, l in enumerate(labels):
    print(f"  {l.upper()}: P={p[i]:.4f} R={r[i]:.4f} F1={f1[i]:.4f}")
print(f"Macro: P={mp:.4f} R={mr:.4f} F1={mf1:.4f}")
print("Confusion Matrix:")
print(pd.DataFrame(cm, index=labels, columns=labels))

if has_retrieval_gt and hit_rates:
    print(f"\nRetrieval Hit Rate@15: {np.mean(hit_rates):.4f}, MRR: {np.mean(mrr_scores):.4f}")
print(f"Average latency: {np.mean(latencies):.2f}s")

df_flat = pd.DataFrame(detailed_results).copy()
for col in ['retrieved_pmcids']:
    if col in df_flat.columns:
        df_flat[col] = df_flat[col].apply(json.dumps)
df_flat.to_csv(EVAL_DIR / "rag_evaluation_detailed.csv", index=False)

with open(EVAL_DIR / "rag_evaluation_detailed.json", 'w', encoding='utf-8') as f:
    json.dump(detailed_results, f, indent=2, ensure_ascii=False)

pd.DataFrame({'question': df_questions['question'], 'true_label': true_labels, 'pred_label': pred_labels})\
    .to_csv(EVAL_DIR / "rag_evaluation_results.csv", index=False)

print(f"\nResults save to {EVAL_DIR}")