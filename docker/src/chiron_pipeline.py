"""
Open‑WebUI manifold pipeline for the Chiron medical RAG system.

This pipeline receives user questions from Open‑WebUI, retrieves the most
relevant text chunks from the local FAISS index (via retrieval.py), builds a
structured prompt with inline‑citation instructions, sends it to Ollama for
full‑text generation, and streams the answer back to the browser together with
a formatted reference list.

The module is loaded and managed by Open‑WebUI's pipeline framework. It
expects the following companion modules to be importable from /app/src:
    config.py      – shared paths and parameters
    retrieval.py   – FAISS‑based chunk retrieval with PubMedBERT embeddings

The LLM (llama3‑gradient:8b) is queried in streaming mode to provide a
responsive user experience. The prompt instructs the model to cite every
fact using bracketed source numbers corresponding to the reference list
appended at the end of the response.
"""

import os
import sys
import json
import requests
import importlib.util
from typing import List, Union, Generator, Iterator

class Pipeline:
    class Valves(json.JSONEncoder):
        pass

    def __init__(self):
        self.type = "manifold"
        self.id = "chiron_medical_rag"
        self.name = "Chiron Medical RAG (Full Text & Citations)"

    def pipelines(self) -> List[dict]:
        return [{"id": self.id, "name": self.name}]

    async def on_startup(self):
        print("Chiron successfully initialized!")

    async def on_shutdown(self):
        pass

    def pipe(self, **kwargs) -> str:
        try:
            user_message = kwargs.get("user_message", "").strip()
            body = kwargs.get("body", {})
            
            if not user_message and body:
                messages = body.get("messages", [])
                if messages:
                    user_message = messages[-1].get("content", "").strip()

            if not user_message:
                return "No question received."

            # Pre-load our config to prevent Open‑WebUI's own /app/config.py from shadowing it.
            if 'config' not in sys.modules or not hasattr(sys.modules['config'], 'EMBEDDING_DIR'):
                config_spec = importlib.util.spec_from_file_location("config", "/app/src/config.py")
                config_module = importlib.util.module_from_spec(config_spec)
                config_spec.loader.exec_module(config_module)
                sys.modules['config'] = config_module

            retrieval_spec = importlib.util.spec_from_file_location("local_retrieval", "/app/src/retrieval.py")
            local_retrieval = importlib.util.module_from_spec(retrieval_spec)
            
            if "/app/src" not in sys.path:
                sys.path.insert(0, "/app/src")
                
            retrieval_spec.loader.exec_module(local_retrieval)

            # Defines the number of retrieved chunks
            results = local_retrieval.retrieve_with_metadata(user_message, k=3)
            
            # Incremental context generation
            kontext_blocks = []
            quellen_verzeichnis = []
            
            for index, item in enumerate(results):
                metadata, text_chunk = item[0], item[1]
                source_num = index + 1
                
                kontext_blocks.append(f"[Source {source_num}]\n{text_chunk}\n")
                
                # Read metadata
                authors = metadata.get("authors", "Unknown Authors")
                title = metadata.get("title", "Unknown Title")
                year = metadata.get("year", "N/A")
                journal = metadata.get("journal", "Unknown Journal")
                doi = metadata.get("doi", "N/A")
                pmcid = metadata.get("pmcid", "Unknown ID")

                quellen_verzeichnis.append(
                    f"[{source_num}] {authors} ({year}). **{title}**. "
                    f"*{journal}*. DOI: {doi} (PMCID: {pmcid})"
                )     

            kontext = "\n".join(kontext_blocks).strip()

            # Promt
            prompt = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                "You are an elite medical research assistant. Your task is to provide a comprehensive, detailed, "
                "and scientifically accurate full-text answer to the user's question based ONLY on the provided context.\n\n"
                "CRITICAL CITATION RULES:\n"
                "- Every fact, claim, or clinical conclusion you make must be directly cited from the context.\n"
                "- Use inline citations like, [2], or [3] exactly where the information is used (e.g., 'Hypertension was found in 45% of patients [1].').\n"
                "- Do not synthesize information without adding the source number.\n"
                "- If the context doesn't contain enough information to answer, state it clearly and cite what is available.\n"
                "- Do not write a summary bibliography at the end, just focus on the cited text response.<|eot_id|>\n"
                "<|start_header_id|>user<|end_header_id|>\n"
                f"Context:\n{kontext}\n\n"
                f"Question: {user_message}<|eot_id|>\n"
                "<|start_header_id|>assistant<|end_header_id|>\n"
            )

            OLLAMA_GENERATE_URL = "http://ollama:11434/api/generate"
            MODEL_NAME = "llama3-gradient:8b"

            payload = {
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": True,  
                "options": {
                    "temperature": 0.2,
                    "num_predict": 500
                }
            }

            # Send stream=True to requests
            resp = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=90, stream=True)
            resp.raise_for_status()
            
            for line in resp.iter_lines():
                if line:
                    chunk_json = json.loads(line.decode("utf-8"))
                    token = chunk_json.get("response", "")
                    if token:
                        yield token
                    if chunk_json.get("done", False):
                        break
            
            # Add literature
            literatur_section = "\n\n---\n### 📚 References\n" + "\n".join(quellen_verzeichnis)
            yield literatur_section
                
        except Exception as e:
            yield f"Error in Chiron-RAG-kernel: {str(e)}"
