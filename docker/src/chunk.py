#!/usr/bin/env python3
"""
Parallel sentence segmentation and chunking of biomedical articles.

Reads the extracted article database (SQLite), splits the text of each
configured section (abstract, introduction, methods, results, discussion,
conclusion) into sentences using spaCy's scientific English model, then
builds overlapping chunks of a fixed number of sentences. Processing is
parallelized across all available CPU cores for maximum throughput.

Input:
    /data/database/articles.sqlite3  (table `articles`)
Output:
    /data/chunks/chunks.pkl          – DataFrame with column "chunk_text"
    /data/chunks/metadaten.pkl       – list of per‑chunk metadata dicts

Parameters (from config.py):
    CHUNK_SIZE : number of sentences per chunk (default 10)
    OVERLAP    : overlapping sentences between consecutive chunks (default 3)
    SEKTIONEN  : list of section names to process

The spaCy pipeline is stripped of unnecessary components (NER, tagger,
lemmatizer) to maximise speed.
"""

from concurrent.futures import ProcessPoolExecutor
import pickle
import pandas as pd
import spacy
from tqdm import tqdm
from pathlib import Path
import os
from config import DB_PATH, CHUNKS_DIR, CHUNK_SIZE, OVERLAP, SEKTIONEN
import sqlite3
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="spacy")

# =============================================================================
# GLOBAL NLP-INITIALIZER
# =============================================================================
nlp = None



# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def init_worker():
    """Initializes spaCy once per CPU core with a minimal pipeline."""
    global nlp
    try:
        # Deactivation of performance killers
        nlp = spacy.load("en_core_sci_sm", disable=["ner", "tagger", "lemmatizer", "attribute_ruler"])
    except OSError:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer", "attribute_ruler"])
    
    # Test for availability
    if "senter" not in nlp.pipe_names and "parser" not in nlp.pipe_names:
        nlp.add_pipe("senter")

def process_paper_chunking(paper_row_tuple):
    """Processes a single paper on an isolated CPU core."""
    idx, row = paper_row_tuple
    
    local_chunks = []
    local_metadata = []
    chunk_counter = 0
    
    for sektion in SEKTIONEN:
        text = row.get(sektion, "")
        if not isinstance(text, str) or not text.strip():
            continue
            
        # Segmentation with SpaCy
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        if not sentences:
            continue
            
        # Originale Chunking-Schleife mit Overlap
        step = CHUNK_SIZE - OVERLAP
        for i in range(0, len(sentences), step):
            chunk_sentences = sentences[i:i + CHUNK_SIZE]
            if chunk_sentences:
                chunk_text = ' '.join(chunk_sentences)
                if not chunk_text.endswith('.'):
                    chunk_text += '.'
                    
                chunk_counter += 1
                local_chunks.append(chunk_text)
                local_metadata.append({
                    'pmcid': row['pmcid'],
                    'title': row['title'],
                    'year': row['year'],
                    'authors': row.get('authors', ''),
                    'journal': row.get('journal', ''),          
                    'doi': row.get('doi', ''),                  
                    'section': sektion,
                    'chunk_index_in_paper': chunk_counter,
                    'chunk_length': len(chunk_text.split())
                })
                
    return local_chunks, local_metadata

def main():
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query("SELECT * FROM articles", conn)
    conn.close()

    print(f"Papers loaded: {len(df)}")
    print(f"Chunking started...")

    # List of tupels necessary
    paper_tasks = list(df.iterrows())
    
    chunks = []
    metadaten = []

    with ProcessPoolExecutor(initializer=init_worker) as executor:
        # Process bar
        results = list(tqdm(
            executor.map(process_paper_chunking, paper_tasks, chunksize=50), 
            total=len(df), 
            desc="Paralleles Chunking"
        ))
        
        # Aggregate results
        for local_chunks, local_metadata in results:
            chunks.extend(local_chunks)
            metadaten.extend(local_metadata)

    # Write Chunks
    print("\nSaving chunks...")
    pd.DataFrame({"chunk_text": chunks}).to_pickle(CHUNKS_DIR / "chunks.pkl")
    with open(CHUNKS_DIR / "metadaten.pkl", "wb") as f:
        pickle.dump(metadaten, f)

    print(f"Created: {len(chunks)} chunks from {len(df)} papers")
    print(f"Average: {len(chunks)/len(df):.1f} chunks/paper")
    print(f"Saved to: {CHUNKS_DIR}")

if __name__ == "__main__":
    main()
