# Chiron – Your Personal Offline RAG System for PubMed Scientific Papers

Chiron builds a custom, offline Retrieval Augmented Generation (RAG) database from PubMed’s open‑access papers. It lets you define exactly which medical topics, journals, and time periods matter to you, downloads only the relevant full‑text articles, and turns them into a searchable index that you can query with any language model you run locally via Ollama.



<p align="center">
<img src="data/golden_data/chiron.png" alt="Description" width="800">
</p>




## What Chiron Does

- **Fetches targeted PubMed papers** – Instead of downloading millions of irrelevant articles, Chiron queries the PubMed API with your search criteria (keywords, journals, date range, MeSH terms). It then streams the matching full‑text XMLs directly from the PMC S3 bucket and processes them on the fly.
- **Extracts structured text** – Each paper is parsed into its sections (abstract, introduction, methods, results, discussion, conclusion) and only high‑quality articles with sufficient content are kept.
- **Chunks and embeds** – The articles are split into overlapping sentence‑level chunks using spaCy’s biomedical model. Every chunk is converted into a 768‑dimensional vector by PubMedBERT.
- **Creates a FAISS index** – The embeddings are stored in a FAISS index for fast cosine‑similarity retrieval.
- **Answers questions (evaluation mode)** – With the golden‑questions test set, Chiron can evaluate how well the system works. It retrieves the most relevant chunks and sends them together with the question to a language model (Biomistral, Mistral, or any Ollama model) to get a yes/no/maybe answer.


## How It Works (Pipeline Overview)

The entire process runs inside Docker and is orchestrated by a single script (`setup.sh`). The main steps are:

1. **Fetch** (`fetch.py`)  
   - Reads the configuration from `db_config.yml`.  
   - Constructs a complex PubMed query (Boolean AND/OR/NOT across keywords, journals, time period).  
   - Retrieves all matching PMC IDs via the NCBI E‑utilities.  
   - Offers an interactive choice: download all papers or a subset (latest N or random N).  
   - Downloads the XML files directly from the AWS S3 bucket and extracts the article data.  
   - Inserts the articles into a local SQLite database (`/data/database/database.db`).  

2. **Merge golden papers** (`merge_golden.py`) *(only if the golden test set is requested)*  
   - Adds the 69 pre‑curated PubMedQA papers into the database so that the evaluation step can use them.  

3. **Chunking** (`chunk.py`)  
   - Reads all articles from the SQLite database.  
   - Splits each article’s textual sections into overlapping chunks of 10 sentences (configurable).  
   - Saves the chunks and their metadata to `/data/chunks/`.  

4. **Embedding** (`embed.py`)  
   - Uses PubMedBERT (via Sentence‑Transformers) to convert every chunk into a 768‑dimensional vector.  
   - Stores the embeddings in `/data/embeddings/`.  

5. **FAISS Index** (`faiss_index.py`)  
   - Normalises the embeddings and builds a FAISS `IndexFlatIP` for fast cosine‑similarity retrieval.  
   - Saves the index to `/data/faiss/`.  

6. **Evaluation** (`evaluation.py`) *(only if golden papers are included)*  
   - Loads the 69 golden questions, retrieves the top‑k chunks, and asks the configured LLM (via Ollama) to answer with “yes”, “no”, or “maybe”.  
   - Calculates accuracy, precision, recall, F1, and a confusion matrix.  
   - Results are saved to `/data/evaluation/`.  



## Getting Started

### Prerequisites
- **Docker** and **Docker Compose**
- About **5 GB free disk space** for the Docker image, models, and your personal database. Depends on the number of downloaded papers. 


#### Optional:  
- An **NCBI account** (free) to get an API key (optional but raises the request rate from 3 to 10 per second).  

#### For GPU utilization only:
- Nvidia container toolkit

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/marlo-80/Chiron
   cd Chiron
    ```
2. **Configure your database**  
   Edit `docker/db_config.yml` to define your desired set of papers.  
   Example:  
   ```yaml
   data_selection:
     time_period: "2015:2026"
     keywords:
       must_contain:
         - "diabetes"
       or_groups:
         - - "complications"
           - "management"
         - - "patients"
           - "cohort"
       must_not_contain:
         - "in vitro"
     journals:
       or_groups:
         - - "PLOS ONE"
           - "Diabetes Care"
     mesh_subjects: {}
   ```
   The schema is flexible: each filter category (`keywords`, `journals`, `mesh_subjects`) supports `must_contain`, `or_groups`, and `must_not_contain`.

3. **Set your NCBI email**  
   In `docker/compose.yml`, replace `your.email@example.com` with your own email (required by NCBI for API access).

4. **(Optional) Adjust model and evaluation settings**  
   In `docker/src/evaluation.py` you can change `MODEL_NAME` to any Ollama model you have pulled. Recommended is `lama3-gradient:8b`.

### Building and Starting the Containers

```bash
docker compose -f docker/compose.yml build
docker compose -f docker/compose.yml up -d
```

This will start three containers:
- **Chiron** – the main pipeline worker.
- **ollama** – the LLM server (GPU‑enabled if correctly configured).
- **openwebui** – a web interface for interacting with Ollama (optional).

### Pull a Language Model

Before evaluating, you need an LLM. Enter the Ollama container and pull your preferred model:

```bash
docker compose -f docker/compose.yml exec ollama ollama pull llama3-gradient:8b
```

(Any model that works with Ollama will work, e.g., `mistral`, `llama3`, `cniongolo/biomistral`.)



## Usage

### Create / Update the Database

Run the main setup script from the project root:

```bash
./setup.sh
```

The script will:
- Ask for the number of papers (subset) and how to select them (latest/random).
- Fetch, process, chunk, embed, and build the index.
- Ask whether you want to add the golden test papers (for evaluation).
- If golden papers were requested, it will also merge them and run the evaluation.

All data is stored in the `data/` directory on your host. The structure is:

```
data/
├── database/        # SQLite article database
├── chunks/          # pickled chunks and metadata
├── embeddings/      # numpy embeddings and metadata
├── faiss/           # FAISS index
├── evaluation/      # evaluation results (CSV, JSON)
├── golden_data/     # the 69 golden questions and their papers
├── dtd/             # JATS DTD files (required for XML parsing)
└── hf_cache/        # HuggingFace model cache (PubMedBERT etc.)
```

### Run Only the Evaluation (after setup)

If you already have a database and index, you can run the evaluation any time with:

```bash
docker compose -f docker/compose.yml exec chiron python /app/src/evaluation.py
```


## Project Structure

```
Chiron/
├── docker/
│   ├── compose.yml
│   ├── dockerfile
│   ├── requirements.txt
│   ├── db_config.yml          # your custom query
│   ├── scripts/
│   │   └── setup.sh           # internal script called by root ./setup.sh
│   └── src/
│       ├── fetch.py
│       ├── merge_golden.py
│       ├── chunk.py
│       ├── embed.py
│       ├── faiss_index.py
│       ├── retrieval.py
│       ├── evaluation.py
│       ├── config.py
│       ├── schema.py
│       └── chiron_pipeline.py # Open WebUI pipeline (optional)
├── data/                      # (created after first run)
├── setup.sh                   # root convenience script
└── README.md
```


## Disclaimer

The PubMed database is far too large for a complete offline system on consumer hardware. Chiron therefore selects a **subset** of papers based on your configuration. Additionally, articles must meet certain formal quality criteria (minimum abstract length, sufficient full‑text content) to be included.

PubMed papers are freely available and are paired with a curated question‑and‑answer dataset (PubMedQA), making them an ideal sandbox for RAG systems. However, the **language model performance is limited**: Chiron’s retrieval reliably finds the correct document chunks, but current offline LLMs still struggle to derive the correct yes/no/maybe answers from those chunks. This project is a demonstration of how a targeted, offline RAG pipeline can be built – not a professional medical decision support system.



<!-- ## License & Acknowledgements

This project is intended for non‑commercial research and educational use only. All PubMed articles retain their original copyright and license terms. Please refer to the individual papers for specific usage conditions. -->
