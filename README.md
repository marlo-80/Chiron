# Chiron: Your Personal Offline RAG System for PubMed Scientific Paper
This projects creates on Retrieval Augmented Generation (RAG) system that answers medical questions on the basis of the PubMed open access medical data. It contains more than 8 million papers comprising the vast majority of research areas in the field of medicine. To reduce those data to an amount a consumer PC can handle you can specify your personal data base. You can use time, scientific field, journal, keywords to define the data you are interested in.


PubMed papers are chunked by meaningful sections and embedded using PubMedBERT. The retrieval is based on FAISS vector search while the Large Language Model (LLM) utilized can be freely defined using OLLAMA.

### Disclaimer
The Pubmed database is too big to use its entirety for an offline RAG system that runs on a consumer PC. So, the aim of this project is not to build a professional expert system for medical questions but to demonstrate how a RAG system for offline usage can be build. Hence, we only take a representative sub sample from the papers. Additionally, papers also need to meet certain formal criteria to be added to the data base. 

PubMed paper contain complex knowledge that can be accessed for free. Additionally, one can find curated question and answer datasets for selected PubMed papers. For these reasons, PubMed papers are a perfect use case to demonstrate the capabilities of this RAG system.

However, the performance of this RAG system is rather poor in answering medical questions. This is not because of the retrieval of relevant information but the LLM performance. Mimir reliably retrieves the correct chunks from the data base but the offline LLM is unable to derive the correct answers from those chunks. 

## Prerquisites
Download PubMed papers for non-commercial use:
`https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/xml/oa_noncomm_xml.PMC003xxxxxx.baseline.2026-01-23.tar.gz`

Install Ollama and add on LLM of your choice.

Download Data
https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/xml/oa_noncomm_xml.PMC003xxxxxx.baseline.2026-01-23.tar.gz
* Big data set to extract training and test data

https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/xml/oa_noncomm_xml.incr.2026-03-04.tar.gz
* Small data set for proof of concept testing

NVIDIA GPU (optional)
Make sure your docker installation can use the GPU

## Set up your Environment

The added [requirements file](requirements.txt) contains all libraries and dependencies we need to execute the notebooks.

### **`macOS`** type the following commands : 

- Install the virtual environment and the required packages by following commands:

    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
### **`WindowsOS`** type the following commands :

- Install the virtual environment and the required packages by following commands.

   For `PowerShell` CLI :

    ```PowerShell
    pyenv local 3.11.3
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    For `Git-Bash` CLI :
    ```
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/Scripts/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

## GPU Utilisation
To use your GPU for faster processing make sure to install the correct PyTorch version for your CUDA installation. By default Mimir installs the most currrent torch version.
If you need a different version adjust the `docker/dockerfile`
  

