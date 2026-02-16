"""
constants.py: Centralized configuration for the RAG benchmarking system.

This module loads necessary environment variables and defines all fixed
configuration settings (API endpoints, model names, default RAG parameters).
It is the single source of truth for global configuration.
"""
# -----------------------------------------------------------------------------
# Reproducibility / scientific reporting notes
# - This file is the single source of truth for experimental defaults.
# - Any change here (models, chunking, paths, language) changes benchmark results.
# - For every run, record:
#     * LLM_MODEL_NAME, EVAL_MODEL_NAME, EMBEDDING_MODEL_NAME
#     * DEFAULT_LANG, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
#     * endpoints (LLM_API_ENDPOINT, SAIA_BASE_URL, SAIA_DOCLING_ENDPOINT)
#     * repository commit hash
#
# GitHub hygiene
# - Never commit secrets: LLM_API_KEY / SAIA_API_KEY must come from environment.
# - Derived artifacts (storage, processed docs, results) belong in .gitignore.
# -----------------------------------------------------------------------------

from dotenv import load_dotenv
import os
import pathlib

# --- Environment Setup (Load .env file) --------------------------------------

env_path = pathlib.Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# --- 1) API Keys and Endpoints ------------------

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_API_ENDPOINT = "https://chat-ai.academiccloud.de/v1/"

    # === SAIA / Docling (GWDG configuration) ===
    # Base URL points to /v1, as in the documentation
SAIA_BASE_URL = os.getenv("SAIA_BASE_URL", "https://chat-ai.academiccloud.de/v1")
SAIA_API_KEY = os.getenv("SAIA_API_KEY", "")
    # docling for PDF
SAIA_DOCLING_ENDPOINT = "documents/convert"

# --- 2) Model Configuration --------------------------------------------------

# Language Model (LLM) for generating answers (e.g. Llama-3.1)

LLM_MODEL_NAME = "meta-llama-3.1-8b-instruct"
#LLM_MODEL_NAME = "llama-3.3-70b-instruct"
#LLM_MODEL_NAME = "GPT-5"
EVAL_MODEL_NAME = "meta-llama-3.1-70b-instruct"
EMBEDDING_MODEL_NAME = "e5-mistral-7b-instruct"
#EMBEDDING_MODEL_NAME = "multilingual-e5-large-instruct"
#EMBEDDING_MODEL_NAME = "qwen3-embedding-4b"


# --- 3) RAG Defaults (Document Processing and Indexing) ----------------------

DEFAULT_LANG = "de"
#DEFAULT_CHUNK_SIZE = 280
#DEFAULT_CHUNK_OVERLAP = 80
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 150
#DEFAULT_CHUNK_SIZE = 700
#DEFAULT_CHUNK_OVERLAP = 250


DEFAULT_TITLE_BOOST = True

# --- File Paths ---
PATH_ROOT = pathlib.Path(__file__).resolve().parent.parent # Assuming /src structure
DATA_DIR = PATH_ROOT / "data" / "raw"
STORAGE_PATH = PATH_ROOT / "data" / "storage"
CACHE_DIR = PATH_ROOT / "data" / "processed"
PROCESSED_DOCS_PATH = CACHE_DIR / "processed_docs.json" # Moved from saia_reader.py
RESULTS_PATH = PATH_ROOT / "results"
QUERY_EMB_CACHE = CACHE_DIR / "query_emb_cache.json"


# --- Benchmark Configuration ---
STRATEGIES = ["sparse", "dense", "hybrid_rrf"]