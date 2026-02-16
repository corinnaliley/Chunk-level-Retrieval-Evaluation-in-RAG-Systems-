"""
saia_reader.py: Adapters and I/O functions for converting external Block data
(from Docling/SAIA) into LlamaIndex Documents, including caching logic.

The core function is document preparation: combining text blocks and enriching
them with section path metadata for improved retrieval context.
"""
# -----------------------------------------------------------------------------
# Reproducibility / scientific reporting notes
# - This module turns Docling/SAIA-derived Blocks into a single LlamaIndex Document.
# - The chosen block types (PARAGRAPH, LIST) and the "Section: ... Text: ..." prefix
#   are part of the experimental setup and influence retrieval quality.
# - PDF conversion depends on an external service (SAIA/Docling) via loaders.convert_with_docling;
#   for scientific runs, log SAIA endpoint/base URL + model/version if available.
#
# GitHub hygiene
# - PROCESSED_DOCS_PATH points to a derived cache artifact (JSON). Do not commit it.
# -----------------------------------------------------------------------------
import json
from pathlib import Path
from typing import List
from llama_index.core import Document
from loaders import convert_with_docling, BlockType
from constants import PROCESSED_DOCS_PATH


# 1) Core Document Processing

def load_document_via_saia(path: str) -> Document:
    """
    Load a single file (PDF/TXT/…) via SAIA Docling and return
    a single LlamaIndex Document.
    """
    blocks = convert_with_docling(path_or_url=path)

    texts = []
    # Iterate over all blocks to collect context from section paths.
    for b in blocks:
        if not b.text.strip():
            continue

        # Select the block types we want to include in the final text
        if b.block_type in (BlockType.PARAGRAPH, BlockType.LIST):
            context_prefix = ""
            # with section path
            # Prepend the section_path (e.g., 'Chapter 1 > Section 2') to the text.
            # This significantly improves the context for the embedding model.
            if b.section_path:
                context_prefix = f"Section: {b.section_path}. Text: "
            # Add the prepared text to the list
            texts.append(context_prefix + b.text.strip())

            # without section path
            # texts.append(b.text.strip())

    full_text = "\n\n".join(texts)

    return Document(
        text=full_text,
        metadata={"source": str(path)},
    )


def load_documents_from_dir_via_saia(data_dir: str) -> List[Document]:
    """
    Load all .pdf and .txt files from a directory via SAIA Docling
    and return a list of LlamaIndex Documents.
    """
    docs: List[Document] = []
    for path in Path(data_dir).rglob("*"):
        if path.suffix.lower() not in {".pdf", ".txt"}:
            continue
        try:
            doc = load_document_via_saia(str(path))
            docs.append(doc)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue
    return docs


# 2) JSON Caching Functions

def save_documents_to_json(docs: List[Document], path: Path = PROCESSED_DOCS_PATH):
    """
    Saves a list of LlamaIndex Document objects to a JSON file.
    """
    try:
        data = [doc.to_json() for doc in docs]
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f)
        print(f"[INFO] Documents saved to: {path}")
    except Exception as e:
        print(f"[ERROR] Failed to save documents to JSON: {e}")

def load_documents_from_json(path: Path = PROCESSED_DOCS_PATH) -> List[Document] | None:
    """
    Loads a list of LlamaIndex Document objects from a JSON cache file.

    Returns: List[Document] if successful, None otherwise.
    """
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        docs = [Document.from_json(item) for item in data]
        print(f"[INFO] Documents loaded from cache: {path}")
        return docs
    except Exception as e:
        print(f"[ERROR] Failed to load documents from JSON: {e}")
        return None