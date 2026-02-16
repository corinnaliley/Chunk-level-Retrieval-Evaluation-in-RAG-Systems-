"""
cached_embedding.py: Persistent caching wrapper for LlamaIndex Embedding objects.

This wrapper stores query embeddings on disk to speed up repeated benchmark runs.
The cache is stored as JSON and keyed by raw query string.

Important: changing the embedding model invalidates the cache.
"""
# -----------------------------------------------------------------------------
# Reproducibility notes
# - This cache is derived data. Do not commit it to Git.
# - Cache validity depends on the embedding model and preprocessing.
#   If EMBEDDING_MODEL_NAME or server config changes, invalidate the cache.
# -----------------------------------------------------------------------------
import json
from pathlib import Path
from typing import Dict, List


class CachedEmbedding:
    """Wrap any embedding object and cache embeddings in JSON."""

    def __init__(self, base_embedding, cache_path: Path):
        self.base = base_embedding
        self.cache_path = cache_path
        self.cache: Dict[str, List[float]] = {}
        self.dirty = False

        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as f:
                self.cache = json.load(f)

    def _get_key(self, text: str) -> str:
        return text.strip()

    def get_text_embedding(self, text: str) -> List[float]:
        key = self._get_key(text)

        if key in self.cache:
            return self.cache[key]

        emb = self.base.get_text_embedding(text)
        self.cache[key] = emb
        self.dirty = True
        return emb

    def persist(self):
        if not self.dirty:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("w", encoding="utf-8") as f:
            json.dump(self.cache, f)
        self.dirty = False