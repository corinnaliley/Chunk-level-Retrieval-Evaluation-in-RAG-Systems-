# -----------------------------------------------------------------------------
# retrievers.py — Retrieval strategy definitions
#
# Scientific / reproducibility notes
# - This module defines the exact retrieval configuration used in experiments.
# - DEFAULT_RETRIEVER_PARAMS materially influence reported metrics.
#   If you change any of these values, treat it as a new experimental condition.
# - For paper reporting, log:
#     * similarity_top_k_dense
#     * similarity_top_k_sparse
#     * rrf_top_k_per_retriever
#     * k_final
#     * bm25_language
# - Deduplication logic (_dedupe) affects ranking and therefore MRR/nDCG.
#
# GitHub hygiene
# - Keep DEFAULT_RETRIEVER_PARAMS stable across comparisons.
# -----------------------------------------------------------------------------
from __future__ import annotations
from typing import List, Optional, Sequence
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle
from llama_index.retrievers.bm25 import BM25Retriever
from constants import DEFAULT_LANG

# Default retriever parameters
DEFAULT_RETRIEVER_PARAMS = {
    "similarity_top_k_dense": 50,
    "similarity_top_k_sparse": 50,
    "rrf_top_k_per_retriever": 80,
    "k_final": 8,
}

def _node_id(node: BaseNode) -> Optional[str]:
    for attr in ("node_id", "id_", "doc_id", "ref_doc_id"):
        v = getattr(node, attr, None)
        if isinstance(v, str) and v:
            return v
    return None


def _dedupe(nodes: Sequence[NodeWithScore]) -> List[NodeWithScore]:
    seen = set()
    out: List[NodeWithScore] = []
    for r in nodes:
        nid = _node_id(r.node) or id(r.node)
        if nid in seen:
            continue
        seen.add(nid)
        out.append(r)
    return out


class TopKFinalWrapper(BaseRetriever):
    def __init__(self, base: BaseRetriever, k_final: int):
        super().__init__()
        self.base = base
        self.k_final = int(k_final)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        res = _dedupe(self.base.retrieve(query_bundle))
        return res[: self.k_final]


def build_retriever(
    strategy: str,
    vindex: VectorStoreIndex,
    *,
    similarity_top_k_dense: int,
    similarity_top_k_sparse: int,
    rrf_top_k_per_retriever: int,
    bm25_language: str = DEFAULT_LANG,
    k_final: int,
) -> BaseRetriever:
    bm25 = BM25Retriever.from_defaults(
        docstore=vindex.docstore,
        similarity_top_k=similarity_top_k_sparse,
        language=bm25_language,
    )

    dense = vindex.as_retriever(similarity_top_k=similarity_top_k_dense)

    if strategy == "sparse":
        return TopKFinalWrapper(bm25, k_final=k_final)

    if strategy == "dense":
        return TopKFinalWrapper(dense, k_final=k_final)

    if strategy == "hybrid_rrf":
        fused = QueryFusionRetriever(
            retrievers=[bm25, dense],
            num_queries=1,
            similarity_top_k=rrf_top_k_per_retriever,
            mode="reciprocal_rerank",
        )
        return TopKFinalWrapper(fused, k_final=k_final)

    raise ValueError("strategy must be one of: sparse, dense, hybrid_rrf")