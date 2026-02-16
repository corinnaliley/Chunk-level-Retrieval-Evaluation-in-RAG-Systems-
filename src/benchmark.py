# -----------------------------------------------------------------------------
# benchmark.py — Retrieval evaluation metrics + strategy loop
#
# Scientific / reproducibility notes
# - This module defines the exact retrieval evaluation protocol:
#     * normalize() (lowercasing, whitespace collapse, german chars kept, punctuation removed)
#     * Hit@k definition: ALL gold snippets must appear in at least one retrieved chunk
#     * MRR: first relevant chunk rank (any gold snippet match)
#     * Chain-MRR: rank where ALL gold snippets are found (max of individual ranks)
#     * nDCG@k: boolean relevance list derived from gold substring matches
# - with_exponential_backoff() introduces variable wall-clock latency by design.
#   Latency metrics in results therefore include:
#     * latency_clean_ms (retrieve runtime excluding backoff sleep)
#     * latency_penalty_ms (sleep time added by backoff)
#     * latency_end_to_end_ms (sum)
# - Randomness:
#     * backoff jitter uses random.uniform(); for strict reproducibility, log the seed
#       or set it at process start. (This file itself does not set a seed.)
#
# -----------------------------------------------------------------------------
from __future__ import annotations
import math
import random
import re
import time
from typing import Dict, List, Tuple
import numpy as np
from llama_index.core import QueryBundle
from llama_index.core.schema import MetadataMode, NodeWithScore
from retrievers import build_retriever, DEFAULT_RETRIEVER_PARAMS


def normalize(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\wäöüß ]", "", text)
    return text.strip()


def split_gold(text: str) -> list[str]:
    parts: list[str] = []
    for line in (text or "").split("\n"):
        line = line.strip()
        if 2 <= len(line) <= 400:
            parts.append(line)
    return parts


def with_exponential_backoff(
    fn,
    *,
    max_retries: int = 10,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: float = 0.1,
):
    penalty_s = 0.0
    for attempt in range(max_retries):
        try:
            return fn(), penalty_s
        except Exception:
            if attempt >= max_retries - 1:
                raise
            delay = min(base_delay * (2**attempt), max_delay)
            delay *= 1 + random.uniform(-jitter, jitter)
            penalty_s += delay
            time.sleep(delay)
    return None


def _chunk_text(n: NodeWithScore) -> str:
    return n.node.get_content(metadata_mode=MetadataMode.NONE) or ""


def compute_chunk_hit_and_rels(nodes: list[NodeWithScore], gold_texts: list[str]) -> tuple[int, list[bool]]:
    if not gold_texts:
        return 0, [False for _ in nodes]

    gold_norm = [normalize(t) for t in gold_texts]
    rels: list[bool] = []

    found_per_gold = [False] * len(gold_norm)

    for n in nodes:
        chunk_norm = normalize(_chunk_text(n))
        hits = [g in chunk_norm for g in gold_norm]
        rels.append(any(hits))
        for i, h in enumerate(hits):
            found_per_gold[i] = found_per_gold[i] or h

    return int(all(found_per_gold)), rels


def compute_chain_mrr(nodes: list[NodeWithScore], gold_texts: list[str]) -> tuple[float, int | None, list[int | None]]:
    if not gold_texts:
        return 0.0, None, []

    gold_norm = [normalize(t) for t in gold_texts]
    gold_ranks: list[int | None] = [None] * len(gold_norm)

    for idx, n in enumerate(nodes, start=1):
        chunk_norm = normalize(_chunk_text(n))
        for i, g in enumerate(gold_norm):
            if gold_ranks[i] is None and g and (g in chunk_norm):
                gold_ranks[i] = idx

    if any(r is None for r in gold_ranks):
        return 0.0, None, gold_ranks

    chain_rank = max(r for r in gold_ranks if r is not None)
    return 1.0 / float(chain_rank), chain_rank, gold_ranks


def ndcg_at_k(rels: list[bool]) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(rels) if r)
    num_rel = sum(rels)
    if num_rel == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(num_rel))
    return dcg / idcg


def eval_strategy(
    strategy: str,
    vindex,
    queries: List[Tuple[str, str]],
    gold_reference: Dict[str, str | list[str]],
    k_eval: int,
    query_meta: Dict[str, Dict],
) -> Dict:
    retriever = build_retriever(strategy, vindex, **DEFAULT_RETRIEVER_PARAMS)

    details = []
    hit_flags: list[int] = []
    mrrs: list[float] = []
    chain_mrrs: list[float] = []
    ndcgs: list[float] = []
    lat_clean_ms: list[float] = []
    lat_penalty_ms: list[float] = []
    lat_e2e_ms: list[float] = []

    for qid, q in queries:
        meta = query_meta[qid]

        def _do_retrieve():
            return retriever.retrieve(QueryBundle(q))

        t0 = time.time()
        nodes, backoff_penalty_s = with_exponential_backoff(_do_retrieve)
        clean_s = time.time() - t0
        nodes = (nodes or [])[:k_eval]

        entry = gold_reference[qid]
        gold_texts: list[str] = []
        if isinstance(entry, list):
            for g in entry:
                gold_texts.extend(split_gold(g))
        else:
            gold_texts = split_gold(entry)

        hit_k, rels = compute_chunk_hit_and_rels(nodes, gold_texts)

        mrr = 0.0
        rel_rank = None
        for idx, r in enumerate(rels, start=1):
            if r:
                mrr = 1.0 / idx
                rel_rank = idx
                break

        chain_mrr, chain_rank, gold_ranks = compute_chain_mrr(nodes, gold_texts)
        ndcg = ndcg_at_k(rels)

        hit_flags.append(hit_k)
        mrrs.append(mrr)
        chain_mrrs.append(chain_mrr)
        ndcgs.append(ndcg)

        lat_clean_ms.append(clean_s * 1000.0)
        lat_penalty_ms.append(backoff_penalty_s * 1000.0)
        lat_e2e_ms.append((clean_s + backoff_penalty_s) * 1000.0)

        details.append(
            {
                "query_id": qid,
                "query": q,
                "query_type": meta["type"],
                "hit_at_k": int(hit_k),
                "mrr": float(mrr),
                "chain_mrr": float(chain_mrr),
                "ndcg_at_k": float(ndcg),
                "relevant_rank": rel_rank,
                "chain_rank": chain_rank,
                "gold_ranks": gold_ranks,
                "latency_clean_ms": int(clean_s * 1000.0),
                "latency_penalty_ms": int(backoff_penalty_s * 1000.0),
                "latency_end_to_end_ms": int((clean_s + backoff_penalty_s) * 1000.0),
            }
        )

    n = max(1, len(details))
    return {
        "strategy": strategy,
        "chunk_hit_rate": float(sum(hit_flags) / n),
        "avg_mrr": float(np.mean(mrrs)) if mrrs else 0.0,
        "avg_chain_mrr": float(np.mean(chain_mrrs)) if chain_mrrs else 0.0,
        "avg_ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "avg_latency_clean_ms": float(np.mean(lat_clean_ms)) if lat_clean_ms else 0.0,
        "avg_latency_penalty_ms": float(np.mean(lat_penalty_ms)) if lat_penalty_ms else 0.0,
        "avg_latency_end_to_end_ms": float(np.mean(lat_e2e_ms)) if lat_e2e_ms else 0.0,
        "details": details,
    }