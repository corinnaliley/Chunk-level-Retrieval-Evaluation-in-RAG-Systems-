"""
run_bench.py: Main orchestration script for the RAG benchmark.

Handles initialization, data loading, index creation/caching, evaluation loop,
and final result export to CSV.
"""
# -----------------------------------------------------------------------------
# Reproducibility / scientific reporting notes
# - This is the main experiment entrypoint. Any change here affects benchmark results.
# - For every reported run, record:
#     * EMBEDDING_MODEL_NAME
#     * DEFAULT_CHUNK_SIZE / DEFAULT_CHUNK_OVERLAP
#     * DEFAULT_RETRIEVER_PARAMS (see retrievers)
#     * STRATEGIES and K_EVAL
#     * llama_index version
#     * Git commit hash of this repository
# - Index caching (STORAGE_PATH / index_<hash>) depends on:
#     * embedding model
#     * chunk size / overlap
#   If these change, a new index_id is generated automatically.
#
# -----------------------------------------------------------------------------
import hashlib
from collections import defaultdict, Counter
import llama_index
from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage,
    VectorStoreIndex
)

import llama_index_init  # sets Settings.llm + Settings.embed_model
from src.retrievers import DEFAULT_RETRIEVER_PARAMS
from testset import TEST_QUERIES, QUERY_META, REFERENCE_ANSWERS
from benchmark import eval_strategy
from constants import DATA_DIR, RESULTS_PATH, EMBEDDING_MODEL_NAME, DEFAULT_CHUNK_SIZE, \
    DEFAULT_CHUNK_OVERLAP, STORAGE_PATH
from saia_reader import (
    load_documents_from_dir_via_saia,
    load_documents_from_json,
    save_documents_to_json,
)
import os
import csv
import json
from datetime import datetime
# === Define retrieval strategies to evaluate ===
STRATEGIES = ["sparse", "dense", "hybrid_rrf"]
K_EVAL = 8  # <-- overwrite k_eval (customize)

# 1) Helper Functions
from pathlib import Path

def _node_text(n) -> str:
    try:
        return n.node.get_content(metadata_mode=MetadataMode.NONE) or ""
    except Exception:
        # fallback
        return getattr(getattr(n, "node", None), "text", "") or ""



def _pick_doc_name_from_nodes(nodes, fallback: str = "unknown") -> str:
    """
    Nimmt den Dokumentnamen aus der Node-Metadatenquelle (tatsächliche Chunk-Basis).
    Fallback: fallback-String.
    """
    for n in nodes or []:
        md = getattr(n, "metadata", None) or {}
        src = (
            md.get("source")
            or md.get("file_name")
            or md.get("filename")
            or md.get("filepath")
        )
        if isinstance(src, str) and src.strip():
            return Path(src).name
    return fallback


def count_chunks(docs):
    parser = Settings.node_parser
    if parser is None:
        raise RuntimeError(
            "Settings.node_parser is None. "
            "Chunk counting requires an active NodeParser."
        )

    total = 0
    per_doc = []

    for d in docs:
        nodes = parser.get_nodes_from_documents([d])
        n = len(nodes)
        total += n

        doc_name = _pick_doc_name_from_nodes(nodes, fallback="unknown")
        per_doc.append((doc_name, n))

    return total, per_doc


def build_index_id():
    cfg = {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
    }
    s = json.dumps(cfg, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]

def _model_name(obj) -> str:
    """Best-effort readable model name for LLM/embeddings."""
    if obj is None:
        return ""
    for attr in ("model_name", "model", "name"):
        if hasattr(obj, attr):
            val = getattr(obj, attr)
            if isinstance(val, str):
                return val
    return obj.__class__.__name__

def _count_ext(docs):
    """Count PDFs vs TXTs from document metadata paths."""
    n_pdf = n_txt = 0
    for d in docs:
        md = d.metadata or {}
        src = (md.get("source")
               or md.get("file_name")
               or md.get("filename")
               or md.get("filepath")
               or "")
        ext = os.path.splitext(str(src))[-1].lower()
        if ext == ".pdf":
            n_pdf += 1
        elif ext == ".txt":
            n_txt += 1
    return n_pdf, n_txt

from benchmark import normalize
from llama_index.core.schema import MetadataMode


def validate_gold_coverage(
    docs,
    gold_reference: dict[str, str | list[str]],
) -> None:
    """
    Validates that every gold passage is fully contained in at least one chunk.

    This is a precondition check for the benchmark:
    if it fails, retrieval metrics would be invalid due to chunking artifacts.

    Raises:
        RuntimeError if any gold passage is not covered.
    """
    parser = Settings.node_parser
    if parser is None:
        raise RuntimeError(
            "Settings.node_parser is None. "
            "Gold coverage validation requires a NodeParser (e.g. SentenceSplitter)."
        )

    # 1) collect all chunk texts
    all_chunks: list[str] = []
    for d in docs:
        nodes = parser.get_nodes_from_documents([d])
        for n in nodes:
            txt = n.get_content(metadata_mode=MetadataMode.NONE)
            if txt:
                all_chunks.append(txt)

    # 2) validate gold coverage
    missing = []

    for query, gold in gold_reference.items():
        gold_texts = gold if isinstance(gold, list) else [gold]

        for g in gold_texts:
            g_norm = normalize(g)

            if not any(g_norm in normalize(chunk) for chunk in all_chunks):
                missing.append((query, g[:100]))

    # 3) hard fail if invalid
    if missing:
        print("\n[ERROR] Gold coverage check failed.")
        for q, g in missing:
            print(f"- Query: {q} | Gold snippet: {g}...")
        raise RuntimeError(
            "Chunking invalid: at least one gold passage is not fully contained "
            "in any chunk. Fix chunking or overlap before running the benchmark."
        )

    print("[OK] Gold coverage validation passed.")


# 2) Main Execution Logic

def main():
    llama_index_init.init()

    # --- Index ID / Storage ---
    index_id = build_index_id()
    storage_dir = STORAGE_PATH / f"index_{index_id}"

    # === Load documents (cache → SAIA Docling) ===
    docs = load_documents_from_json()

    if not docs:
        print("[INFO] Document cache not found. Processing via SAIA Docling...")
        docs = load_documents_from_dir_via_saia(DATA_DIR)

        if not docs:
            print("[ERROR] No documents loaded. Aborting.")
            return

        save_documents_to_json(docs)

    # === Debug sources ===
    sources = []
    for d in docs:
        src = (
                d.metadata.get("source")
                or d.metadata.get("file_name")
                or d.metadata.get("filename")
                or d.metadata.get("filepath")
        )
        if isinstance(src, str):
            sources.append(src)

    print(f"[DEBUG] Loaded {len(docs)} documents.")
    ext_counts = Counter(Path(s).suffix.lower() for s in sources)
    print("[DEBUG] Extensions in docs:", ext_counts)

    # Chunk-Counting
    total_chunks, per_doc = count_chunks(docs)

    print(f"[chunks] Total chunks: {total_chunks}")
    print(f"[chunks] Avg per document: {total_chunks / len(docs):.1f}")
    print("[chunks] Per document:")
    for name, n in per_doc:
        print(f"  - {name}: {n}")

    out = Path.cwd() / "chunks_debug.txt"
    print("DEBUG: schreibe nach", out)

    parser = Settings.node_parser

    with out.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(f"\n=== {d.metadata.get('source')} ===\n")
            nodes = parser.get_nodes_from_documents([d])
            for i, n in enumerate(nodes):
                f.write(f"\n--- Chunk {i} ---\n")
                f.write(n.get_content(metadata_mode=MetadataMode.NONE))
                f.write("\n")

    print("DEBUG: chunks_debug.txt geschrieben")

    # === Gold coverage validation (run once) ===
    validate_gold_coverage(
        docs=docs,
        gold_reference=REFERENCE_ANSWERS,
    )

    # === Load or build index ===
    print(f"[index] Using index dir: {storage_dir}")

    if storage_dir.exists():
        print("[index] Loading existing index")
        sc = StorageContext.from_defaults(persist_dir=str(storage_dir))
        vindex = load_index_from_storage(sc)
    else:
        print("[index] Building new index")
        storage_dir.mkdir(parents=True, exist_ok=True)
        vindex = VectorStoreIndex.from_documents(docs)
        vindex.storage_context.persist(persist_dir=str(storage_dir))

    # === Run evaluation ===
    results = []
    for strategy in STRATEGIES:
        res = eval_strategy(
            strategy,
            vindex,
            queries=TEST_QUERIES,
            gold_reference=REFERENCE_ANSWERS,
            k_eval=K_EVAL,
            query_meta=QUERY_META,
        )

        results.append(res)

    # === Print summary ===
    print("\n=== SUMMARY ===")
    for r in results:
        print(
            f"- {r['strategy']}: "
            f"hit={r['chunk_hit_rate']:.2f}, "
            f"mrr={r['avg_mrr']:.3f}, "
            f"chain_mrr={r.get('avg_chain_mrr', 0.0):.3f}, "
            f"ndcg={r['avg_ndcg']:.3f}, "
            f"clean={r['avg_latency_clean_ms']:.0f} ms, "
            f"e2e={r['avg_latency_end_to_end_ms']:.0f} ms"
        )

        per_type_hits = defaultdict(list)
        per_type_mrr = defaultdict(list)
        per_type_chain_mrr = defaultdict(list)
        per_type_ndcg = defaultdict(list)

        for d in r["details"]:
            qtype = d["query_type"]
            per_type_hits[qtype].append(d["hit_at_k"])
            per_type_mrr[qtype].append(d["mrr"])
            per_type_chain_mrr[qtype].append(d.get("chain_mrr", 0.0))
            per_type_ndcg[qtype].append(d["ndcg_at_k"])

        print(f"\n--- Breakdown for strategy {r['strategy']} ---")
        for qtype in per_type_hits:
            print(
                f"{qtype:15s}: "
                f"hit={sum(per_type_hits[qtype]) / len(per_type_hits[qtype]):.2f} | "
                f"mrr={sum(per_type_mrr[qtype]) / len(per_type_mrr[qtype]):.3f} | "
                f"chain_mrr={sum(per_type_chain_mrr[qtype]) / len(per_type_chain_mrr[qtype]):.3f} | "
                f"ndcg={sum(per_type_ndcg[qtype]) / len(per_type_ndcg[qtype]):.3f}"
            )

    # === Run metadata ===
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    embed_model_name = _model_name(Settings.embed_model)

    # === Export CSV ===
    out_dir = RESULTS_PATH
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"retrieval_results_{run_id}.csv"

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")

        writer.writerow([
            "dataset",
            "query_id",
            "query_type",
            "strategy",
            "k_eval",

            "hit_at_k",
            "mrr",
            "chain_mrr",
            "ndcg_at_k",
            "relevant_rank",
            "chain_rank",

            "latency_clean_ms",
            "latency_penalty_ms",
            "latency_end_to_end_ms",

            "index_id",
            "chunk_size",
            "chunk_overlap",
            "relevance_method",

            "similarity_top_k_dense",
            "similarity_top_k_sparse",
            "rrf_top_k_per_retriever",
            "mmr_alpha",
            "k_final",

            "embed_model",
            "llama_index_version",
        ])

        for res in results:
            strategy = res["strategy"]
            for d in res.get("details", []):
                writer.writerow([
                    "local_saia_docs",
                    d["query_id"],
                    d["query_type"],
                    strategy,
                    K_EVAL,

                    d["hit_at_k"],
                    d["mrr"],
                    d.get("chain_mrr", 0.0),
                    d["ndcg_at_k"],
                    d["relevant_rank"],
                    d.get("chain_rank", None),

                    d["latency_clean_ms"],
                    d["latency_penalty_ms"],
                    d["latency_end_to_end_ms"],

                    index_id,
                    DEFAULT_CHUNK_SIZE,
                    DEFAULT_CHUNK_OVERLAP,
                    "exact_normalized_substring",

                    DEFAULT_RETRIEVER_PARAMS["similarity_top_k_dense"],
                    DEFAULT_RETRIEVER_PARAMS["similarity_top_k_sparse"],
                    DEFAULT_RETRIEVER_PARAMS["rrf_top_k_per_retriever"],
                    DEFAULT_RETRIEVER_PARAMS["mmr_alpha"],
                    DEFAULT_RETRIEVER_PARAMS["k_final"],

                    embed_model_name,
                    llama_index.core.__version__,
                ])

    print(f"\n[INFO] Wrote detailed results to: {out_path}")


if __name__ == "__main__":
    main()