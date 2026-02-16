# Chunk-Level Retrieval Evaluation in RAG Systems

Dieses Repository enthält ein reproduzierbares Benchmark-Setup zur Evaluation von Retrieval-Strategien auf Chunk-Ebene für ein RAG-System (Retrieval-Augmented Generation).

Verglichen werden:

- Sparse Retrieval (BM25)
- Dense Retrieval (Embedding-basiert)
- Hybrid Retrieval (Reciprocal Rank Fusion)

Die Evaluation erfolgt auf Chunk-Level mit:
- Hit@k
- MRR
- Chain-MRR (für Multi-Hop)
- nDCG@k
- Latenzmessung (clean / penalty / end-to-end)

---

# Projektstruktur

src/
│
├── run_bench.py # Hauptskript (Orchestrierung)
├── benchmark.py # Evaluationsmetriken
├── retrievers.py # Retrieval-Strategien
├── saia_reader.py # SAIA/Docling → LlamaIndex Dokumente
├── loaders.py # Dokument-Konvertierung
├── llama_index_init.py # LLM + Embedding Initialisierung
├── cached_embedding.py # Embedding Cache
├── testset.py # Benchmark-Testset (Gold-Referenzen)
├── constants.py # Zentrale Konfiguration
└── vllm_embeddings.py # vLLM Embedding Wrapper

---
## Setup

### 1) Python
Empfohlen: Python 3.10+

### 2) Installation
```
pip install -r requirements.txt
```

### 3) .env Datei
Im Projektroot eine `.env` Datei anlegen:

```
LLM_API_KEY=your_api_key_here
SAIA_API_KEY=your_saia_key_here
```

Optional:
```
SAIA_BASE_URL=https://chat-ai.academiccloud.de/v1
```

---

## Benchmark ausführen

```
python src/run_bench.py
```

Outputs:
- Ergebnisse: `results/`
- Index-Cache: `data/storage/`
- Dokument-Cache: `data/processed/`

---

## Evaluationsdefinition

### Normalisierung
- Lowercasing
- Whitespace-Kompaktion
- Entfernen von Satzzeichen
- Beibehaltung deutscher Umlaute

### Hit@k
Alle Gold-Snippets müssen in mindestens einem der Top-k Chunks enthalten sein.

### MRR
Inverse Rank des ersten relevanten Chunks.

### Chain-MRR
Inverse Rank des Chunks, bei dem das letzte noch fehlende Gold-Snippet gefunden wird.

### nDCG@k
Boolean-Relevanzliste auf Chunk-Ebene.

---

## Nicht versionieren (.gitignore)

```
.env
data/storage/
data/processed/
results/
__pycache__/
*.pyc
chunks_debug.txt
```

---

## Hinweis zur PDF-Verarbeitung

Die PDF-Verarbeitung erfolgt über die SAIA Docling API.
Die Qualität der erzeugten Blockstruktur hängt von der Server-Version und Konfiguration ab.

Bei Änderung des Embedding-Modells oder der Chunking-Parameter wird automatisch eine neue Index-ID erzeugt.
