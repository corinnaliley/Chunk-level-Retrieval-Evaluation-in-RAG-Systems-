"""
loaders.py — Unified document loading & conversion to normalized text blocks
---------------------------------------------------------------------------
Goals:
  - Provide a single entry-point to convert heterogeneous files (PDF/DOCX/HTML/TXT)
    into a normalized list of "Block" objects
  - Prefer hosted Docling (SAIA) if available (resource-friendly)
  - Fallback to local Docling if installed
  - Fallback for plain text/markdown
  - Offer JSONL caching helpers

IMPORTANT:
  - The exact SAIA Docling endpoint path/JSON schema may vary in your platform.
    Adapt `SAIA_DOCILING_ENDPOINT` and `parse_saia_docling_response(...)` to match.
  - Set your API key via argument or env var SAIA_API_KEY.

Typical use:
  from loaders import convert_with_docling, save_blocks_jsonl, load_blocks_jsonl

  blocks = convert_with_docling(
      path_or_url="data/raw/student_regulations.pdf",
      use_saia=True,
      saia_base_url="https://<your-saia-host>/api",
      saia_api_key=os.getenv("SAIA_API_KEY"),
  )
  save_blocks_jsonl(blocks, "data/processed/student_regulations.jsonl")
"""

# -----------------------------------------------------------------------------
# Reproducibility / scientific reporting notes
# - PDF conversion relies on an external SAIA/Docling service (network + server-side
#   versions). For comparable experiments, log:
#     * SAIA_BASE_URL, SAIA_DOCLING_ENDPOINT
#     * Docling/SAIA server version (if available)
#     * any server-side conversion parameters (OCR, layout, etc.)
# - The markdown-to-block parsing logic (_blocks_from_markdown) defines your
#   "document representation" and therefore affects retrieval. Changes here are
#   experimental changes and should be tracked (git commit + changelog).
#
# -----------------------------------------------------------------------------

from __future__ import annotations
import os
import json
import enum
import pathlib
import mimetypes
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Iterable
from constants import SAIA_BASE_URL, SAIA_API_KEY, SAIA_DOCLING_ENDPOINT

# Optional dependency: requests for SAIA HTTP calls
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # We'll fail gracefully if used without being installed.

# Optional dependency: local Docling
try:
    from docling.document_converter import DocumentConverter, ConvertSettings  # type: ignore
    _HAS_DOCLING = True
except Exception:  # pragma: no cover
    _HAS_DOCLING = False


# -----------------------------
# Normalized Block definitions
# -----------------------------

class BlockType(str, enum.Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    OTHER = "other"


@dataclass
class Block:
    doc_id: str                   # Stable document id (e.g., filename or URL)
    page: Optional[int]           # Page index (0-based) if known
    block_type: BlockType         # Semantic type of block
    section_path: Optional[str]   # e.g., "§3 > Abs. 2" or "H1: Prüfungsfristen > H2: Verlängerungen"
    text: str                     # Visible, human-readable text
    meta: Dict[str, Any]          # Extra metadata (coordinates, ids, confidence, etc.)


# -----------------------------
# Public API
# -----------------------------
def convert_with_docling(
    path_or_url: str,
    doc_id: Optional[str] = None,
) -> List[Block]:
    """
    Convert a document (local file path or URL) into a list of Block objects.

    Behavior:
      - For PDFs: use the SAIA Docling API (https://chat-ai.academiccloud.de/v1/documents/convert).
        If this fails, we raise a clear error and DO NOT fall back to plaintext.
      - For plain text / markdown files: use a simple plaintext conversion.
      - For all other file types: we try plaintext as a last resort.

    This function is intentionally strict for PDFs:
    they MUST go through Docling to avoid HTML/Login pages or binary garbage
    being indexed as text.
    """

    # Derive a stable document id if none was given
    if doc_id is None:
        doc_id = pathlib.Path(path_or_url).stem

    # Normalized lowercase file name for type decisions
    lower_name = path_or_url.lower()

    # --- 1. PDFs: always use SAIA Docling ---
    if lower_name.endswith(".pdf"):
        try:
            return _convert_via_saia(
                path_or_url,
                doc_id=doc_id,
                saia_api_key=SAIA_API_KEY,
                saia_base_url=SAIA_BASE_URL,
            )
        except Exception as e:
            # PDF must not silently fall back to plaintext
            raise RuntimeError(
                f"PDF conversion failed for '{path_or_url}': "
                f"{e}. PDF files require a working SAIA Docling endpoint."
            ) from e

    # --- 2. Text-like files: plaintext conversion ---
    try:
        return _convert_plaintext(path_or_url, doc_id)
    except Exception as e:
        raise RuntimeError(
            f"Plaintext conversion failed for '{path_or_url}': {e}"
        ) from e

def save_blocks_jsonl(blocks: Iterable[Block], out_path: str) -> None:
    """
    Save blocks to JSONL (one JSON object per line).
    """
    out_path = str(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for b in blocks:
            obj = asdict(b)
            # Ensure enum -> str
            obj["block_type"] = str(b.block_type.value)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_blocks_jsonl(in_path: str) -> List[Block]:
    """
    Load blocks from JSONL produced by save_blocks_jsonl.
    """
    items: List[Block] = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            items.append(
                Block(
                    doc_id=obj.get("doc_id", ""),
                    page=obj.get("page", None),
                    block_type=BlockType(obj.get("block_type", "other")),
                    section_path=obj.get("section_path"),
                    text=obj.get("text", ""),
                    meta=obj.get("meta", {}) or {},
                )
            )
    return items


# -----------------------------
# SAIA / Hosted Docling helpers
# -----------------------------

# Adjust this relative path to match your SAIA deployment.
# Example assumption: POST {saia_base_url}/docling/convert with either a file upload
# or a JSON body when passing URLs.

def _convert_via_saia(
    path_or_url: str,
    doc_id: str,
    saia_api_key: str,
    saia_base_url: str,
    timeout_s: int = 60, #change if the output says "[saia_reader] failed for"
) -> List[Block]:
    """
    Call SAIA Docling to convert a document to markdown.

    Endpoint (from SAIA documentation):
        POST {SAIA_BASE_URL}/documents/convert
      e.g.:
        https://chat-ai.academiccloud.de/v1/documents/convert

    The response is JSON like:
        {
          "response_type": "MARKDOWN",
          "filename": "example_document",
          "images": [...],
          "markdown": "# Your Markdown File"
        }
    """

    if requests is None:
        raise RuntimeError("requests is not installed")

    # Build full URL, e.g. "https://chat-ai.academiccloud.de/v1/documents/convert"
    url = f"{saia_base_url.rstrip('/')}/{SAIA_DOCLING_ENDPOINT.lstrip('/')}"
    headers = {
        "accept": "application/json",
    }
    if saia_api_key:
        headers["Authorization"] = f"Bearer {saia_api_key}"

    # SAIA Docling expects multipart/form-data with a "document" field
    if _looks_like_url(path_or_url):
        # If you really want to support remote URLs, you would need a different API shape.
        # The official doc example uses only file upload, so we keep it simple.
        raise RuntimeError("SAIA Docling URL-based conversion not supported in this helper")
    else:
        with open(path_or_url, "rb") as fh:
            files = {
                "document": (
                    os.path.basename(path_or_url),
                    fh,
                    _guess_mime(path_or_url),
                )
            }
            resp = requests.post(
                url,
                headers=headers,
                files=files,
                timeout=timeout_s,
            )

    if resp.status_code != 200:
        raise RuntimeError(
            f"SAIA Docling HTTP {resp.status_code}: {resp.text[:300]}"
        )

    # Try to parse JSON as described in the documentation
    try:
        payload = resp.json()
    except Exception as e:
        # If we cannot parse JSON, print a helpful debug snippet
        snippet = resp.text[:300]
        raise RuntimeError(
            f"SAIA Docling did not return valid JSON. Error: {e}; body snippet: {snippet}"
        )

    # Safety: check for HTML/login pages (should not happen with correct endpoint)
    text_snippet = (resp.text or "")[:200]
    if "<!DOCTYPE html>" in text_snippet or "<html" in text_snippet.lower():
        raise RuntimeError(
            "SAIA Docling returned HTML (likely a login page), not JSON."
        )

    # Extract markdown field
    md = payload.get("markdown", "")
    if not isinstance(md, str) or not md.strip():
        raise RuntimeError(
            f"SAIA Docling JSON has no non-empty 'markdown' field. Payload keys: {list(payload.keys())}"
        )

    # Convert markdown string into Blocks
    return _blocks_from_markdown(md, doc_id=doc_id)



def parse_saia_docling_response(payload: Dict[str, Any], doc_id: str) -> List[Block]:
    """
    SAIA /documents/convert liefert u. a.:
    {
      "response_type": "MARKDOWN" | "HTML" | "JSON" | "TOKENS",
      "filename": "...",
      "images": [...],
      "markdown": "...",   # falls response_type=markdown
      "html": "...",       # falls response_type=html
      "json": {...}        # falls response_type=json
    }
    """
    blocks: List[Block] = []

    md = payload.get("markdown")
    if isinstance(md, str) and md.strip():
        return _blocks_from_markdown(md, doc_id)

    html = payload.get("html")
    if isinstance(html, str) and html.strip():
        import re as _re
        text = _re.sub(r"<[^>]+>", "", html)
        return _paragraphize_plaintext(text, doc_id)

    js = payload.get("json")
    if isinstance(js, dict):
        # sehr einfacher Fallback: alle String-Felder konkatenieren
        parts = [v for v in js.values() if isinstance(v, str) and v.strip()]
        if parts:
            return _paragraphize_plaintext("\n\n".join(parts), doc_id)

    # letzter Fallback: alle stringigen Top-Level-Felder zusammenziehen
    parts = [v for v in payload.values() if isinstance(v, str) and v.strip()]
    if parts:
        return _paragraphize_plaintext("\n\n".join(parts), doc_id)
    return blocks


# -----------------------------
# Plain text / markdown fallback
# -----------------------------

def _convert_plaintext(path_or_url: str, doc_id: str) -> List[Block]:
    """
    Fallback converter for plain text-like files (e.g., .txt, .md).

    This function is only meant for files that are actually stored as text on disk.
    It should NOT be used for binary formats like PDF – those must be handled
    via Docling (remote SAIA or local) to avoid garbage output.
    """
    # URLs are not supported in plaintext mode – they must be downloaded or
    # processed via Docling before calling this function.
    if _looks_like_url(path_or_url):
        raise RuntimeError(
            "Plaintext fallback does not support URLs. "
            "Download the file or use Docling/SAIA for remote content."
        )

    path = pathlib.Path(path_or_url)
    if not path.exists():
        raise FileNotFoundError(path)

    # Guess the MIME type of the file (e.g., text/plain, text/markdown, application/pdf, ...).
    mt = _guess_mime(str(path))

    # PDFs are binary and require a proper PDF parser like Docling.
    # Using read_text() on them would produce broken or meaningless content.
    if mt == "application/pdf":
        raise RuntimeError(
            f"Cannot process PDF via plaintext fallback: {path}. "
            "PDFs must be converted via SAIA Docling or local Docling."
        )

    # Optionally, restrict this fallback to known text formats only.
    # You can relax this if you want to be more permissive.
    if mt not in ("text/plain", "text/markdown"):
        raise RuntimeError(
            f"Unsupported MIME type for plaintext fallback: {mt}. "
            "This converter is only intended for real text files like .txt or .md."
        )

    # Read the file as UTF-8 text. `errors='ignore'` prevents crashes on minor
    # encoding issues and simply drops problematic bytes.
    text = path.read_text(encoding="utf-8", errors="ignore")

    # Split the text into paragraphs and wrap them into Block objects.
    # This keeps the representation consistent with Docling-based converters.
    blocks = _paragraphize_plaintext(text, doc_id)
    return blocks



def _paragraphize_plaintext(text: str, doc_id: str) -> List[Block]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    blocks: List[Block] = []
    for i, para in enumerate(paragraphs):
        blocks.append(Block(
            doc_id=doc_id,
            page=None,
            block_type=BlockType.PARAGRAPH,
            section_path=None,
            text=para,
            meta={"para_index": i},
        ))
    return blocks


# -----------------------------
# Utilities
# -----------------------------

def _derive_doc_id(path_or_url: str) -> str:
    if _looks_like_url(path_or_url):
        return path_or_url
    return pathlib.Path(path_or_url).name

def _looks_like_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")

def _looks_like_file(s: str) -> bool:
    try:
        return pathlib.Path(s).exists()
    except Exception:
        return False

def _guess_mime(path: str) -> str:
    mt, _ = mimetypes.guess_type(path)
    return mt or "application/octet-stream"

def _blocks_from_markdown(md: str, doc_id: str) -> List[Block]:
    """
    Wandelt SAIA-Markdown in strukturierte Blocks um:
      - '#', '##', ... => HEADING (section_path wird aufgebaut)
      - '- ' oder '* ' => LIST
      - sonst: PARAGRAPH (Absätze)
    """
    lines = md.splitlines()
    blocks: List[Block] = []
    section_stack: List[str] = []
    para_buf: List[str] = []

    def flush_para():
        if para_buf:
            text = " ".join(s.strip() for s in para_buf).strip()
            if text:
                blocks.append(Block(
                    doc_id=doc_id, page=None, block_type=BlockType.PARAGRAPH,
                    section_path=" > ".join(section_stack) if section_stack else None,
                    text=text, meta={}
                ))
            para_buf.clear()

    import re as _re
    for line in lines:
        if not line.strip():
            flush_para(); continue
        m = _re.match(r"^(#{1,6})\s+(.*)$", line.strip())
        if m:
            flush_para()
            level = len(m.group(1))
            title = m.group(2).strip()
            section_stack = section_stack[:level-1] + [title]
            blocks.append(Block(
                doc_id=doc_id, page=None, block_type=BlockType.HEADING,
                section_path=" > ".join(section_stack), text=title, meta={"level": level}
            ))
            continue
        if line.lstrip().startswith(("-", "*")):
            flush_para()
            item = line.lstrip("-* ").strip()
            blocks.append(Block(
                doc_id=doc_id, page=None, block_type=BlockType.LIST,
                section_path=" > ".join(section_stack) if section_stack else None,
                text=item, meta={}
            ))
            continue
        para_buf.append(line)
    flush_para()
    return blocks