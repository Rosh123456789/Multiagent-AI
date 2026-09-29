from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Suppress the cosmetic posthog telemetry errors from chromadb 0.5.x.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

# ---------------------------------------------------------------------------
# ChromaDB integration (optional — graceful fallback when unavailable)
# ---------------------------------------------------------------------------
try:
    import chromadb  # type: ignore[import-not-found]
    from chromadb.utils import embedding_functions  # type: ignore[import-not-found]

    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False
    logger.info("chromadb not installed — KnowledgeBase will use keyword fallback.")

_COLLECTION_NAME = "deepshield_runbooks"

# Ollama model used for embeddings.  Overridable via env var.
_EMBED_MODEL = os.getenv("DEEPSHIELD_EMBED_MODEL", "nomic-embed-text")
_OLLAMA_BASE_URL = os.getenv("DEEPSHIELD_OLLAMA_URL", "http://localhost:11434")

# ---------------------------------------------------------------------------
# Keyword scoring weights used by the fallback engine.
# Each tuple: (keyword_list, weight)
# ---------------------------------------------------------------------------
_KEYWORD_WEIGHTS: List[Tuple[List[str], float]] = [
    (["mysql", "database", "db connection", "cannot connect", "database_connection"], 3.0),
    (["crashloop", "crash loop", "restart loop", "oomkilled", "restarting"], 2.5),
    (["high cpu", "cpu usage", "cpu spike", "throttl", "high_cpu"], 2.5),
    (["high memory", "memory leak", "oom", "heap", "high_memory"], 2.5),
    (["service down", "health check", "unhealthy", "not responding", "service_down"], 2.0),
]

# Severity hints associated with each runbook filename.
_FILE_SEVERITY: Dict[str, str] = {
    "database_connection.md": "critical",
    "crashloop.md": "high",
    "high_cpu.md": "high",
    "high_memory.md": "high",
    "service_down.md": "high",
}


# ---------------------------------------------------------------------------
# Tiny TF-IDF-style embedding for zero-dependency fallback
# ---------------------------------------------------------------------------

class _TinyEmbeddingFunction:
    """
    Deterministic bag-of-words embedding used when Ollama is not reachable.
    Produces sparse float vectors of fixed length (1024).
    NOT semantically meaningful — exists only so ChromaDB can initialise
    and the keyword fallback handles the actual retrieval.
    """

    _DIM = 1024

    def __call__(self, input: List[str]) -> List[List[float]]:  # noqa: A002
        return [self._embed(text) for text in input]

    def _embed(self, text: str) -> List[float]:
        vec = [0.0] * self._DIM
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        for token in tokens:
            idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % self._DIM
            vec[idx] += 1.0
        # L2 normalise.
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_runbook(file_path: Path) -> Dict[str, Any]:
    """Parse a markdown runbook into structured fields."""
    text = file_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    title = file_path.stem.replace("_", " ").title()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            break

    # First non-empty, non-heading paragraph → summary.
    summary_lines: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            summary_lines.append(stripped)
            if len(summary_lines) >= 2:
                break
    summary = " ".join(summary_lines) if summary_lines else "Operational guidance available."

    # Separate causes vs recommended actions by section heading.
    # Handles both markdown headings (## Causes) and plain-text labels (Possible causes:).
    causes: List[str] = []
    recommended: List[str] = []
    all_bullets: List[str] = []
    current_section = "other"

    _SECTION_RE = re.compile(r"^#{1,3}\s+|^[A-Z][^:]{2,40}:\s*$", re.IGNORECASE)

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        # Detect section label — either a markdown heading or a plain "Label:" line.
        if _SECTION_RE.match(stripped):
            if "cause" in lower or "potential" in lower or "possible" in lower:
                current_section = "causes"
            elif "recommend" in lower or "action" in lower or "step" in lower:
                current_section = "recommended"
            else:
                current_section = "other"
            continue
        if re.match(r"^\s*[-•*]\s+", line):
            bullet = re.sub(r"^\s*[-•*]\s+", "", line).strip()
            all_bullets.append(bullet)
            if current_section == "causes":
                causes.append(bullet)
            elif current_section == "recommended":
                recommended.append(bullet)

    # Fall back to all bullets if section parsing found nothing.
    if not recommended:
        recommended = all_bullets

    severity = _FILE_SEVERITY.get(file_path.name, "medium")

    return {
        "title": title,
        "summary": summary,
        "causes": causes,
        "recommended_actions": recommended,
        "severity_hint": severity,
        "filename": file_path.name,
        "full_text": text,
    }


def _doc_id(file_path: Path) -> str:
    """Stable document ID based on the file path."""
    return hashlib.sha1(str(file_path).encode()).hexdigest()[:16]


def _build_embedding_function() -> Any:
    """
    Return the best available embedding function for ChromaDB.

    Priority:
    1. OllamaEmbeddingFunction (uses the already-installed Ollama service —
       no extra downloads needed, works offline after Ollama is pulled once).
    2. _TinyEmbeddingFunction (deterministic BoW — last resort so ChromaDB
       can always initialise; real retrieval falls back to keyword search).
    """
    if not _CHROMA_AVAILABLE:
        return None

    try:
        ef = embedding_functions.OllamaEmbeddingFunction(
            url=f"{_OLLAMA_BASE_URL}/api/embeddings",
            model_name=_EMBED_MODEL,
        )
        # Probe with a tiny string to confirm Ollama is reachable.
        ef(["test"])
        logger.info("ChromaDB: using Ollama embedding function (model=%s).", _EMBED_MODEL)
        return ef
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Ollama embedding function unavailable (%s) — using tiny BoW fallback.", exc
        )

    return _TinyEmbeddingFunction()


# ---------------------------------------------------------------------------
# KnowledgeBase
# ---------------------------------------------------------------------------

class KnowledgeBase:
    """
    Loads operational runbooks from a markdown directory and exposes them via:

    1. **ChromaDB vector search** (primary): embeds runbook text at startup and
       retrieves the closest match by cosine similarity at query time.
       Uses ``OllamaEmbeddingFunction`` (no internet download required) when
       Ollama is running; falls back to a deterministic BoW embedding otherwise.

    2. **Keyword scoring fallback** (secondary): when ChromaDB is unavailable
       or returns no results above the similarity threshold.

    Usage
    -----
    ::

        kb = KnowledgeBase("knowledge")
        runbook = kb.get_runbook("Payment API cannot connect to MySQL")
    """

    def __init__(
        self,
        knowledge_dir: str | Path,
        *,
        chroma_persist_dir: Optional[str] = None,
        similarity_threshold: float = 0.30,
    ) -> None:
        self.knowledge_dir = Path(knowledge_dir)
        self.similarity_threshold = similarity_threshold
        self._runbooks: Dict[str, Dict[str, Any]] = {}   # filename → parsed runbook
        self._chroma_client: Any = None
        self._collection: Any = None

        self._load_runbooks()
        if _CHROMA_AVAILABLE:
            self._init_chroma(chroma_persist_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, n_results: int = 3) -> Dict[str, Any]:
        """
        Return matching runbook filenames ranked by relevance.

        Tries ChromaDB first, falls back to keyword scoring.
        """
        if not self._runbooks:
            return {"query": query, "matches": [], "source": "none"}

        if self._collection is not None:
            results = self._search_chroma(query, n_results)
            if results["matches"]:
                return results

        return self._search_keywords(query, n_results)

    def get_runbook(self, query: str) -> Dict[str, Any]:
        """
        Retrieve the best-matching runbook for a natural-language query.

        Returns a rich dict with title, summary, causes, recommended_actions,
        severity_hint, filename, retrieval_source, and score.
        """
        result = self.search(query, n_results=1)
        matches = result.get("matches", [])

        if not matches:
            return {
                "title": "No matching runbook",
                "summary": "No known runbook found for this issue.",
                "causes": [],
                "recommended_actions": [
                    "Inspect container logs manually.",
                    "Check service health endpoints.",
                    "Escalate to on-call engineer.",
                ],
                "severity_hint": "unknown",
                "filename": None,
                "retrieval_source": "none",
                "score": 0.0,
            }

        best = matches[0]
        filename = best["filename"]
        runbook = self._runbooks.get(filename, {})

        return {
            **runbook,
            "retrieval_source": result.get("source", "keyword"),
            "score": best.get("score", 1.0),
        }

    def list_runbooks(self) -> List[str]:
        """Return all loaded runbook filenames."""
        return sorted(self._runbooks.keys())

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _load_runbooks(self) -> None:
        if not self.knowledge_dir.exists():
            logger.warning("Knowledge directory '%s' does not exist.", self.knowledge_dir)
            return

        for file_path in sorted(self.knowledge_dir.glob("*.md")):
            try:
                self._runbooks[file_path.name] = _parse_runbook(file_path)
                logger.debug("Loaded runbook: %s", file_path.name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to parse runbook '%s': %s", file_path.name, exc)

        logger.info("KnowledgeBase: loaded %d runbook(s).", len(self._runbooks))

    def _init_chroma(self, persist_dir: Optional[str]) -> None:
        """Initialise the ChromaDB client and upsert all runbooks."""
        try:
            ef = _build_embedding_function()

            chroma_settings = chromadb.Settings(anonymized_telemetry=False)

            if persist_dir:
                self._chroma_client = chromadb.PersistentClient(
                    path=persist_dir, settings=chroma_settings
                )
            else:
                self._chroma_client = chromadb.EphemeralClient(settings=chroma_settings)

            collection_kwargs: Dict[str, Any] = {
                "name": _COLLECTION_NAME,
                "metadata": {"hnsw:space": "cosine"},
            }
            if ef is not None:
                collection_kwargs["embedding_function"] = ef

            self._collection = self._chroma_client.get_or_create_collection(
                **collection_kwargs
            )
            self._upsert_runbooks()
            logger.info(
                "ChromaDB collection '%s' ready with %d document(s).",
                _COLLECTION_NAME,
                self._collection.count(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ChromaDB initialisation failed (%s) — keyword search only.", exc
            )
            self._chroma_client = None
            self._collection = None

    def _upsert_runbooks(self) -> None:
        """Insert or update all runbook documents in the ChromaDB collection."""
        if not self._collection or not self._runbooks:
            return

        ids, documents, metadatas = [], [], []
        for filename, runbook in self._runbooks.items():
            file_path = self.knowledge_dir / filename
            ids.append(_doc_id(file_path))
            documents.append(runbook["full_text"])
            metadatas.append(
                {
                    "filename": filename,
                    "title": runbook["title"],
                    "severity_hint": runbook["severity_hint"],
                }
            )

        self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    # ------------------------------------------------------------------
    # ChromaDB search
    # ------------------------------------------------------------------

    def _search_chroma(self, query: str, n_results: int) -> Dict[str, Any]:
        try:
            count = self._collection.count()
            if count == 0:
                return {"query": query, "matches": [], "source": "chromadb_empty"}

            raw = self._collection.query(
                query_texts=[query],
                n_results=min(n_results, count),
                include=["metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("ChromaDB query failed (%s) — falling back to keywords.", exc)
            return {"query": query, "matches": [], "source": "chroma_error"}

        matches: List[Dict[str, Any]] = []
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        for meta, distance in zip(metadatas, distances):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite.
            # Convert to a 0–1 similarity score.
            similarity = max(0.0, 1.0 - (distance / 2.0))
            if similarity >= self.similarity_threshold:
                matches.append(
                    {
                        "filename": meta.get("filename"),
                        "title": meta.get("title"),
                        "severity_hint": meta.get("severity_hint"),
                        "score": round(similarity, 4),
                    }
                )

        logger.debug("ChromaDB search: query=%r  matches=%d", query, len(matches))
        return {"query": query, "matches": matches, "source": "chromadb"}

    # ------------------------------------------------------------------
    # Keyword fallback
    # ------------------------------------------------------------------

    def _search_keywords(self, query: str, n_results: int) -> Dict[str, Any]:
        query_lower = query.lower()
        scored: List[Tuple[float, str]] = []

        for filename, runbook in self._runbooks.items():
            score = self._keyword_score(query_lower, filename, runbook["full_text"].lower())
            if score > 0:
                scored.append((score, filename))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:n_results]

        matches = [
            {
                "filename": filename,
                "title": self._runbooks[filename]["title"],
                "severity_hint": self._runbooks[filename]["severity_hint"],
                "score": round(score / 10.0, 4),  # Normalise roughly to 0–1.
            }
            for score, filename in top
        ]

        logger.debug("Keyword search: query=%r  matches=%d", query, len(matches))
        return {"query": query, "matches": matches, "source": "keyword"}

    @staticmethod
    def _keyword_score(query_lower: str, filename: str, content_lower: str) -> float:
        """Return a weighted relevance score for a single runbook."""
        score = 0.0
        # Filename stem match is a strong signal.
        stem = filename.replace(".md", "").replace("_", " ")
        if stem in query_lower:
            score += 5.0

        for keywords, weight in _KEYWORD_WEIGHTS:
            query_hits = sum(1 for kw in keywords if kw in query_lower)
            content_hits = sum(1 for kw in keywords if kw in content_lower)
            if query_hits > 0 and content_hits > 0:
                score += weight * query_hits

        return score
