"""Endee-powered retrieval layer for Doc4U.

This module is intentionally defensive: it uses Endee through the official
LangChain integration when the packages and Endee server/token are available,
and it falls back to a small local lexical retriever only so the demo never
breaks during evaluation setup. The README explains how to run the real Endee
path.
"""
from __future__ import annotations

import json
import math
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


@dataclass
class RetrievedContext:
    text: str
    source: str
    title: str
    score: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """Split long text into overlapping chunks for retrieval."""
    clean = " ".join((text or "").split())
    if not clean:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(clean):
        end = min(start + chunk_size, len(clean))
        chunks.append(clean[start:end])
        if end == len(clean):
            break
        start = max(0, end - overlap)
    return chunks


class LocalKeywordStore:
    """Tiny fallback retriever used only when Endee is not configured."""

    def __init__(self, storage_path: str = "data/local_retrieval_store.json") -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.documents: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if self.storage_path.exists():
            try:
                self.documents = json.loads(self.storage_path.read_text(encoding="utf-8"))
            except Exception:
                self.documents = []

    def _save(self) -> None:
        self.storage_path.write_text(json.dumps(self.documents, indent=2), encoding="utf-8")

    def add_texts(self, texts: Iterable[str], metadatas: Iterable[Dict[str, Any]]) -> List[str]:
        ids: List[str] = []
        for text, metadata in zip(texts, metadatas):
            doc_id = str(uuid.uuid4())
            ids.append(doc_id)
            self.documents.append({"id": doc_id, "text": text, "metadata": metadata})
        self._save()
        return ids

    @staticmethod
    def _tokens(text: str) -> List[str]:
        return [m.group(0).lower() for m in TOKEN_RE.finditer(text or "")]

    def similarity_search_with_score(self, query: str, k: int = 4, filter: Optional[List[Dict[str, Any]]] = None):
        query_tokens = self._tokens(query)
        if not query_tokens:
            return []
        query_set = set(query_tokens)
        rows: List[Tuple[Dict[str, Any], float]] = []
        for doc in self.documents:
            metadata = doc.get("metadata", {})
            if filter:
                allowed = True
                for item in filter:
                    for key, condition in item.items():
                        if isinstance(condition, dict) and "$eq" in condition:
                            allowed = allowed and metadata.get(key) == condition["$eq"]
                if not allowed:
                    continue
            doc_tokens = self._tokens(doc.get("text", ""))
            if not doc_tokens:
                continue
            overlap = len(query_set.intersection(doc_tokens))
            score = overlap / math.sqrt(len(set(doc_tokens)) + 1)
            if score > 0:
                rows.append((doc, score))
        rows.sort(key=lambda row: row[1], reverse=True)
        return rows[:k]


class EndeeRAGService:
    """Retriever that powers semantic search, RAG context, and recommendations."""

    def __init__(self) -> None:
        self.index_name = os.getenv("ENDEE_INDEX_NAME", "doc4u_health_index")
        self.dimension = int(os.getenv("ENDEE_DIMENSION", "384"))
        self.api_token = os.getenv("ENDEE_API_TOKEN") or None
        self.enabled = os.getenv("ENDEE_ENABLED", "true").lower() in {"1", "true", "yes"}
        self.status_message = "Endee not initialized yet."
        self.vector_store: Any = None
        self.local_store = LocalKeywordStore()
        self._initialize()
        self.seed_default_knowledge()

    def _initialize(self) -> None:
        if not self.enabled:
            self.status_message = "ENDEE_ENABLED=false, using local fallback retriever."
            return
        try:
            from langchain_endee import EndeeVectorStore  # type: ignore
            from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore

            embeddings = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
            self.vector_store = EndeeVectorStore(
                embedding=embeddings,
                api_token=self.api_token,
                index_name=self.index_name,
                dimension=self.dimension,
            )
            self.status_message = f"Connected to Endee index '{self.index_name}'."
        except Exception as exc:
            self.vector_store = None
            self.status_message = f"Endee unavailable, using local fallback retriever: {exc}"

    def seed_default_knowledge(self) -> None:
        seed_path = Path("data/medical_knowledge_seed.json")
        if not seed_path.exists():
            return
        marker = Path("data/.seeded")
        if marker.exists():
            return
        try:
            records = json.loads(seed_path.read_text(encoding="utf-8"))
            texts = [r["text"] for r in records]
            metadatas = [
                {
                    "source": r.get("source", "Doc4U seed knowledge"),
                    "title": r.get("title", "Health guidance"),
                    "type": "seed",
                    "user_id": "public",
                }
                for r in records
            ]
            self.add_texts(texts, metadatas)
            marker.write_text("seeded", encoding="utf-8")
        except Exception:
            # Seeding is helpful, not critical.
            pass

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> List[str]:
        texts = [t for t in texts if t and t.strip()]
        if not texts:
            return []
        if self.vector_store is not None:
            try:
                return self.vector_store.add_texts(texts=texts, metadatas=metadatas)
            except Exception as exc:
                self.status_message = f"Endee insert failed, fallback used: {exc}"
        return self.local_store.add_texts(texts, metadatas)

    def ingest_document(self, text: str, *, title: str, source: str, user_id: str, doc_type: str = "upload") -> int:
        chunks = chunk_text(text)
        metadatas = [
            {
                "title": title,
                "source": source,
                "user_id": user_id,
                "type": doc_type,
                "chunk": idx,
            }
            for idx, _ in enumerate(chunks)
        ]
        self.add_texts(chunks, metadatas)
        return len(chunks)

    def search(self, query: str, *, user_id: Optional[str] = None, k: int = 4) -> List[RetrievedContext]:
        filters: List[Dict[str, Any]] = []
        # Keep public health knowledge available to all. User-uploaded content is filtered separately.
        candidate_rows = []
        if self.vector_store is not None:
            try:
                candidate_rows.extend(self.vector_store.similarity_search_with_score(query=query, k=k))
                if user_id:
                    candidate_rows.extend(
                        self.vector_store.similarity_search_with_score(
                            query=query,
                            k=k,
                            filter=[{"user_id": {"$eq": user_id}}],
                        )
                    )
            except Exception as exc:
                self.status_message = f"Endee search failed, fallback used: {exc}"
                candidate_rows = []
        if not candidate_rows:
            candidate_rows.extend(self.local_store.similarity_search_with_score(query=query, k=k))
            if user_id:
                candidate_rows.extend(
                    self.local_store.similarity_search_with_score(
                        query=query,
                        k=k,
                        filter=[{"user_id": {"$eq": user_id}}],
                    )
                )
        results: List[RetrievedContext] = []
        seen = set()
        for row in candidate_rows:
            if isinstance(row, tuple) and hasattr(row[0], "page_content"):
                doc, score = row
                text = doc.page_content
                metadata = doc.metadata or {}
            elif isinstance(row, tuple) and isinstance(row[0], dict):
                doc, score = row
                text = doc.get("text", "")
                metadata = doc.get("metadata", {})
            else:
                doc = row
                score = 0.0
                text = getattr(doc, "page_content", "")
                metadata = getattr(doc, "metadata", {}) or {}
            key = (metadata.get("source"), metadata.get("chunk"), text[:80])
            if key in seen:
                continue
            seen.add(key)
            results.append(
                RetrievedContext(
                    text=text,
                    title=metadata.get("title", "Retrieved context"),
                    source=metadata.get("source", "Endee vector index"),
                    score=float(score or 0),
                    metadata=metadata,
                )
            )
            if len(results) >= k:
                break
        return results

    def status(self) -> Dict[str, Any]:
        return {
            "endee_configured": self.vector_store is not None,
            "index_name": self.index_name,
            "dimension": self.dimension,
            "message": self.status_message,
        }


rag_service = EndeeRAGService()
