"""
rag_engine.py - RAG Vector Database Engine (FAISS)

This module provides the foundation for the RAG (Retrieval-Augmented Generation)
pipeline integrated into the Property Valuation Agentic Advisor. It enables
semantic search over real estate documents (market trends, property laws,
investment guides) to ground AI-generated advice in factual, retrieved context.

Architecture:
    1. Document Loading  - Ingest PDFs, DOCX, and TXT files from knowledge_base/
    2. Text Chunking     - Split documents into overlapping chunks for indexing
    3. Embedding         - Encode chunks as dense vectors (Sentence-Transformers)
    4. FAISS Indexing    - Store and retrieve vectors via similarity search
    5. RAG Chain         - Combine retrieved context with LLM for grounded answers

Usage:
    engine = RAGEngine(docs_dir="data/knowledge_base/")
    engine.build_index()
    answer = engine.query("What are the current trends in Bangalore property market?")
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------
KNOWLEDGE_BASE_DIR = Path("data/knowledge_base")
INDEX_FILE = Path("models/knowledge_index.joblib")
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80
TOP_K_RESULTS = 3


class RAGEngine:
    """
    Lightweight retrieval engine over the local knowledge base.

    Attributes:
        docs_dir (Path): Directory containing source documents.
        index_file (Path): Path where the TF-IDF index is persisted.

    Note:
        This implementation intentionally avoids LangChain/LLMs. It provides
        fast, deterministic retrieval (TF-IDF + cosine similarity) to ground
        the advisory layer in the project knowledge base.
    """

    def __init__(
        self,
        docs_dir: str | Path = KNOWLEDGE_BASE_DIR,
        index_file: str | Path = INDEX_FILE,
    ) -> None:
        self.docs_dir = Path(docs_dir)
        self.index_file = Path(index_file)
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self._chunks: list[dict[str, str]] = []

    def _index_exists(self) -> bool:
        return self.index_file.exists()

    def _load_documents(self):
        if not self.docs_dir.exists():
            logging.warning(f"Knowledge base directory not found: {self.docs_dir}")
            return []

        logging.info(f"Loading documents from {self.docs_dir}...")
        documents: list[tuple[str, str]] = []
        for path in sorted(self.docs_dir.glob("**/*.txt")):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception as exc:
                logging.warning("Failed reading %s: %s", path, exc)
                continue
            documents.append((str(path), text))
        return documents

    def _chunk_text(self, text: str) -> list[str]:
        normalized = re.sub(r"\r\n?", "\n", text).strip()
        if not normalized:
            return []

        parts = [p.strip() for p in normalized.split("\n\n") if p.strip()]
        chunks: list[str] = []
        for part in parts:
            if len(part) <= CHUNK_SIZE:
                chunks.append(part)
                continue

            start = 0
            while start < len(part):
                end = min(len(part), start + CHUNK_SIZE)
                chunk = part[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                if end >= len(part):
                    break
                start = max(0, end - CHUNK_OVERLAP)

        return chunks

    def build_index(self, force: bool = False) -> bool:
        """Load documents, chunk them, and build a persisted TF-IDF index."""
        if not force and self._load_index():
            logging.info("Reusing persisted knowledge index.")
            return True

        documents = self._load_documents()
        
        if not documents:
            logging.warning("No documents found to build the knowledge base.")
            return False

        chunks: list[dict[str, str]] = []
        for source, text in documents:
            for chunk in self._chunk_text(text):
                chunks.append({"source": source, "text": chunk})

        if not chunks:
            logging.warning("No chunks produced from knowledge base.")
            return False

        texts = [c["text"] for c in chunks]
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_features=50_000,
        )
        matrix = vectorizer.fit_transform(texts)

        payload = {
            "vectorizer": vectorizer,
            "matrix": matrix,
            "chunks": chunks,
        }
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, self.index_file)

        self._vectorizer = vectorizer
        self._matrix = matrix
        self._chunks = chunks
        logging.info("Knowledge index saved to %s (%d chunks).", self.index_file, len(chunks))
        return True

    def query(self, question: str, top_k: int = TOP_K_RESULTS) -> str:
        """
        Retrieve relevant document chunks and return raw context.

        Args:
            question: Natural language question about real estate.
            top_k: Number of top relevant chunks to retrieve.

        Returns:
            A newline-joined string of the top retrieved document chunks.
        """
        if not self._load_index() and not self.build_index():
            logging.warning("Knowledge base unavailable. Returning fallback context.")
            return "Knowledge base is currently unavailable."

        if not self._vectorizer or self._matrix is None:
            return "Knowledge base is currently unavailable."

        query_vec = self._vectorizer.transform([question])
        scores = cosine_similarity(self._matrix, query_vec).reshape(-1)
        if scores.size == 0:
            return "No relevant context found."

        k = max(1, min(int(top_k), scores.size))
        top_idx = np.argpartition(-scores, kth=k - 1)[:k]
        top_sorted = top_idx[np.argsort(-scores[top_idx])]
        return "\n\n".join(self._chunks[i]["text"] for i in top_sorted)

    def retrieve_comps(self, property_details: dict, top_k: int = 3) -> list[dict]:
        """
        Retrieve the top-k most similar comparable property sales from
        `comparable_sales.txt` using a simple distance-based scorer.
        
        Args:
            property_details: Dictionary with keys like area, bedrooms, bathrooms, 
                            basement, airconditioning, mainroad, guestroom, etc.
            top_k: Number of comparable sales to retrieve (default 3).
        
        Returns:
            List of comparable property transactions with details extracted.
        """
        comps_path = self.docs_dir / "comparable_sales.txt"
        if not comps_path.exists():
            logging.warning("Comparable sales file missing at %s", comps_path)
            return []

        raw_text = comps_path.read_text(encoding="utf-8", errors="ignore")
        blocks = re.split(r"\n(?=Comparable Property\s+\d+)", raw_text)
        parsed: list[dict] = []
        for idx, block in enumerate(blocks):
            comp = self._parse_comp_from_text(block, idx + 1)
            if comp:
                parsed.append(comp)

        if not parsed:
            return []

        target_area = float(property_details.get("area") or 0)
        target_beds = float(property_details.get("bedrooms") or 0)
        target_baths = float(property_details.get("bathrooms") or 0)
        target_basement = str(property_details.get("basement") or "No").lower() == "yes"
        target_ac = str(property_details.get("airconditioning") or "No").lower() == "yes"
        target_mainroad = str(property_details.get("mainroad") or "No").lower() == "yes"

        def score(comp: dict) -> float:
            comp_area = float(comp.get("area") or 0)
            comp_beds = float(comp.get("bedrooms") or 0)
            comp_baths = float(comp.get("bathrooms") or 0)

            # Relative numeric distance with small eps for stability.
            eps = 1e-6
            area_dist = abs(target_area - comp_area) / max(target_area, comp_area, eps)
            bed_dist = abs(target_beds - comp_beds) / max(target_beds, comp_beds, eps)
            bath_dist = abs(target_baths - comp_baths) / max(target_baths, comp_baths, eps)

            basement_dist = 0.0 if (str(comp.get("basement") or "No").lower() == "yes") == target_basement else 1.0
            ac_dist = 0.0 if (str(comp.get("airconditioning") or "No").lower() == "yes") == target_ac else 1.0
            mainroad_dist = 0.0 if (str(comp.get("mainroad") or "No").lower() == "yes") == target_mainroad else 1.0

            return (
                0.55 * area_dist
                + 0.20 * bed_dist
                + 0.15 * bath_dist
                + 0.04 * basement_dist
                + 0.04 * ac_dist
                + 0.02 * mainroad_dist
            )

        ranked = sorted(parsed, key=score)
        return ranked[: max(1, min(int(top_k), len(ranked)))]

    def _parse_comp_from_text(self, text: str, comp_number: int) -> dict | None:
        """
        Parse comparable property information from document text.
        Extracts key fields like date, location, price, bedrooms, bathrooms, etc.
        """
        
        try:
            comp = {
                "comp_number": comp_number,
                "location": self._extract_field(text, r"Location:\s*([^\n]+)"),
                "date": self._extract_field(text, r"Date:\s*([^\n]+)"),
                "area": self._extract_numeric(text, r"Area:\s*([\d,]+)"),
                "bedrooms": self._extract_numeric(text, r"Bedrooms:\s*(\d+)"),
                "bathrooms": self._extract_numeric(text, r"Bathrooms:\s*(\d+)"),
                "price": self._extract_field(text, r"Sale Price:\s*([^\n]+)"),
                "basement": self._extract_field(text, r"Basement:\s*(Yes|No)"),
                "airconditioning": self._extract_field(text, r"Air Conditioning:\s*(Yes|No)"),
                "mainroad": self._extract_field(text, r"Main Road Access:\s*(Yes|No)"),
                "raw_text": text[:200] + "..." if len(text) > 200 else text
            }
            
            # Only return if we found at least a price
            if comp.get("price"):
                return comp
        except Exception as e:
            logging.warning(f"Failed to parse comp #{comp_number}: {e}")
        
        return None

    def _extract_field(self, text: str, pattern: str) -> str | None:
        """Extract a string field from text using regex."""
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else None

    def _extract_numeric(self, text: str, pattern: str) -> int | None:
        """Extract a numeric field from text using regex."""
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1).replace(",", ""))
            except ValueError:
                return None
        return None

    @property
    def is_ready(self) -> bool:
        """Returns True if a knowledge index is available in memory or on disk."""
        return self._vectorizer is not None or self._index_exists()

    def _load_index(self) -> bool:
        if self._vectorizer is not None and self._matrix is not None and self._chunks:
            return True
        if not self._index_exists():
            return False
        try:
            payload = joblib.load(self.index_file)
            self._vectorizer = payload.get("vectorizer")
            self._matrix = payload.get("matrix")
            self._chunks = payload.get("chunks") or []
            return self._vectorizer is not None and self._matrix is not None and bool(self._chunks)
        except Exception as exc:
            logging.warning("Failed to load knowledge index %s: %s", self.index_file, exc)
            return False
