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

Usage (future implementation):
    engine = RAGEngine(docs_dir="data/knowledge_base/")
    engine.build_index()
    answer = engine.query("What are the current trends in Bangalore property market?")
"""

from __future__ import annotations

from pathlib import Path


# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------
KNOWLEDGE_BASE_DIR = Path("data/knowledge_base")
FAISS_INDEX_DIR = Path("models/faiss_index")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # Lightweight, CPU-friendly
CHUNK_SIZE = 500                         # Characters per chunk
CHUNK_OVERLAP = 50                       # Overlap to preserve context
TOP_K_RESULTS = 3                        # Number of docs to retrieve


class RAGEngine:
    """
    Retrieval-Augmented Generation engine backed by a FAISS vector store.

    Attributes:
        docs_dir (Path): Directory containing source documents.
        index_dir (Path): Directory where the FAISS index is persisted.
        embedding_model (str): HuggingFace model name for text embeddings.

    Note:
        Full implementation (document loading, chunking, indexing, and
        retrieval logic) will be added in the next phase of development.
    """

    def __init__(
        self,
        docs_dir: str | Path = KNOWLEDGE_BASE_DIR,
        index_dir: str | Path = FAISS_INDEX_DIR,
        embedding_model: str = EMBEDDING_MODEL,
    ) -> None:
        self.docs_dir = Path(docs_dir)
        self.index_dir = Path(index_dir)
        self.embedding_model = embedding_model
        self._vector_store = None

    # ----------------------------------------------------------
    # Phase 2: Full implementation coming in next PR
    # ----------------------------------------------------------

    def build_index(self) -> None:
        """Load documents, chunk them, and build the FAISS vector index."""
        raise NotImplementedError(
            "build_index() will be implemented in Phase 2. "
            "Add real estate PDFs to data/knowledge_base/ to get started."
        )

    def query(self, question: str, top_k: int = TOP_K_RESULTS) -> str:
        """
        Retrieve relevant document chunks and generate a grounded answer.

        Args:
            question: Natural language question about real estate.
            top_k: Number of top relevant chunks to retrieve.

        Returns:
            LLM-generated answer grounded in retrieved context.
        """
        raise NotImplementedError(
            "query() will be implemented in Phase 2 alongside the FAISS index."
        )

    @property
    def is_ready(self) -> bool:
        """Returns True if a FAISS index has been built and loaded."""
        return self._vector_store is not None
