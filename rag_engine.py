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

import logging
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from llm_config import get_groq_model, has_groq_api_key

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

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
        self.trusted = True  # Default to True for local dev; gate in production
        self._vector_store = None

    # ----------------------------------------------------------
    # Phase 2: Full implementation coming in next PR
    # ----------------------------------------------------------

    def build_index(self) -> None:
        """Load documents, chunk them, and build the FAISS vector index."""
        if not self.docs_dir.exists():
            logging.error(f"Knowledge base directory not found: {self.docs_dir}")
            return
            
        logging.info(f"Loading documents from {self.docs_dir}...")
        loader = DirectoryLoader(str(self.docs_dir), glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()
        
        if not documents:
            logging.warning("No documents found to build the knowledge base.")
            return

        logging.info(f"Loaded {len(documents)} documents. Splitting into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        )
        splits = text_splitter.split_documents(documents)
        
        logging.info(f"Initializing HuggingFace Embeddings: {self.embedding_model}")
        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)

        logging.info("Building FAISS Vector Store...")
        self._vector_store = FAISS.from_documents(documents=splits, embedding=embeddings)
        
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._vector_store.save_local(str(self.index_dir))
        logging.info(f"FAISS index successfully saved to {self.index_dir}")

    def query(self, question: str, top_k: int = TOP_K_RESULTS) -> str:
        """
        Retrieve relevant document chunks and return either a grounded answer
        from Groq or the raw retrieved context when Groq is unavailable.

        Args:
            question: Natural language question about real estate.
            top_k: Number of top relevant chunks to retrieve.

        Returns:
            A Groq-generated answer grounded in retrieved context, or a
            newline-joined string of the top retrieved document chunks when
            GROQ_API_KEY is not configured.
        """
        if self._vector_store is None:
            if not self.index_dir.exists():
                raise RuntimeError("FAISS index not found. Run build_index() first.")
            
            logging.info("Loading existing FAISS index...")
            embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
            self._vector_store = FAISS.load_local(
                str(self.index_dir), 
                embeddings, 
                allow_dangerous_deserialization=self.trusted
            )

        retriever = self._vector_store.as_retriever(search_kwargs={"k": top_k})
        
        # When Groq is unavailable, return raw retrieved context instead of a generated answer.
        if not has_groq_api_key():
            logging.warning("No GROQ_API_KEY found. Returning raw context chunks instead of LLM answer.")
            docs = retriever.invoke(question)
            return "\n\n".join([d.page_content for d in docs])
            
        llm = ChatGroq(model=get_groq_model(), temperature=0.0)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a real estate investment advisor. Answer the user's question based strictly on the provided context. If the answer is not in the context, say you don't know.\n\nContext: {context}"),
            ("human", "{input}")
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        logging.info(f"Querying LLM for: {question}")
        response = rag_chain.invoke({"input": question})
        return response["answer"]

    def retrieve_comps(self, property_details: dict, top_k: int = 3) -> list[dict]:
        """
        Retrieve the top-k most similar comparable property sales.
        
        Args:
            property_details: Dictionary with keys like area, bedrooms, bathrooms, 
                            basement, airconditioning, mainroad, guestroom, etc.
            top_k: Number of comparable sales to retrieve (default 3).
        
        Returns:
            List of comparable property transactions with details extracted.
        """
        if self._vector_store is None:
            if not self.index_dir.exists():
                raise RuntimeError("FAISS index not found. Run build_index() first.")
            
            logging.info("Loading existing FAISS index for comps retrieval...")
            embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
            self._vector_store = FAISS.load_local(
                str(self.index_dir), 
                embeddings, 
                allow_dangerous_deserialization=self.trusted
            )
        
        # Build a natural language query from property details
        query_parts = []
        if property_details.get("bedrooms"):
            query_parts.append(f"{property_details['bedrooms']}-bedroom")
        if property_details.get("area"):
            query_parts.append(f"{property_details['area']} sq ft")
        if property_details.get("basement") == "Yes":
            query_parts.append("with basement")
        if property_details.get("airconditioning") == "Yes":
            query_parts.append("with air conditioning")
        if property_details.get("mainroad") == "Yes":
            query_parts.append("on main road")
        
        search_query = " ".join(query_parts) + " recent sale comparable transaction"
        
        logging.info(f"Searching for comps with query: {search_query}")
        
        retriever = self._vector_store.as_retriever(search_kwargs={"k": top_k})
        docs = retriever.invoke(search_query)
        
        # Parse document content into structured comps
        comps = []
        for i, doc in enumerate(docs):
            content = doc.page_content
            comp = self._parse_comp_from_text(content, i + 1)
            if comp:
                comps.append(comp)
        
        return comps

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
        """Returns True if a FAISS index has been built and loaded."""
        return self._vector_store is not None
