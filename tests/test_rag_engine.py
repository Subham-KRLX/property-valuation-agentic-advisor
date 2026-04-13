import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from rag_engine import RAGEngine

@pytest.fixture
def mock_embeddings():
    with patch("rag_engine.HuggingFaceEmbeddings") as mock:
        yield mock

@pytest.fixture
def mock_faiss():
    with patch("rag_engine.FAISS") as mock:
        yield mock

@pytest.fixture
def mock_chat_groq():
    with patch("rag_engine.ChatGroq") as mock:
        yield mock

@pytest.fixture
def rag_engine(tmp_path):
    docs_dir = tmp_path / "knowledge_base"
    docs_dir.mkdir()
    (docs_dir / "test.txt").write_text("Location: Bangalore\nSale Price: 1,00,00,000\nArea: 2000")
    
    index_dir = tmp_path / "faiss_index"
    return RAGEngine(docs_dir=docs_dir, index_dir=index_dir)

def test_index_exists_false(rag_engine):
    assert rag_engine._index_exists() is False

def test_build_index(rag_engine, mock_embeddings, mock_faiss):
    # Mock FAISS.from_documents to return a mock vector store
    mock_vs = MagicMock()
    mock_faiss.from_documents.return_value = mock_vs
    
    success = rag_engine.build_index()
    
    assert success is True
    assert mock_faiss.from_documents.called
    assert mock_vs.save_local.called
    assert rag_engine.index_dir.exists()

def test_query_fallback_no_groq(rag_engine, mock_embeddings, mock_faiss):
    # Mock index loading
    rag_engine.index_dir.mkdir(parents=True, exist_ok=True)
    (rag_engine.index_dir / "index.faiss").write_text("dummy") # Just to make it non-empty
    
    mock_vs = MagicMock()
    mock_faiss.load_local.return_value = mock_vs
    
    # Mock retriever
    mock_doc = MagicMock()
    mock_doc.page_content = "Retrieved content"
    mock_vs.as_retriever.return_value.invoke.return_value = [mock_doc]
    
    with patch("rag_engine.has_groq_api_key", return_value=False):
        answer = rag_engine.query("What is the price?")
        assert answer == "Retrieved content"

def test_parse_comp_from_text(rag_engine):
    text = """
    Location: Whitefield, Bangalore
    Date: 2024-01-15
    Area: 2,500 sq ft
    Bedrooms: 3
    Bathrooms: 3
    Sale Price: INR 1.5 Cr
    Basement: No
    Air Conditioning: Yes
    Main Road Access: Yes
    """
    comp = rag_engine._parse_comp_from_text(text, 1)
    
    assert comp["location"] == "Whitefield, Bangalore"
    assert comp["area"] == 2500
    assert comp["bedrooms"] == 3
    assert comp["price"] == "INR 1.5 Cr"
    assert comp["airconditioning"] == "Yes"

def test_parse_comp_missing_data(rag_engine):
    text = "Just some random text with no fields."
    comp = rag_engine._parse_comp_from_text(text, 1)
    assert comp is None
