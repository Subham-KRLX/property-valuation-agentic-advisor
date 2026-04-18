from pathlib import Path

import pytest

from src.rag_engine import RAGEngine


@pytest.fixture
def rag_engine(tmp_path: Path) -> RAGEngine:
    docs_dir = tmp_path / "knowledge_base"
    docs_dir.mkdir()
    (docs_dir / "market_trends.txt").write_text(
        "Bangalore market trends:\n\n"
        "Properties with main road access often command a premium.\n\n"
        "Air conditioning increases tenant appeal in warmer months.",
        encoding="utf-8",
    )
    (docs_dir / "comparable_sales.txt").write_text(
        "Recent Comparable Property Sales\n\n"
        "Comparable Property 1\n"
        "Location: Whitefield, Bangalore\n"
        "Date: January 2026\n"
        "Area: 4200\n"
        "Bedrooms: 3\n"
        "Bathrooms: 3\n"
        "Sale Price: 8,200,000\n"
        "Basement: Yes\n"
        "Air Conditioning: Yes\n"
        "Main Road Access: No\n\n"
        "Comparable Property 2\n"
        "Location: Electronic City, Bangalore\n"
        "Date: December 2025\n"
        "Area: 3900\n"
        "Bedrooms: 3\n"
        "Bathrooms: 2\n"
        "Sale Price: 7,650,000\n"
        "Basement: No\n"
        "Air Conditioning: Yes\n"
        "Main Road Access: Yes\n",
        encoding="utf-8",
    )

    index_file = tmp_path / "knowledge_index.joblib"
    engine = RAGEngine(docs_dir=docs_dir, index_file=index_file)
    return engine


def test_index_exists_false(rag_engine: RAGEngine):
    assert rag_engine._index_exists() is False


def test_build_index_and_query(rag_engine: RAGEngine):
    assert rag_engine.build_index() is True
    assert rag_engine._index_exists() is True
    result = rag_engine.query("Does main road access have a premium?", top_k=1)
    assert "premium" in result.lower()


def test_parse_comp_from_text(rag_engine: RAGEngine):
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


def test_parse_comp_missing_data(rag_engine: RAGEngine):
    text = "Just some random text with no fields."
    comp = rag_engine._parse_comp_from_text(text, 1)
    assert comp is None


def test_retrieve_comps(rag_engine: RAGEngine):
    comps = rag_engine.retrieve_comps(
        {
            "area": 4100,
            "bedrooms": 3,
            "bathrooms": 3,
            "basement": "Yes",
            "airconditioning": "Yes",
            "mainroad": "No",
        },
        top_k=1,
    )
    assert len(comps) == 1
    assert comps[0]["location"] == "Whitefield, Bangalore"
