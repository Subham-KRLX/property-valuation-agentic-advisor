import pytest
from src.pdf_report import build_property_report, _format_currency, _format_label, _format_value

def test_format_currency():
    assert _format_currency(1234567) == "INR 1,234,567"
    assert _format_currency(0) == "INR 0"

def test_format_label():
    assert _format_label("area_sqft") == "Area Sqft"
    assert _format_label("bedrooms") == "Bedrooms"

def test_format_value():
    assert _format_value("area", 1500) == "1,500 sq ft"
    assert _format_value("bedrooms", 3.0) == "3"
    assert _format_value("location", "Bangalore") == "Bangalore"

def test_build_property_report_basic():
    property_details = {
        "area": 2000,
        "bedrooms": 3,
        "bathrooms": 2,
    }
    estimated_price = 15000000.0
    advisory_text = "Good investment."
    
    pdf_bytes = build_property_report(
        property_details=property_details,
        estimated_price=estimated_price,
        advisory_text=advisory_text
    )
    
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0
    # Minimal PDF header check
    assert pdf_bytes.startswith(b"%PDF")

def test_build_property_report_full():
    property_details = {"area": 2500, "bedrooms": 4}
    estimated_price = 20000000.0
    advisory_text = "Highly recommended.\n\nGreat location."
    validation_warnings = ["Unusually large area"]
    metadata = {
        "metrics": {"r2": 0.85, "mae": 500000, "rmse": 700000}
    }
    comps = [
        {"location": "Nearby", "price": "1.8 Cr", "date": "2023-12-10", "area": 2400, "bedrooms": 4, "bathrooms": 3}
    ]
    
    pdf_bytes = build_property_report(
        property_details=property_details,
        estimated_price=estimated_price,
        advisory_text=advisory_text,
        validation_warnings=validation_warnings,
        metadata=metadata,
        comps=comps
    )
    
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0


def test_get_advisory_mode():
    from src.pdf_report import _get_advisory_mode
    assert _get_advisory_mode("This is a template advisory context.") == "Template-based advisory"
    assert _get_advisory_mode("Advisory unavailable: API error.") == "Advisory unavailable"
