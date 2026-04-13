import pytest
from validator import PropertyInputValidator, ValidationResult

@pytest.fixture
def validator():
    return PropertyInputValidator()

def test_valid_inputs(validator):
    inputs = {
        "area": 5000,
        "bedrooms": 3,
        "bathrooms": 2,
        "stories": 2,
        "parking": 2,
    }
    result = validator.validate(inputs)
    assert result.is_valid is True
    assert len(result.errors) == 0
    # 5000/3 = 1666 sqft per bed, which is > 350, so no warnings there.
    # baths/beds = 2/3 = 0.66, no warnings there.
    assert len(result.warnings) == 0

def test_missing_field(validator):
    inputs = {
        "area": 5000,
        "bedrooms": 3,
        # "bathrooms" missing
        "stories": 2,
        "parking": 2,
    }
    result = validator.validate(inputs)
    assert result.is_valid is False
    assert any("Missing field: bathrooms" in err for err in result.errors)

def test_hard_bounds_failure(validator):
    inputs = {
        "area": 50000, # Max is 25000
        "bedrooms": 3,
        "bathrooms": 2,
        "stories": 2,
        "parking": 2,
    }
    result = validator.validate(inputs)
    assert result.is_valid is False
    assert any("Area (50000) is out of range" in err for err in result.errors)

def test_soft_bounds_warning(validator):
    inputs = {
        "area": 1600, # Lower soft bound is 1650
        "bedrooms": 3,
        "bathrooms": 2,
        "stories": 2,
        "parking": 2,
    }
    result = validator.validate(inputs)
    assert result.is_valid is True
    assert any("Area (1600) is unusual" in warn for warn in result.warnings)

def test_cross_field_ratio_warning(validator):
    inputs = {
        "area": 5000,
        "bedrooms": 3,
        "bathrooms": 4, # More bathrooms than bedrooms
        "stories": 2,
        "parking": 2,
    }
    result = validator.validate(inputs)
    assert result.is_valid is True
    assert any("More bathrooms than bedrooms" in warn for warn in result.warnings)

def test_cross_field_too_many_beds(validator):
    inputs = {
        "area": 1500,
        "bedrooms": 5, # Too many for 1500 sqft
        "bathrooms": 2,
        "stories": 2,
        "parking": 1,
    }
    result = validator.validate(inputs)
    assert result.is_valid is False
    assert any("Too many bedrooms for this area" in err for err in result.errors)

def test_non_numeric_input(validator):
    inputs = {
        "area": "large",
        "bedrooms": 3,
        "bathrooms": 2,
        "stories": 2,
        "parking": 1,
    }
    result = validator.validate(inputs)
    assert result.is_valid is False
    assert any("Area must be a numeric value" in err for err in result.errors)
