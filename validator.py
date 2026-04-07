from dataclasses import dataclass, field

# Physical limits
HARD_BOUNDS = {
    "area": (300, 25000),
    "bedrooms": (1, 10),
    "bathrooms": (1, 6),
    "stories": (1, 5),
    "parking": (0, 5),
}

# Training data distribution limits 
SOFT_BOUNDS = {
    "area": (1650, 16200),
    "bedrooms": (1, 5),
    "bathrooms": (1, 4),
    "stories": (1, 4),
    "parking": (0, 3),
}

@dataclass
class ValidationResult:
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, message: str):
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str):
        self.warnings.append(message)

class PropertyInputValidator:
    def _coerce_numeric(self, feat: str, val, result: ValidationResult):
        try:
            return float(val)
        except (TypeError, ValueError):
            result.add_error(f"{feat.capitalize()} must be a numeric value")
            return None

    def validate(self, inputs: dict) -> ValidationResult:
        result = ValidationResult()
        numeric_inputs = {}

        expected_fields = set(HARD_BOUNDS) | set(SOFT_BOUNDS)
        for feat in expected_fields:
            val = inputs.get(feat)
            if val is None:
                if feat in HARD_BOUNDS:
                    result.add_error(f"Missing field: {feat}")
                continue

            numeric_val = self._coerce_numeric(feat, val, result)
            if numeric_val is not None:
                numeric_inputs[feat] = numeric_val
        
        # Check hard bounds
        for feat, (lo, hi) in HARD_BOUNDS.items():
            if feat not in numeric_inputs:
                continue
            val = numeric_inputs[feat]
            if not (lo <= val <= hi):
                result.add_error(f"{feat.capitalize()} ({inputs.get(feat)}) is out of range [{lo}-{hi}]")

        # Check soft bounds
        for feat, (lo, hi) in SOFT_BOUNDS.items():
            if feat not in numeric_inputs:
                continue
            val = numeric_inputs[feat]
            if val is not None and not (lo <= val <= hi):
                result.add_warning(f"{feat.capitalize()} ({inputs.get(feat)}) is unusual for this dataset")

        # Cross-field logic only runs when core inputs are present and within hard bounds.
        cross_fields = ("area", "bedrooms", "bathrooms", "stories")
        all_valid = all(
            feat in numeric_inputs and HARD_BOUNDS[feat][0] <= numeric_inputs[feat] <= HARD_BOUNDS[feat][1]
            for feat in cross_fields
        )

        if all_valid:
            area = numeric_inputs["area"]
            beds = numeric_inputs["bedrooms"]
            baths = numeric_inputs["bathrooms"]
            stories = numeric_inputs["stories"]

            sqft_per_bed = area / beds
            if sqft_per_bed < 200:
                result.add_error(f"Area per bedroom ({sqft_per_bed:.0f}) is too low")
            elif sqft_per_bed < 350:
                result.add_warning("Property seems unusually cramped")

            if baths / beds > 2.0:
                result.add_warning("High bathroom to bedroom ratio")
            elif baths > beds:
                result.add_warning("More bathrooms than bedrooms")

            if stories >= 3 and area < 1500:
                result.add_warning("Multiple stories with small area - please verify")

            if beds >= 5 and area < 2000:
                result.add_error("Too many bedrooms for this area")

        return result
