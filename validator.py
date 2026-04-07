"""
validator.py - Input Validation Module for Property Valuation

Provides rule-based and statistical validation of property feature inputs
before they are passed to the ML model, preventing noisy or physically
implausible data from corrupting predictions.

Validation is split into two tiers:
  - ERROR   : Input is physically/logically impossible. Prediction is blocked.
  - WARNING : Input is unusual relative to training data. Prediction proceeds
              but the user is notified.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Hard bounds – absolute physical limits; prediction refused if violated.
HARD_BOUNDS: dict[str, tuple[int | float, int | float]] = {
    "area":      (300,    25_000),
    "bedrooms":  (1,      10),
    "bathrooms": (1,      6),
    "stories":   (1,      5),
    "parking":   (0,      5),
}

# Soft bounds – derived from the Kaggle Housing Prices training distribution
# (545 records, Indian real estate market).  Out-of-range → warning only.
SOFT_BOUNDS: dict[str, tuple[int | float, int | float]] = {
    "area":      (1_650,  16_200),
    "bedrooms":  (1,      5),
    "bathrooms": (1,      4),
    "stories":   (1,      4),
    "parking":   (0,      3),
}

# Cross-field thresholds
MIN_SQFT_PER_BEDROOM  = 200   # sq ft – below this is physically impossible
WARN_SQFT_PER_BEDROOM = 350   # sq ft – below this is unusually cramped
MAX_BATH_BED_RATIO    = 2.0   # bathrooms / bedrooms – above this is unusual
MIN_AREA_HIGH_RISE    = 1_500 # sq ft – warn if 3+ stories below this
MAX_BEDS_TINY_HOUSE   = 5     # bedrooms – error if ≥ this in < 2,000 sq ft
MAX_AREA_TINY_HOUSE   = 2_000 # sq ft


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """
    Encapsulates the outcome of a validation run.

    Attributes:
        is_valid  : False if any ERROR-level check failed; True otherwise.
        errors    : List of human-readable error messages (block prediction).
        warnings  : List of human-readable warning messages (allow prediction).
    """

    is_valid: bool = True
    errors:   list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        """Register a blocking error."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Register a non-blocking warning."""
        self.warnings.append(message)

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Validator
# ─────────────────────────────────────────────────────────────────────────────

class PropertyInputValidator:
    """
    Validates property feature inputs before passing them to the ML model.

    Checks performed (in order):
        1. Hard bounds   – absolute physical/logical limits.
        2. Soft bounds   – out-of-training-distribution detection.
        3. Cross-field   – logical consistency between related features.

    Usage::

        validator = PropertyInputValidator()
        result = validator.validate(input_dict)

        if not result.is_valid:
            # show result.errors and abort prediction
            ...
        if result.has_warnings:
            # show result.warnings but allow prediction
            ...
    """

    def validate(self, inputs: dict) -> ValidationResult:
        """
        Run all validation checks on the provided input dictionary.

        Args:
            inputs: A dict with keys matching FEATURES in train_model.py
                    (area, bedrooms, bathrooms, stories, parking,
                     mainroad, guestroom, basement, hotwaterheating,
                     airconditioning).

        Returns:
            A ValidationResult with accumulated errors and warnings.
        """
        result = ValidationResult()
        self._check_hard_bounds(inputs, result)
        self._check_soft_bounds(inputs, result)
        self._check_cross_field_logic(inputs, result)
        return result

    # ── Private helpers ──────────────────────────────────────────────────────

    def _check_hard_bounds(
        self, inputs: dict, result: ValidationResult
    ) -> None:
        """Block values that are physically impossible."""
        for feat, (lo, hi) in HARD_BOUNDS.items():
            value = inputs.get(feat)
            if value is None:
                result.add_error(f"Missing required field: '{feat}'.")
                continue
            if not (lo <= value <= hi):
                result.add_error(
                    f"**{feat.capitalize()}** value of `{value}` is outside "
                    f"the acceptable range [{lo} – {hi}]. "
                    "Please enter a realistic value."
                )

    def _check_soft_bounds(
        self, inputs: dict, result: ValidationResult
    ) -> None:
        """Warn about values that are out-of-distribution relative to training data."""
        for feat, (lo, hi) in SOFT_BOUNDS.items():
            value = inputs.get(feat)
            if value is None:
                continue  # already caught by hard-bound check
            if not (lo <= value <= hi):
                result.add_warning(
                    f"**{feat.capitalize()}** value of `{value}` is outside "
                    f"the typical range seen in training data [{lo} – {hi}]. "
                    "The model's accuracy may be reduced for this input."
                )

    def _check_cross_field_logic(
        self, inputs: dict, result: ValidationResult
    ) -> None:
        """Enforce logical consistency between related property features."""
        area      = inputs.get("area",      0)
        bedrooms  = inputs.get("bedrooms",  1)
        bathrooms = inputs.get("bathrooms", 1)
        stories   = inputs.get("stories",   1)

        # 1. Area-per-bedroom check
        if bedrooms > 0:
            sqft_per_bed = area / bedrooms
            if sqft_per_bed < MIN_SQFT_PER_BEDROOM:
                result.add_error(
                    f"Area per bedroom is only **{sqft_per_bed:.0f} sq ft** "
                    f"({area:,} sq ft ÷ {bedrooms} bedrooms). "
                    f"The minimum realistic value is {MIN_SQFT_PER_BEDROOM} sq ft. "
                    "Please reduce the number of bedrooms or increase the total area."
                )
            elif sqft_per_bed < WARN_SQFT_PER_BEDROOM:
                result.add_warning(
                    f"Area per bedroom is **{sqft_per_bed:.0f} sq ft** — "
                    "unusually small for a residential property. "
                    f"Typical properties have at least {WARN_SQFT_PER_BEDROOM} sq ft "
                    "per bedroom."
                )

        # 2. Bathrooms-to-bedrooms ratio
        if bedrooms > 0:
            ratio = bathrooms / bedrooms
            if ratio > MAX_BATH_BED_RATIO:
                result.add_warning(
                    f"**{bathrooms} bathrooms** for **{bedrooms} bedroom(s)** "
                    f"gives a {ratio:.1f}× ratio, which is unusually high. "
                    "Please verify the property details."
                )
            # Simpler check: any case where bathrooms exceed bedrooms
            elif bathrooms > bedrooms:
                result.add_warning(
                    f"Bathrooms ({bathrooms}) exceed bedrooms ({bedrooms}). "
                    "This is uncommon in residential properties — "
                    "please double-check the entry."
                )

        # 3. High-rise with tiny footprint
        if stories >= 3 and area < MIN_AREA_HIGH_RISE:
            result.add_warning(
                f"A **{stories}-story** building with only **{area:,} sq ft** "
                "total area is unusual. Please confirm that 'Area' represents "
                "the total built-up area across all floors."
            )

        # 4. Too many bedrooms for a tiny house
        if bedrooms >= MAX_BEDS_TINY_HOUSE and area < MAX_AREA_TINY_HOUSE:
            result.add_error(
                f"**{bedrooms} bedrooms** in **{area:,} sq ft** is not "
                "physically plausible. Please reduce the number of bedrooms "
                "or increase the total area."
            )
