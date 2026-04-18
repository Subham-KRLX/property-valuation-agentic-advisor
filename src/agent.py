from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import re
from typing import Any

from src.rag_engine import RAGEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _format_inr(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    try:
        return f"₹{float(value):,.0f}"
    except Exception:
        return str(value)


def _parse_inr(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not math.isnan(float(value)):
        return float(value)

    text = str(value).strip().lower()
    if not text:
        return None

    # Handle formats like "8,200,000" or "INR 1.5 Cr".
    digits = re.sub(r"[^0-9.,]", "", text)
    digits = digits.replace(",", "")
    try:
        number = float(digits) if digits else None
    except ValueError:
        number = None

    if number is None:
        return None

    if "cr" in text or "crore" in text:
        return number * 10_000_000
    if "lac" in text or "lakh" in text:
        return number * 100_000

    return number


@dataclass(frozen=True)
class AdvisoryResult:
    advice: str
    comps: list[dict[str, Any]]
    context: str


class PropertyAdvisorAgent:
    """
    Traditional (non-LLM) advisor.

    - Uses deterministic retrieval from the local knowledge base to surface
      market context.
    - Uses a distance-based comparable-sales matcher for comps.
    - Produces a concise, template-based investment summary.
    """

    def __init__(self) -> None:
        self.rag_engine = RAGEngine()
        try:
            if not self.rag_engine.is_ready:
                self.rag_engine.build_index()
        except Exception as exc:
            logging.warning("Could not initialize knowledge index: %s", exc)

    def run(self, property_details: dict[str, Any], predicted_price: float) -> tuple[str, list]:
        context = self._retrieve_context(property_details)
        comps = self.rag_engine.retrieve_comps(property_details, top_k=3)
        advice = self._generate_advice(
            property_details=property_details,
            predicted_price=float(predicted_price),
            context=context,
            comps=comps,
        )
        return advice, comps

    def _retrieve_context(self, details: dict[str, Any]) -> str:
        query_parts = ["property investment trends"]
        if details.get("mainroad") == "Yes":
            query_parts.append("main road access premium")
        if details.get("airconditioning") == "Yes":
            query_parts.append("air conditioning demand")
        if details.get("basement") == "Yes":
            query_parts.append("basement value")
        query_parts.append("Bangalore market")  # knowledge base is Bangalore-centric
        query = " | ".join(query_parts)
        return self.rag_engine.query(query, top_k=2)

    def _generate_advice(
        self,
        *,
        property_details: dict[str, Any],
        predicted_price: float,
        context: str,
        comps: list[dict[str, Any]],
    ) -> str:
        area = property_details.get("area")
        beds = property_details.get("bedrooms")
        baths = property_details.get("bathrooms")
        stories = property_details.get("stories")

        comp_prices = [_parse_inr(c.get("price")) for c in comps]
        comp_prices = [p for p in comp_prices if p is not None]

        comp_line = "Comparable sales were not available for this run."
        recommendation = "Hold"
        rationale = "Insufficient comparable evidence to make a strong call."

        if comp_prices:
            median = sorted(comp_prices)[len(comp_prices) // 2]
            delta = (predicted_price - median) / max(median, 1.0)
            delta_pct = delta * 100
            comp_line = (
                f"Based on {len(comp_prices)} nearby comparable sale(s), the median comp is "
                f"{_format_inr(median)}; the model estimate is {delta_pct:+.1f}% vs that median."
            )

            # Simple, explicit decision rule.
            if delta <= -0.08:
                recommendation = "Buy"
                rationale = "The estimate is meaningfully below comparable sales, suggesting relative value."
            elif delta >= 0.12:
                recommendation = "Pass"
                rationale = "The estimate is meaningfully above comparable sales, suggesting limited upside."
            else:
                recommendation = "Hold"
                rationale = "The estimate is broadly in-line with comparable sales; negotiate and validate."

        amenities = []
        for k, label in [
            ("mainroad", "main road access"),
            ("guestroom", "guest room"),
            ("basement", "basement"),
            ("hotwaterheating", "hot water heating"),
            ("airconditioning", "air conditioning"),
        ]:
            if property_details.get(k) == "Yes":
                amenities.append(label)

        amenity_text = ", ".join(amenities) if amenities else "no major amenities flagged"

        paragraph_1 = (
            f"Property summary: {area} sq ft, {beds} bed / {baths} bath, {stories} storey(ies), "
            f"with {amenity_text}. The ML valuation estimate is **{_format_inr(predicted_price)}**."
        )

        paragraph_2 = (
            f"{comp_line}\n\n"
            f"Retrieved market context (knowledge base):\n{context}"
        )

        paragraph_3 = (
            f"Recommendation: **{recommendation}**. {rationale} "
            "Next steps: verify legal/encumbrance status, inspect construction quality, and validate the "
            "local pricing with at least 2 on-ground brokers."
        )

        return "\n\n".join([paragraph_1, paragraph_2, paragraph_3])
