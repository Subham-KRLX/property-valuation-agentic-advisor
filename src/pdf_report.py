from __future__ import annotations

from datetime import datetime
from html import escape
from io import BytesIO
import re

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _format_currency(value: float) -> str:
    return f"INR {value:,.0f}"


def _format_label(key: str) -> str:
    return key.replace("_", " ").title()


def _format_value(key: str, value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)

    if key == "area" and isinstance(value, (int, float)):
        return f"{value:,} sq ft"

    return str(value)


def _clean_advice(text: str) -> list[str]:
    cleaned = re.sub(r"[*_`#]", "", text).strip()
    paragraphs = [part.strip() for part in cleaned.split("\n\n") if part.strip()]
    return paragraphs or ["No advisory summary was available."]


def _get_advisory_mode(advisory_text: str) -> str:
    normalized = advisory_text.lower()
    if "advisory unavailable" in normalized:
        return "Advisory unavailable"
    return "Template-based advisory"


def build_property_report(
    property_details: dict[str, object],
    estimated_price: float,
    advisory_text: str,
    validation_warnings: list[str] | None = None,
    metadata: dict | None = None,
    comps: list[dict] | None = None,
) -> bytes:
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    title_style.textColor = colors.HexColor("#0F172A")
    title_style.spaceAfter = 8

    section_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1D4ED8"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1F2937"),
        spaceAfter=6,
    )
    note_style = ParagraphStyle(
        "Note",
        parent=styles["BodyText"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#475569"),
    )

    story = [
        Paragraph("Property Investment Brief", title_style),
        Paragraph(
            f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            note_style,
        ),
        Spacer(1, 8),
    ]

    headline_table = Table(
        [
            ["Estimated Property Value", _format_currency(estimated_price)],
            ["Report Type", "Property Investment Brief"],
        ],
        colWidths=[70 * mm, 90 * mm],
    )
    headline_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EFF6FF")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#0F172A")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#BFDBFE")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BFDBFE")),
            ]
        )
    )
    story.extend([headline_table, Spacer(1, 10)])

    story.append(Paragraph("Investment Breakdown", section_style))
    breakdown_rows = [
        ["Signal", "Summary"],
        ["Valuation Estimate", _format_currency(estimated_price)],
        ["Validation Warnings", str(len(validation_warnings or []))],
        ["Comparable Sales Included", str(min(len(comps or []), 3))],
        ["Advisory Mode", _get_advisory_mode(advisory_text)],
    ]
    breakdown_table = Table(breakdown_rows, colWidths=[55 * mm, 105 * mm], hAlign="LEFT")
    breakdown_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DBEAFE")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BFDBFE")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend([breakdown_table, Spacer(1, 10)])

    story.append(Paragraph("Property Details", section_style))
    property_rows = [["Feature", "Value"]]
    for key, value in property_details.items():
        property_rows.append([_format_label(key), _format_value(key, value)])

    property_table = Table(property_rows, colWidths=[55 * mm, 105 * mm], hAlign="LEFT")
    property_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend([property_table, Spacer(1, 10)])

    if validation_warnings:
        story.append(Paragraph("Validation Warnings", section_style))
        for warning in validation_warnings:
            story.append(Paragraph(f"- {escape(warning)}", body_style))
        story.append(Spacer(1, 4))

    if comps and len(comps) > 0:
        story.append(Paragraph("Comparable Properties (Comps)", section_style))
        comps_rows = [["Location", "Price", "Date", "Area", "Beds/Baths"]]
        for comp in comps[:3]:
            location = escape(str(comp.get("location") or "N/A"))
            price = escape(str(comp.get("price") or "N/A"))
            date = escape(str(comp.get("date") or "N/A"))
            area = str(comp.get("area") or "N/A")
            beds = str(comp.get("bedrooms") or "N/A")
            baths = str(comp.get("bathrooms") or "N/A")
            comps_rows.append([location, price, date, f"{area} sq ft", f"{beds}/{baths}"])
        
        comps_table = Table(comps_rows, colWidths=[35 * mm, 30 * mm, 25 * mm, 30 * mm, 20 * mm], hAlign="LEFT")
        comps_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#FEF3C7")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#78350F")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 9),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 1), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#FCD34D")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FFFBEB")]),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.extend([comps_table, Spacer(1, 10)])

    if metadata and metadata.get("metrics"):
        story.append(Paragraph("Model Performance Snapshot", section_style))
        metrics = metadata["metrics"]
        metrics_rows = [
            ["Metric", "Value"],
            ["R2 Score", f"{metrics['r2']:.3f}"],
            ["MAE", _format_currency(metrics["mae"])],
            ["RMSE", _format_currency(metrics["rmse"])],
        ]
        metrics_table = Table(metrics_rows, colWidths=[45 * mm, 50 * mm], hAlign="LEFT")
        metrics_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCFCE7")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#14532D")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BBF7D0")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0FDF4")]),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.extend([metrics_table, Spacer(1, 10)])

    story.append(Paragraph("Investment Advisory", section_style))
    for paragraph in _clean_advice(advisory_text):
        story.append(Paragraph(escape(paragraph), body_style))

    document.build(story)
    return buffer.getvalue()
