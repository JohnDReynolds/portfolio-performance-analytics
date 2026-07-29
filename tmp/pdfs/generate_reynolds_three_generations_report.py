"""Generate the three-generation Reynolds mortgage-market history PDF."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from generate_leon_reynolds_report import (
    BLUE,
    GOLD,
    INK,
    MUTED,
    NAVY,
    PALE_BLUE,
    PALE_GOLD,
    RULE,
    ResearchDocTemplate,
    _add_markdown_body,
    _styles,
    _toc,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MARKDOWN_PATH = (
    PROJECT_ROOT
    / "output"
    / "pdf"
    / "Reynolds_Family_Three_Generations_Mortgage_Markets.md"
)
PDF_PATH = (
    PROJECT_ROOT
    / "output"
    / "pdf"
    / "Reynolds_Family_Three_Generations_Mortgage_Markets.pdf"
)


def _page_number(canvas: Any, doc: Any) -> None:
    """Draw the report header and footer on non-cover pages."""
    if doc.page <= 1:
        return
    canvas.saveState()
    width, height = letter

    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.45)
    canvas.line(
        doc.leftMargin,
        height - 0.48 * inch,
        width - doc.rightMargin,
        height - 0.48 * inch,
    )
    canvas.setFont("Helvetica", 7.3)
    canvas.setFillColor(MUTED)
    canvas.drawString(
        doc.leftMargin,
        height - 0.39 * inch,
        "REYNOLDS FAMILY | THREE GENERATIONS IN MORTGAGE MARKETS",
    )

    canvas.drawRightString(
        width - doc.rightMargin,
        0.42 * inch,
        f"Page {doc.page}",
    )
    canvas.setStrokeColor(RULE)
    canvas.line(
        doc.leftMargin,
        0.54 * inch,
        width - doc.rightMargin,
        0.54 * inch,
    )
    canvas.restoreState()


def _cover(story: list[Any], styles: dict[str, Any]) -> None:
    """Add the cover and the three-generation summary."""
    story.extend(
        [
            Spacer(1, 0.20 * inch),
            Paragraph("FAMILY AND FINANCIAL HISTORY", styles["cover_kicker"]),
            Paragraph(
                "Three Generations in the<br/>American Mortgage Market",
                styles["cover_title"],
            ),
            Paragraph(
                "The Reynolds family and the changing boundary between "
                "public purpose and private risk",
                styles["cover_subtitle"],
            ),
            Paragraph(
                "Prepared July 28, 2026 | Revised John Reynolds role",
                styles["cover_date"],
            ),
            HRFlowable(
                width="48%",
                thickness=1.4,
                color=GOLD,
                spaceBefore=0,
                spaceAfter=15,
                hAlign="CENTER",
            ),
        ]
    )

    header_style = styles["cover_fact_label"].clone("GenerationHeader")
    header_style.textColor = colors.white
    header_style.fontSize = 8
    cell_style = styles["cover_fact_value"].clone("GenerationCell")
    cell_style.fontSize = 8
    cell_style.leading = 10.2

    rows = [
        [
            Paragraph("Generation", header_style),
            Paragraph("Mortgage-market role", header_style),
            Paragraph("Family experience", header_style),
        ],
        [
            Paragraph("<b>Kirk Reynolds</b><br/>1889-1963", cell_style),
            Paragraph(
                "Loan inspector, thrift advocate, FHA underwriter, "
                "Home Loan Bank director",
                cell_style,
            ),
            Paragraph(
                "Helped operate the Depression-era mortgage rescue system",
                cell_style,
            ),
        ],
        [
            Paragraph("<b>Daniel Reynolds</b><br/>1932-2025", cell_style),
            Paragraph(
                "Residential appraiser, including work at Benj. Franklin",
                cell_style,
            ),
            Paragraph(
                "Lost more than 80 percent of his money after the "
                "1990 federal takeover (family testimony)",
                cell_style,
            ),
        ],
        [
            Paragraph("<b>John Reynolds</b>", cell_style),
            Paragraph(
                "Analyst at American Underwriters in the mid-1980s",
                cell_style,
            ),
            Paragraph(
                "Roven group held a large Citadel position; analyst role is "
                "family testimony",
                cell_style,
            ),
        ],
    ]
    generation_table = Table(
        rows,
        colWidths=[1.25 * inch, 2.56 * inch, 2.70 * inch],
        hAlign="CENTER",
        repeatRows=1,
    )
    generation_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("BACKGROUND", (0, 1), (-1, -1), PALE_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.6, RULE),
                ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#D4DEE5")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.extend([generation_table, Spacer(1, 14)])

    thesis = (
        "<b>Central finding:</b> Depression-era intervention repaired a broken "
        "mortgage market. Later crises arose when public guarantees and "
        "regulatory concessions interacted with private leverage, interest-rate "
        "risk, weak supervision, abrupt rule changes, and in some institutions "
        "fraud. Across three generations, the family experienced government as "
        "stabilizer, rule-maker, source of loss, and creator of tradable market "
        "distortions."
    )
    scope = (
        "Family recollections are preserved explicitly as family testimony. "
        "Documented facts, reasoned interpretations, and contested causal "
        "claims are identified separately."
    )
    story.extend(
        [
            Table(
                [[Paragraph(thesis, styles["cover_scope"])]],
                colWidths=[6.45 * inch],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), PALE_GOLD),
                        ("BOX", (0, 0), (-1, -1), 0.55, GOLD),
                        ("LEFTPADDING", (0, 0), (-1, -1), 9),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                        ("TOPPADDING", (0, 0), (-1, -1), 7),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ]
                ),
            ),
            Spacer(1, 10),
            Paragraph(scope, styles["cover_scope"]),
            PageBreak(),
        ]
    )


def _document() -> None:
    """Build the family and financial history report."""
    styles = _styles()
    markdown = MARKDOWN_PATH.read_text(encoding="utf-8")

    doc = ResearchDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        rightMargin=0.72 * inch,
        leftMargin=0.72 * inch,
        topMargin=0.67 * inch,
        bottomMargin=0.68 * inch,
        title="Three Generations in the American Mortgage Market",
        author="Prepared for the Reynolds family",
        subject=(
            "Family history of Kirk, Daniel, and John Reynolds in American "
            "mortgage finance"
        ),
        creator="ReportLab",
    )

    story: list[Any] = []
    _cover(story, styles)
    _toc(story, styles)
    _add_markdown_body(
        story,
        styles,
        markdown,
        page_break_sections=("13. ", "15. "),
    )

    doc.multiBuild(
        story,
        onFirstPage=_page_number,
        onLaterPages=_page_number,
    )


if __name__ == "__main__":
    _document()
