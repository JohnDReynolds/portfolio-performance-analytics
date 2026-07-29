"""Generate the Leon Kirkwood Reynolds biographical research PDF."""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MARKDOWN_PATH = (
    PROJECT_ROOT
    / "output"
    / "pdf"
    / "Leon_Kirkwood_Reynolds_Biographical_Research.md"
)
PDF_PATH = (
    PROJECT_ROOT
    / "output"
    / "pdf"
    / "Leon_Kirkwood_Reynolds_Biographical_Research.pdf"
)

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2A648C")
GOLD = colors.HexColor("#C39A52")
INK = colors.HexColor("#202A33")
MUTED = colors.HexColor("#5D6974")
PALE_BLUE = colors.HexColor("#EDF4F8")
PALE_GOLD = colors.HexColor("#F7F0E3")
RULE = colors.HexColor("#C9D3DB")


class ResearchDocTemplate(SimpleDocTemplate):
    """Document template that registers headings for the table of contents."""

    def afterFlowable(self, flowable: Any) -> None:
        """Register section headings as outline and table-of-contents entries."""
        if not isinstance(flowable, Paragraph):
            return
        if flowable.style.name not in {"SectionHeading", "Subheading"}:
            return

        level = 0 if flowable.style.name == "SectionHeading" else 1
        text = flowable.getPlainText()
        key = f"heading-{self.seq.nextf('heading')}"
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(text, key, level=level, closed=False)
        self.notify("TOCEntry", (level, text, self.page, key))


def _styles() -> dict[str, ParagraphStyle]:
    """Create the report's paragraph styles."""
    base = getSampleStyleSheet()
    return {
        "cover_kicker": ParagraphStyle(
            "CoverKicker",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=11,
            textColor=GOLD,
            spaceAfter=12,
            alignment=TA_CENTER,
            uppercase=True,
        ),
        "cover_title": ParagraphStyle(
            "CoverTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=27,
            leading=31,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=9,
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=13,
            leading=17,
            textColor=BLUE,
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "cover_date": ParagraphStyle(
            "CoverDate",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=18,
        ),
        "cover_scope": ParagraphStyle(
            "CoverScope",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=13.2,
            textColor=INK,
            spaceAfter=8,
        ),
        "cover_fact_label": ParagraphStyle(
            "CoverFactLabel",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.2,
            leading=10.5,
            textColor=NAVY,
        ),
        "cover_fact_value": ParagraphStyle(
            "CoverFactValue",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8.2,
            leading=10.5,
            textColor=INK,
        ),
        "toc_title": ParagraphStyle(
            "TOCTitle",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=NAVY,
            spaceAfter=14,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Times-Roman",
            fontSize=9.3,
            leading=13.1,
            textColor=INK,
            spaceAfter=6.2,
            allowWidows=0,
            allowOrphans=0,
        ),
        "section": ParagraphStyle(
            "SectionHeading",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            textColor=NAVY,
            spaceBefore=10,
            spaceAfter=7,
            keepWithNext=True,
        ),
        "subheading": ParagraphStyle(
            "Subheading",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=13,
            textColor=BLUE,
            spaceBefore=7,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontName="Times-Roman",
            fontSize=9.1,
            leading=12.7,
            textColor=INK,
            leftIndent=16,
            firstLineIndent=0,
            rightIndent=2,
            spaceAfter=3.4,
            bulletIndent=3,
            allowWidows=0,
            allowOrphans=0,
        ),
        "quote": ParagraphStyle(
            "Quote",
            parent=base["BodyText"],
            fontName="Times-Italic",
            fontSize=8.8,
            leading=12.5,
            textColor=NAVY,
            leftIndent=13,
            rightIndent=10,
            spaceAfter=7,
        ),
        "url": ParagraphStyle(
            "URL",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.1,
            leading=9.2,
            textColor=BLUE,
            leftIndent=8,
            rightIndent=5,
            spaceAfter=6,
            wordWrap="CJK",
        ),
        "source_note": ParagraphStyle(
            "SourceNote",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.7,
            leading=10.3,
            textColor=MUTED,
            spaceAfter=5,
        ),
    }


def _inline_markup(text: str) -> str:
    """Convert the limited inline Markdown used in the report."""
    escaped = html.escape(text, quote=False)
    escaped = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        r'<link href="\2" color="#2A648C"><u>\1</u></link>',
        escaped,
    )
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(
        r"`([^`]+)`",
        r'<font name="Courier" size="8">\1</font>',
        escaped,
    )
    return escaped


def _page_number(canvas: Any, doc: Any) -> None:
    """Draw the header and footer on non-cover pages."""
    if doc.page <= 1:
        return
    canvas.saveState()
    width, height = letter

    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.45)
    canvas.line(doc.leftMargin, height - 0.48 * inch, width - doc.rightMargin, height - 0.48 * inch)
    canvas.setFont("Helvetica", 7.3)
    canvas.setFillColor(MUTED)
    canvas.drawString(
        doc.leftMargin,
        height - 0.39 * inch,
        "LEON KIRKWOOD REYNOLDS | BIOGRAPHICAL RESEARCH",
    )

    page_text = f"Page {doc.page}"
    canvas.drawRightString(width - doc.rightMargin, 0.42 * inch, page_text)
    canvas.setStrokeColor(RULE)
    canvas.line(doc.leftMargin, 0.54 * inch, width - doc.rightMargin, 0.54 * inch)
    canvas.restoreState()


def _cover(story: list[Any], styles: dict[str, ParagraphStyle]) -> None:
    """Add the cover page."""
    story.extend(
        [
            Spacer(1, 0.38 * inch),
            Paragraph("FAMILY HISTORY RESEARCH REPORT", styles["cover_kicker"]),
            Paragraph("Leon Kirkwood Reynolds", styles["cover_title"]),
            Paragraph("(1889-1963)", styles["cover_subtitle"]),
            Paragraph(
                "A documented biographical and genealogical research report",
                styles["cover_subtitle"],
            ),
            Paragraph("Prepared July 26, 2026", styles["cover_date"]),
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

    facts = [
        ("Born", "December 9, 1889 | Arcadia, Valley County, Nebraska"),
        ("Known as", "Kirk Reynolds; also L. Kirk Reynolds and Leon K. Reynolds"),
        ("Married", "Philomene Barnes | March 3, 1921 | Portland, Oregon"),
        ("Son", "Daniel Kirkwood Reynolds | born April 7, 1932"),
        (
            "World War I",
            "Served in France without seeing combat (family testimony)",
        ),
        (
            "Career",
            "Mortgage inspection; savings-and-loan leadership; FHA underwriting; "
            "Federal Home Loan Bank governance; First Federal presidency",
        ),
        (
            "Honolulu",
            "Government bungalow near Royal Hawaiian; Kirk remained through "
            "December 7, 1941 (family testimony)",
        ),
        (
            "Died",
            "June 11, 1963 | pneumonia after fractured hip; Parkinson's disease "
            "(family testimony)",
        ),
        ("Buried", "Clackamas Cemetery | Clackamas County, Oregon"),
    ]
    fact_rows = [
        [
            Paragraph(label, styles["cover_fact_label"]),
            Paragraph(value, styles["cover_fact_value"]),
        ]
        for label, value in facts
    ]
    fact_table = Table(fact_rows, colWidths=[0.95 * inch, 5.57 * inch], hAlign="CENTER")
    fact_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.55, RULE),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#DCE5EB")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 4.4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4.4),
            ]
        )
    )
    story.extend([fact_table, Spacer(1, 14)])

    scope = (
        "This report consolidates the information located to date about Leon "
        "Kirkwood Reynolds. It preserves established facts, family testimony, "
        "and promising unresolved clues while distinguishing evidence from "
        "inference."
    )
    family = (
        "The research subject is the grandfather of the person who requested this "
        "report. Family testimony identifies Daniel as Leon and Philomene's only "
        "child; records Leon's noncombat World War I service in France; and preserves "
        "details of the family's Honolulu residence, travel, and separation shortly "
        "before Pearl Harbor."
    )
    note = (
        "<b>Research note:</b> Historical directories and newspapers were often "
        "searched through optical character recognition. OCR errors are common. "
        "The absence of a search result is not proof that a person was absent."
    )
    story.extend(
        [
            Paragraph(scope, styles["cover_scope"]),
            Paragraph(family, styles["cover_scope"]),
            Table(
                [[Paragraph(note, styles["source_note"])]],
                colWidths=[6.45 * inch],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), PALE_GOLD),
                        ("BOX", (0, 0), (-1, -1), 0.45, GOLD),
                        ("LEFTPADDING", (0, 0), (-1, -1), 8),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ]
                ),
            ),
            PageBreak(),
        ]
    )


def _toc(story: list[Any], styles: dict[str, ParagraphStyle]) -> None:
    """Add a generated table of contents."""
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(
            "TOCSection",
            fontName="Helvetica",
            fontSize=8.8,
            leading=12,
            textColor=INK,
            leftIndent=0,
            firstLineIndent=0,
            spaceBefore=2,
        ),
        ParagraphStyle(
            "TOCSubsection",
            fontName="Helvetica",
            fontSize=7.8,
            leading=10,
            textColor=MUTED,
            leftIndent=13,
            firstLineIndent=0,
            spaceBefore=1,
        ),
    ]
    story.extend(
        [
            Paragraph("Contents", styles["toc_title"]),
            Paragraph(
                "Major sections and dated or topical subsections are listed below. "
                "The PDF bookmarks provide the same navigation.",
                styles["source_note"],
            ),
            Spacer(1, 6),
            toc,
            PageBreak(),
        ]
    )


def _flush_paragraph(
    story: list[Any],
    paragraph_lines: list[str],
    styles: dict[str, ParagraphStyle],
) -> None:
    """Add accumulated Markdown paragraph lines to the story."""
    if not paragraph_lines:
        return
    text = " ".join(line.strip() for line in paragraph_lines)
    story.append(Paragraph(_inline_markup(text), styles["body"]))
    paragraph_lines.clear()


def _flush_list_item(
    story: list[Any],
    item_lines: list[str],
    item_marker: list[str],
    styles: dict[str, ParagraphStyle],
) -> None:
    """Add an accumulated bullet or numbered item to the story."""
    if not item_lines:
        return
    text = " ".join(line.strip() for line in item_lines)
    marker = item_marker[0]
    bullet_text = "\u2022" if marker == "bullet" else marker
    story.append(
        Paragraph(
            _inline_markup(text),
            styles["bullet"],
            bulletText=bullet_text,
        )
    )
    item_lines.clear()
    item_marker.clear()


def _add_markdown_body(
    story: list[Any],
    styles: dict[str, ParagraphStyle],
    markdown: str,
    page_break_sections: tuple[str, ...] = ("18. ", "23. "),
) -> None:
    """Convert the report's block-level Markdown to ReportLab flowables."""
    lines = markdown.splitlines()
    start_index = next(
        index for index, line in enumerate(lines) if line.startswith("## 1. ")
    )
    paragraph_lines: list[str] = []
    item_lines: list[str] = []
    item_marker: list[str] = []

    for line in lines[start_index:]:
        stripped = line.strip()

        if not stripped:
            _flush_paragraph(story, paragraph_lines, styles)
            _flush_list_item(story, item_lines, item_marker, styles)
            continue

        if stripped == "---":
            _flush_paragraph(story, paragraph_lines, styles)
            _flush_list_item(story, item_lines, item_marker, styles)
            story.append(
                HRFlowable(
                    width="100%",
                    thickness=0.55,
                    color=RULE,
                    spaceBefore=6,
                    spaceAfter=8,
                )
            )
            continue

        if stripped.startswith("## "):
            _flush_paragraph(story, paragraph_lines, styles)
            _flush_list_item(story, item_lines, item_marker, styles)
            heading = stripped[3:]
            if heading.startswith(page_break_sections):
                story.append(PageBreak())
            story.append(Paragraph(_inline_markup(heading), styles["section"]))
            continue

        if stripped.startswith("### "):
            _flush_paragraph(story, paragraph_lines, styles)
            _flush_list_item(story, item_lines, item_marker, styles)
            story.append(
                Paragraph(_inline_markup(stripped[4:]), styles["subheading"])
            )
            continue

        if stripped.startswith("- "):
            _flush_paragraph(story, paragraph_lines, styles)
            _flush_list_item(story, item_lines, item_marker, styles)
            item_lines.append(stripped[2:])
            item_marker.append("bullet")
            continue

        numbered_match = re.match(r"^(\d+)\.\s+(.+)$", stripped)
        if numbered_match:
            _flush_paragraph(story, paragraph_lines, styles)
            _flush_list_item(story, item_lines, item_marker, styles)
            number, text = numbered_match.groups()
            item_lines.append(text)
            item_marker.append(f"{number}.")
            continue

        if stripped.startswith("> "):
            _flush_paragraph(story, paragraph_lines, styles)
            _flush_list_item(story, item_lines, item_marker, styles)
            quote = Paragraph(_inline_markup(stripped[2:]), styles["quote"])
            story.append(
                Table(
                    [[quote]],
                    colWidths=[6.12 * inch],
                    style=TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
                            ("LINEBEFORE", (0, 0), (0, -1), 2.0, BLUE),
                            ("LEFTPADDING", (0, 0), (-1, -1), 8),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                            ("TOPPADDING", (0, 0), (-1, -1), 5),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                        ]
                    ),
                )
            )
            story.append(Spacer(1, 5))
            continue

        if stripped.startswith(("http://", "https://")):
            _flush_paragraph(story, paragraph_lines, styles)
            _flush_list_item(story, item_lines, item_marker, styles)
            safe_url = html.escape(stripped, quote=True)
            story.append(
                Paragraph(
                    f'<link href="{safe_url}" color="#2A648C"><u>{safe_url}</u></link>',
                    styles["url"],
                )
            )
            continue

        if item_lines:
            item_lines.append(stripped)
        else:
            paragraph_lines.append(stripped)

    _flush_paragraph(story, paragraph_lines, styles)
    _flush_list_item(story, item_lines, item_marker, styles)


def _document() -> None:
    """Build the report."""
    styles = _styles()
    markdown = MARKDOWN_PATH.read_text(encoding="utf-8")

    doc = ResearchDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        rightMargin=0.72 * inch,
        leftMargin=0.72 * inch,
        topMargin=0.67 * inch,
        bottomMargin=0.68 * inch,
        title="Leon Kirkwood Reynolds (1889-1963): Biographical Research Report",
        author="Prepared for the Reynolds family",
        subject="Biographical, genealogical, and professional-history research",
        creator="ReportLab",
    )

    story: list[Any] = []
    _cover(story, styles)
    _toc(story, styles)
    _add_markdown_body(story, styles, markdown)

    doc.multiBuild(
        story,
        onFirstPage=_page_number,
        onLaterPages=_page_number,
    )


if __name__ == "__main__":
    _document()
