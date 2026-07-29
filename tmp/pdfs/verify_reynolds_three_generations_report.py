"""Verify the final Reynolds family history PDF and its Markdown source."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from PIL import Image, ImageChops
from pypdf import PdfReader


def _flatten_outline(items: list[object]) -> int:
    """Count all PDF outline entries recursively."""
    count = 0
    for item in items:
        if isinstance(item, list):
            count += _flatten_outline(item)
        else:
            count += 1
    return count


def main() -> int:
    """Run semantic, source-reference, and rendered-page checks."""
    pdf_path = Path(sys.argv[1])
    markdown_path = Path(sys.argv[2])
    render_dir = Path(sys.argv[3])

    reader = PdfReader(pdf_path)
    page_text = [(page.extract_text() or "").strip() for page in reader.pages]
    full_text = "\n".join(page_text)
    markdown = markdown_path.read_text(encoding="ascii")

    assert len(reader.pages) == 21
    assert all(len(text) >= 50 for text in page_text)

    required_phrases = (
        "analyst at American Underwriters",
        "John's analyst work",
        "American Underwriters in the mid-1980s",
        "9.1 percent",
        "9.8 percent",
        "The documented institutional setting",
        "Overall conclusion",
        "S22. American Underwriters",
        "S23. Los Angeles Times",
    )
    for phrase in required_phrases:
        assert phrase in full_text, phrase

    forbidden_phrases = (
        "Freddie",
        "Fannie",
        "Citadel Holding Corp. v. Roven",
        "Section 16(b)",
    )
    for phrase in forbidden_phrases:
        assert phrase not in full_text, phrase
        assert phrase not in markdown, phrase

    sections = [
        int(value)
        for value in re.findall(r"(?m)^## (\d+)\. ", markdown)
    ]
    assert sections == list(range(1, 17)), sections

    source_definitions = {
        int(value)
        for value in re.findall(r"(?m)^### S(\d{2})\. ", markdown)
    }
    source_references = {
        int(value)
        for value in re.findall(r"\[S(\d{2})(?:,|\])", markdown)
    }
    assert source_definitions == set(range(1, 24)), source_definitions
    assert source_references <= source_definitions

    rendered_pages = sorted(render_dir.glob("page-*.png"))
    assert len(rendered_pages) == 21
    for image_path in rendered_pages:
        with Image.open(image_path).convert("RGB") as image:
            background = Image.new("RGB", image.size, "white")
            content_box = ImageChops.difference(image, background).getbbox()
            assert content_box is not None, image_path
            left, top, right, bottom = content_box
            assert left >= 8 and top >= 8
            assert right <= image.width - 8 and bottom <= image.height - 8

    outline_count = _flatten_outline(reader.outline)
    print(
        "PASS:"
        f" {len(reader.pages)} pages,"
        f" {len(rendered_pages)} rendered pages,"
        f" {len(source_definitions)} sources,"
        f" {outline_count} outline entries"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
