"""Render the root README as a screen-oriented landscape PDF."""

# The PDF and image renderers intentionally use the same self-contained Chrome
# discovery and isolation flags so either release-asset script can run independently.
# pylint: disable=duplicate-code

from __future__ import annotations

# Python imports
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

# Third-party imports
from markdown_it import MarkdownIt


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_README_PATH = _PROJECT_ROOT / "README.md"
_DEFAULT_OUTPUT_PATH = _PROJECT_ROOT / "PPAR.pdf"
_CHROME_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome",
    "chromium",
    "chrome",
)


def main() -> None:
    """Render the current root README to a Letter-landscape PDF."""
    args = _parse_args()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = _readme_html()

    with tempfile.TemporaryDirectory(prefix="ppar_readme_pdf_") as directory:
        temporary_directory = Path(directory)
        html_path = temporary_directory / "README.html"
        html_path.write_text(html, encoding="utf-8")
        temporary_pdf_path = temporary_directory / "PPAR.pdf"
        _print_pdf(_find_chrome(), html_path, temporary_pdf_path, temporary_directory)
        temporary_pdf_path.replace(output_path)

    print(output_path.relative_to(_PROJECT_ROOT))


def _parse_args() -> argparse.Namespace:
    """Return command-line arguments for README PDF rendering."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--output",
        default=_DEFAULT_OUTPUT_PATH,
        help="Destination PDF path. Defaults to PPAR.pdf in the project root.",
    )
    return parser.parse_args()


def _readme_html() -> str:
    """Return a standalone, GitHub-like HTML rendering of the root README."""
    markdown = _README_PATH.read_text(encoding="utf-8")
    body = MarkdownIt("commonmark", {"html": True}).render(markdown)
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<base href="{_PROJECT_ROOT.as_uri()}/">
<style>
@page {{ size: Letter landscape; margin: 0.3in; }}
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; }}
body {{
  color: #1f2328;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 15px;
  line-height: 1.5;
}}
h1, h2, h3 {{ break-after: avoid-page; }}
h1 {{ border-bottom: 1px solid #d0d7de; font-size: 30px; padding-bottom: 6px; }}
h2 {{ border-bottom: 1px solid #d0d7de; font-size: 24px; padding-bottom: 5px; }}
h3 {{ font-size: 19px; }}
p, li {{ orphans: 3; widows: 3; }}
a {{ color: #0969da; text-decoration: none; }}
code {{ background: #eff1f3; border-radius: 4px; padding: 2px 4px; }}
pre {{ background: #f6f8fa; padding: 12px; white-space: pre-wrap; }}
img {{ display: block; height: auto; margin: 12px auto; max-width: 100%; }}
hr {{ border: 0; border-top: 1px solid #d0d7de; margin: 22px 0; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def _print_pdf(
    chrome_path: str,
    html_path: Path,
    pdf_path: Path,
    temporary_directory: Path,
) -> None:
    """Print one local HTML document to PDF with headless Chrome."""
    command = [
        chrome_path,
        "--headless=new",
        "--disable-gpu",
        "--disable-background-networking",
        "--disable-component-update",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-sync",
        "--no-pdf-header-footer",
        f"--user-data-dir={temporary_directory / 'chrome_profile'}",
        f"--print-to-pdf={pdf_path}",
        html_path.as_uri(),
    ]
    try:
        subprocess.run(command, check=True, timeout=30)
    except subprocess.TimeoutExpired:
        if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
            raise
    if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
        raise OSError("Chrome did not create the README PDF")


def _find_chrome() -> str:
    """Return the available Chrome or Chromium executable."""
    for candidate in _CHROME_CANDIDATES:
        path = shutil.which(candidate) if "/" not in candidate else candidate
        if path and Path(path).exists():
            return path
    raise RuntimeError("Could not find Chrome or Chromium for PDF rendering.")


if __name__ == "__main__":
    main()
