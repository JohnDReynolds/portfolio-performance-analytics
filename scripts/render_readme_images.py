"""Render README chart and table images from packaged demo data.

The script uses bundled demo data, headless Chrome, and Pillow cropping to
refresh the chart images and table screenshots referenced by ``README.md``.
"""

# This script is meant to run directly from the repository checkout. Insert the
# repository root before importing ppar so the local source tree is used even
# when the package has not been installed. The ppar imports below therefore
# intentionally sit after executable bootstrap code; `noqa: E402` suppresses
# the "module import not at top of file" warning for those lines.
# pylint: disable=wrong-import-order,wrong-import-position

# Python Imports
import argparse
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Sequence

# Third-Party Imports
from PIL import Image, ImageChops

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
Image.MAX_IMAGE_PIXELS = None
_CACHE_DIR = _REPO_ROOT / "_demo_output" / "readme_image_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR / "xdg"))

# Project Imports
from ppar.analytics import Analytics  # noqa: E402
from ppar.analytics.attribution import Attribution, Chart, View  # noqa: E402
from ppar.analytics.frequency import Frequency  # noqa: E402
import ppar.utilities as util  # noqa: E402

_IMAGE_DIR = _REPO_ROOT / "docs" / "images" / "readme"
_GENERIC_ANALYTICS_TEMPLATE_DIR = _REPO_ROOT / "ppar" / "setup_templates" / "generic_analytics"
_CHROME_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome",
    "chromium",
    "chrome",
)
_RENDER_CONFIG = {
    "OverallAttributionBySecurity": (5200, 7600),
    "CumulativeAttributionByEconomicSector": (5200, 3600),
    "OverallAttributionByEconomicSector": (5200, 3200),
    "RiskStatistics": (3000, 4800),
    "PerformanceAuditPortfolio": (3200, 4400),
}
_PORTFOLIO_PERFORMANCE_AUDIT_HTML = (
    _REPO_ROOT / "_demo_output" / "performance_comparison_portfolio" / "report.html"
)
_PORTFOLIO_PERFORMANCE_AUDIT_SECTIONS = (
    ("performance-differences", "Performance Differences"),
    ("performance-difference-causes", "Performance Difference Causes"),
    ("data-audit-issues", "Data Audit Issues"),
)


def main() -> None:
    """Render all README table screenshots.

    Raises:
        RuntimeError: If Chrome or Chromium cannot be found.
        subprocess.CalledProcessError: If Chrome cannot render a screenshot.
        OSError: If temporary or generated image files cannot be read or
            written.
    """
    args = _parse_args()
    chrome_path = _find_chrome()
    with tempfile.TemporaryDirectory(prefix="ppar_readme_images_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        if args.only in ("all", "analytics"):
            analytics, sector = _analytics_outputs()
            _write_chart_images(sector)
            html_paths = _write_html_inputs(temp_dir, analytics, sector)
            for name, html_path in html_paths.items():
                _render_cropped_jpg(chrome_path, html_path, temp_dir, name)
        if args.only in ("all", "performance-comparison"):
            html_inputs = {
                "PerformanceAuditPortfolio": _write_report_sections_input(
                    _PORTFOLIO_PERFORMANCE_AUDIT_HTML,
                    temp_dir / "PerformanceAuditPortfolio.html",
                    sections=_PORTFOLIO_PERFORMANCE_AUDIT_SECTIONS,
                    title="Portfolio Performance Audit",
                    extra_style="""
                    <style>
                    html,
                    body {
                      background: #ffffff;
                    }
                    .pc-col-review-key {
                      display: none;
                    }
                    </style>
                    """,
                )
            }
            for name, html_path in html_inputs.items():
                _render_cropped_jpg(chrome_path, html_path, temp_dir, name)


def _parse_args() -> argparse.Namespace:
    """Parse README image-rendering arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Render README analytics and performance-comparison images.",
    )
    parser.add_argument(
        "--only",
        choices=("all", "analytics", "performance-comparison"),
        default="all",
        help="Limit rendering to one image family. Defaults to all.",
    )
    return parser.parse_args()


def _analytics_outputs() -> tuple[Analytics, Attribution]:
    """Return Analytics and sector-attribution outputs for README rendering.

    Returns:
        The packaged Mega-Cap analytics object and its economic-sector
        attribution output.

    Raises:
        PpaError: If construction of demonstration analytics output fails.
    """
    analytics = Analytics(
        _performance_data_source("Mega-Cap Alpha Portfolio.csv"),
        _performance_data_source("Mega-Cap Benchmark.csv"),
        portfolio_classification_name="Security",
        benchmark_classification_name="Security",
        frequency=Frequency.QUARTERLY,
    )
    classification_name = "Economic Sector"
    sector = analytics.get_attribution(
        classification_name,
        _classification_data_source(classification_name),
        _mapping_data_sources(analytics, classification_name),
    )
    return analytics, sector


def _performance_data_source(file_name: str) -> Path:
    """Return one packaged generic analytics performance CSV path."""
    return _GENERIC_ANALYTICS_TEMPLATE_DIR / "performance" / file_name


def _classification_data_source(classification_name: str) -> Path:
    """Return one packaged generic analytics classification CSV path."""
    return _GENERIC_ANALYTICS_TEMPLATE_DIR / "classifications" / f"{classification_name}.csv"


def _mapping_data_sources(
    analytics: Analytics,
    to_classification_name: str,
) -> tuple[Path | None, Path | None]:
    """Return packaged mapping CSV paths for README analytics rendering."""
    mapping_paths: list[Path | None] = []
    for from_classification_name in analytics.classification_names():
        if from_classification_name == to_classification_name:
            mapping_paths.append(None)
        else:
            mapping_paths.append(
                _GENERIC_ANALYTICS_TEMPLATE_DIR
                / "mappings"
                / f"{from_classification_name}--to--{to_classification_name}.csv"
            )
    return (mapping_paths[0], mapping_paths[1])


def _write_chart_images(sector: Attribution) -> None:
    """Write README chart PNG images from sector attribution output.

    Args:
        sector: Attribution output grouped by Economic Sector.

    Raises:
        OSError: If generated image files cannot be written.
        PpaError: If chart rendering fails.
    """
    chart_files = {
        Chart.OVERALL_ATTRIBUTION: "OverallAttributionByEconomicSector.png",
        Chart.OVERALL_CONTRIBUTION: "OverallContributionByEconomicSector.png",
        Chart.SUBPERIOD_ATTRIBUTION: "SubPeriodAttributionEffectsByEconomicSector.png",
        Chart.SUBPERIOD_RETURN: "SubPeriodReturns.png",
        Chart.HEATMAP_ACTIVE_CONTRIBUTION: "ActiveContributionsByEconomicSector.png",
        Chart.HEATMAP_ATTRIBUTION: "TotalAttributionEffectsByEconomicSector.png",
        Chart.CUMULATIVE_ATTRIBUTION: "CumulativeAttributionEffectsByEconomicSector.png",
        Chart.CUMULATIVE_RETURN: "CumulativeReturns.png",
    }
    for chart, file_name in chart_files.items():
        path = _IMAGE_DIR / file_name
        path.write_bytes(sector.to_chart(chart))
        print(f"{path.relative_to(_REPO_ROOT)}")


def _write_html_inputs(
    temp_dir: Path,
    analytics: Analytics,
    sector: Attribution,
) -> dict[str, Path]:
    """Write temporary HTML inputs for the README table images.

    Args:
        temp_dir: Temporary directory in which to write HTML files.
        analytics: Packaged Mega-Cap analytics output.
        sector: Attribution output grouped by Economic Sector.

    Returns:
        Mapping from README image stem to its temporary HTML input path.

    Raises:
        OSError: If an HTML input file cannot be written.
        PpaError: If construction of demonstration analytics output fails.
    """
    security = analytics.get_attribution()
    html_by_name = {
        "OverallAttributionBySecurity": security.to_html(View.OVERALL_ATTRIBUTION),
        "CumulativeAttributionByEconomicSector": sector.to_html(View.CUMULATIVE_ATTRIBUTION),
        "OverallAttributionByEconomicSector": sector.to_html(View.OVERALL_ATTRIBUTION),
        "RiskStatistics": analytics.get_riskstatistics().to_html(),
    }

    html_paths: dict[str, Path] = {}
    for name, html in html_by_name.items():
        html_path = temp_dir / f"{name}.html"
        with io.open(html_path, "w", encoding=util.ENCODING, newline="\n") as file:
            file.write(html)
        html_paths[name] = html_path
    return html_paths


def _render_png(
    chrome_path: str,
    html_path: Path,
    png_path: Path,
    window_size: Sequence[int],
    user_data_dir: Path,
) -> None:
    """Render one HTML file to PNG using headless Chrome.

    Args:
        chrome_path: Path or executable name for Chrome or Chromium.
        html_path: HTML document to render.
        png_path: PNG output path.
        window_size: Browser viewport width and height in pixels.
        user_data_dir: Isolated Chrome profile directory for this render.

    Raises:
        subprocess.CalledProcessError: If the Chrome rendering process fails.
    """
    command = [
        chrome_path,
        "--headless=new",
        "--disable-gpu",
        "--disable-background-networking",
        "--disable-component-update",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-sync",
        "--hide-scrollbars",
        "--force-device-scale-factor=2",
        f"--user-data-dir={user_data_dir}",
        f"--screenshot={png_path}",
        f"--window-size={window_size[0]},{window_size[1]}",
        html_path.resolve().as_uri(),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            timeout=30,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        if png_path.exists() and png_path.stat().st_size > 0:
            return
        raise


def _render_cropped_jpg(
    chrome_path: str,
    html_path: Path,
    temp_dir: Path,
    name: str,
) -> None:
    """Render one README HTML source and save it as a cropped JPEG.

    Args:
        chrome_path: Path or executable name for Chrome or Chromium.
        html_path: HTML document to render.
        temp_dir: Temporary directory in which to render the intermediate PNG.
        name: README image stem and render-configuration key.

    Raises:
        subprocess.CalledProcessError: If the Chrome rendering process fails.
        OSError: If temporary or generated image files cannot be read or
            written.
    """
    png_path = temp_dir / f"{name}.png"
    user_data_dir = temp_dir / f"{name}_chrome_profile"
    _render_png(
        chrome_path,
        html_path,
        png_path,
        _RENDER_CONFIG[name],
        user_data_dir,
    )
    _crop_and_save_jpg(png_path, _IMAGE_DIR / f"{name}.jpg")


def _write_report_sections_input(
    source_html_path: Path,
    destination_html_path: Path,
    *,
    sections: Sequence[tuple[str, str]],
    title: str,
    extra_style: str = "",
) -> Path:
    """Write a temporary HTML page containing selected report sections.

    Args:
        source_html_path: Existing report HTML file to read.
        destination_html_path: Temporary HTML file to write.
        sections: Section HTML ids and titles to extract from the source report.
        title: Temporary page title.
        extra_style: Additional CSS to append for README-only presentation.

    Returns:
        Path to the temporary report-section HTML file.

    Raises:
        FileNotFoundError: If ``source_html_path`` does not exist.
        ValueError: If a requested section cannot be found.
        OSError: If source or destination HTML cannot be read or written.
    """
    source_html = source_html_path.read_text(encoding=util.ENCODING)
    style_html = _html_between(source_html, "<style>", "</style>") or ""
    section_html = "\n".join(
        _extract_report_section(source_html, source_html_path, section_id)
        for section_id, _section_title in sections
    )
    destination_html_path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html>",
                "<head>",
                f"<title>{title}</title>",
                style_html,
                extra_style,
                "</head>",
                "<body>",
                section_html,
                "</body>",
                "</html>",
            ]
        ),
        encoding=util.ENCODING,
    )
    return destination_html_path


def _extract_report_section(
    source_html: str,
    source_html_path: Path,
    section_id: str,
) -> str:
    """Return one report section from a generated HTML report."""
    section_marker = f'<section class="pc-section" id="{section_id}">'
    section_start = source_html.find(section_marker)
    if section_start < 0:
        raise ValueError(f"{source_html_path} does not contain section {section_id!r}")
    section_end = source_html.find("</section>", section_start)
    if section_end < 0:
        raise ValueError(f"{source_html_path} has an unterminated section {section_id!r}")
    return source_html[section_start : section_end + len("</section>")]


def _html_between(html: str, start_marker: str, end_marker: str) -> str | None:
    """Return inclusive HTML text between two markers, if present."""
    start = html.find(start_marker)
    if start < 0:
        return None
    end = html.find(end_marker, start)
    if end < 0:
        return None
    return html[start : end + len(end_marker)]


def _crop_and_save_jpg(png_path: Path, jpg_path: Path) -> None:
    """Crop screenshot margins and save a high-quality JPEG.

    Args:
        png_path: Rendered PNG screenshot to open.
        jpg_path: Destination JPEG path.

    Raises:
        OSError: If the source image cannot be opened or the destination image
            cannot be written.
    """
    image = Image.open(png_path).convert("RGB")
    background_color = image.getpixel((0, 0))
    background = Image.new("RGB", image.size, background_color)
    diff = ImageChops.difference(image, background)
    mask = diff.convert("L").point(lambda value: 255 if value > 6 else 0)
    bbox = mask.getbbox()
    cropped = image
    if bbox is not None:
        pad = 48
        cropped = image.crop(
            (
                max(0, bbox[0] - pad),
                max(0, bbox[1] - pad),
                min(image.width, bbox[2] + pad),
                min(image.height, bbox[3] + pad),
            )
        )

    cropped.save(jpg_path, quality=95, optimize=True)
    print(f"{jpg_path.relative_to(_REPO_ROOT)} {cropped.width}x{cropped.height}")


def _find_chrome() -> str:
    """Return the Chrome executable used for screenshot rendering.

    Returns:
        Existing Chrome or Chromium executable path or discoverable command.

    Raises:
        RuntimeError: If no configured Chrome or Chromium executable exists.
    """
    for candidate in _CHROME_CANDIDATES:
        path = shutil.which(candidate) if "/" not in candidate else candidate
        if path and Path(path).exists():
            return path
    raise RuntimeError("Could not find Chrome or Chromium for image rendering.")


if __name__ == "__main__":
    main()
