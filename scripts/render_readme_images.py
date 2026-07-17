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
from lxml import html as lxml_html
from lxml.html import HtmlElement
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
    # This report contains long, wrapping tables. A 1x, browser-sized viewport
    # keeps the text readable and avoids the very large bitmap produced by a
    # wide 2x screenshot.
    "PerformanceAuditPortfolio": (1440, 16000),
}
_DEVICE_SCALE_FACTOR_BY_NAME = {"PerformanceAuditPortfolio": 1}
_MINIMUM_CROPPED_WIDTH_BY_NAME = {"RiskStatistics": 2000}
_EXPECTED_HTML_MARKER_BY_NAME = {
    "OverallAttributionBySecurity": "Overall Attribution by Security",
    "CumulativeAttributionByEconomicSector": "Cumulative Attribution by Economic Sector",
    "OverallAttributionByEconomicSector": "Overall Attribution by Economic Sector",
    "RiskStatistics": "Ex-Post Risk Statistics",
    "PerformanceAuditPortfolio": "Performance Differences",
}
_FORBIDDEN_HTML_MARKER_BY_NAME = {
    "OverallAttributionBySecurity": "Portfolio Differences",
    "CumulativeAttributionByEconomicSector": "Portfolio Differences",
    "OverallAttributionByEconomicSector": "Portfolio Differences",
    "RiskStatistics": "Portfolio Differences",
}
_MINIMUM_IMAGE_SIZE_BY_NAME = {
    "OverallAttributionBySecurity": (2000, 1000),
    "CumulativeAttributionByEconomicSector": (2500, 1000),
    "OverallAttributionByEconomicSector": (1800, 900),
    "RiskStatistics": (1500, 2000),
    "PerformanceAuditPortfolio": (1200, 1000),
}
_PERFORMANCE_AUDIT_SCENARIOS = {
    ("ALPHA", "2026-01-31", "2026-02-27"),
    ("ALPHA", "2026-05-01", "2026-05-29"),
    ("BALANCED", "2026-04-01", "2026-04-30"),
    ("BALANCED", "2026-05-09", "2026-05-14"),
    ("INCOME", "2026-01-01", "2026-01-30"),
    ("INCOME", "2026-02-28", "2026-03-31"),
    ("INCOME", "2026-04-01", "2026-04-30"),
}
_PERFORMANCE_AUDIT_ISSUE_TYPES = {
    "missing_dividend",
    "holdings_accrued_rate",
    "pa_sa_rate",
    "transactions_price_range",
    "dividend_rate",
    "holdings_price_range",
}
_PORTFOLIO_PERFORMANCE_AUDIT_HTML = (
    _REPO_ROOT
    / "_demo_output"
    / "audit_portfolio"
    / "portfolio_audit.html"
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
        if args.only == "security-attribution":
            analytics, sector = _analytics_outputs()
            html_path = _write_html_inputs(temp_dir, analytics, sector)[
                "OverallAttributionBySecurity"
            ]
            _render_cropped_jpg(
                chrome_path,
                html_path,
                temp_dir,
                "OverallAttributionBySecurity",
            )
        if args.only == "risk-statistics":
            analytics, _sector = _analytics_outputs()
            html_path = temp_dir / "RiskStatistics.html"
            html_path.write_text(
                analytics.get_riskstatistics().to_html(),
                encoding=util.ENCODING,
            )
            _render_cropped_jpg(chrome_path, html_path, temp_dir, "RiskStatistics")
        if args.only in ("all", "audit"):
            preview_html_path = temp_dir / "PerformanceAuditPortfolio.html"
            _write_performance_audit_preview(
                _PORTFOLIO_PERFORMANCE_AUDIT_HTML,
                preview_html_path,
            )
            _render_cropped_jpg(
                chrome_path,
                preview_html_path,
                temp_dir,
                "PerformanceAuditPortfolio",
            )


def _parse_args() -> argparse.Namespace:
    """Parse README image-rendering arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Render README Analytics and Audit images.",
    )
    parser.add_argument(
        "--only",
        choices=(
            "all",
            "analytics",
            "security-attribution",
            "risk-statistics",
            "audit",
        ),
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
    security = analytics.get_attribution(
        "Security",
        _classification_data_source("Security"),
    )
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
        if name == "OverallAttributionBySecurity":
            _write_security_attribution_preview(html_path)
        html_paths[name] = html_path
    return html_paths


def _write_security_attribution_preview(html_path: Path) -> None:
    """Trim the README-only Security table while retaining its Total row."""
    document = lxml_html.document_fromstring(html_path.read_text(encoding=util.ENCODING))
    tables = document.xpath("//table")
    if len(tables) != 1:
        raise ValueError(f"Security attribution preview expected one table, found {len(tables)}")
    body = tables[0].xpath("./tbody")
    if len(body) != 1:
        raise ValueError("Security attribution preview is missing its table body")
    rows = body[0].xpath("./tr")
    preview_row_count = 9
    if len(rows) <= (preview_row_count * 2) + 1:
        raise ValueError("Security attribution preview does not have enough rows to trim")

    security_rows = rows[:-1]
    total_row = rows[-1]
    retained_rows = [
        *security_rows[:preview_row_count],
        *security_rows[-preview_row_count:],
    ]
    for row in rows:
        body[0].remove(row)
    for row in retained_rows[:preview_row_count]:
        body[0].append(row)
    body[0].append(_security_ellipsis_row())
    for row in retained_rows[preview_row_count:]:
        body[0].append(row)
    body[0].append(total_row)

    html_path.write_text(
        lxml_html.tostring(document, encoding="unicode", doctype="<!doctype html>"),
        encoding=util.ENCODING,
    )


def _security_ellipsis_row() -> HtmlElement:
    """Return the explanatory omission row used by the Security preview."""
    row = lxml_html.Element("tr")
    cell = lxml_html.Element("td")
    cell.set("colspan", "14")
    cell.set("style", "font-style: italic; text-align: center;")
    cell.text = "… additional securities not shown …"
    row.append(cell)
    return row


def _render_png(
    chrome_path: str,
    html_path: Path,
    png_path: Path,
    window_size: Sequence[int],
    user_data_dir: Path,
    device_scale_factor: int = 2,
) -> None:
    """Render one HTML file to PNG using headless Chrome.

    Args:
        chrome_path: Path or executable name for Chrome or Chromium.
        html_path: HTML document to render.
        png_path: PNG output path.
        window_size: Browser viewport width and height in pixels.
        user_data_dir: Isolated Chrome profile directory for this render.
        device_scale_factor: Pixel density at which Chrome captures the page.

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
        f"--force-device-scale-factor={device_scale_factor}",
        f"--user-data-dir={user_data_dir}",
        f"--screenshot={png_path}",
        f"--window-size={window_size[0]},{window_size[1]}",
        html_path.resolve().as_uri(),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            timeout=120,
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
            written, or if validation rejects the source or rendered image.
    """
    _validate_html_source(html_path, name)
    png_path = temp_dir / f"{name}.png"
    temporary_jpg_path = temp_dir / f"{name}.jpg"
    destination_jpg_path = _IMAGE_DIR / f"{name}.jpg"
    user_data_dir = temp_dir / f"{name}_chrome_profile"
    _render_png(
        chrome_path,
        html_path,
        png_path,
        _RENDER_CONFIG[name],
        user_data_dir,
        _DEVICE_SCALE_FACTOR_BY_NAME.get(name, 2),
    )
    image_size = _crop_and_save_jpg(
        png_path,
        temporary_jpg_path,
        minimum_width=_MINIMUM_CROPPED_WIDTH_BY_NAME.get(name),
    )
    _validate_image_size(name, image_size)
    temporary_jpg_path.replace(destination_jpg_path)
    print(
        f"{destination_jpg_path.relative_to(_REPO_ROOT)} "
        f"{image_size[0]}x{image_size[1]}"
    )


def _write_performance_audit_preview(source_path: Path, destination_path: Path) -> None:
    """Write a shorter README preview without modifying the native report."""
    document = lxml_html.document_fromstring(source_path.read_text(encoding=util.ENCODING))
    differences = document.get_element_by_id("performance-differences")
    causes = document.get_element_by_id("performance-difference-causes")
    issues = document.get_element_by_id("data-audit-issues")

    _retain_scenario_rows(differences)
    _retain_scenario_rows(causes)
    _retain_one_row_per_issue_type(issues)
    destination_path.write_text(
        lxml_html.tostring(document, encoding="unicode", doctype="<!doctype html>"),
        encoding=util.ENCODING,
    )


def _retain_scenario_rows(section: HtmlElement) -> None:
    """Retain rows belonging to the selected complete review scenarios."""
    kept_rows = 0
    retained_scenarios: set[tuple[str, str, str]] = set()
    for row in section.xpath(".//tbody/tr"):
        values = [_normalized_cell_text(cell) for cell in row.xpath("./td")]
        scenario = (values[0], values[1], values[2])
        if scenario in _PERFORMANCE_AUDIT_SCENARIOS:
            kept_rows += 1
            retained_scenarios.add(scenario)
        else:
            row.getparent().remove(row)
    if retained_scenarios != _PERFORMANCE_AUDIT_SCENARIOS:
        missing = sorted(_PERFORMANCE_AUDIT_SCENARIOS - retained_scenarios)
        raise ValueError(f"Performance Audit preview is missing scenarios: {missing}")
    _set_section_row_count(section, kept_rows)


def _retain_one_row_per_issue_type(section: HtmlElement) -> None:
    """Retain one issue per type connected to a displayed review scenario."""
    retained_issue_types: set[str] = set()
    for row in section.xpath(".//tbody/tr"):
        values = [_normalized_cell_text(cell) for cell in row.xpath("./td")]
        issue_type = values[5]
        if (
            issue_type in _PERFORMANCE_AUDIT_ISSUE_TYPES
            and issue_type not in retained_issue_types
            and _issue_matches_selected_scenario(values)
        ):
            retained_issue_types.add(issue_type)
        else:
            row.getparent().remove(row)
    if retained_issue_types != _PERFORMANCE_AUDIT_ISSUE_TYPES:
        missing = sorted(_PERFORMANCE_AUDIT_ISSUE_TYPES - retained_issue_types)
        raise ValueError(f"Performance Audit preview is missing issue types: {missing}")
    _set_section_row_count(section, len(retained_issue_types))


def _issue_matches_selected_scenario(values: Sequence[str]) -> bool:
    """Return whether one Data Issues row belongs to a selected period."""
    portfolio_id = values[1]
    as_of_date = values[2]
    return any(
        portfolio_id == scenario_portfolio
        and from_date <= as_of_date <= thru_date
        for scenario_portfolio, from_date, thru_date in _PERFORMANCE_AUDIT_SCENARIOS
    )


def _normalized_cell_text(cell: HtmlElement) -> str:
    """Return one HTML table cell as normalized plain text."""
    return " ".join(cell.text_content().split())


def _set_section_row_count(section: HtmlElement, row_count: int) -> None:
    """Update a report section's visible row-count label."""
    labels = section.xpath('.//p[contains(@class, "pc-table-meta")]')
    if labels:
        labels[0].text = f"Rows: {row_count}"


def _validate_html_source(html_path: Path, name: str) -> None:
    """Reject an HTML source that does not match its requested image."""
    html = html_path.read_text(encoding=util.ENCODING)
    expected_marker = _EXPECTED_HTML_MARKER_BY_NAME[name]
    forbidden_marker = _FORBIDDEN_HTML_MARKER_BY_NAME.get(name)
    if expected_marker not in html:
        raise OSError(f"{name} source HTML is missing {expected_marker!r}")
    if forbidden_marker is not None and forbidden_marker in html:
        raise OSError(f"{name} source HTML unexpectedly contains {forbidden_marker!r}")


def _validate_image_size(name: str, image_size: tuple[int, int]) -> None:
    """Reject a rendered image that is implausibly small for its report."""
    minimum_width, minimum_height = _MINIMUM_IMAGE_SIZE_BY_NAME[name]
    width, height = image_size
    if width < minimum_width or height < minimum_height:
        raise OSError(
            f"{name} rendered at {width}x{height}; expected at least "
            f"{minimum_width}x{minimum_height}"
        )


def _crop_and_save_jpg(
    png_path: Path,
    jpg_path: Path,
    *,
    minimum_width: int | None = None,
) -> tuple[int, int]:
    """Crop screenshot margins and save a high-quality JPEG.

    Args:
        png_path: Rendered PNG screenshot to open.
        jpg_path: Destination JPEG path.
        minimum_width: Smallest retained image width. This prevents narrow
            reports from being enlarged when displayed at full container width.

    Returns:
        Width and height of the cropped JPEG.

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
        left = max(0, bbox[0] - pad)
        right = min(image.width, bbox[2] + pad)
        if minimum_width is not None and right - left < minimum_width:
            extra_width = minimum_width - (right - left)
            left = max(0, left - extra_width // 2)
            right = min(image.width, left + minimum_width)
            left = max(0, right - minimum_width)
        cropped = image.crop(
            (
                left,
                max(0, bbox[1] - pad),
                right,
                min(image.height, bbox[3] + pad),
            )
        )

    cropped.save(jpg_path, quality=95, optimize=True)
    return cropped.width, cropped.height


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
