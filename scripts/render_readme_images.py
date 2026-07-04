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
import ppar.demos.generic_analytics_data_sources as demo_data  # noqa: E402
from ppar.analytics.frequency import Frequency  # noqa: E402
import ppar.utilities as util  # noqa: E402

_IMAGE_DIR = _REPO_ROOT / "docs" / "images" / "readme"
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
}


def main() -> None:
    """Render all README table screenshots.

    Raises:
        RuntimeError: If Chrome or Chromium cannot be found.
        subprocess.CalledProcessError: If Chrome cannot render a screenshot.
        OSError: If temporary or generated image files cannot be read or
            written.
    """
    chrome_path = _find_chrome()
    with tempfile.TemporaryDirectory(prefix="ppar_readme_images_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        analytics, sector = _analytics_outputs()
        _write_chart_images(sector)
        html_paths = _write_html_inputs(temp_dir, analytics, sector)
        for name, html_path in html_paths.items():
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


def _analytics_outputs() -> tuple[Analytics, Attribution]:
    """Return Analytics and sector-attribution outputs for README rendering.

    Returns:
        The packaged Mega-Cap analytics object and its economic-sector
        attribution output.

    Raises:
        PpaError: If construction of demonstration analytics output fails.
    """
    analytics = Analytics(
        demo_data.performance_data_source("Mega-Cap Alpha Portfolio.csv"),
        demo_data.performance_data_source("Mega-Cap Benchmark.csv"),
        portfolio_classification_name="Security",
        benchmark_classification_name="Security",
        frequency=Frequency.QUARTERLY,
    )
    classification_name = "Economic Sector"
    sector = analytics.get_attribution(
        classification_name,
        demo_data.classification_data_source(classification_name),
        demo_data.mapping_data_sources(analytics, classification_name),
    )
    return analytics, sector


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


def _crop_and_save_jpg(png_path: Path, jpg_path: Path) -> None:
    """Crop white screenshot margins and save a high-quality JPEG.

    Args:
        png_path: Rendered PNG screenshot to open.
        jpg_path: Destination JPEG path.

    Raises:
        OSError: If the source image cannot be opened or the destination image
            cannot be written.
    """
    image = Image.open(png_path).convert("RGB")
    white = Image.new("RGB", image.size, (255, 255, 255))
    bbox = ImageChops.difference(image, white).getbbox()
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
