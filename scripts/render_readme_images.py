"""Render README table images from the current HTML output.

The script uses bundled demo data, headless Chrome, and Pillow cropping to
refresh the table screenshots referenced by ``README.md``.
"""

# Imports below the repository path bootstrap are intentional for direct execution.
# pylint: disable=wrong-import-order,wrong-import-position

# Python Imports
import io
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Sequence

# Third-Party Imports
from PIL import Image, ImageChops

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
Image.MAX_IMAGE_PIXELS = None

# Project Imports
from ppar.analytics import Analytics  # noqa: E402
from ppar.attribution import View  # noqa: E402
import ppar.demo_data_sources as demo_data  # noqa: E402
from ppar.frequency import Frequency  # noqa: E402
import ppar.utilities as util  # noqa: E402

IMAGE_DIR = REPO_ROOT / "images"
CHROME_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome",
    "chromium",
    "chrome",
)
RENDER_CONFIG = {
    "OverallAttributionBySecurity": (5200, 7600),
    "CumulativeAttributionByEconomicSector": (5200, 3600),
    "OverallAttributionByEconomicSector": (5200, 3200),
    "RiskStatistics": (3000, 4800),
}


def main() -> None:
    """Render all README table screenshots."""
    chrome_path = _find_chrome()
    with tempfile.TemporaryDirectory(prefix="ppar_readme_images_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        html_paths = _write_html_inputs(temp_dir)
        for name, html_path in html_paths.items():
            png_path = temp_dir / f"{name}.png"
            _render_png(chrome_path, html_path, png_path, RENDER_CONFIG[name])
            _crop_and_save_jpg(png_path, IMAGE_DIR / f"{name}.jpg")


def _write_html_inputs(temp_dir: Path) -> dict[str, Path]:
    """Write temporary HTML files for the README table images."""
    analytics = Analytics(
        demo_data.performance_data_source("Large-Cap Alpha Portfolio.csv"),
        demo_data.performance_data_source("Large-Cap Benchmark.csv"),
        portfolio_classification_name="Security",
        benchmark_classification_name="Security",
        beginning_date="2022-12-31",
        ending_date="2024-02-29",
        frequency=Frequency.MONTHLY,
    )
    security = analytics.get_attribution()
    classification_name = "Economic Sector"
    sector = analytics.get_attribution(
        classification_name,
        demo_data.classification_data_source(classification_name),
        demo_data.mapping_data_sources(analytics, classification_name),
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
        html_paths[name] = html_path
    return html_paths


def _render_png(
    chrome_path: str,
    html_path: Path,
    png_path: Path,
    window_size: Sequence[int],
) -> None:
    """Render one HTML file to PNG using headless Chrome."""
    command = [
        chrome_path,
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        "--force-device-scale-factor=2",
        f"--screenshot={png_path}",
        f"--window-size={window_size[0]},{window_size[1]}",
        html_path.resolve().as_uri(),
    ]
    subprocess.run(command, check=True)


def _crop_and_save_jpg(png_path: Path, jpg_path: Path) -> None:
    """Crop white screenshot margins and save a high-quality JPG."""
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
    print(f"{jpg_path.relative_to(REPO_ROOT)} {cropped.width}x{cropped.height}")


def _find_chrome() -> str:
    """Return the Chrome executable used for screenshot rendering."""
    for candidate in CHROME_CANDIDATES:
        path = shutil.which(candidate) if "/" not in candidate else candidate
        if path and Path(path).exists():
            return path
    raise RuntimeError("Could not find Chrome or Chromium for image rendering.")


if __name__ == "__main__":
    main()
