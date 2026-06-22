"""Demonstrate Axys-backed analytics and attribution output."""

# Python imports
import datetime as dt
from importlib.resources import as_file, files
from pathlib import Path
import time

# Project imports
from ppar.analytics.attribution import View
from ppar.axys import AxysData
import ppar.utilities as util

_OUTPUT_DIRECTORY = Path("_demo_output") / "axys_analytics"


def main() -> None:
    """Run the Axys analytics demonstration.

    Raises:
        PpaError: If Axys source validation, reconciliation, or analytics
            calculations fail.
    """
    time_start = time.perf_counter()

    with as_file(files("ppar.demos.data") / "axys") as axys_data_root:
        axys_data = AxysData(axys_data_root / "axys_column_mappings.yaml")

        for portfolio_code in ("PORT_SMALL", "PORT_LARGE"):
            # Specify dates and classification.
            portfolio = axys_data.get_portfolio(
                portfolio_code,
                from_date=dt.date(2024, 1, 1),
                thru_date=dt.date(2025, 12, 31),
                classification_name="Sector",
            )
            analytics = portfolio.to_analytics()
            attribution = analytics.get_attribution()
            _ = attribution.to_html(View.OVERALL_ATTRIBUTION)

        # Default to the dates and classification in the YAML file.
        portfolio = axys_data.get_portfolio("PORT_SMALL")
        benchmark = axys_data.get_portfolio("PORT_LARGE")
        analytics = portfolio.to_analytics(benchmark)
        attribution = analytics.get_attribution()
        html = attribution.to_html(View.OVERALL_ATTRIBUTION)
        output_path = _write_html("overall_attribution.html", html)
        _print_demo_handoff(output_path)

    print("Time:", time.perf_counter() - time_start)


def _write_html(file_name: str, html: str) -> Path:
    """Write one Axys analytics demo HTML artifact."""
    path = _OUTPUT_DIRECTORY / file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding=util.ENCODING)
    return path


def _print_demo_handoff(path: Path) -> None:
    """Print generated Axys analytics demo artifact path."""
    print(f"Axys analytics demo output written to: {path.parent.resolve()}")
    print(f"Open {path.resolve()} to review the demo output.")


if __name__ == "__main__":
    main()
