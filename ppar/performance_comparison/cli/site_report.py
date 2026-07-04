"""Write performance-comparison reports for a configured site folder."""

from __future__ import annotations

# Python imports
import argparse
from pathlib import Path
import sys
from typing import Final

# Project imports
from ppar.errors import PpaError
from ppar.performance_comparison import (
    compare_snapshots,
    write_performance_comparison_report_bundle,
)
from ppar.performance_comparison.specification import (
    PORTFOLIO_COMPARISON_LEVEL,
    SECURITY_COMPARISON_LEVEL,
)

_CONFIG_FILE_NAME: Final[str] = "ppar.yaml"
_OUTPUT_DIR: Final[str] = "output"
_REPORT_CHOICES: Final[tuple[str, ...]] = (
    PORTFOLIO_COMPARISON_LEVEL,
    SECURITY_COMPARISON_LEVEL,
    "both",
)


def main(argv: list[str] | None = None) -> int:
    """Run the site report command.

    Args:
        argv: Optional command-line arguments excluding the executable name.

    Returns:
        Process exit code. ``0`` indicates that requested report bundles were
        written.
    """
    args = _argument_parser().parse_args(argv)
    try:
        result = run_report(args.site_directory, report=args.report)
    except PpaError as error:
        print(f"Report failed: {error}", file=sys.stderr)
        return 1

    _print_success(result)
    return 0


def run_report(site_directory: Path, *, report: str = "both") -> dict[str, Path | str]:
    """Write one or more report bundles for a configured site folder.

    Args:
        site_directory: Folder containing ``ppar.yaml``.
        report: Report family to generate: ``"portfolio"``, ``"security"``,
            or ``"both"``. Defaults to ``"both"``.

    Returns:
        Paths for the site folder, config file, and generated workbooks.

    Raises:
        PpaError: If the site folder/config file is missing, the report family
            is invalid, or report generation fails.
    """
    if report not in _REPORT_CHOICES:
        raise PpaError(
            f"report must be one of: {', '.join(_REPORT_CHOICES)}.",
            504,
        )

    site_path = Path(site_directory).expanduser()
    if not site_path.is_dir():
        raise PpaError(f"{site_path} is not a directory. Run setup first.", 802)
    config_path = site_path / _CONFIG_FILE_NAME
    if not config_path.exists():
        raise PpaError(f"{config_path} is missing. Run setup first.", 802)

    result: dict[str, Path] = {
        "site_directory": site_path,
        "config_path": config_path,
    }
    if report in ("both", PORTFOLIO_COMPARISON_LEVEL):
        result["portfolio_report"] = _write_report_bundle(
            config_path,
            site_path / _OUTPUT_DIR / PORTFOLIO_COMPARISON_LEVEL,
            comparison_level=PORTFOLIO_COMPARISON_LEVEL,
        )
    if report in ("both", SECURITY_COMPARISON_LEVEL):
        try:
            result["security_report"] = _write_report_bundle(
                config_path,
                site_path / _OUTPUT_DIR / SECURITY_COMPARISON_LEVEL,
                comparison_level=SECURITY_COMPARISON_LEVEL,
            )
        except PpaError as error:
            if report == SECURITY_COMPARISON_LEVEL or not _is_missing_security_data(error):
                raise
            result["security_status"] = (
                "skipped because files.security_performance is not available"
            )
    return result


def _argument_parser() -> argparse.ArgumentParser:
    """Return the site report argument parser."""
    parser = argparse.ArgumentParser(
        description="Write performance-comparison report bundles for a site setup.",
    )
    parser.add_argument(
        "site_directory",
        type=Path,
        help="Folder containing ppar.yaml.",
    )
    parser.add_argument(
        "--report",
        choices=_REPORT_CHOICES,
        default="both",
        help="Report family to generate. Defaults to both.",
    )
    return parser


def _write_report_bundle(
    config_path: Path,
    output_directory: Path,
    *,
    comparison_level: str,
) -> Path:
    """Write one report bundle and return the workbook path."""
    findings = compare_snapshots(
        config_path,
        comparison_level=comparison_level,
    )
    title = (
        "Portfolio Performance Comparison"
        if comparison_level == PORTFOLIO_COMPARISON_LEVEL
        else "Security Performance Comparison"
    )
    paths = write_performance_comparison_report_bundle(
        findings,
        output_directory,
        title=title,
        include_workbook=True,
        comparison_path=config_path,
        comparison_level=comparison_level,
    )
    workbook = paths.get("review_workbook")
    if workbook is None:
        raise PpaError(f"Report bundle did not write report.xlsx in {output_directory}.", 999)
    return workbook


def _is_missing_security_data(error: PpaError) -> bool:
    """Return whether a security report failed because secperf is absent."""
    message = str(error)
    return (
        "files.security_performance" in message
        and ("is required" in message or "is missing" in message)
    )


def _print_success(result: dict[str, Path | str]) -> None:
    """Print a concise user handoff."""
    print("PPAR performance comparison complete")
    print(f"Site folder: {result['site_directory']}")
    print(f"Config: {result['config_path']}")
    if "portfolio_report" in result:
        print(f"Portfolio report: {result['portfolio_report']}")
    if "security_report" in result:
        print(f"Security report: {result['security_report']}")
    if "security_status" in result:
        print(f"Security report: {result['security_status']}")


if __name__ == "__main__":
    raise SystemExit(main())
