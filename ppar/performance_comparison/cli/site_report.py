"""Write Performance Auditing reports for a configured site folder."""

from __future__ import annotations

# Python imports
import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Final

# Project imports
from ppar.errors import PpaError
from ppar.performance_comparison import (
    compare_snapshots,
    write_performance_comparison_report_bundle,
)
from ppar.performance_comparison import source_loader
from ppar.performance_comparison import x_ref as _pc_x_ref
from ppar.performance_comparison.specification import (
    PORTFOLIO_COMPARISON_LEVEL,
    SECURITY_COMPARISON_LEVEL,
)

_CONFIG_FILE_NAME: Final[str] = "ppar.yaml"
_OUTPUT_DIR: Final[str] = "output"
_DEFAULT_SITE_DIRECTORY: Final[str] = "audit"
_REPORT_CHOICES: Final[tuple[str, ...]] = (
    PORTFOLIO_COMPARISON_LEVEL,
    SECURITY_COMPARISON_LEVEL,
    "both",
)


@dataclass(frozen=True)
class AuditRunSettings:
    """Resolved presentation and validation settings for one audit run."""

    report: str
    output_directory: Path
    title: str | None
    exclude_suppressed: bool
    include_reconstruction_diagnostics: bool
    require_causal_attribution: bool
    allow_incomplete_yaml: bool
    include_workbook: bool


def main(
    argv: list[str] | None = None,
    *,
    prog: str = "ppar audit",
) -> int:
    """Run the site report command.

    Args:
        argv: Optional command-line arguments excluding the executable name.

    Returns:
        Process exit code. ``0`` indicates that requested report bundles were
        written.
    """
    args = _argument_parser(prog=prog).parse_args(argv)
    try:
        result = run_report(
            _default_site_directory(args.site_directory),
            report=args.report,
            output_directory=args.output,
            title=args.title,
            exclude_suppressed=args.exclude_suppressed,
            include_reconstruction_diagnostics=args.include_reconstruction_diagnostics,
            require_causal_attribution=args.require_causal_attribution,
        allow_incomplete_yaml=args.allow_incomplete_yaml,
        include_workbook=not args.no_xlsx,
        )
    except PpaError as error:
        print(f"Report failed: {error}", file=sys.stderr)
        return 1

    _print_success(result)
    return 0


@source_loader.source_frame_cache()
def run_report(
    site_directory: Path | str,
    *,
    report: str = "both",
    output_directory: Path | None = None,
    title: str | None = None,
    top_evidence_limit: int = 10,
    exclude_suppressed: bool = False,
    include_reconstruction_diagnostics: bool = False,
    require_causal_attribution: bool = False,
    allow_incomplete_yaml: bool = False,
    include_workbook: bool = True,
) -> dict[str, Any]:
    """Write one or more report bundles for a configured site folder.

    Args:
        site_directory: Folder containing ``ppar.yaml``. Accepts a ``Path`` or
            string path.
        report: Report family to generate: ``"portfolio"``, ``"security"``,
            or ``"both"``. Defaults to ``"both"``.
        output_directory: Optional base output directory override.
        title: Optional report-title override.
        top_evidence_limit: Maximum top-evidence rows per performance period.
        exclude_suppressed: Whether to exclude suppressed findings.
        include_reconstruction_diagnostics: Whether to add reconstruction
            diagnostic sheets, sections, and CSV artifacts.
        require_causal_attribution: Whether supported causal attribution setup
            must be complete.
        allow_incomplete_yaml: Whether diagnostic output may bypass the complete
            YAML setup guardrail.
        include_workbook: Whether to write ``report.xlsx`` in addition to HTML
            and supporting files.

    Returns:
        Paths for the site folder, config file, and generated review artifacts.

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
        raise PpaError(f"{site_path} is not a directory.", 802)
    config_path = site_path / _CONFIG_FILE_NAME
    if not config_path.exists():
        raise PpaError(
            f"{config_path} is missing. Run from the audit folder "
            "or pass the folder. For first-time setup, run: "
            "ppar setup ./my_ppar_data",
            802,
        )

    result: dict[str, Any] = {
        "site_directory": site_path,
        "config_path": config_path,
        "review_paths": [],
    }
    output_root = output_directory or site_path / _OUTPUT_DIR
    x_ref_issues = _pc_x_ref.x_ref_issues_table(config_path)
    if report in ("both", PORTFOLIO_COMPARISON_LEVEL):
        result["portfolio_report_paths"] = _write_report_bundle(
            config_path,
            output_root / PORTFOLIO_COMPARISON_LEVEL,
            comparison_level=PORTFOLIO_COMPARISON_LEVEL,
            title=title,
            top_evidence_limit=top_evidence_limit,
            exclude_suppressed=exclude_suppressed,
            include_reconstruction_diagnostics=include_reconstruction_diagnostics,
            require_causal_attribution=require_causal_attribution,
            allow_incomplete_yaml=allow_incomplete_yaml,
            _x_ref_issues=x_ref_issues,
            include_workbook=include_workbook,
        )
        result["review_paths"].extend(result["portfolio_report_paths"])
    if report in ("both", SECURITY_COMPARISON_LEVEL):
        try:
            result["security_report_paths"] = _write_report_bundle(
                config_path,
                output_root / SECURITY_COMPARISON_LEVEL,
                comparison_level=SECURITY_COMPARISON_LEVEL,
                title=title,
                top_evidence_limit=top_evidence_limit,
                exclude_suppressed=exclude_suppressed,
                include_reconstruction_diagnostics=include_reconstruction_diagnostics,
                require_causal_attribution=require_causal_attribution,
                allow_incomplete_yaml=allow_incomplete_yaml,
                _x_ref_issues=x_ref_issues,
                include_workbook=include_workbook,
            )
            result["review_paths"].extend(result["security_report_paths"])
        except PpaError as error:
            if report == SECURITY_COMPARISON_LEVEL or not _is_missing_security_data(error):
                raise
            result["security_status"] = (
                "skipped because files.security_performance is not available"
            )
    return result


def _argument_parser(
    *,
    prog: str = "ppar audit",
    include_site_directory: bool = True,
) -> argparse.ArgumentParser:
    """Return the site report argument parser."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Write Performance Auditing report bundles for a site setup.",
        epilog=(
            (
                "Examples:\n"
                "  ppar audit ./my_ppar_data/audit\n"
                "  ppar audit --report portfolio"
            )
            if include_site_directory
            else (
                "Examples:\n"
                "  python run_audit.py\n"
                "  python run_audit.py --report portfolio"
            )
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    if include_site_directory:
        parser.add_argument(
            "site_directory",
            nargs="?",
            type=Path,
            help="Folder containing ppar.yaml. Defaults to the current folder.",
        )
    parser.add_argument(
        "--report",
        choices=_REPORT_CHOICES,
        default="both",
        help="Report family to generate. Defaults to both.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory. Defaults to audit/output.",
    )
    parser.add_argument(
        "--title",
        help="Visible report title. Defaults to the standard portfolio/security title.",
    )
    parser.add_argument(
        "--no-xlsx",
        action="store_true",
        help=(
            "Skip report.xlsx and write report.html plus supporting files. "
            "Use this for faster, lighter runs when HTML and CSV output are "
            "sufficient. Disabled by default."
        ),
    )
    parser.add_argument(
        "--exclude_suppressed",
        action="store_true",
        help=(
            "Omit findings marked suppressed by ppar.yaml rules from the bundle. "
            "Use this to focus review output on findings that still require "
            "attention. Source-data is still processed. Disabled by default."
        ),
    )
    parser.add_argument(
        "--include-reconstruction-diagnostics",
        action="store_true",
        help=(
            "Add detailed return-reconstruction checks to report.xlsx, report.html, "
            "and supporting CSVs. Useful for investigating how holdings and flows "
            "reproduce reported returns. Disabled by default."
        ),
    )
    parser.add_argument(
        "--require-causal-attribution",
        action="store_true",
        help=(
            "Make validation stricter: fail when a changed period lacks YAML setup "
            "required by a supported explanation method. This does not require every "
            "difference to be fully explained. Disabled by default."
        ),
    )
    parser.add_argument(
        "--allow-incomplete-yaml",
        action="store_true",
        help=(
            "Relax validation: allow a diagnostic report when changed fields lack "
            "explicit additive, evidence-only, or suppression treatment in ppar.yaml. "
            "The result may be incomplete and is not a finalized audit. Disabled by "
            "default."
        ),
    )
    return parser


def _default_site_directory(site_directory: Path | None) -> Path:
    """Return the explicit or conventional Performance Auditing site directory."""
    if site_directory is not None:
        return site_directory
    return Path.cwd()


def script_run_settings(
    site_directory: Path,
    argv: list[str] | None = None,
) -> AuditRunSettings:
    """Resolve script arguments using the same rules as ``ppar audit``."""
    arguments = _argument_parser(
        prog="python run_audit.py",
        include_site_directory=False,
    ).parse_args(argv)
    return AuditRunSettings(
        report=arguments.report,
        output_directory=arguments.output or site_directory / _OUTPUT_DIR,
        title=arguments.title,
        exclude_suppressed=arguments.exclude_suppressed,
        include_reconstruction_diagnostics=(
            arguments.include_reconstruction_diagnostics
        ),
        require_causal_attribution=arguments.require_causal_attribution,
        allow_incomplete_yaml=arguments.allow_incomplete_yaml,
        include_workbook=not arguments.no_xlsx,
    )


def is_missing_security_data(error: PpaError) -> bool:
    """Return whether security output failed because secperf is unavailable."""
    return _is_missing_security_data(error)


def _write_report_bundle(
    config_path: Path,
    output_directory: Path,
    *,
    comparison_level: str,
    title: str | None,
    top_evidence_limit: int,
    exclude_suppressed: bool,
    include_reconstruction_diagnostics: bool,
    require_causal_attribution: bool,
    allow_incomplete_yaml: bool,
    _x_ref_issues: Any,
    include_workbook: bool,
) -> list[Path]:
    """Write one report bundle and return the primary workbook path."""
    findings = compare_snapshots(
        config_path,
        include_suppressed=not exclude_suppressed,
        require_causal_attribution=require_causal_attribution,
        comparison_level=comparison_level,
    )
    report_title = title or (
        "Portfolio Performance Auditing Report"
        if comparison_level == PORTFOLIO_COMPARISON_LEVEL
        else "Security Performance Auditing Report"
    )
    paths = write_performance_comparison_report_bundle(
        findings,
        output_directory,
        title=report_title,
        top_evidence_limit=top_evidence_limit,
        include_workbook=include_workbook,
        require_complete_yaml_setup=not allow_incomplete_yaml,
        require_causal_attribution=require_causal_attribution,
        comparison_path=config_path,
        comparison_level=comparison_level,
        include_reconstruction_diagnostics=include_reconstruction_diagnostics,
        _x_ref_issues=_x_ref_issues,
    )
    workbook = paths.get("review_workbook")
    if include_workbook and workbook is None:
        raise PpaError(f"Report bundle did not write report.xlsx in {output_directory}.", 999)
    html_report = paths.get("html_report")
    if html_report is None:
        raise PpaError(f"Report bundle did not write report.html in {output_directory}.", 999)
    return [workbook] if workbook is not None else [html_report]


def _is_missing_security_data(error: PpaError) -> bool:
    """Return whether a security report failed because secperf is absent."""
    message = str(error)
    return (
        "files.security_performance" in message
        and ("is required" in message or "is missing" in message)
    )


def _print_success(result: dict[str, Any]) -> None:
    """Print a concise user handoff."""
    print("Open these files to review Performance Auditing output:")
    for path in result["review_paths"]:
        print(f"  {path}")
    if "security_status" in result:
        print()
        print("Security output skipped because files.security_performance is not available.")


if __name__ == "__main__":
    raise SystemExit(main())
