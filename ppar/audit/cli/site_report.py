"""Write Audit reports for a configured site folder."""

from __future__ import annotations

# Python imports
import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Final

# Third-party imports
import polars as pl

# Project imports
from ppar.errors import PpaError
from ppar.audit import (
    write_audit_report_bundle,
)
from ppar.audit import review_model as _pc_review_model
from ppar.audit import source_loader
from ppar.audit import workbook_tables as _pc_workbook_tables
from ppar.audit.data_issues import checks as _data_issue_checks
from ppar.audit.runner import AuditComparisonViews
from ppar.audit.specification import (
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
_CSV_REVIEW_ARTIFACTS: Final[tuple[str, ...]] = (
    _pc_review_model.PERFORMANCE_DIFFERENCES_ARTIFACT,
    _pc_review_model.PERFORMANCE_DIFFERENCE_CAUSES_ARTIFACT,
    _pc_review_model.DATA_ISSUES_ARTIFACT,
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
    include_html_output: bool
    expand_all_supporting_files: bool


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
            include_workbook=not args.no_xlsx_output,
            include_html_output=not args.no_html_output,
            expand_all_supporting_files=args.expand_all_supporting_files,
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
    include_html_output: bool = True,
    expand_all_supporting_files: bool = False,
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
        include_workbook: Whether to write the level-specific XLSX audit.
        include_html_output: Whether to write the browser HTML review report.
            When both primary presentation formats are disabled, Audit promotes
            its canonical CSV review tables instead.
        expand_all_supporting_files: Whether to retain individual supporting
            CSV and JSON files instead of the default compact ZIP archive.

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
    data_issues = _data_issue_checks.data_issues_table(config_path)
    comparison_views = AuditComparisonViews(
        config_path,
        include_suppressed=not exclude_suppressed,
        require_causal_attribution=require_causal_attribution,
    )
    reconstruction_cache = _pc_workbook_tables.WorkbookReconstructionCache(
        config_path
    )
    if report in ("both", PORTFOLIO_COMPARISON_LEVEL):
        result["portfolio_report_paths"] = _write_report_bundle(
            config_path,
            comparison_views.findings(PORTFOLIO_COMPARISON_LEVEL),
            output_root / PORTFOLIO_COMPARISON_LEVEL,
            comparison_level=PORTFOLIO_COMPARISON_LEVEL,
            title=title,
            top_evidence_limit=top_evidence_limit,
            include_reconstruction_diagnostics=include_reconstruction_diagnostics,
            require_causal_attribution=require_causal_attribution,
            allow_incomplete_yaml=allow_incomplete_yaml,
            _data_issues=data_issues,
            include_workbook=include_workbook,
            include_html_output=include_html_output,
            expand_all_supporting_files=expand_all_supporting_files,
            _reconstruction_cache=reconstruction_cache,
        )
        result["review_paths"].extend(result["portfolio_report_paths"])
    if report in ("both", SECURITY_COMPARISON_LEVEL):
        try:
            result["security_report_paths"] = _write_report_bundle(
                config_path,
                comparison_views.findings(SECURITY_COMPARISON_LEVEL),
                output_root / SECURITY_COMPARISON_LEVEL,
                comparison_level=SECURITY_COMPARISON_LEVEL,
                title=title,
                top_evidence_limit=top_evidence_limit,
                include_reconstruction_diagnostics=include_reconstruction_diagnostics,
                require_causal_attribution=require_causal_attribution,
                allow_incomplete_yaml=allow_incomplete_yaml,
                _data_issues=data_issues,
                include_workbook=include_workbook,
                include_html_output=include_html_output,
                expand_all_supporting_files=expand_all_supporting_files,
                _reconstruction_cache=reconstruction_cache,
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
        description="Write Audit report bundles for a site setup.",
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
        "--no-xlsx-output",
        action="store_true",
        help=(
            "Do not write the level-specific XLSX audit. HTML remains enabled "
            "unless --no-html-output is also supplied."
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
            "Add detailed return-reconstruction checks to the audit workbook, HTML "
            "report, and supporting CSVs. Useful for investigating how holdings and "
            "flows reproduce reported returns. Disabled by default."
        ),
    )
    parser.add_argument(
        "--no-html-output",
        action="store_true",
        help=(
            "Do not write the browser HTML audit. XLSX remains enabled unless "
            "--no-xlsx-output is also supplied. Supplying both options writes "
            "a CSV-only audit."
        ),
    )
    parser.add_argument(
        "--expand-all-supporting-files",
        action="store_true",
        help=(
            "Write every supporting CSV and JSON file under supporting_files "
            "instead of the default audit_support.zip archive."
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
    """Return the explicit or conventional Audit site directory."""
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
        include_workbook=not arguments.no_xlsx_output,
        include_html_output=not arguments.no_html_output,
        expand_all_supporting_files=arguments.expand_all_supporting_files,
    )


def is_missing_security_data(error: PpaError) -> bool:
    """Return whether security output failed because secperf is unavailable."""
    return _is_missing_security_data(error)


def _write_report_bundle(
    config_path: Path,
    findings: pl.DataFrame,
    output_directory: Path,
    *,
    comparison_level: str,
    title: str | None,
    top_evidence_limit: int,
    include_reconstruction_diagnostics: bool,
    require_causal_attribution: bool,
    allow_incomplete_yaml: bool,
    _data_issues: Any,
    include_workbook: bool,
    include_html_output: bool,
    expand_all_supporting_files: bool,
    _reconstruction_cache: _pc_workbook_tables.WorkbookReconstructionCache,
) -> list[Path]:
    """Write one report bundle and return its primary review paths."""
    report_title = title or (
        "Portfolio Audit Report"
        if comparison_level == PORTFOLIO_COMPARISON_LEVEL
        else "Security Audit Report"
    )
    paths = write_audit_report_bundle(
        findings,
        output_directory,
        title=report_title,
        top_evidence_limit=top_evidence_limit,
        include_workbook=include_workbook,
        include_html_output=include_html_output,
        require_complete_yaml_setup=not allow_incomplete_yaml,
        require_causal_attribution=require_causal_attribution,
        comparison_path=config_path,
        comparison_level=comparison_level,
        include_reconstruction_diagnostics=include_reconstruction_diagnostics,
        _data_issues=_data_issues,
        _reconstruction_cache=_reconstruction_cache,
        expand_all_supporting_files=expand_all_supporting_files,
    )
    review_paths: list[Path] = []
    workbook = paths.get(_pc_review_model.REVIEW_WORKBOOK_ARTIFACT)
    if include_workbook and workbook is None:
        workbook_name = _pc_review_model.review_workbook_file_name(comparison_level)
        raise PpaError(
            f"Report bundle did not write {workbook_name} in {output_directory}.",
            999,
        )
    if workbook is not None:
        review_paths.append(workbook)
    html_report = paths.get(_pc_review_model.HTML_REPORT_ARTIFACT)
    if include_html_output and html_report is None:
        html_name = _pc_review_model.html_report_file_name(comparison_level)
        raise PpaError(
            f"Report bundle did not write {html_name} in {output_directory}.",
            999,
        )
    if html_report is not None:
        review_paths.append(html_report)
    if not include_workbook and not include_html_output:
        for artifact_name in _CSV_REVIEW_ARTIFACTS:
            csv_path = paths.get(artifact_name)
            if csv_path is None:
                raise PpaError(
                    f"CSV-only report bundle did not write {artifact_name}.csv "
                    f"in {output_directory}.",
                    999,
                )
            review_paths.append(csv_path)
    if review_paths:
        return review_paths
    raise PpaError("Report bundle did not write primary review output.", 999)


def _is_missing_security_data(error: PpaError) -> bool:
    """Return whether a security report failed because secperf is absent."""
    message = str(error)
    return (
        "files.security_performance" in message
        and ("is required" in message or "is missing" in message)
    )


def _print_success(result: dict[str, Any]) -> None:
    """Print a concise user handoff."""
    print("Open these files to review Audit output:")
    for path in result["review_paths"]:
        print(f"  {path}")
    if "security_status" in result:
        print()
        print("Security output skipped because files.security_performance is not available.")


if __name__ == "__main__":
    raise SystemExit(main())
