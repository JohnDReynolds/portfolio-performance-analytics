"""Write Audit reports for a configured site folder."""

from __future__ import annotations

# Python imports
import argparse
from pathlib import Path
import sys
from typing import Any, Final

# Third-party imports
import polars as pl
import yaml

# Project imports
from ppar.errors import PpaError
from ppar.audit import (
    write_audit_report_bundle,
)
from ppar.audit import review_model as _pc_review_model
from ppar.audit import source_loader
from ppar.audit import workbook_tables as _pc_workbook_tables
from ppar.audit import workbook_reconstruction as _pc_workbook_reconstruction
from ppar.audit.data_issues import checks as _data_issue_checks
from ppar.audit.runner import AuditComparisonViews
from ppar.audit.run_settings import (
    AuditRunSettings,
    audit_settings as _audit_settings,
    resolve_settings as _resolve_settings,
)
from ppar.audit.specification import (
    PORTFOLIO_COMPARISON_LEVEL,
    SECURITY_COMPARISON_LEVEL,
)
import ppar.utilities as util

_CONFIG_FILE_NAME: Final[str] = "ppar.yaml"
_CSV_REVIEW_ARTIFACTS: Final[tuple[str, ...]] = (
    _pc_review_model.PERFORMANCE_DIFFERENCES_ARTIFACT,
    _pc_review_model.PERFORMANCE_DIFFERENCE_CAUSES_ARTIFACT,
    _pc_review_model.DATA_ISSUES_ARTIFACT,
)


def main(
    argv: list[str] | None = None,
    *,
    prog: str = "ppar audit",
) -> int:
    """Run the site report command.

    Args:
        argv: Optional command-line arguments excluding the executable name.

    Returns:
        Process exit code. ``0`` indicates that available report bundles were
        written.
    """
    args = _argument_parser(prog=prog).parse_args(argv)
    try:
        result = run_report(
            _default_site_directory(args.site_directory),
            output_directory=args.output_directory,
            title=args.title,
            exclude_suppressed=args.exclude_suppressed,
            include_reconstruction_diagnostics=args.reconstruction_diagnostics,
            require_causal_attribution=args.require_causal_attribution,
            allow_incomplete_yaml=args.allow_incomplete_yaml,
            include_workbook=args.xlsx_output,
            include_html_output=args.html_output,
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
    output_directory: Path | None = None,
    title: str | None = None,
    top_evidence_limit: int = 10,
    exclude_suppressed: bool | None = None,
    include_reconstruction_diagnostics: bool | None = None,
    require_causal_attribution: bool | None = None,
    allow_incomplete_yaml: bool = False,
    include_workbook: bool | None = None,
    include_html_output: bool | None = None,
    expand_all_supporting_files: bool | None = None,
) -> dict[str, Any]:
    """Write one or more report bundles for a configured site folder.

    Args:
        site_directory: Folder containing ``ppar.yaml``. Accepts a ``Path`` or
            string path.
        output_directory: Optional one-run output-directory override.
        title: Optional one-run report-title override.
        top_evidence_limit: Maximum top-evidence rows per performance period.
        exclude_suppressed: Optional one-run suppressed-finding override.
        include_reconstruction_diagnostics: Optional one-run reconstruction-
            diagnostics override.
        require_causal_attribution: Optional one-run causal-attribution override.
        allow_incomplete_yaml: Whether diagnostic output may bypass the complete
            YAML setup guardrail.
        include_workbook: Optional one-run XLSX-output override.
        include_html_output: Optional one-run HTML-output override.
            When both primary presentation formats are disabled, Audit promotes
            its canonical CSV review tables instead.
        expand_all_supporting_files: Optional one-run supporting-file-layout
            override.

    Returns:
        Paths for the site folder, config file, and generated review artifacts.

    Raises:
        PpaError: If the site folder/config file is missing or report generation
            fails.
    """
    site_path = Path(site_directory).expanduser()
    if not site_path.is_dir():
        raise PpaError(f"{site_path} is not a directory.", 802)
    config_path = site_path / _CONFIG_FILE_NAME
    if not config_path.exists():
        raise PpaError(
            f"{config_path} is missing. Run from the Audit workspace "
            "or pass its folder. For first-time setup, run: "
            "ppar setup ./my_ppar_audit",
            802,
        )
    settings = _resolve_settings(
        site_path,
        _audit_settings(_load_config_values(config_path), required=True),
        output_directory=output_directory,
        title=title,
        exclude_suppressed=exclude_suppressed,
        include_reconstruction_diagnostics=include_reconstruction_diagnostics,
        require_causal_attribution=require_causal_attribution,
        allow_incomplete_yaml=allow_incomplete_yaml,
        include_workbook=include_workbook,
        include_html_output=include_html_output,
        expand_all_supporting_files=expand_all_supporting_files,
    )

    result: dict[str, Any] = {
        "site_directory": site_path,
        "config_path": config_path,
        "review_paths": [],
    }
    output_root = settings.output_directory
    data_issues = _data_issue_checks.data_issues_table(config_path)
    comparison_views = AuditComparisonViews(
        config_path,
        include_suppressed=not settings.exclude_suppressed,
        require_causal_attribution=settings.require_causal_attribution,
    )
    reconstruction_cache = _pc_workbook_reconstruction.WorkbookReconstructionCache(
        config_path
    )
    result["portfolio_report_paths"] = _write_report_bundle(
        config_path,
        comparison_views.findings(PORTFOLIO_COMPARISON_LEVEL),
        output_root / PORTFOLIO_COMPARISON_LEVEL,
        comparison_level=PORTFOLIO_COMPARISON_LEVEL,
        title=settings.title,
        top_evidence_limit=top_evidence_limit,
        include_reconstruction_diagnostics=settings.include_reconstruction_diagnostics,
        require_causal_attribution=settings.require_causal_attribution,
        allow_incomplete_yaml=settings.allow_incomplete_yaml,
        _data_issues=data_issues,
        include_workbook=settings.include_workbook,
        include_html_output=settings.include_html_output,
        expand_all_supporting_files=settings.expand_all_supporting_files,
        _reconstruction_cache=reconstruction_cache,
    )
    result["review_paths"].extend(result["portfolio_report_paths"])
    try:
        result["security_report_paths"] = _write_report_bundle(
            config_path,
            comparison_views.findings(SECURITY_COMPARISON_LEVEL),
            output_root / SECURITY_COMPARISON_LEVEL,
            comparison_level=SECURITY_COMPARISON_LEVEL,
            title=settings.title,
            top_evidence_limit=top_evidence_limit,
            include_reconstruction_diagnostics=(
                settings.include_reconstruction_diagnostics
            ),
            require_causal_attribution=settings.require_causal_attribution,
            allow_incomplete_yaml=settings.allow_incomplete_yaml,
            _data_issues=data_issues,
            include_workbook=settings.include_workbook,
            include_html_output=settings.include_html_output,
            expand_all_supporting_files=settings.expand_all_supporting_files,
            _reconstruction_cache=reconstruction_cache,
        )
        result["review_paths"].extend(result["security_report_paths"])
    except PpaError as error:
        if not _is_missing_security_data(error):
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
        allow_abbrev=False,
        description="Write Audit report bundles for a site setup.",
        epilog=(
            (
                "Examples:\n"
                "  ppar audit ./my_ppar_audit"
            )
            if include_site_directory
            else (
                "Examples:\n"
                "  python run_audit.py"
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
        "--output-directory",
        type=Path,
        help="Override YAML audit.output_directory for this run.",
    )
    parser.add_argument(
        "--title",
        help="Override YAML audit.title for this run.",
    )
    parser.add_argument(
        "--xlsx-output",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override YAML audit.xlsx_output for this run. Disabling both XLSX "
            "and HTML writes a CSV-only audit."
        ),
    )
    parser.add_argument(
        "--exclude-suppressed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override YAML audit.exclude_suppressed for this run. Source-data is "
            "still processed when suppressed findings are omitted."
        ),
    )
    parser.add_argument(
        "--reconstruction-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override YAML audit.reconstruction_diagnostics for this run. These "
            "diagnostics show how holdings and flows reproduce reported returns."
        ),
    )
    parser.add_argument(
        "--html-output",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override YAML audit.html_output for this run. Disabling both HTML "
            "and XLSX writes a CSV-only audit."
        ),
    )
    parser.add_argument(
        "--expand-all-supporting-files",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override YAML audit.expand_all_supporting_files for this run. Expanded "
            "files replace the compact audit_support.zip archive; source_detail.csv "
            "always remains at the report root."
        ),
    )
    parser.add_argument(
        "--require-causal-attribution",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override YAML audit.require_causal_attribution for this run. Enabling "
            "it fails when a changed period lacks setup required by a supported "
            "explanation method."
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
    config_path = (site_directory / _CONFIG_FILE_NAME).resolve()
    return _resolve_settings(
        site_directory,
        _audit_settings(_load_config_values(config_path), required=True),
        output_directory=arguments.output_directory,
        title=arguments.title,
        exclude_suppressed=arguments.exclude_suppressed,
        include_reconstruction_diagnostics=arguments.reconstruction_diagnostics,
        require_causal_attribution=arguments.require_causal_attribution,
        allow_incomplete_yaml=arguments.allow_incomplete_yaml,
        include_workbook=arguments.xlsx_output,
        include_html_output=arguments.html_output,
        expand_all_supporting_files=arguments.expand_all_supporting_files,
    )


def _load_config_values(config_path: Path) -> dict[str, Any]:
    """Load an Audit YAML file and return its root mapping."""
    if not config_path.exists():
        raise PpaError(f"{config_path} does not exist.", 504)
    with open(config_path, "r", encoding=util.ENCODING) as file:
        try:
            values: Any = yaml.safe_load(file)
        except Exception as error:
            raise PpaError(f"Invalid YAML in {config_path}: {error}", 504) from error
    if not isinstance(values, dict):
        raise PpaError(f"{config_path} must contain a YAML mapping.", 504)
    return values


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
    _reconstruction_cache: _pc_workbook_reconstruction.WorkbookReconstructionCache,
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
