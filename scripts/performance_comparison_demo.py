"""Demonstrate performance comparison findings output."""

# This script is meant to run directly from the repository checkout. Insert the
# repository root before importing ppar so the local source tree is used even
# when the package has not been installed. The ppar imports below therefore
# intentionally sit after executable bootstrap code; `noqa: E402` suppresses
# the "module import not at top of file" warning for those lines.
# pylint: disable=wrong-import-order,wrong-import-position

# Python imports
from pathlib import Path
import sys

# Third-party imports
import polars as pl

_REPO_ROOT = Path(__file__).resolve().parents[1]
_AXYS_DATA_ROOT = _REPO_ROOT / "tests" / "data" / "axys"
sys.path.insert(0, str(_REPO_ROOT))

# Project imports
from ppar.performance_comparison import (  # noqa: E402
    compact_findings_table,
    compare_snapshots,
    performance_comparison_markdown_report,
    portfolio_period_cause_summary,
    portfolio_period_contribution_candidates,
    portfolio_period_evidence_breakdown,
    portfolio_period_impact_coverage_summary,
    portfolio_period_summary,
    rank_portfolio_period_evidence,
    security_period_evidence_breakdown,
    security_period_summary,
    summarize_findings,
    transaction_activity_summary,
    write_performance_comparison_html_report,
    write_performance_comparison_markdown_report,
    write_performance_comparison_report_bundle,
)


def _print_table(title: str, table: pl.DataFrame, *, wide: bool = False) -> None:
    """Print a titled table using demo-friendly Polars display settings."""
    print(title)
    if wide:
        with pl.Config(
            tbl_cols=-1,
            tbl_rows=-1,
            tbl_width_chars=240,
            fmt_str_lengths=80,
        ):
            print(table)
    else:
        with pl.Config(tbl_cols=-1, tbl_rows=-1, tbl_width_chars=160):
            print(table)
    print()


def main() -> None:
    """Run the performance comparison demonstration."""
    comparison_path = _AXYS_DATA_ROOT / "ppar_performance_comparison_restatement.yaml"
    report_path = _REPO_ROOT / "_demo_output" / "performance_comparison_restatement.md"
    html_report_path = _REPO_ROOT / "_demo_output" / "performance_comparison_restatement.html"
    bundle_path = _REPO_ROOT / "_demo_output" / "performance_comparison_bundle"
    suppressed_comparison_path = (
        _AXYS_DATA_ROOT / "ppar_performance_comparison_suppressed.yaml"
    )
    findings = compare_snapshots(comparison_path)
    active_findings = compare_snapshots(comparison_path, include_suppressed=False)
    compact_active_findings = compact_findings_table(findings)
    period_summary = portfolio_period_summary(findings)
    security_summary = security_period_summary(findings)
    evidence_breakdown = portfolio_period_evidence_breakdown(findings)
    evidence_ranking = rank_portfolio_period_evidence(findings)
    contribution_candidates = portfolio_period_contribution_candidates(findings)
    cause_summary = portfolio_period_cause_summary(findings)
    impact_coverage = portfolio_period_impact_coverage_summary(findings)
    transaction_summary = transaction_activity_summary(findings)
    security_evidence_breakdown = security_period_evidence_breakdown(findings)
    summaries = summarize_findings(findings)
    active_summaries = summarize_findings(active_findings)
    suppressed_findings = compare_snapshots(suppressed_comparison_path)
    suppressed_active_findings = compare_snapshots(
        suppressed_comparison_path,
        include_suppressed=False,
    )
    suppressed_summaries = summarize_findings(suppressed_findings)
    suppressed_active_summaries = summarize_findings(suppressed_active_findings)
    markdown_report = performance_comparison_markdown_report(findings)
    written_report_path = write_performance_comparison_markdown_report(
        findings,
        report_path,
    )
    written_html_report_path = write_performance_comparison_html_report(
        findings,
        html_report_path,
    )
    written_bundle_paths = write_performance_comparison_report_bundle(
        findings,
        bundle_path,
    )
    print("Restatement comparison")
    print()
    print(f"Markdown report written to: {written_report_path}")
    print(f"HTML report written to: {written_html_report_path}")
    print(f"Report bundle written to: {written_bundle_paths['manifest'].parent}")
    print()
    print(markdown_report)
    print()
    _print_table("Finding count by code", summaries["by_code"])
    _print_table("Finding count by dataset", summaries["by_dataset"])
    _print_table("Finding count by evidence role", summaries["by_evidence_role"])
    _print_table("Finding count by suppression state", summaries["by_suppressed"])
    _print_table("Finding count by code and suppression state", summaries["by_code_suppressed"])
    _print_table("Active finding count by code", active_summaries["by_code"])
    _print_table("Portfolio-period summary", period_summary)
    _print_table("Security-period summary", security_summary)
    _print_table("Portfolio-period evidence breakdown", evidence_breakdown)
    _print_table("Portfolio-period evidence ranking", evidence_ranking, wide=True)
    _print_table(
        "Portfolio-period contribution candidates",
        contribution_candidates,
        wide=True,
    )
    _print_table("Portfolio-period cause summary", cause_summary, wide=True)
    _print_table("Portfolio-period impact coverage", impact_coverage, wide=True)
    _print_table("Transaction activity summary", transaction_summary, wide=True)
    _print_table("Security-period evidence breakdown", security_evidence_breakdown)
    _print_table("Compact active findings", compact_active_findings, wide=True)
    _print_table("Full audit findings", findings, wide=True)

    print("Suppressed restatement comparison")
    print()
    print(f"All findings: {suppressed_findings.height}")
    print(f"Active findings: {suppressed_active_findings.height}")
    print()
    _print_table("Finding count by suppression state", suppressed_summaries["by_suppressed"])
    _print_table(
        "Finding count by code and suppression state",
        suppressed_summaries["by_code_suppressed"],
    )
    _print_table("Active finding count by code", suppressed_active_summaries["by_code"])


if __name__ == "__main__":
    main()
