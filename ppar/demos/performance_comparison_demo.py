"""Demonstrate performance comparison findings output."""

# Python imports
from importlib.resources import as_file, files
from pathlib import Path

# Third-party imports
import polars as pl

# Project imports
from ppar.performance_comparison import (
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

_DemoTable = tuple[str, pl.DataFrame, bool]


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
    with as_file(files("ppar.demo_data") / "axys") as axys_data_root:
        comparison_path = axys_data_root / "ppar_performance_comparison_restatement.yaml"
        suppressed_comparison_path = (
            axys_data_root / "ppar_performance_comparison_suppressed.yaml"
        )

        output_root = Path.cwd() / "_demo_output"
        report_path = output_root / "performance_comparison_restatement.md"
        html_report_path = output_root / "performance_comparison_restatement.html"
        bundle_path = output_root / "performance_comparison_bundle"

        _run_comparison_demo(
            comparison_path=comparison_path,
            suppressed_comparison_path=suppressed_comparison_path,
            report_path=report_path,
            html_report_path=html_report_path,
            bundle_path=bundle_path,
        )


def _run_comparison_demo(
    *,
    comparison_path: Path,
    suppressed_comparison_path: Path,
    report_path: Path,
    html_report_path: Path,
    bundle_path: Path,
) -> None:
    """Run the performance comparison workflow for resolved demo paths."""
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
    _print_bundle_handoff(written_bundle_paths)
    print(markdown_report)
    print()
    _print_restatement_tables(
        _restatement_demo_tables(
            summaries=summaries,
            active_summaries=active_summaries,
            detail_tables={
                "period_summary": period_summary,
                "security_summary": security_summary,
                "evidence_breakdown": evidence_breakdown,
                "evidence_ranking": evidence_ranking,
                "contribution_candidates": contribution_candidates,
                "cause_summary": cause_summary,
                "impact_coverage": impact_coverage,
                "transaction_summary": transaction_summary,
                "security_evidence_breakdown": security_evidence_breakdown,
                "compact_active_findings": compact_active_findings,
                "findings": findings,
            },
        )
    )
    _print_suppressed_summary(
        suppressed_findings=suppressed_findings,
        suppressed_active_findings=suppressed_active_findings,
        suppressed_summaries=suppressed_summaries,
        suppressed_active_summaries=suppressed_active_summaries,
    )


def _restatement_demo_tables(
    *,
    summaries: dict[str, pl.DataFrame],
    active_summaries: dict[str, pl.DataFrame],
    detail_tables: dict[str, pl.DataFrame],
) -> list[_DemoTable]:
    """Return the detailed restatement comparison tables for printing."""
    return [
        ("Finding count by code", summaries["by_code"], False),
        ("Finding count by dataset", summaries["by_dataset"], False),
        ("Finding count by evidence role", summaries["by_evidence_role"], False),
        ("Finding count by suppression state", summaries["by_suppressed"], False),
        (
            "Finding count by code and suppression state",
            summaries["by_code_suppressed"],
            False,
        ),
        ("Active finding count by code", active_summaries["by_code"], False),
        ("Portfolio-period summary", detail_tables["period_summary"], False),
        ("Security-period summary", detail_tables["security_summary"], False),
        ("Portfolio-period evidence breakdown", detail_tables["evidence_breakdown"], False),
        ("Portfolio-period evidence ranking", detail_tables["evidence_ranking"], True),
        (
            "Portfolio-period contribution candidates",
            detail_tables["contribution_candidates"],
            True,
        ),
        ("Portfolio-period cause summary", detail_tables["cause_summary"], True),
        ("Portfolio-period impact coverage", detail_tables["impact_coverage"], True),
        ("Transaction activity summary", detail_tables["transaction_summary"], True),
        (
            "Security-period evidence breakdown",
            detail_tables["security_evidence_breakdown"],
            False,
        ),
        ("Compact active findings", detail_tables["compact_active_findings"], True),
        ("Full audit findings", detail_tables["findings"], True),
    ]


def _print_restatement_tables(tables: list[_DemoTable]) -> None:
    """Print the detailed restatement comparison tables."""
    for title, table, wide in tables:
        _print_table(title, table, wide=wide)


def _print_suppressed_summary(
    *,
    suppressed_findings: pl.DataFrame,
    suppressed_active_findings: pl.DataFrame,
    suppressed_summaries: dict[str, pl.DataFrame],
    suppressed_active_summaries: dict[str, pl.DataFrame],
) -> None:
    """Print suppression-focused demo output."""
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


def _print_bundle_handoff(bundle_paths: dict[str, Path]) -> None:
    """Print generated bundle artifacts and the recommended review path."""
    print("Bundle artifacts:")
    for artifact_name, artifact_path in sorted(bundle_paths.items()):
        print(f"- {artifact_name}: {artifact_path.name}")
    print()
    needs_review_summary = pl.read_csv(bundle_paths["needs_review_summary"])
    _print_table("Needs review summary", needs_review_summary, wide=True)
    print("Recommended review path")
    print(f"1. Open {bundle_paths['html_report']}")
    print(f"2. Inspect {bundle_paths['needs_review_summary']}")
    print(f"3. Use {bundle_paths['manifest']} to audit generated artifacts")
    print()


if __name__ == "__main__":
    main()
