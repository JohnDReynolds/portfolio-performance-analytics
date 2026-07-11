"""Run temporary, conservative scale checks for Analytics and Auditing.

This adjunct release check expands packaged CSV rows into unrelated portfolio
copies. It exercises large-site loading and filtering without maintaining a
second permanent data fixture or acting as a general benchmarking framework.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import polars as pl
from polars.testing import assert_frame_equal

from ppar.analytics.attribution import View
from ppar.analytics.frequency import Frequency
from ppar.axys import AxysData


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ANALYTICS_TEMPLATE = (
    _PROJECT_ROOT / "ppar" / "setup_templates" / "axysapx_analytics"
)
_AUDIT_TEMPLATE = (
    _PROJECT_ROOT / "ppar" / "setup_templates" / "axysapx_performance_comparison"
)
_BASELINE_TIMEOUT_SECONDS = 60
_ALLOWED_LARGE_SITE_SCALES = tuple(range(10, 101, 10))
_SELECTED_WORKLOAD_SCALE = 10
_LONG_HISTORY_SCALE = 5
_HISTORY_BLOCK_YEARS = 5
_SCALING_WARNING_MULTIPLIER = 1.05
_SCALING_FAILURE_MULTIPLIER = 1.10
_ANALYTICS_SCALING_WARNING_RATIO = 1.05
_ANALYTICS_SCALING_FAILURE_RATIO = 1.10


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Return command-line settings for the scale check."""
    parser = argparse.ArgumentParser(
        description="Run temporary Analytics and Audit large-site scale checks.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        choices=_ALLOWED_LARGE_SITE_SCALES,
        default=10,
        help="Large-site multiplier from 10 through 100 in increments of 10.",
    )
    return parser.parse_args(argv)


def _run(
    command: Sequence[str | Path],
    *,
    timeout_seconds: float = _BASELINE_TIMEOUT_SECONDS,
) -> float:
    """Run one timed command and return elapsed seconds."""
    started = time.perf_counter()
    subprocess.run(
        [str(part) for part in command],
        cwd=_PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return time.perf_counter() - started


def _analytics_scaling_result(
    baseline_elapsed: float,
    scaled_elapsed: float,
) -> tuple[str, float]:
    """Return the Analytics large-site status and elapsed-time ratio.

    Raises:
        RuntimeError: If the scaled run exceeds 1.10 times baseline duration.
    """
    if baseline_elapsed <= 0:
        raise ValueError("Analytics baseline time must be greater than zero.")
    ratio = scaled_elapsed / baseline_elapsed
    if ratio > _ANALYTICS_SCALING_FAILURE_RATIO:
        raise RuntimeError(
            "Analytics large-site scaling exceeded the 1.10x failure threshold: "
            f"baseline={baseline_elapsed:.2f}s, scaled={scaled_elapsed:.2f}s, "
            f"ratio={ratio:.2f}x."
        )
    status = "WARN" if ratio > _ANALYTICS_SCALING_WARNING_RATIO else "PASS"
    return status, ratio


def _workload_scaling_result(
    scenario: str,
    factor: int,
    baseline_elapsed: float,
    scaled_elapsed: float,
) -> tuple[str, float, float, float]:
    """Return factor-based status, ratio, warning cap, and error cap."""
    if baseline_elapsed <= 0:
        raise ValueError(f"{scenario} baseline time must be greater than zero.")
    ratio = scaled_elapsed / baseline_elapsed
    warning_ratio = factor * _SCALING_WARNING_MULTIPLIER
    error_ratio = factor * _SCALING_FAILURE_MULTIPLIER
    if ratio > error_ratio:
        raise RuntimeError(
            f"{scenario} exceeded the {error_ratio:.2f}x time-ratio error cap: "
            f"baseline={baseline_elapsed:.2f}s, scaled={scaled_elapsed:.2f}s, "
            f"ratio={ratio:.2f}x."
        )
    status = "WARN" if ratio > warning_ratio else "PASS"
    return status, ratio, warning_ratio, error_ratio


def _sublinear_scaling_result(
    scenario: str,
    factor: int,
    baseline_elapsed: float,
    scaled_elapsed: float,
) -> tuple[str, float, float, float]:
    """Return caps for Analytics workloads whose time growth is sublinear."""
    expected_ratio = 1.0 + factor / 10.0
    warning_ratio = expected_ratio * _SCALING_WARNING_MULTIPLIER
    error_ratio = expected_ratio * _SCALING_FAILURE_MULTIPLIER
    if baseline_elapsed <= 0:
        raise ValueError(f"{scenario} baseline time must be greater than zero.")
    ratio = scaled_elapsed / baseline_elapsed
    if ratio > error_ratio:
        raise RuntimeError(
            f"{scenario} exceeded the {error_ratio:.2f}x time-ratio error cap: "
            f"baseline={baseline_elapsed:.2f}s, scaled={scaled_elapsed:.2f}s, "
            f"ratio={ratio:.2f}x."
        )
    status = "WARN" if ratio > warning_ratio else "PASS"
    return status, ratio, warning_ratio, error_ratio


def _scaled_timeout(baseline_elapsed: float, error_ratio: float) -> float:
    """Return the scaled subprocess timeout derived from its measured baseline."""
    return baseline_elapsed * error_ratio


# The explicit fields keep the scenario call sites and terminal output readable.
# pylint: disable-next=too-many-arguments
def _print_scale_result(
    scenario: str,
    factor: int,
    baseline_rows: int,
    scaled_rows: int,
    baseline_elapsed: float,
    scaled_elapsed: float,
    *,
    status: str = "PASS",
    warning_cap: str = "none",
    error_cap: str,
) -> None:
    """Print one consistent baseline, scaled, ratio, and limits summary."""
    row_ratio = scaled_rows / baseline_rows
    time_ratio = scaled_elapsed / baseline_elapsed
    print(f"{status} {scenario} {factor}x")
    print(
        f"  baseline 1x: rows={baseline_rows:,}, time={baseline_elapsed:.2f}s"
    )
    print(
        f"  scaled {factor}x: rows={scaled_rows:,}, time={scaled_elapsed:.2f}s"
    )
    print(f"  ratios: rows={row_ratio:.2f}x, time={time_ratio:.2f}x")
    print(f"  time ratio caps: warning={warning_cap}, error={error_cap}")


def _print_timeout_result(
    scenario: str,
    factor: int,
    baseline_rows: int,
    scaled_rows: int,
    baseline_elapsed: float,
    timeout_seconds: float,
    warning_ratio: float,
    error_ratio: float,
) -> None:
    """Print a consistent failed summary when a scaled subprocess times out."""
    print(f"FAIL {scenario} {factor}x")
    print(f"  baseline 1x: rows={baseline_rows:,}, time={baseline_elapsed:.2f}s")
    print(
        f"  scaled {factor}x: rows={scaled_rows:,}, "
        f"time=>{timeout_seconds:.2f}s (timed out)"
    )
    print(
        f"  ratios: rows={scaled_rows / baseline_rows:.2f}x, "
        f"time=>{timeout_seconds / baseline_elapsed:.2f}x"
    )
    print(
        f"  time ratio caps: warning=>{warning_ratio:.2f}x, "
        f"error=>{error_ratio:.2f}x"
    )


def _expanded_frame(
    source_path: Path,
    scale: int,
    portfolio_columns: tuple[str, ...],
) -> pl.DataFrame:
    """Return rows copied across consistently suffixed portfolio identifiers."""
    source = pl.read_csv(source_path)
    available_columns = [name for name in portfolio_columns if name in source.columns]
    if not available_columns:
        return source

    copies: list[pl.DataFrame] = [source]
    for copy_number in range(1, scale):
        suffix = f"_SCALE_{copy_number:03d}"
        copies.append(
            source.with_columns(
                [
                    (pl.col(name).cast(pl.String) + suffix).alias(name)
                    for name in available_columns
                ]
            )
        )
    return pl.concat(copies, how="vertical")


def _require_workspace_path(workspace: Path, path: Path) -> None:
    """Reject any generated-data path outside the temporary workspace."""
    if not path.resolve().is_relative_to(workspace.resolve()):
        raise ValueError(f"Scale-check path is outside its workspace: {path}")


def _expanded_selected_analytics_frames(
    security_performance: pl.DataFrame,
    security_reference: pl.DataFrame,
    scale: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return proportionally scaled performance and aligned reference rows."""
    performance_copies: list[pl.DataFrame] = []
    reference_copies: list[pl.DataFrame] = []
    for copy_number in range(scale):
        suffix = f"_LOAD_{copy_number:02d}"
        performance_copies.append(
            security_performance.with_columns(
                (pl.col("SECURITY_ID") + suffix).alias("SECURITY_ID"),
                (pl.col("BEGIN_WEIGHT") / scale).alias("BEGIN_WEIGHT"),
                (pl.col("CONTRIBUTION") / scale).alias("CONTRIBUTION"),
            )
        )
        reference_copies.append(
            security_reference.with_columns(
                (pl.col("SECURITY_ID") + suffix).alias("SECURITY_ID"),
                (pl.col("SECURITY_NAME") + suffix).alias("SECURITY_NAME"),
            )
        )
    return (
        pl.concat(performance_copies, how="vertical"),
        pl.concat(reference_copies, how="vertical"),
    )


def _expanded_history_frame(source: pl.DataFrame, scale: int) -> pl.DataFrame:
    """Return chronologically shifted, non-overlapping copies of period rows."""
    dated = source.with_columns(
        pl.col("FROM_DATE").str.to_date(),
        pl.col("THRU_DATE").str.to_date(),
    )
    copies: list[pl.DataFrame] = []
    for copy_number in range(scale):
        offset = f"{copy_number * _HISTORY_BLOCK_YEARS}y"
        copies.append(
            dated.with_columns(
                pl.col("FROM_DATE").dt.offset_by(offset),
                pl.col("THRU_DATE").dt.offset_by(offset),
            )
        )
    return pl.concat(copies, how="vertical")


def _expanded_audit_history_frame(source: pl.DataFrame, scale: int) -> pl.DataFrame:
    """Return consistently date-shifted copies of one Audit source table."""
    date_columns = [column for column in source.columns if column.endswith("_DATE")]
    if not date_columns:
        return pl.concat([source] * scale, how="vertical")
    dated = source.with_columns(
        [pl.col(column).str.to_date() for column in date_columns]
    )
    copies = []
    for copy_number in range(scale):
        offset = f"{copy_number * _HISTORY_BLOCK_YEARS}y"
        copies.append(
            dated.with_columns(
                [pl.col(column).dt.offset_by(offset) for column in date_columns]
            )
        )
    return pl.concat(copies, how="vertical")


def _prepare_analytics(directory: Path, scale: int) -> tuple[Path, int]:
    """Write one temporary Analytics site and return its secperf row count."""
    shutil.copytree(_ANALYTICS_TEMPLATE, directory)
    (directory / "axysapx_analytics.yaml").rename(directory / "ppar.yaml")
    for file_name in ("portperf.csv", "secperf.csv"):
        frame = _expanded_frame(
            directory / file_name,
            scale,
            ("PORTFOLIO_CODE",),
        )
        frame.write_csv(directory / file_name)
    return directory, pl.read_csv(directory / "secperf.csv").height


def _prepare_audit(directory: Path, scale: int) -> tuple[Path, int]:
    """Write one temporary Audit site and return its aggregate CSV row count."""
    shutil.copytree(_AUDIT_TEMPLATE, directory)
    (directory / "axysapx_performance_comparison.yaml").rename(
        directory / "ppar.yaml"
    )
    row_count = 0
    for snapshot_name in ("snapshot_a", "snapshot_b"):
        snapshot = directory / snapshot_name
        for path in snapshot.glob("*.csv"):
            frame = _expanded_frame(
                path,
                scale,
                ("PORTFOLIO_CODE", "PORT"),
            )
            frame.write_csv(path)
            row_count += frame.height
    return directory, row_count


def _prepare_long_history_audit(directory: Path) -> tuple[Path, int, set[int]]:
    """Write a fixed 5x Audit history and return rows and expected years."""
    shutil.copytree(_AUDIT_TEMPLATE, directory)
    (directory / "axysapx_performance_comparison.yaml").rename(
        directory / "ppar.yaml"
    )
    row_count = 0
    expected_years: set[int] = set()
    for snapshot_name in ("snapshot_a", "snapshot_b"):
        snapshot = directory / snapshot_name
        for path in snapshot.glob("*.csv"):
            expanded = _expanded_audit_history_frame(
                pl.read_csv(path),
                _LONG_HISTORY_SCALE,
            )
            expanded.write_csv(path)
            row_count += expanded.height
            if snapshot_name == "snapshot_a" and path.name == "portperf.csv":
                expected_years = set(
                    expanded.get_column("FROM_DATE").dt.year().to_list()
                )
    return directory, row_count, expected_years


def _prepare_selected_analytics(directory: Path, scale: int) -> tuple[Path, int]:
    """Write an Analytics site whose selected pair has unique copied securities."""
    shutil.copytree(_ANALYTICS_TEMPLATE, directory)
    (directory / "axysapx_analytics.yaml").rename(directory / "ppar.yaml")

    security_performance = pl.read_csv(directory / "secperf.csv")
    security_reference = pl.read_csv(directory / "secref.csv")
    expanded_performance, expanded_reference = _expanded_selected_analytics_frames(
        security_performance,
        security_reference,
        scale,
    )
    expanded_performance.write_csv(directory / "secperf.csv")
    expanded_reference.write_csv(directory / "secref.csv")
    return directory, expanded_performance.height


def _prepare_long_history_analytics(directory: Path) -> tuple[Path, int, int]:
    """Write a fixed 5x Analytics history and return rows and period count."""
    shutil.copytree(_ANALYTICS_TEMPLATE, directory)
    yaml_path = directory / "axysapx_analytics.yaml"
    yaml_text = yaml_path.read_text(encoding="utf-8")
    yaml_text = yaml_text.replace(
        "thru_date: 2026-05-29",
        "thru_date: 2046-05-29",
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")
    yaml_path.rename(directory / "ppar.yaml")

    period_count = 0
    row_count = 0
    for file_name in ("portperf.csv", "secperf.csv"):
        expanded = _expanded_history_frame(
            pl.read_csv(directory / file_name),
            _LONG_HISTORY_SCALE,
        )
        expanded.write_csv(directory / file_name)
        row_count += expanded.height
        if file_name == "portperf.csv":
            period_count = (
                expanded.filter(pl.col("PORTFOLIO_CODE") == "MEGA_ALPHA")
                .select("FROM_DATE", "THRU_DATE")
                .unique()
                .height
            )
    return directory, row_count, period_count


def _selected_analytics_tables(
    site: Path,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Return Security, sector, and risk tables for one Analytics site."""
    source = AxysData(site / "ppar.yaml")
    security_portfolio = source.get_portfolio(
        "MEGA_ALPHA", classification_name="Security"
    )
    security_benchmark = source.get_portfolio(
        "MEGA_BENCH", classification_name="Security"
    )
    security_analytics = security_portfolio.to_analytics(
        security_benchmark,
        frequency=Frequency.QUARTERLY,
    )
    security_overall = security_analytics.get_attribution().to_polars(
        View.OVERALL_ATTRIBUTION
    )

    sector_portfolio = source.get_portfolio(
        "MEGA_ALPHA", classification_name="Economic Sector"
    )
    sector_benchmark = source.get_portfolio(
        "MEGA_BENCH", classification_name="Economic Sector"
    )
    sector_analytics = sector_portfolio.to_analytics(
        sector_benchmark,
        frequency=Frequency.QUARTERLY,
    )
    sector_overall = sector_analytics.get_attribution().to_polars(
        View.OVERALL_ATTRIBUTION
    )
    risk = sector_analytics.get_riskstatistics().to_polars()
    return security_overall, sector_overall, risk


def _html_outputs(directory: Path) -> dict[str, bytes]:
    """Return generated HTML files keyed by filename."""
    return {path.name: path.read_bytes() for path in directory.glob("*.html")}


def _check_analytics(workspace: Path, scale: int) -> tuple[int, float]:
    """Run baseline and scaled Analytics and require identical HTML output."""
    baseline_path = workspace / "analytics_baseline"
    scaled_path = workspace / "analytics_scaled"
    _require_workspace_path(workspace, baseline_path)
    _require_workspace_path(workspace, scaled_path)
    baseline_site, baseline_rows = _prepare_analytics(baseline_path, 1)
    scaled_site, row_count = _prepare_analytics(scaled_path, scale)
    baseline_elapsed = _run([sys.executable, "-m", "ppar.cli", "analytics", baseline_site])
    scaled_elapsed = _run(
        [sys.executable, "-m", "ppar.cli", "analytics", scaled_site],
        timeout_seconds=_scaled_timeout(
            baseline_elapsed,
            _ANALYTICS_SCALING_FAILURE_RATIO,
        ),
    )
    baseline_html = _html_outputs(baseline_site / "output")
    scaled_html = _html_outputs(scaled_site / "output")
    if not baseline_html or baseline_html != scaled_html:
        raise RuntimeError("Scaled Analytics HTML differs from the 1x baseline.")
    status, _ = _analytics_scaling_result(baseline_elapsed, scaled_elapsed)
    _print_scale_result(
        "Analytics large-site",
        scale,
        baseline_rows,
        row_count,
        baseline_elapsed,
        scaled_elapsed,
        status=status,
        warning_cap=f">{_ANALYTICS_SCALING_WARNING_RATIO:.2f}x",
        error_cap=f">{_ANALYTICS_SCALING_FAILURE_RATIO:.2f}x",
    )
    return row_count, scaled_elapsed


def _check_audit(workspace: Path, scale: int) -> tuple[int, float]:
    """Run scaled Audit and validate its portfolio and security bundles."""
    baseline_path = workspace / "audit_baseline"
    scaled_path = workspace / "audit_scaled"
    _require_workspace_path(workspace, baseline_path)
    _require_workspace_path(workspace, scaled_path)
    baseline_site, baseline_rows = _prepare_audit(baseline_path, 1)
    site, row_count = _prepare_audit(scaled_path, scale)
    baseline_elapsed = _run(
        [sys.executable, "-m", "ppar.cli", "audit", baseline_site]
    )
    error_ratio = scale * _SCALING_FAILURE_MULTIPLIER
    warning_ratio = scale * _SCALING_WARNING_MULTIPLIER
    timeout_seconds = _scaled_timeout(baseline_elapsed, error_ratio)
    try:
        elapsed = _run(
            [sys.executable, "-m", "ppar.cli", "audit", site],
            timeout_seconds=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        _print_timeout_result(
            "Audit large-site",
            scale,
            baseline_rows,
            row_count,
            baseline_elapsed,
            timeout_seconds,
            warning_ratio,
            error_ratio,
        )
        raise RuntimeError(
            "Audit large-site exceeded its execution-time error cap."
        ) from error
    for audit_site in (baseline_site, site):
        for report_name in ("portfolio", "security"):
            _run(
                [
                    sys.executable,
                    "-m",
                    "ppar.performance_comparison.cli.validate_bundle",
                    audit_site / "output" / report_name,
                ]
            )
    status, _, warning_ratio, error_ratio = _workload_scaling_result(
        "Audit large-site",
        scale,
        baseline_elapsed,
        elapsed,
    )
    _print_scale_result(
        "Audit large-site",
        scale,
        baseline_rows,
        row_count,
        baseline_elapsed,
        elapsed,
        status=status,
        warning_cap=f">{warning_ratio:.2f}x",
        error_cap=f">{error_ratio:.2f}x",
    )
    return row_count, elapsed


def _check_long_history_audit(workspace: Path) -> tuple[int, float]:
    """Run a fixed 5x date-shifted history through the standard Audit command."""
    baseline_path = workspace / "audit_history_baseline"
    site_path = workspace / "audit_long_history"
    _require_workspace_path(workspace, baseline_path)
    _require_workspace_path(workspace, site_path)
    baseline_site, baseline_rows = _prepare_audit(baseline_path, 1)
    site, row_count, _ = _prepare_long_history_audit(site_path)
    if row_count != baseline_rows * _LONG_HISTORY_SCALE:
        raise RuntimeError(
            "Audit long-history row count differs from the expected 5x history: "
            f"expected={baseline_rows * _LONG_HISTORY_SCALE}, actual={row_count}."
        )

    baseline_elapsed = _run(
        [sys.executable, "-m", "ppar.cli", "audit", baseline_site]
    )
    error_ratio = _LONG_HISTORY_SCALE * _SCALING_FAILURE_MULTIPLIER
    elapsed = _run(
        [sys.executable, "-m", "ppar.cli", "audit", site],
        timeout_seconds=_scaled_timeout(baseline_elapsed, error_ratio),
    )
    for report_name in ("portfolio", "security"):
        report_path = site / "output" / report_name
        baseline_report_path = baseline_site / "output" / report_name
        _run(
            [
                sys.executable,
                "-m",
                "ppar.performance_comparison.cli.validate_bundle",
                report_path,
            ]
        )
        findings = pl.read_csv(
            report_path / "supporting_files" / "findings.csv",
            try_parse_dates=True,
        )
        baseline_findings = pl.read_csv(
            baseline_report_path / "supporting_files" / "findings.csv",
            try_parse_dates=True,
        )
        baseline_years = set(
            baseline_findings.get_column("from_date")
            .drop_nulls()
            .dt.year()
            .to_list()
        )
        expected_years = {
            year + copy_number * _HISTORY_BLOCK_YEARS
            for year in baseline_years
            for copy_number in range(_LONG_HISTORY_SCALE)
        }
        actual_years = set(
            findings.get_column("from_date").drop_nulls().dt.year().to_list()
        )
        if not expected_years.issubset(actual_years):
            missing_years = sorted(expected_years - actual_years)
            raise RuntimeError(
                f"Audit long-history {report_name} output is missing years: "
                f"{missing_years}."
            )

    status, _, warning_ratio, error_ratio = _workload_scaling_result(
        "Audit long-history",
        _LONG_HISTORY_SCALE,
        baseline_elapsed,
        elapsed,
    )
    _print_scale_result(
        "Audit long-history",
        _LONG_HISTORY_SCALE,
        baseline_rows,
        row_count,
        baseline_elapsed,
        elapsed,
        status=status,
        warning_cap=f">{warning_ratio:.2f}x",
        error_cap=f">{error_ratio:.2f}x",
    )
    return row_count, elapsed


def _check_selected_analytics(workspace: Path) -> tuple[int, float]:
    """Run the 10x selected-workload calculation and verify financial results."""
    baseline_path = workspace / "selected_baseline"
    scaled_path = workspace / "selected_scaled"
    _require_workspace_path(workspace, baseline_path)
    _require_workspace_path(workspace, scaled_path)
    baseline_site, _ = _prepare_analytics(baseline_path, 1)
    scaled_site, row_count = _prepare_selected_analytics(
        scaled_path,
        _SELECTED_WORKLOAD_SCALE,
    )
    baseline_started = time.perf_counter()
    baseline_security, baseline_sector, baseline_risk = _selected_analytics_tables(
        baseline_site
    )
    baseline_elapsed = time.perf_counter() - baseline_started
    started = time.perf_counter()
    scaled_security, scaled_sector, scaled_risk = _selected_analytics_tables(
        scaled_site
    )
    elapsed = time.perf_counter() - started

    expected_security_rows = (
        baseline_security.filter(pl.col("Classification_Identifier").is_not_null()).height
        * _SELECTED_WORKLOAD_SCALE
    )
    actual_security_rows = scaled_security.filter(
        pl.col("Classification_Identifier").is_not_null()
    ).height
    if actual_security_rows != expected_security_rows:
        raise RuntimeError(
            "Selected Analytics security rows differ from the expected "
            f"10x count: expected={expected_security_rows}, actual={actual_security_rows}."
        )
    assert_frame_equal(
        baseline_sector,
        scaled_sector,
        check_exact=False,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    assert_frame_equal(
        baseline_risk,
        scaled_risk,
        check_exact=False,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    baseline_rows = pl.read_csv(baseline_site / "secperf.csv").height
    status, _, warning_ratio, error_ratio = _sublinear_scaling_result(
        "Analytics selected-workload",
        _SELECTED_WORKLOAD_SCALE,
        baseline_elapsed,
        elapsed,
    )
    _print_scale_result(
        "Analytics selected-workload",
        _SELECTED_WORKLOAD_SCALE,
        baseline_rows,
        row_count,
        baseline_elapsed,
        elapsed,
        status=status,
        warning_cap=f">{warning_ratio:.2f}x",
        error_cap=f">{error_ratio:.2f}x",
    )
    return row_count, elapsed


def _check_long_history_analytics(workspace: Path) -> tuple[int, float]:
    """Run the fixed 5x history through the standard Analytics command."""
    baseline_path = workspace / "long_history_baseline"
    site_path = workspace / "long_history"
    _require_workspace_path(workspace, baseline_path)
    _require_workspace_path(workspace, site_path)
    baseline_site, baseline_security_rows = _prepare_analytics(baseline_path, 1)
    site, row_count, period_count = _prepare_long_history_analytics(site_path)
    if period_count != 60 * _LONG_HISTORY_SCALE:
        raise RuntimeError(
            "Long-history period count differs from the expected 5x history: "
            f"expected={60 * _LONG_HISTORY_SCALE}, actual={period_count}."
        )

    performance = pl.read_csv(site / "portperf.csv").filter(
        pl.col("PORTFOLIO_CODE") == "MEGA_ALPHA"
    )
    periods = performance.select("FROM_DATE", "THRU_DATE").sort("FROM_DATE")
    if not periods["FROM_DATE"].is_sorted() or not (
        periods["FROM_DATE"].slice(1)
        > periods["THRU_DATE"].slice(0, periods.height - 1)
    ).all():
        raise RuntimeError("Long-history periods overlap or are not chronological.")

    baseline_elapsed = _run(
        [sys.executable, "-m", "ppar.cli", "analytics", baseline_site]
    )
    expected_ratio = 1.0 + _LONG_HISTORY_SCALE / 10.0
    error_ratio = expected_ratio * _SCALING_FAILURE_MULTIPLIER
    elapsed = _run(
        [sys.executable, "-m", "ppar.cli", "analytics", site],
        timeout_seconds=_scaled_timeout(baseline_elapsed, error_ratio),
    )
    artifacts = [path for path in (site / "output").iterdir() if path.is_file()]
    if len(artifacts) != 11 or any(path.stat().st_size == 0 for path in artifacts):
        raise RuntimeError("Long-history Analytics artifacts are incomplete.")
    baseline_rows = baseline_security_rows + pl.read_csv(
        baseline_site / "portperf.csv"
    ).height
    status, _, warning_ratio, error_ratio = _sublinear_scaling_result(
        "Analytics long-history",
        _LONG_HISTORY_SCALE,
        baseline_elapsed,
        elapsed,
    )
    _print_scale_result(
        "Analytics long-history",
        _LONG_HISTORY_SCALE,
        baseline_rows,
        row_count,
        baseline_elapsed,
        elapsed,
        status=status,
        warning_cap=f">{warning_ratio:.2f}x",
        error_cap=f">{error_ratio:.2f}x",
    )
    return row_count, elapsed


def main(argv: Sequence[str] | None = None) -> int:
    """Run temporary large-site filtering checks and print compact timings."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        with tempfile.TemporaryDirectory(prefix="ppar_scale_check_") as directory:
            workspace = Path(directory)
            _check_analytics(workspace, args.scale)
            _check_audit(workspace, args.scale)
            _check_selected_analytics(workspace)
            _check_long_history_analytics(workspace)
            _check_long_history_audit(workspace)
    except (RuntimeError, subprocess.SubprocessError) as error:
        print(f"Scale checks failed: {error}", file=sys.stderr)
        return 1
    print(f"Scale checks passed at {args.scale}x.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
