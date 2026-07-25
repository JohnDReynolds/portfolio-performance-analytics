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
import statistics
import subprocess
import sys
import tempfile
import time

import polars as pl
from polars.testing import assert_frame_equal

from ppar.analytics.attribution import View
from ppar.analytics.frequency import Frequency
from ppar.axys_apx import AxysData

try:
    from scripts import audit_scale_contract
except ModuleNotFoundError:  # Direct ``python scripts/check_scale.py`` execution.
    import audit_scale_contract  # type: ignore[no-redef]


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ANALYTICS_TEMPLATE = (
    _PROJECT_ROOT / "ppar" / "setup_templates" / "axys_apx_analytics"
)
_AUDIT_TEMPLATE = (
    _PROJECT_ROOT / "ppar" / "setup_templates" / "axys_apx_audit"
)
_BASELINE_TIMEOUT_SECONDS = 60
_PERFORMANCE_TIMING_SAMPLES = 3
_PROCESS_TIMEOUT_GRACE_SECONDS = 5.0
_ALLOWED_LARGE_SITE_SCALES = (*range(10, 101, 10), 500, 1000)
_MAX_ANALYTICS_LARGE_SITE_SCALE = 500
_EXTREME_AUDIT_SCALE = 1000
# A fully changed 1000x copy would intentionally exceed Audit's production
# 100,000-row reviewer-output ceiling. Keep the full input volume while
# limiting changed portfolios so this controlled stress check can complete
# without weakening that safety guard.
_EXTREME_AUDIT_CHANGED_PORTFOLIOS = frozenset({"BALANCED"})
_SELECTED_WORKLOAD_SCALE = 10
_LONG_HISTORY_SCALE = 5
_HISTORY_BLOCK_YEARS = 5
_SCALING_WARNING_MULTIPLIER = 1.05
_SCALING_FAILURE_MULTIPLIER = 1.10
_ANALYTICS_SCALING_WARNING_RATIO = 1.05
_ANALYTICS_SCALING_FAILURE_RATIO = 1.10
# The 10x through 500x measurements follow approximately ``1 + scale / 7``.
# The 5% warning and 10% failure margins remain explicit regression headroom,
# rather than preserving the much looser pre-optimization growth curve.
_AUDIT_LARGE_SITE_SCALE_DIVISOR = 7.0
_EXTREME_AUDIT_WARNING_RATIO = 85.0
_EXTREME_AUDIT_FAILURE_RATIO = 95.0
_AUDIT_HISTORY_WARNING_RATIO = 1.75
_AUDIT_HISTORY_FAILURE_RATIO = 2.00


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
        help=(
            "Large-site multiplier from 10 through 100 in increments of 10, "
            "the 500x release-candidate stress level, or the controlled 1000x "
            "Audit large-input stress level (Analytics remains at 500x)."
        ),
    )
    return parser.parse_args(argv)


def _run(
    command: Sequence[str | Path],
    *,
    timeout_seconds: float = _BASELINE_TIMEOUT_SECONDS,
) -> float:
    """Run one timed command and return elapsed seconds."""
    started = time.perf_counter()
    normalized_command = [str(part) for part in command]
    try:
        subprocess.run(
            normalized_command,
            cwd=_PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.CalledProcessError as error:
        details = (error.stderr or error.stdout or "No child-process output.").strip()
        raise RuntimeError(
            f"Command failed: {' '.join(normalized_command)}\n{details}"
        ) from error
    return time.perf_counter() - started


def _run_median_elapsed(
    command: Sequence[str | Path],
    *,
    timeout_seconds: float = _BASELINE_TIMEOUT_SECONDS,
) -> float:
    """Return the median elapsed time from repeated short command runs.

    Three samples keep the tight Analytics ratio gate from depending on one
    unusually fast or slow process startup. Each sample still receives the
    same safety timeout, and the unchanged performance ratio is applied to the
    median after all samples complete.
    """
    samples = [
        _run(command, timeout_seconds=timeout_seconds)
        for _ in range(_PERFORMANCE_TIMING_SAMPLES)
    ]
    return statistics.median(samples)


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


def _audit_large_site_scaling_result(
    factor: int,
    baseline_elapsed: float,
    scaled_elapsed: float,
) -> tuple[str, float, float, float]:
    """Return tighter caps for the observed Audit large-site growth curve."""
    warning_ratio, error_ratio = _audit_large_site_caps(factor)
    if baseline_elapsed <= 0:
        raise ValueError("Audit large-site baseline time must be greater than zero.")
    ratio = scaled_elapsed / baseline_elapsed
    if ratio > error_ratio:
        raise RuntimeError(
            f"Audit large-site exceeded the {error_ratio:.2f}x time-ratio error cap: "
            f"baseline={baseline_elapsed:.2f}s, scaled={scaled_elapsed:.2f}s, "
            f"ratio={ratio:.2f}x."
        )
    status = "WARN" if ratio > warning_ratio else "PASS"
    return status, ratio, warning_ratio, error_ratio


def _audit_large_site_caps(factor: int) -> tuple[float, float]:
    """Return measured-curve caps for one Audit large-site workload.

    The controlled 1000x fixture has materially fewer changed rows than the
    fully changed 10x through 500x fixtures, so it has separately measured
    caps instead of extrapolating either workload beyond its observed shape.
    """
    if factor == _EXTREME_AUDIT_SCALE:
        return _EXTREME_AUDIT_WARNING_RATIO, _EXTREME_AUDIT_FAILURE_RATIO
    expected_ratio = 1.0 + factor / _AUDIT_LARGE_SITE_SCALE_DIVISOR
    return (
        expected_ratio * _SCALING_WARNING_MULTIPLIER,
        expected_ratio * _SCALING_FAILURE_MULTIPLIER,
    )


def _audit_history_scaling_result(
    baseline_elapsed: float,
    scaled_elapsed: float,
) -> tuple[str, float, float, float]:
    """Return fixed caps for the complete fivefold Audit history workload."""
    if baseline_elapsed <= 0:
        raise ValueError("Audit long-history baseline time must be greater than zero.")
    ratio = scaled_elapsed / baseline_elapsed
    if ratio > _AUDIT_HISTORY_FAILURE_RATIO:
        raise RuntimeError(
            "Audit long-history exceeded the "
            f"{_AUDIT_HISTORY_FAILURE_RATIO:.2f}x time-ratio error cap: "
            f"baseline={baseline_elapsed:.2f}s, scaled={scaled_elapsed:.2f}s, "
            f"ratio={ratio:.2f}x."
        )
    status = "WARN" if ratio > _AUDIT_HISTORY_WARNING_RATIO else "PASS"
    return (
        status,
        ratio,
        _AUDIT_HISTORY_WARNING_RATIO,
        _AUDIT_HISTORY_FAILURE_RATIO,
    )


def _scaled_timeout(baseline_elapsed: float, error_ratio: float) -> float:
    """Return a process-safety timeout beyond the performance failure boundary.

    The grace interval allows the command to finish and report its measured
    ratio normally. It does not change the performance threshold applied after
    completion.
    """
    return baseline_elapsed * error_ratio + _PROCESS_TIMEOUT_GRACE_SECONDS


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
    security_master: pl.DataFrame,
    scale: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return proportionally scaled performance and aligned reference rows."""
    performance_copies: list[pl.DataFrame] = []
    reference_copies: list[pl.DataFrame] = []
    for copy_number in range(scale):
        suffix = f"_LOAD_{copy_number:02d}"
        performance_copies.append(
            security_performance.with_columns(
                (pl.col("Security Symbol") + suffix).alias("Security Symbol"),
                (pl.col("Beginning Weight") / scale).alias("Beginning Weight"),
                (pl.col("Contribution") / scale).alias("Contribution"),
            )
        )
        reference_copies.append(
            security_master.with_columns(
                (pl.col("Security Symbol") + suffix).alias("Security Symbol"),
                (pl.col("Security Name") + suffix).alias("Security Name"),
            )
        )
    return (
        pl.concat(performance_copies, how="vertical"),
        pl.concat(reference_copies, how="vertical"),
    )


def _expanded_history_frame(source: pl.DataFrame, scale: int) -> pl.DataFrame:
    """Return chronologically shifted, non-overlapping copies of period rows."""
    dated = source.with_columns(
        pl.col("From Date").str.to_date(),
        pl.col("Thru Date").str.to_date(),
    )
    copies: list[pl.DataFrame] = []
    for copy_number in range(scale):
        offset = f"{copy_number * _HISTORY_BLOCK_YEARS}y"
        copies.append(
            dated.with_columns(
                pl.col("From Date").dt.offset_by(offset),
                pl.col("Thru Date").dt.offset_by(offset),
            )
        )
    return pl.concat(copies, how="vertical")


def _expanded_audit_history_frame(source: pl.DataFrame, scale: int) -> pl.DataFrame:
    """Return consistently date-shifted copies of one Audit source table."""
    date_columns = [column for column in source.columns if column.endswith(" Date")]
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
    (directory / "axys_apx_analytics.yaml").rename(directory / "ppar.yaml")
    for file_name in ("portperf.csv", "secperf.csv"):
        frame = _expanded_frame(
            directory / file_name,
            scale,
            ("Portfolio Code",),
        )
        frame.write_csv(directory / file_name)
    return directory, pl.read_csv(directory / "secperf.csv").height


def _prepare_audit(
    directory: Path,
    scale: int,
    *,
    changed_portfolios: frozenset[str] | None = None,
) -> tuple[Path, int]:
    """Write one temporary Audit site and return its aggregate CSV row count."""
    shutil.copytree(_AUDIT_TEMPLATE, directory)
    (directory / "axys_apx_audit.yaml").rename(
        directory / "ppar.yaml"
    )
    if changed_portfolios is not None:
        _retain_audit_changes_for_portfolios(directory, changed_portfolios)
    row_count = 0
    for snapshot_name in ("snapshot_a", "snapshot_b"):
        snapshot = directory / snapshot_name
        for path in snapshot.glob("*.csv"):
            frame = _expanded_frame(
                path,
                scale,
                ("Portfolio Code",),
            )
            frame.write_csv(path)
            row_count += frame.height
    return directory, row_count


def _retain_audit_changes_for_portfolios(
    site: Path,
    changed_portfolios: frozenset[str],
) -> None:
    """Make nonselected Snapshot B portfolios identical to Snapshot A.

    The controlled 1000x workload still reads and compares every copied input
    row. Restricting actual differences prevents synthetic reviewer output from
    crossing the same 100,000-row production ceiling that this test must retain.
    """
    snapshot_a = site / "snapshot_a"
    snapshot_b = site / "snapshot_b"
    for snapshot_b_path in snapshot_b.glob("*.csv"):
        snapshot_a_path = snapshot_a / snapshot_b_path.name
        if not snapshot_a_path.is_file():
            continue
        frame_a = pl.read_csv(snapshot_a_path)
        frame_b = pl.read_csv(snapshot_b_path)
        portfolio_column = next(
            (
                column_name
                for column_name in ("Portfolio Code",)
                if column_name in frame_a.columns and column_name in frame_b.columns
            ),
            None,
        )
        if portfolio_column is None:
            continue
        selected = list(changed_portfolios)
        retained_changes = frame_b.filter(pl.col(portfolio_column).is_in(selected))
        unchanged_rows = frame_a.filter(~pl.col(portfolio_column).is_in(selected))
        pl.concat((retained_changes, unchanged_rows)).write_csv(snapshot_b_path)


def _audit_changed_portfolio_scope(scale: int) -> frozenset[str] | None:
    """Return the controlled changed-portfolio scope for an Audit scale."""
    if scale == _EXTREME_AUDIT_SCALE:
        return _EXTREME_AUDIT_CHANGED_PORTFOLIOS
    return None


def _analytics_large_site_scale(requested_scale: int) -> int:
    """Return the established Analytics workload for a combined scale run."""
    return min(requested_scale, _MAX_ANALYTICS_LARGE_SITE_SCALE)


def _prepare_long_history_audit(directory: Path) -> tuple[Path, int, int]:
    """Write a fixed 5x Audit history and return rows and static-reference rows."""
    shutil.copytree(_AUDIT_TEMPLATE, directory)
    (directory / "axys_apx_audit.yaml").rename(
        directory / "ppar.yaml"
    )
    row_count = 0
    static_reference_rows = 0
    for snapshot_name in ("snapshot_a", "snapshot_b"):
        snapshot = directory / snapshot_name
        for path in snapshot.glob("*.csv"):
            source = pl.read_csv(path)
            if path.name == "secmast.csv":
                expanded = source
                static_reference_rows += source.height
            else:
                expanded = _expanded_audit_history_frame(
                    source,
                    _LONG_HISTORY_SCALE,
                )
            expanded.write_csv(path)
            row_count += expanded.height
    return directory, row_count, static_reference_rows


def _prepare_selected_analytics(directory: Path, scale: int) -> tuple[Path, int]:
    """Write an Analytics site whose selected pair has unique copied securities."""
    shutil.copytree(_ANALYTICS_TEMPLATE, directory)
    (directory / "axys_apx_analytics.yaml").rename(directory / "ppar.yaml")

    security_performance = pl.read_csv(directory / "secperf.csv")
    security_master = pl.read_csv(directory / "secmast.csv")
    expanded_performance, expanded_reference = _expanded_selected_analytics_frames(
        security_performance,
        security_master,
        scale,
    )
    expanded_performance.write_csv(directory / "secperf.csv")
    expanded_reference.write_csv(directory / "secmast.csv")
    return directory, expanded_performance.height


def _prepare_long_history_analytics(directory: Path) -> tuple[Path, int, int]:
    """Write a fixed 5x Analytics history and return rows and period count."""
    shutil.copytree(_ANALYTICS_TEMPLATE, directory)
    yaml_path = directory / "axys_apx_analytics.yaml"
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
                expanded.filter(pl.col("Portfolio Code") == "MEGA_ALPHA")
                .select("From Date", "Thru Date")
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
    baseline_command = [sys.executable, "-m", "ppar.cli", "analytics", baseline_site]
    scaled_command = [sys.executable, "-m", "ppar.cli", "analytics", scaled_site]
    baseline_elapsed = _run_median_elapsed(baseline_command)
    timeout_seconds = _scaled_timeout(
        baseline_elapsed,
        _ANALYTICS_SCALING_FAILURE_RATIO,
    )
    try:
        scaled_elapsed = _run_median_elapsed(
            scaled_command,
            timeout_seconds=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        _print_timeout_result(
            "Analytics large-site",
            scale,
            baseline_rows,
            row_count,
            baseline_elapsed,
            timeout_seconds,
            _ANALYTICS_SCALING_WARNING_RATIO,
            _ANALYTICS_SCALING_FAILURE_RATIO,
        )
        raise RuntimeError(
            "Analytics large-site exceeded its process-safety timeout."
        ) from error
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
    changed_portfolios = _audit_changed_portfolio_scope(scale)
    baseline_site, baseline_rows = _prepare_audit(
        baseline_path,
        1,
        changed_portfolios=changed_portfolios,
    )
    site, row_count = _prepare_audit(
        scaled_path,
        scale,
        changed_portfolios=changed_portfolios,
    )
    baseline_elapsed = _run(
        [sys.executable, "-m", "ppar.cli", "audit", baseline_site]
    )
    warning_ratio, error_ratio = _audit_large_site_caps(scale)
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
    for report_name in ("portfolio", "security"):
        baseline_report_path = baseline_site / "output" / report_name
        scaled_report_path = site / "output" / report_name
        for report_path in (baseline_report_path, scaled_report_path):
            _run(
                [
                    sys.executable,
                    "-m",
                    "ppar.audit.cli.validate_bundle",
                    report_path,
                ]
            )
        audit_scale_contract.assert_scaled_audit_equivalent(
            baseline_report_path,
            scaled_report_path,
            scale,
        )
        audit_scale_contract.print_output_metrics(report_name, scaled_report_path)
    status, _, warning_ratio, error_ratio = _audit_large_site_scaling_result(
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
    site, row_count, static_reference_rows = _prepare_long_history_audit(site_path)
    expected_rows = (
        (baseline_rows - static_reference_rows) * _LONG_HISTORY_SCALE
        + static_reference_rows
    )
    if row_count != expected_rows:
        raise RuntimeError(
            "Audit long-history row count differs from the expected 5x history: "
            f"expected={expected_rows}, actual={row_count}."
        )

    baseline_elapsed = _run(
        [sys.executable, "-m", "ppar.cli", "audit", baseline_site]
    )
    error_ratio = _AUDIT_HISTORY_FAILURE_RATIO
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
                "ppar.audit.cli.validate_bundle",
                report_path,
            ]
        )
        findings = audit_scale_contract.read_supporting_csv(
            report_path,
            "findings.csv",
        )
        baseline_findings = audit_scale_contract.read_supporting_csv(
            baseline_report_path,
            "findings.csv",
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

    status, _, warning_ratio, error_ratio = _audit_history_scaling_result(
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
        pl.col("Portfolio Code") == "MEGA_ALPHA"
    )
    periods = performance.select("From Date", "Thru Date").sort("From Date")
    if not periods["From Date"].is_sorted() or not (
        periods["From Date"].slice(1)
        > periods["Thru Date"].slice(0, periods.height - 1)
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
            analytics_scale = _analytics_large_site_scale(args.scale)
            if analytics_scale != args.scale:
                print(
                    f"INFO Analytics large-site remains {analytics_scale}x; "
                    f"Audit large-site runs at {args.scale}x."
                )
            _check_analytics(workspace, analytics_scale)
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
