"""Audit or rebuild derived performance-comparison demo CSV files.

The packaged performance-comparison demos keep user-visible operational inputs
in ``holdings.csv``, ``prices.csv``, and ``transactions.csv``. The
``secperf.csv`` and ``portperf.csv`` files are derived review targets. This
script keeps the derived performance files internally aligned by:

1. recomputing security beginning weights and contributions from ``secperf``;
2. deriving portfolio performance rows from the resulting security rows; and
3. reporting whether the checked-in files already match those derived values.

By default, the script audits without writing. Pass ``--write`` to update the
packaged demo files.
"""

from __future__ import annotations

# Python imports
import argparse
import json
from pathlib import Path
from typing import Final

# Third-party imports
import pandas as pd


_REPO_ROOT: Final = Path(__file__).resolve().parents[2]
_DEFAULT_AXYS_DIRECTORY: Final = _REPO_ROOT / "ppar" / "demos" / "data" / "axys"
_SNAPSHOT_DIRECTORIES: Final = ("axys_full_spec_a", "axys_full_spec_b")
_PERIOD_KEY: Final = ["PORTFOLIO_CODE", "FROM_DATE", "THRU_DATE"]
_PORTPERF_COLUMNS: Final = [
    "END_MV",
    "FLOW",
    "INCOME",
    "GAIN_LOSS",
    "PORTFOLIO_CODE",
    "PORTFOLIO_NAME",
    "PERIOD_ID",
    "FROM_DATE",
    "THRU_DATE",
    "BEGIN_MV",
    "PORT_RETURN",
]
_SECPERF_NUMERIC_COLUMNS: Final = [
    "END_MV",
    "INCOME",
    "GAIN_LOSS",
    "BEGIN_WEIGHT",
    "BEGIN_MV",
    "SEC_RETURN",
    "CONTRIBUTION",
]
_PORTPERF_NUMERIC_COLUMNS: Final = [
    "END_MV",
    "FLOW",
    "INCOME",
    "GAIN_LOSS",
    "BEGIN_MV",
    "PORT_RETURN",
]
_CHECK_TOLERANCE: Final = 0.000000001


def main() -> int:
    """Audit or rewrite packaged performance-comparison demo performance files."""
    args = _parse_args()
    summary = rebuild_demo_performance_files(args.axys_directory, write=args.write)
    print(json.dumps(summary, indent=2))

    if args.write:
        return 0
    if any(snapshot["has_drift"] for snapshot in summary["snapshots"]):
        return 1
    return 0


def rebuild_demo_performance_files(
    axys_directory: Path,
    *,
    write: bool = False,
) -> dict[str, object]:
    """Return audit summary, optionally rewriting derived performance files.

    Args:
        axys_directory: Directory containing ``axys_full_spec_a`` and
            ``axys_full_spec_b``.
        write: Whether to write rebuilt ``secperf.csv`` and ``portperf.csv``.

    Returns:
        JSON-serializable audit summary with one entry per snapshot.
    """
    snapshots: list[dict[str, object]] = []
    for snapshot_name in _SNAPSHOT_DIRECTORIES:
        snapshot_directory = axys_directory / snapshot_name
        current_secperf = pd.read_csv(snapshot_directory / "secperf.csv")
        current_portperf = pd.read_csv(snapshot_directory / "portperf.csv")

        rebuilt_secperf = _rebuild_security_performance(current_secperf)
        rebuilt_portperf = _rebuild_portfolio_performance(
            current_portperf,
            rebuilt_secperf,
        )
        secperf_delta = _max_numeric_delta(
            current_secperf,
            rebuilt_secperf,
            _SECPERF_NUMERIC_COLUMNS,
        )
        portperf_delta = _max_numeric_delta(
            current_portperf,
            rebuilt_portperf,
            _PORTPERF_NUMERIC_COLUMNS,
        )
        has_drift = (
            secperf_delta > _CHECK_TOLERANCE or portperf_delta > _CHECK_TOLERANCE
        )
        if write:
            rebuilt_secperf.to_csv(snapshot_directory / "secperf.csv", index=False)
            rebuilt_portperf.to_csv(snapshot_directory / "portperf.csv", index=False)

        snapshots.append(
            {
                "snapshot": snapshot_name,
                "secperf_rows": int(rebuilt_secperf.shape[0]),
                "portperf_rows": int(rebuilt_portperf.shape[0]),
                "max_secperf_numeric_delta": secperf_delta,
                "max_portperf_numeric_delta": portperf_delta,
                "has_drift": has_drift,
                "written": write,
            }
        )

    return {
        "axys_directory": str(axys_directory),
        "mode": "write" if write else "check",
        "snapshots": snapshots,
    }


def _rebuild_security_performance(secperf: pd.DataFrame) -> pd.DataFrame:
    """Return security rows with weights and contributions recomputed."""
    rebuilt = secperf.copy()
    period_begin_market_value = rebuilt.groupby(_PERIOD_KEY)["BEGIN_MV"].transform("sum")
    rebuilt["BEGIN_WEIGHT"] = (rebuilt["BEGIN_MV"] / period_begin_market_value).round(10)
    rebuilt["CONTRIBUTION"] = (rebuilt["BEGIN_WEIGHT"] * rebuilt["SEC_RETURN"]).round(10)
    return rebuilt


def _rebuild_portfolio_performance(
    portperf: pd.DataFrame,
    secperf: pd.DataFrame,
) -> pd.DataFrame:
    """Return portfolio rows derived from security performance rows."""
    aggregate = (
        secperf.groupby(_PERIOD_KEY, as_index=False)
        .agg(
            BEGIN_MV=("BEGIN_MV", "sum"),
            END_MV=("END_MV", "sum"),
            INCOME=("INCOME", "sum"),
            GAIN_LOSS=("GAIN_LOSS", "sum"),
            PORT_RETURN=("CONTRIBUTION", "sum"),
        )
        .reset_index(drop=True)
    )
    rebuilt = (
        portperf.drop(
            columns=["BEGIN_MV", "END_MV", "INCOME", "GAIN_LOSS", "PORT_RETURN"]
        )
        .merge(aggregate, on=_PERIOD_KEY, how="left")
        .reset_index(drop=True)
    )
    rebuilt["FLOW"] = 0.0
    rebuilt = rebuilt[_PORTPERF_COLUMNS]
    for column in ["END_MV", "FLOW", "INCOME", "GAIN_LOSS", "BEGIN_MV"]:
        rebuilt[column] = rebuilt[column].round(2)
    rebuilt["PORT_RETURN"] = rebuilt["PORT_RETURN"].round(10)
    return rebuilt


def _max_numeric_delta(
    current: pd.DataFrame,
    rebuilt: pd.DataFrame,
    numeric_columns: list[str],
) -> float:
    """Return the maximum absolute numeric difference between aligned frames."""
    if current.shape != rebuilt.shape:
        return float("inf")

    max_delta = 0.0
    for column in numeric_columns:
        current_values = pd.to_numeric(current[column], errors="coerce")
        rebuilt_values = pd.to_numeric(rebuilt[column], errors="coerce")
        column_delta = (current_values - rebuilt_values).abs().max()
        if pd.notna(column_delta):
            max_delta = max(max_delta, float(column_delta))
    return max_delta


def _parse_args() -> argparse.Namespace:
    """Return command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Audit or rebuild derived portperf/secperf files for the packaged "
            "performance-comparison demos."
        )
    )
    parser.add_argument(
        "--axys-directory",
        type=Path,
        default=_DEFAULT_AXYS_DIRECTORY,
        help="Directory containing axys_full_spec_a and axys_full_spec_b.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Rewrite secperf.csv and portperf.csv instead of audit-only mode.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
