"""Load normalized FX rate comparison sources."""

from __future__ import annotations

# Third-party imports
import polars as pl

# Project imports
from ppar.errors import PpaError
from ppar.audit import aliases
from ppar.audit import schema as pc_cols
from ppar.audit.currency_basis import normalize_currency_columns
from ppar.audit import source_loader
from ppar.audit.portfolio_performance import SnapshotKey
from ppar.audit.specification import AuditSpecification
import ppar.utilities as util


_OPTIONAL_KEY_COLUMNS = (
    pc_cols.PORTFOLIO_ID,
    pc_cols.RATE_SOURCE,
    pc_cols.RATE_TYPE,
)


def _validate_required_values(
    frame: pl.DataFrame,
    *,
    path: util.PathLike,
    specification_path: util.PathLike,
) -> None:
    """Validate required normalized FX values and rate semantics.

    Args:
        frame: Normalized FX rate rows.
        path: Source CSV path.
        specification_path: Comparison YAML path for error context.

    Raises:
        PpaError: If a currency or date is blank, or a rate is not finite and
            strictly positive.
    """
    for column in (pc_cols.FROM_CURRENCY, pc_cols.TO_CURRENCY):
        invalid_rows = frame.filter(
            pl.col(column).is_null()
            | (pl.col(column).cast(pl.String).str.strip_chars() == "")
        )
        if not invalid_rows.is_empty():
            raise PpaError(
                (
                    f"{specification_path}: fx_rates column {column!r} contains "
                    f"a blank value in {str(path)!r}."
                ),
                502,
            )

    if frame.filter(pl.col(pc_cols.RATE_DATE).is_null()).height:
        raise PpaError(
            (
                f"{specification_path}: fx_rates column {pc_cols.RATE_DATE!r} "
                f"contains a blank value in {str(path)!r}."
            ),
            502,
        )

    numeric_rate = pl.col(pc_cols.FX_RATE).cast(pl.Float64, strict=False)
    invalid_rates = frame.filter(
        numeric_rate.is_null()
        | numeric_rate.is_nan()
        | numeric_rate.is_infinite()
        | (numeric_rate <= 0.0)
    )
    if invalid_rates.is_empty():
        return

    invalid_rate = invalid_rates.get_column(pc_cols.FX_RATE)[0]
    raise PpaError(
        (
            f"{specification_path}: fx_rates column {pc_cols.FX_RATE!r} must "
            f"contain finite positive rates; found {invalid_rate!r} in "
            f"{str(path)!r}."
        ),
        502,
    )


def _validate_unique_rows(
    frame: pl.DataFrame,
    *,
    path: util.PathLike,
) -> None:
    """Reject duplicate rows for the normalized FX comparison key.

    Args:
        frame: Validated normalized FX rate rows.
        path: Source CSV path.

    Raises:
        PpaError: If more than one row has the same pair, date, and available
            source/type provenance.
    """
    key_columns = (
        pc_cols.FROM_CURRENCY,
        pc_cols.TO_CURRENCY,
        pc_cols.RATE_DATE,
        *(column for column in _OPTIONAL_KEY_COLUMNS if column in frame.columns),
    )
    duplicate_keys = (
        frame.group_by(list(key_columns))
        .len(name="_duplicate_count")
        .filter(pl.col("_duplicate_count") > 1)
    )
    if duplicate_keys.is_empty():
        return

    duplicate_key = duplicate_keys.select(key_columns).row(0, named=True)
    raise PpaError(
        f"fx_rates contains duplicate rows in {str(path)!r}: {duplicate_key}",
        112,
    )


class FxRatesLoader:
    """Load normalized FX rate rows for comparison snapshots.

    Attributes:
        _specification: Parsed comparison specification.
    """

    def __init__(self, specification: AuditSpecification) -> None:
        """Initialize the FX rates loader.

        Args:
            specification: Parsed comparison specification containing resolved
                snapshot and file paths.
        """
        self._specification = specification

    def load(self, snapshot_key: SnapshotKey) -> pl.DataFrame | None:
        """Load one snapshot's normalized FX rate rows.

        Args:
            snapshot_key: Snapshot side to load, either ``"a"`` or ``"b"``.

        Returns:
            FX rate rows with normalized comparison column names, or ``None``
            when the optional dataset is omitted or missing.

        Raises:
            PpaError: If the source exists but required columns cannot be
                resolved.
        """
        path = source_loader.optional_file_path(
            self._specification,
            pc_cols.FX_RATES,
            snapshot_key,
        )
        if path is None or not util.file_path_exists(path):
            return None
        cached = source_loader.cached_normalized_frame(
            self._specification.path,
            pc_cols.FX_RATES,
            snapshot_key,
            path,
        )
        if cached is not None:
            return cached

        frame = source_loader.read_mapped_csv(
            path,
            pc_cols.FX_RATES_COLUMNS,
            pc_cols.FX_RATES,
            aliases.FX_RATES_REQUIRED_ALIASES,
            aliases.FX_RATES_OPTIONAL_ALIASES,
            self._specification.path,
        ).with_columns(
            pl.col(pc_cols.RATE_DATE).str.strptime(pl.Date, "%Y-%m-%d", strict=True),
        )
        frame = normalize_currency_columns(
            source_loader.require_numeric_columns(
                frame,
                columns=(pc_cols.FX_RATE, pc_cols.LOCAL_EXPOSURE),
                dataset_name=pc_cols.FX_RATES,
                path=path,
                specification_path=self._specification.path,
            )
        )
        _validate_required_values(
            frame,
            path=path,
            specification_path=self._specification.path,
        )
        _validate_unique_rows(frame, path=path)
        return source_loader.cache_normalized_frame(
            self._specification.path,
            pc_cols.FX_RATES,
            snapshot_key,
            path,
            frame,
        )
