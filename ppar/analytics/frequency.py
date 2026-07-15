"""Define reporting frequencies and calendar-aligned period helpers."""

# Python imports
import calendar
import datetime as dt
from enum import Enum
from typing import Sequence

# Project imports
from ppar.errors import PpaError


class Frequency(Enum):
    """Enumeration of supported reporting frequencies.

    The enumeration values are used throughout the analytics pipeline to
    determine how performance data should be grouped and consolidated.

    Attributes:
        AS_OFTEN_AS_POSSIBLE: Use the native frequency of the supplied data
            without additional consolidation.
        MONTHLY: Consolidate data to calendar month-end periods.
        QUARTERLY: Consolidate data to calendar quarter-end periods.
        YEARLY: Consolidate data to calendar year-end periods.
    """

    AS_OFTEN_AS_POSSIBLE = "Periodic"  # As often as possible based on the frequency of the data.
    MONTHLY = "Monthly"  # Currently only supports calendar month-end, not business month-end.
    QUARTERLY = "Quarterly"  # Calendar quarters.
    YEARLY = "Yearly"  # Calendar years.


def date_matches_frequency(date: dt.date, frequency: Frequency) -> bool:
    """Determine whether a date aligns with a reporting frequency.

    Args:
        date: The date to evaluate.
        frequency: The reporting frequency to test against.

    Returns:
        ``True`` if the date aligns with the specified reporting frequency;
        otherwise, ``False``.
    """
    match frequency:
        case Frequency.AS_OFTEN_AS_POSSIBLE:
            return True
        case Frequency.MONTHLY:
            return _is_calendar_month_end(date)
        case Frequency.QUARTERLY:
            return date.month in (3, 6, 9, 12) and _is_calendar_month_end(date)
        case Frequency.YEARLY:
            return date.month == 12 and _is_calendar_month_end(date)


def _is_calendar_month_end(date: dt.date) -> bool:
    """Determine whether a date is a calendar month-end date.

    Args:
        date: The date to evaluate.

    Returns:
        ``True`` if the date is the final calendar day of the month;
        otherwise, ``False``.
    """
    if date.day < 28:
        return False
    return date.day == calendar.monthrange(date.year, date.month)[1]


def periods_per_year(frequency: Frequency) -> int:
    """Return the number of reporting periods in a calendar year.

    Args:
        frequency: The reporting frequency.

    Returns:
        The number of periods per calendar year for the specified frequency.

    Raises:
        PpaError: If ``frequency`` is
            ``Frequency.AS_OFTEN_AS_POSSIBLE`` because a fixed annual period
            count cannot be determined for that frequency.
    """
    match frequency:
        case Frequency.MONTHLY:
            return 12
        case Frequency.QUARTERLY:
            return 4
        case Frequency.YEARLY:
            return 1
        case _:  # Frequency.AS_OFTEN_AS_POSSIBLE
            # This method requires a fixed reporting frequency.
            raise PpaError(f"Unhandled Frequency {frequency}", 999)


def validate_frequency_coverage(
    periods: Sequence[tuple[dt.date, dt.date]],
    frequency: Frequency,
) -> None:
    """Validate that fixed-frequency source data skips no reporting bucket.

    Args:
        periods: Ordered inclusive source ``(from_date, thru_date)`` periods.
        frequency: Frequency that the dates must follow.

    Raises:
        PpaError: If no source period overlaps a required calendar month,
            quarter, or year between the first and last source periods.

    Notes:
        A source period need not end on a calendar boundary, and business-day
        data legitimately omits weekends and holidays. Coverage is therefore
        evaluated at the requested reporting frequency rather than by requiring
        adjacent calendar dates.
    """
    if frequency == Frequency.AS_OFTEN_AS_POSSIBLE or not periods:
        return

    def _bucket(date: dt.date) -> int:
        if frequency == Frequency.MONTHLY:
            return date.year * 12 + date.month - 1
        if frequency == Frequency.QUARTERLY:
            return date.year * 4 + (date.month - 1) // 3
        return date.year

    def _bucket_label(bucket: int) -> str:
        if frequency == Frequency.MONTHLY:
            year, zero_based_month = divmod(bucket, 12)
            return f"{year:04d}-{zero_based_month + 1:02d}"
        if frequency == Frequency.QUARTERLY:
            year, zero_based_quarter = divmod(bucket, 4)
            return f"{year:04d}-Q{zero_based_quarter + 1}"
        return str(bucket)

    covered_buckets: set[int] = set()
    for from_date, thru_date in periods:
        covered_buckets.update(range(_bucket(from_date), _bucket(thru_date) + 1))
    first_bucket = min(_bucket(from_date) for from_date, _ in periods)
    last_bucket = max(_bucket(thru_date) for _, thru_date in periods)
    missing_buckets = [
        bucket
        for bucket in range(first_bucket, last_bucket + 1)
        if bucket not in covered_buckets
    ]
    if missing_buckets:
        missing_labels = [_bucket_label(bucket) for bucket in missing_buckets]
        raise PpaError(
            f"missing {frequency.value.lower()} coverage for {missing_labels}.",
            253,
            context={
                "frequency": frequency.value,
                "missing_periods": missing_labels,
            },
        )
