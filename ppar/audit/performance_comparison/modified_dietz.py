"""Modified Dietz helper calculations for performance comparison diagnostics."""

from __future__ import annotations

# Python imports
import datetime as dt
import math

# Project imports
from ppar.performance_comparison.methods import ModifiedDietzInclusionRule


def modified_dietz_external_flow_impact(
    *,
    flow_delta: float,
    denominator: float,
    from_date: dt.date,
    thru_date: dt.date,
    flow_date: dt.date,
    inclusion_rule: str,
) -> float:
    """Return a Modified Dietz cross-check estimate for one external flow.

    Args:
        flow_delta: Snapshot B minus snapshot A external-flow amount.
        denominator: Return denominator for the portfolio period.
        from_date: Portfolio-period start date.
        thru_date: Portfolio-period end date.
        flow_date: Date used to time the external flow.
        inclusion_rule: Whether the flow is treated as beginning- or end-of-day.

    Returns:
        Decimal return impact estimate for the flow.

    Raises:
        ValueError: If the denominator is zero, the period is invalid, the flow
            date is outside the period, or the inclusion rule is unsupported.

    Notes:
        This helper supports the cross-check-only Modified Dietz diagnostic
        path. It should not be treated as regular additive attribution unless
        double-counting behavior is explicitly modeled.
    """
    if not math.isfinite(flow_delta):
        raise ValueError("flow_delta must be finite")
    if not math.isfinite(denominator) or denominator == 0:
        raise ValueError("denominator must be finite and nonzero")
    flow_weight = modified_dietz_flow_weight(
        from_date=from_date,
        thru_date=thru_date,
        flow_date=flow_date,
        inclusion_rule=inclusion_rule,
    )
    return flow_delta * flow_weight / denominator


def modified_dietz_flow_weight(
    *,
    from_date: dt.date,
    thru_date: dt.date,
    flow_date: dt.date,
    inclusion_rule: str,
) -> float:
    """Return the actual-days Modified Dietz flow weight.

    Args:
        from_date: Portfolio-period start date.
        thru_date: Portfolio-period end date.
        flow_date: Date used to time the external flow.
        inclusion_rule: Whether the flow is treated as beginning- or end-of-day.

    Returns:
        Weight applied to the external-flow amount.

    Raises:
        ValueError: If the period is invalid, the flow date is outside the
            period, or the inclusion rule is unsupported.
    """
    period_days = (thru_date - from_date).days + 1
    if period_days <= 0:
        raise ValueError("period must include at least one day")
    if not from_date <= flow_date <= thru_date:
        raise ValueError("flow_date must be inside the period")

    remaining_days = (thru_date - flow_date).days
    if inclusion_rule == ModifiedDietzInclusionRule.BEGINNING_OF_DAY.value:
        remaining_days += 1
    elif inclusion_rule != ModifiedDietzInclusionRule.END_OF_DAY.value:
        raise ValueError("inclusion_rule must be beginning_of_day or end_of_day")
    return remaining_days / period_days


def usable_modified_dietz_denominator(value: object) -> bool:
    """Return whether a configured Modified Dietz denominator is usable.

    Args:
        value: Candidate denominator value.

    Returns:
        ``True`` when the value is finite, numeric, non-boolean, and nonzero.
    """
    number = modified_dietz_float(value)
    return number is not None and number != 0


def usable_modified_dietz_number(value: object) -> bool:
    """Return whether a value can be used in Modified Dietz arithmetic.

    Args:
        value: Candidate numeric value.

    Returns:
        ``True`` when the value is finite, numeric, and non-boolean.
    """
    return modified_dietz_float(value) is not None


def modified_dietz_float(value: object) -> float | None:
    """Return a float for finite non-boolean numeric Modified Dietz values.

    Args:
        value: Candidate numeric value.

    Returns:
        Float value, or ``None`` for booleans, nonnumeric values, NaN, or
        infinity.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else None
    return None
