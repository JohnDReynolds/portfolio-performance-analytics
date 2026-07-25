"""Build Modified Dietz formula rows for Audit workbook review tables."""

from __future__ import annotations

# Python imports
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

# Project imports
import ppar.common as util
from ppar.audit import schema as audit_schema
from ppar.audit import workbook_layout as layout
from ppar.audit import workbook_reconstruction
from ppar.audit import workbook_rows as rows
from ppar.audit.performance_comparison import explain
from ppar.audit.performance_comparison import findings
from ppar.audit.performance_comparison import return_reconstruction
from ppar.audit.rendering import format_value

FORMULA_FINDING_CODE = "reconstruction_formula_input"
FORMULA_ROW_TYPE = "Formula Input"
RECONSTRUCTION_COMPONENTS = "_workbook_reconstruction_components"

BEGINNING_VALUE_FIELD = "beginning_market_value"
ENDING_VALUE_FIELD = "ending_market_value"
NET_FLOW_FIELD = "net_flow"
WEIGHTED_FLOW_FIELD = "weighted_flow"
INCOME_FIELD = "income"

__all__ = [
    "BEGINNING_VALUE_FIELD",
    "ENDING_VALUE_FIELD",
    "FORMULA_FINDING_CODE",
    "FORMULA_ROW_TYPE",
    "INCOME_FIELD",
    "NET_FLOW_FIELD",
    "RECONSTRUCTION_COMPONENTS",
    "WEIGHTED_FLOW_FIELD",
    "portfolio_reconstruction_formula_rows",
    "security_reconstruction_formula_rows",
]

_INPUT_ROLE_PERFORMANCE_INPUT = "Performance Input"
_IMPACT_STATUS_ESTIMATED = "Estimated"
_UNEXPLAINED_TOLERANCE = 0.0000005
_ROLE_METADATA = {
    BEGINNING_VALUE_FIELD: (
        audit_schema.HOLDINGS,
        BEGINNING_VALUE_FIELD,
        "Beginning holdings market value",
    ),
    ENDING_VALUE_FIELD: (
        audit_schema.HOLDINGS,
        ENDING_VALUE_FIELD,
        "Ending holdings market value",
    ),
    NET_FLOW_FIELD: (
        audit_schema.TRANSACTIONS,
        NET_FLOW_FIELD,
        "Transaction net flow",
    ),
    WEIGHTED_FLOW_FIELD: (
        audit_schema.TRANSACTIONS,
        WEIGHTED_FLOW_FIELD,
        "Transaction weighted flow",
    ),
    INCOME_FIELD: (
        audit_schema.TRANSACTIONS,
        INCOME_FIELD,
        "Transaction income",
    ),
}


@dataclass(frozen=True)
class _FormulaRowValues:
    """Values that vary across a generated reconstruction formula row.

    Attributes:
        dataset: Source dataset containing the formula input.
        source_column: Source-facing formula input field.
        role_label: Reviewer-facing formula input label.
        snapshot_a_value: Formula input value from snapshot A.
        snapshot_b_value: Formula input value from snapshot B.
        difference: Snapshot B minus snapshot A input difference.
        estimated_impact: Estimated return impact of the input difference.
        as_of_date: Date associated with the formula input.
        security_id: Optional security identifier for security-level review.
        guidance: Reviewer guidance for interpreting the input difference.
    """

    dataset: str
    source_column: str
    role_label: str
    snapshot_a_value: object
    snapshot_b_value: object
    difference: object
    estimated_impact: float
    as_of_date: object
    security_id: object
    guidance: str


def portfolio_reconstruction_formula_rows(
    comparison_path: util.PathLike | None,
    *,
    active_keys: set[tuple[object, object, object]] | None = None,
    reconstruction_cache: workbook_reconstruction.WorkbookReconstructionCache | None = None,
) -> list[dict[str, object]]:
    """Return portfolio reconstruction formula rows for cause review.

    Args:
        comparison_path: Optional path to the Audit comparison specification.
        active_keys: Optional portfolio-period keys that should be included.
        reconstruction_cache: Optional shared reconstruction result cache.

    Returns:
        Formula component rows whose value or estimated impact changed.
    """
    resolved_cache = workbook_reconstruction.resolved_reconstruction_cache(
        comparison_path,
        reconstruction_cache,
    )
    checks = resolved_cache.portfolio_checks()
    if checks.is_empty():
        return []

    formula_rows: list[dict[str, object]] = []
    for source_row in checks.iter_rows(named=True):
        if active_keys is not None and rows.review_period_key(source_row) not in active_keys:
            continue
        formula_rows.extend(
            _formula_rows_for_check(
                source_row,
                row_factory=_portfolio_formula_row,
            )
        )
    return formula_rows


def security_reconstruction_formula_rows(
    comparison_path: util.PathLike | None,
    *,
    active_keys: set[tuple[object, object, object, object]] | None = None,
    reconstruction_cache: workbook_reconstruction.WorkbookReconstructionCache | None = None,
) -> list[dict[str, object]]:
    """Return security reconstruction formula rows for cause review.

    Args:
        comparison_path: Optional path to the Audit comparison specification.
        active_keys: Optional security-period keys that should be included.
        reconstruction_cache: Optional shared reconstruction result cache.

    Returns:
        Formula component rows whose value or estimated impact changed.
    """
    resolved_cache = workbook_reconstruction.resolved_reconstruction_cache(
        comparison_path,
        reconstruction_cache,
    )
    checks = resolved_cache.security_checks(active_keys=active_keys)
    if checks.is_empty():
        return []

    formula_rows: list[dict[str, object]] = []
    for source_row in checks.iter_rows(named=True):
        if (
            active_keys is not None
            and rows.security_review_period_key(source_row) not in active_keys
        ):
            continue
        formula_rows.extend(
            _formula_rows_for_check(
                source_row,
                row_factory=_security_formula_row,
            )
        )
    return formula_rows


def _formula_rows_for_check(
    source_row: Mapping[str, object],
    *,
    row_factory: Callable[..., dict[str, object]],
) -> list[dict[str, object]]:
    numerator_b = rows.number_or_none(
        source_row.get(return_reconstruction.DERIVED_NUMERATOR_B)
    )
    denominator_a = rows.number_or_none(
        source_row.get(return_reconstruction.DERIVED_DENOMINATOR_A)
    )
    denominator_b = rows.number_or_none(
        source_row.get(return_reconstruction.DERIVED_DENOMINATOR_B)
    )
    if (
        numerator_b is None
        or denominator_a is None
        or denominator_b is None
        or denominator_a == 0.0
        or denominator_b == 0.0
    ):
        return []

    denominator_effect = numerator_b * ((1.0 / denominator_b) - (1.0 / denominator_a))
    beginning_denominator_effect, weighted_flow_denominator_effect = (
        _denominator_component_effects(source_row, denominator_effect)
    )
    beginning_value_difference = rows.number_or_none(
        source_row.get(return_reconstruction.BEGIN_VALUE_DIFFERENCE)
    )
    ending_value_difference = rows.number_or_none(
        source_row.get(return_reconstruction.END_VALUE_DIFFERENCE)
    )
    net_flow_difference = rows.number_or_none(
        source_row.get(return_reconstruction.NET_FLOW_DIFFERENCE)
    )
    income_difference = rows.number_or_none(
        source_row.get(return_reconstruction.INCOME_DIFFERENCE)
    )
    formula_rows = [
        row_factory(
            source_row,
            field=BEGINNING_VALUE_FIELD,
            snapshot_a_value=source_row.get(return_reconstruction.BEGIN_VALUE_A),
            snapshot_b_value=source_row.get(return_reconstruction.BEGIN_VALUE_B),
            difference=beginning_value_difference,
            estimated_impact=(
                _component_impact(_negated_difference(beginning_value_difference), denominator_a)
                + beginning_denominator_effect
            ),
            as_of_date=source_row.get(return_reconstruction.BEGIN_VALUE_DATE_B),
        ),
        row_factory(
            source_row,
            field=ENDING_VALUE_FIELD,
            snapshot_a_value=source_row.get(return_reconstruction.END_VALUE_A),
            snapshot_b_value=source_row.get(return_reconstruction.END_VALUE_B),
            difference=ending_value_difference,
            estimated_impact=_component_impact(ending_value_difference, denominator_a),
            as_of_date=source_row.get(return_reconstruction.END_VALUE_DATE_B),
        ),
        row_factory(
            source_row,
            field=NET_FLOW_FIELD,
            snapshot_a_value=source_row.get(return_reconstruction.NET_FLOW_A),
            snapshot_b_value=source_row.get(return_reconstruction.NET_FLOW_B),
            difference=net_flow_difference,
            estimated_impact=_component_impact(
                -net_flow_difference if net_flow_difference is not None else None,
                denominator_a,
            ),
            as_of_date=source_row.get(return_reconstruction.RECONSTRUCTION_THRU_DATE),
        ),
        row_factory(
            source_row,
            field=WEIGHTED_FLOW_FIELD,
            snapshot_a_value=source_row.get(return_reconstruction.WEIGHTED_FLOW_A),
            snapshot_b_value=source_row.get(return_reconstruction.WEIGHTED_FLOW_B),
            difference=source_row.get(return_reconstruction.WEIGHTED_FLOW_DIFFERENCE),
            estimated_impact=weighted_flow_denominator_effect,
            as_of_date=source_row.get(return_reconstruction.RECONSTRUCTION_THRU_DATE),
        ),
    ]
    if income_difference is not None:
        formula_rows.append(
            row_factory(
                source_row,
                field=INCOME_FIELD,
                snapshot_a_value=source_row.get(return_reconstruction.INCOME_A),
                snapshot_b_value=source_row.get(return_reconstruction.INCOME_B),
                difference=income_difference,
                estimated_impact=_component_impact(income_difference, denominator_a),
                as_of_date=source_row.get(return_reconstruction.RECONSTRUCTION_THRU_DATE),
            )
        )
    return _nonzero_formula_rows(formula_rows)


def _denominator_component_effects(
    source_row: Mapping[str, object],
    denominator_effect: float,
) -> tuple[float, float]:
    """Return denominator effect allocated to beginning value and weighted flow."""
    beginning_value_difference = (
        rows.number_or_none(source_row.get(return_reconstruction.BEGIN_VALUE_DIFFERENCE))
        or 0.0
    )
    weighted_flow_difference = (
        rows.number_or_none(source_row.get(return_reconstruction.WEIGHTED_FLOW_DIFFERENCE))
        or 0.0
    )
    denominator_difference = rows.number_or_none(
        source_row.get(return_reconstruction.DERIVED_DENOMINATOR_DIFFERENCE)
    )
    if denominator_difference is None or abs(denominator_difference) <= _UNEXPLAINED_TOLERANCE:
        return 0.0, 0.0
    return (
        denominator_effect * (beginning_value_difference / denominator_difference),
        denominator_effect * (weighted_flow_difference / denominator_difference),
    )


def _component_impact(component_difference: float | None, denominator_a: float) -> float:
    """Return return impact for a numerator component difference."""
    if component_difference is None:
        return 0.0
    return component_difference / denominator_a


def _negated_difference(component_difference: float | None) -> float | None:
    """Return the opposite sign for a formula component difference."""
    if component_difference is None:
        return None
    return -component_difference


def _nonzero_formula_rows(
    formula_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Return formula rows with meaningful value or impact differences."""
    return [
        formula_row
        for formula_row in formula_rows
        if (
            abs(rows.number_or_none(formula_row.get(layout.CHANGE)) or 0.0)
            > _UNEXPLAINED_TOLERANCE
            or abs(rows.number_or_none(formula_row.get(layout.ESTIMATED_IMPACT)) or 0.0)
            > _UNEXPLAINED_TOLERANCE
        )
    ]


def _portfolio_formula_row(
    source_row: Mapping[str, object],
    *,
    field: str,
    snapshot_a_value: object,
    snapshot_b_value: object,
    difference: object,
    estimated_impact: float,
    as_of_date: object,
) -> dict[str, object]:
    """Return one promoted portfolio return-reconstruction formula row."""
    dataset, source_column, role_label = _ROLE_METADATA[field]
    return _formula_row(
        source_row,
        _FormulaRowValues(
            dataset=dataset,
            source_column=source_column,
            role_label=role_label,
            snapshot_a_value=snapshot_a_value,
            snapshot_b_value=snapshot_b_value,
            difference=difference,
            estimated_impact=estimated_impact,
            as_of_date=as_of_date,
            security_id=None,
            guidance=_portfolio_formula_guidance(field, role_label, difference),
        ),
    )


def _security_formula_row(
    source_row: Mapping[str, object],
    *,
    field: str,
    snapshot_a_value: object,
    snapshot_b_value: object,
    difference: object,
    estimated_impact: float,
    as_of_date: object,
) -> dict[str, object]:
    """Return one promoted security return-reconstruction formula row."""
    dataset, source_column, role_label = _ROLE_METADATA[field]
    security_id = source_row.get(return_reconstruction.RECONSTRUCTION_SECURITY_ID)
    return _formula_row(
        source_row,
        _FormulaRowValues(
            dataset=dataset,
            source_column=source_column,
            role_label=role_label,
            snapshot_a_value=snapshot_a_value,
            snapshot_b_value=snapshot_b_value,
            difference=difference,
            estimated_impact=estimated_impact,
            as_of_date=as_of_date,
            security_id=security_id,
            guidance=_security_formula_guidance(
                field,
                role_label,
                format_value(security_id),
                difference,
            ),
        ),
    )


def _formula_row(
    source_row: Mapping[str, object],
    values: _FormulaRowValues,
) -> dict[str, object]:
    """Return fields shared by portfolio and security formula rows."""
    return {
        findings.PORTFOLIO_ID: source_row.get(
            return_reconstruction.RECONSTRUCTION_PORTFOLIO_ID
        ),
        findings.FROM_DATE: source_row.get(return_reconstruction.RECONSTRUCTION_FROM_DATE),
        findings.THRU_DATE: source_row.get(return_reconstruction.RECONSTRUCTION_THRU_DATE),
        layout.AS_OF_DATE: values.as_of_date,
        layout.USE: rows.USE_EXPLAINS_CHANGE,
        layout.CHANGE_LABEL: f"{values.role_label} changed",
        layout.DATASET_FIELD: f"{values.dataset}.{values.source_column}",
        findings.SECURITY_ID: values.security_id,
        layout.ROW_TYPE: FORMULA_ROW_TYPE,
        findings.SNAPSHOT_A_VALUE: values.snapshot_a_value,
        findings.SNAPSHOT_B_VALUE: values.snapshot_b_value,
        layout.CHANGE: values.difference,
        findings.IMPACT_INPUT_VALUE: values.snapshot_a_value,
        layout.ESTIMATED_IMPACT: values.estimated_impact,
        layout.INPUT_ROLE: _INPUT_ROLE_PERFORMANCE_INPUT,
        layout.IMPACT_STATUS: _IMPACT_STATUS_ESTIMATED,
        layout.REVIEW_NOTE: "",
        layout.REVIEW_GUIDANCE: values.guidance,
        findings.DATASET: values.dataset,
        findings.SOURCE_COLUMN: values.source_column,
        findings.FINDING_CODE: FORMULA_FINDING_CODE,
        explain.REVIEW_RANK: -100,
        layout.USE_PRIORITY: rows.use_priority(rows.USE_EXPLAINS_CHANGE),
        layout.REVIEW_KEY: source_row.get(return_reconstruction.RECONSTRUCTION_REVIEW_KEY),
        RECONSTRUCTION_COMPONENTS: values.source_column,
    }


def _portfolio_formula_guidance(field: str, role_label: str, difference: object) -> str:
    """Return deterministic guidance for portfolio reconstruction formula rows."""
    change_text = rows.change_amount_text(difference)
    if field == BEGINNING_VALUE_FIELD:
        return (
            f"Beginning portfolio value {rows.increased_or_decreased(difference)} "
            f"by {change_text}. A higher beginning value lowers the calculated "
            "return. This value is retained because it is an input to Modified Dietz."
        )
    if field == ENDING_VALUE_FIELD:
        return (
            f"Ending portfolio value {rows.increased_or_decreased(difference)} "
            f"by {change_text}."
        )
    if field == NET_FLOW_FIELD:
        return f"Net external flows {rows.increased_or_decreased(difference)} by {change_text}."
    if field == WEIGHTED_FLOW_FIELD:
        return (
            f"Weighted external flows {rows.increased_or_decreased(difference)} "
            f"by {change_text}."
        )
    if field == INCOME_FIELD:
        return f"Income {rows.increased_or_decreased(difference)} by {change_text}."
    return f"{role_label} {rows.increased_or_decreased(difference)} by {change_text}."


def _security_formula_guidance(
    field: str,
    role_label: str,
    security_id: str,
    difference: object,
) -> str:
    """Return deterministic guidance for security reconstruction formula rows."""
    security_prefix = f"{security_id} " if security_id else ""
    change_text = rows.change_amount_text(difference)
    if field == BEGINNING_VALUE_FIELD:
        return (
            f"{security_prefix}beginning value {rows.increased_or_decreased(difference)} "
            f"by {change_text}. A higher beginning value lowers the calculated return."
        )
    if field == ENDING_VALUE_FIELD:
        return (
            f"{security_prefix}ending value {rows.increased_or_decreased(difference)} "
            f"by {change_text}."
        )
    if field == NET_FLOW_FIELD:
        return (
            f"{security_prefix}buy/sell flow was {change_text} "
            f"{rows.higher_or_lower(difference)}."
        )
    if field == WEIGHTED_FLOW_FIELD:
        return (
            f"{security_prefix}date-weighted buy/sell flow was {change_text} "
            f"{rows.higher_or_lower(difference)}."
        )
    if field == INCOME_FIELD:
        return (
            f"{security_prefix}income {rows.increased_or_decreased(difference)} "
            f"by {change_text}."
        )
    return (
        f"{security_prefix}{role_label.lower()} "
        f"{rows.increased_or_decreased(difference)} by {change_text}."
    )
