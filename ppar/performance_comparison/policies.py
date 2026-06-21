"""Validate performance comparison impact policy configuration."""

from __future__ import annotations

# Python imports
from collections.abc import Mapping
import datetime as dt
from dataclasses import dataclass
from typing import Final, cast

# Project imports
from ppar.errors import PpaError
from ppar.performance_comparison import schema as pc_cols
from ppar.performance_comparison.findings import (
    IMPACT_POLICY_CASH_BALANCE,
    IMPACT_POLICY_CASH_MARKET_VALUE,
    IMPACT_POLICY_EVIDENCE_ONLY_PREFIX,
    IMPACT_POLICY_POSITION_ACCRUED,
    IMPACT_POLICY_POSITION_MARKET_VALUE,
    IMPACT_POLICY_POSITION_QUANTITY_UNIT_MARKET_VALUE,
    IMPACT_POLICY_PRICE_WEIGHTED,
    IMPACT_POLICY_PORTFOLIO_SOURCE_FIELD,
    IMPACT_POLICY_SECURITY_CONTRIBUTION,
    IMPACT_POLICY_SECURITY_RETURN_WEIGHTED,
    TRANSACTION_IMPACT_POLICY_EXTERNAL_FLOW_EVIDENCE_ONLY,
    TRANSACTION_IMPACT_POLICY_PERFORMANCE_AMOUNT_DELTA,
)
from ppar.performance_comparison.methods import (
    CashImpactMethod,
    ContributionImpactMethod,
    FxRateImpactMethod,
    ModifiedDietzDayCount,
    ModifiedDietzDoubleCountPolicy,
    ModifiedDietzFlowTiming,
    ModifiedDietzInclusionRule,
    PositionImpactMethod,
    PriceImpactMethod,
    SecurityMasterImpactMethod,
    TransactionImpactMethod,
)
from ppar.performance_comparison.specification import PerformanceComparisonSpecification
from ppar.performance_comparison.transactions import (
    TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL,
)

_TRANSACTION_IMPACT_METHODS_KEY: Final[str] = "transaction_impact_methods"
_CONTRIBUTION_IMPACT_METHODS_KEY: Final[str] = "contribution_impact_methods"
_POSITION_IMPACT_METHODS_KEY: Final[str] = "position_impact_methods"
_PRICE_IMPACT_METHODS_KEY: Final[str] = "price_impact_methods"
_CASH_IMPACT_METHODS_KEY: Final[str] = "cash_impact_methods"
_FX_RATE_IMPACT_METHODS_KEY: Final[str] = "fx_rate_impact_methods"
_SECURITY_MASTER_IMPACT_METHODS_KEY: Final[str] = "security_master_impact_methods"
_EVIDENCE_ONLY_IMPACT_METHODS_KEY: Final[str] = "evidence_only_impact_methods"
_PORTFOLIO_SOURCE_FIELD_KEY: Final[str] = "portfolio_source_field"
_SECURITY_CONTRIBUTION_KEY: Final[str] = "security_contribution"
_SECURITY_RETURN_KEY: Final[str] = "security_return"
_MARKET_VALUE_KEY: Final[str] = "market_value"
_EXTERNAL_FLOW_KEY: Final[str] = "external_flow"
_PERFORMANCE_KEY: Final[str] = "performance"
_TRANSACTION_QUANTITY_KEY: Final[str] = pc_cols.QUANTITY
_TRANSACTION_PRICE_KEY: Final[str] = pc_cols.PRICE
_TRANSACTION_COMMISSION_KEY: Final[str] = pc_cols.COMMISSION
_METHOD_KEY: Final[str] = "method"
_SOURCE_FIELDS_KEY: Final[str] = "source_fields"
_WEIGHT_SOURCE_KEY: Final[str] = "weight_source"
_EVIDENCE_ONLY_METHOD: Final[str] = TransactionImpactMethod.EVIDENCE_ONLY.value
_VENDOR_CONTRIBUTION_DELTA_METHOD: Final[str] = (
    ContributionImpactMethod.VENDOR_CONTRIBUTION_DELTA.value
)
_SECURITY_RETURN_DELTA_TIMES_WEIGHT_METHOD: Final[str] = (
    ContributionImpactMethod.SECURITY_RETURN_DELTA_TIMES_WEIGHT.value
)
_SOURCE_FIELD_DELTA_OVER_BEGIN_MV_METHOD: Final[str] = (
    ContributionImpactMethod.SOURCE_FIELD_DELTA_OVER_BEGIN_MARKET_VALUE.value
)
_MODIFIED_DIETZ_METHOD: Final[str] = TransactionImpactMethod.MODIFIED_DIETZ.value
_TRANSACTION_AMOUNT_DELTA_METHOD: Final[str] = (
    TransactionImpactMethod.TRANSACTION_AMOUNT_DELTA_OVER_RETURN_DENOMINATOR.value
)
_POSITION_MARKET_VALUE_DELTA_METHOD: Final[str] = (
    PositionImpactMethod.MARKET_VALUE_DELTA_OVER_RETURN_DENOMINATOR.value
)
_POSITION_ACCRUED_DELTA_METHOD: Final[str] = (
    PositionImpactMethod.ACCRUED_DELTA_OVER_RETURN_DENOMINATOR.value
)
_POSITION_EVIDENCE_ONLY_METHOD: Final[str] = PositionImpactMethod.EVIDENCE_ONLY.value
_POSITION_QUANTITY_UNIT_MARKET_VALUE_METHOD: Final[str] = (
    PositionImpactMethod[
        "QUANTITY_DELTA_TIMES_SNAPSHOT_A_UNIT_MARKET_VALUE_OVER_RETURN_DENOMINATOR"
    ].value
)
_PRICE_DELTA_OVER_SNAPSHOT_A_PRICE_TIMES_WEIGHT_METHOD: Final[str] = (
    PriceImpactMethod.PRICE_DELTA_OVER_SNAPSHOT_A_PRICE_TIMES_WEIGHT.value
)
_CASH_DELTA_OVER_RETURN_DENOMINATOR_METHOD: Final[str] = (
    CashImpactMethod.CASH_DELTA_OVER_RETURN_DENOMINATOR.value
)
_FX_RATE_EVIDENCE_ONLY_METHOD: Final[str] = FxRateImpactMethod.EVIDENCE_ONLY.value
_SECURITY_MASTER_EVIDENCE_ONLY_METHOD: Final[str] = (
    SecurityMasterImpactMethod.EVIDENCE_ONLY.value
)
_FLOW_TIMING_KEY: Final[str] = "flow_timing"
_DAY_COUNT_KEY: Final[str] = "day_count"
_INCLUSION_RULE_KEY: Final[str] = "inclusion_rule"
_DENOMINATOR_SOURCE_KEY: Final[str] = "denominator_source"
_DOUBLE_COUNT_POLICY_KEY: Final[str] = "double_count_policy"
_MODIFIED_DIETZ_FLOW_TIMINGS: Final[frozenset[str]] = frozenset(
    member.value for member in ModifiedDietzFlowTiming
)
_MODIFIED_DIETZ_DAY_COUNTS: Final[frozenset[str]] = frozenset(
    member.value for member in ModifiedDietzDayCount
)
_MODIFIED_DIETZ_INCLUSION_RULES: Final[frozenset[str]] = frozenset(
    member.value for member in ModifiedDietzInclusionRule
)
_MODIFIED_DIETZ_DOUBLE_COUNT_POLICIES: Final[frozenset[str]] = frozenset(
    member.value for member in ModifiedDietzDoubleCountPolicy
)
_MODIFIED_DIETZ_REQUIRED_KEYS: Final[frozenset[str]] = frozenset(
    {
        _METHOD_KEY,
        _FLOW_TIMING_KEY,
        _DAY_COUNT_KEY,
        _INCLUSION_RULE_KEY,
        _DENOMINATOR_SOURCE_KEY,
        _DOUBLE_COUNT_POLICY_KEY,
    }
)
_PERFORMANCE_AMOUNT_REQUIRED_KEYS: Final[frozenset[str]] = frozenset(
    {
        _METHOD_KEY,
        _DENOMINATOR_SOURCE_KEY,
    }
)
_TRANSACTION_EVIDENCE_ONLY_REQUIRED_KEYS: Final[frozenset[str]] = frozenset(
    {
        _METHOD_KEY,
    }
)
_POSITION_MARKET_VALUE_REQUIRED_KEYS: Final[frozenset[str]] = frozenset(
    {
        _METHOD_KEY,
        _DENOMINATOR_SOURCE_KEY,
    }
)
_POSITION_ACCRUED_REQUIRED_KEYS: Final[frozenset[str]] = frozenset(
    {
        _METHOD_KEY,
        _DENOMINATOR_SOURCE_KEY,
    }
)
_POSITION_EVIDENCE_ONLY_REQUIRED_KEYS: Final[frozenset[str]] = frozenset(
    {
        _METHOD_KEY,
    }
)
_POSITION_QUANTITY_REQUIRED_KEYS: Final[frozenset[str]] = frozenset(
    {
        _METHOD_KEY,
        _DENOMINATOR_SOURCE_KEY,
    }
)
_PRICE_REQUIRED_KEYS: Final[frozenset[str]] = frozenset(
    {
        _METHOD_KEY,
        _WEIGHT_SOURCE_KEY,
    }
)
_CASH_REQUIRED_KEYS: Final[frozenset[str]] = frozenset(
    {
        _METHOD_KEY,
        _DENOMINATOR_SOURCE_KEY,
    }
)
_FX_RATE_EVIDENCE_ONLY_REQUIRED_KEYS: Final[frozenset[str]] = frozenset(
    {
        _METHOD_KEY,
    }
)
_SECURITY_MASTER_EVIDENCE_ONLY_REQUIRED_KEYS: Final[frozenset[str]] = frozenset(
    {
        _METHOD_KEY,
    }
)
_MODIFIED_DIETZ_ALLOWED_VALUES: Final[dict[str, frozenset[str]]] = {
    _FLOW_TIMING_KEY: _MODIFIED_DIETZ_FLOW_TIMINGS,
    _DAY_COUNT_KEY: _MODIFIED_DIETZ_DAY_COUNTS,
    _INCLUSION_RULE_KEY: _MODIFIED_DIETZ_INCLUSION_RULES,
    _DENOMINATOR_SOURCE_KEY: frozenset({"begin_market_value"}),
    _DOUBLE_COUNT_POLICY_KEY: _MODIFIED_DIETZ_DOUBLE_COUNT_POLICIES,
}
_RESERVED_EXTERNAL_FLOW_METHODS: Final[frozenset[str]] = frozenset(
    {
        _MODIFIED_DIETZ_METHOD,
        "subperiod_linked",
        "unweighted_flow_delta",
    }
)
_PERFORMANCE_AMOUNT_ALLOWED_VALUES: Final[dict[str, frozenset[str]]] = {
    _DENOMINATOR_SOURCE_KEY: frozenset({"begin_market_value"}),
}
_POSITION_MARKET_VALUE_ALLOWED_VALUES: Final[dict[str, frozenset[str]]] = {
    _DENOMINATOR_SOURCE_KEY: frozenset({"begin_market_value"}),
}
_POSITION_ACCRUED_ALLOWED_VALUES: Final[dict[str, frozenset[str]]] = {
    _DENOMINATOR_SOURCE_KEY: frozenset({"begin_market_value"}),
}
_POSITION_QUANTITY_ALLOWED_VALUES: Final[dict[str, frozenset[str]]] = {
    _DENOMINATOR_SOURCE_KEY: frozenset({"begin_market_value"}),
}
_PRICE_ALLOWED_VALUES: Final[dict[str, frozenset[str]]] = {
    _WEIGHT_SOURCE_KEY: frozenset({"snapshot_a_weight"}),
}
_CASH_ALLOWED_VALUES: Final[dict[str, frozenset[str]]] = {
    _DENOMINATOR_SOURCE_KEY: frozenset({"begin_market_value"}),
}
_EVIDENCE_ONLY_REQUIRED_KEYS: Final[frozenset[str]] = frozenset(
    {
        _METHOD_KEY,
        _SOURCE_FIELDS_KEY,
    }
)
_EVIDENCE_ONLY_SUPPORTED_SOURCE_FIELDS: Final[dict[str, frozenset[str]]] = {
    pc_cols.CASH: frozenset({pc_cols.CASH_BALANCE, pc_cols.MARKET_VALUE}),
    pc_cols.FX_RATES: frozenset({pc_cols.FX_RATE}),
    pc_cols.POSITIONS: frozenset(
        {pc_cols.QUANTITY, pc_cols.MARKET_VALUE, pc_cols.COST, pc_cols.ACCRUED}
    ),
    pc_cols.PRICES: frozenset({pc_cols.PRICE}),
    pc_cols.SECURITY_MASTER: frozenset(
        {
            pc_cols.SECURITY_NAME,
            pc_cols.TICKER,
            pc_cols.CUSIP,
            pc_cols.ISIN,
            pc_cols.CURRENCY,
            pc_cols.COUNTRY,
            pc_cols.SECTOR,
            pc_cols.INDUSTRY,
            pc_cols.ASSET_CLASS,
        }
    ),
    pc_cols.TRANSACTIONS: frozenset(
        {pc_cols.AMOUNT, pc_cols.QUANTITY, pc_cols.PRICE, pc_cols.COMMISSION}
    ),
}
_PORTFOLIO_SOURCE_FIELD_REQUIRED_KEYS: Final[frozenset[str]] = frozenset(
    {
        _METHOD_KEY,
        _DENOMINATOR_SOURCE_KEY,
        _SOURCE_FIELDS_KEY,
    }
)
_SECURITY_CONTRIBUTION_REQUIRED_KEYS: Final[frozenset[str]] = frozenset(
    {_METHOD_KEY}
)
_SECURITY_RETURN_REQUIRED_KEYS: Final[frozenset[str]] = frozenset(
    {
        _METHOD_KEY,
        _WEIGHT_SOURCE_KEY,
    }
)
_PORTFOLIO_SOURCE_FIELD_ALLOWED_VALUES: Final[dict[str, frozenset[str]]] = {
    _DENOMINATOR_SOURCE_KEY: frozenset({"begin_market_value"}),
}
_SECURITY_RETURN_ALLOWED_VALUES: Final[dict[str, frozenset[str]]] = {
    _WEIGHT_SOURCE_KEY: frozenset({"snapshot_a_weight"}),
}
_PORTFOLIO_SOURCE_FIELD_ALLOWED_SOURCE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        pc_cols.INCOME,
        pc_cols.GAIN_LOSS,
    }
)

@dataclass(frozen=True)
class _TransactionImpactPolicy:
    """Carry explicitly configured transaction impact-method settings.

    Attributes:
        method: YAML method name.
        finding_label: Stable finding-table label exposed to reports.
        flow_timing: Date field used to time external flows.
        day_count: Day-count convention for timing weights.
        inclusion_rule: Beginning/end-of-day flow inclusion convention.
        denominator_source: YAML-selected return denominator source.
        double_count_policy: Rule for handling overlap with portfolio-level
            flow deltas.
    """

    method: str
    finding_label: str
    flow_timing: str | None = None
    day_count: str | None = None
    inclusion_rule: str | None = None
    denominator_source: str | None = None
    double_count_policy: str | None = None


@dataclass(frozen=True)
class _ModifiedDietzEligibility:
    """Describe whether one external-flow row has all Modified Dietz inputs.

    Attributes:
        eligible: Whether the row has every explicitly configured input needed
            for a Modified Dietz cross-check estimate.
        missing_inputs: Human-readable missing or disqualifying inputs.
        flow_date: YAML-selected transaction flow date, when available.
    """

    eligible: bool
    missing_inputs: tuple[str, ...] = ()
    flow_date: dt.date | None = None


def _transaction_impact_policies(
    specification: PerformanceComparisonSpecification,
) -> dict[str, _TransactionImpactPolicy]:
    """Return validated YAML-configured transaction impact policies.

    Args:
        specification: Parsed comparison specification.

    Returns:
        Transaction impact policies keyed by normalized performance-flow
        treatment. Missing configuration returns an empty mapping.

    Raises:
        PpaError: If transaction impact method configuration is malformed or
            names an unsupported method.
    """
    methods_value = specification.values.get(_TRANSACTION_IMPACT_METHODS_KEY, {})
    if methods_value is None:
        return {}
    if not isinstance(methods_value, dict):
        raise PpaError(
            (
                f"{specification.path}: {_TRANSACTION_IMPACT_METHODS_KEY} "
                "must be a mapping."
            ),
            504,
        )

    unsupported_keys = set(methods_value) - {
        _EXTERNAL_FLOW_KEY,
        _PERFORMANCE_KEY,
        _TRANSACTION_QUANTITY_KEY,
        _TRANSACTION_PRICE_KEY,
        _TRANSACTION_COMMISSION_KEY,
    }
    if unsupported_keys:
        unsupported = ", ".join(sorted(str(key) for key in unsupported_keys))
        raise PpaError(
            (
                f"{specification.path}: unsupported "
                f"{_TRANSACTION_IMPACT_METHODS_KEY} keys: {unsupported}."
            ),
            504,
        )

    policies: dict[str, _TransactionImpactPolicy] = {}
    external_flow_value = methods_value.get(_EXTERNAL_FLOW_KEY)
    if external_flow_value is not None and not isinstance(external_flow_value, dict):
        raise PpaError(
            (
                f"{specification.path}: "
                f"{_TRANSACTION_IMPACT_METHODS_KEY}.{_EXTERNAL_FLOW_KEY} "
                "must be a mapping."
            ),
            504,
        )
    if isinstance(external_flow_value, dict):
        policies[_EXTERNAL_FLOW_KEY] = _validated_external_flow_policy(
            specification,
            external_flow_value,
        )

    performance_value = methods_value.get(_PERFORMANCE_KEY)
    if performance_value is not None and not isinstance(performance_value, dict):
        raise PpaError(
            (
                f"{specification.path}: "
                f"{_TRANSACTION_IMPACT_METHODS_KEY}.{_PERFORMANCE_KEY} "
                "must be a mapping."
            ),
            504,
        )
    if isinstance(performance_value, dict):
        policies[_PERFORMANCE_KEY] = _validated_performance_amount_policy(
            specification,
            performance_value,
        )
    for policy_key in (
        _TRANSACTION_QUANTITY_KEY,
        _TRANSACTION_PRICE_KEY,
        _TRANSACTION_COMMISSION_KEY,
    ):
        evidence_only_value = methods_value.get(policy_key)
        if evidence_only_value is not None and not isinstance(evidence_only_value, dict):
            raise PpaError(
                (
                    f"{specification.path}: "
                    f"{_TRANSACTION_IMPACT_METHODS_KEY}.{policy_key} "
                    "must be a mapping."
                ),
                504,
            )
        if isinstance(evidence_only_value, dict):
            policies[policy_key] = _validated_transaction_evidence_only_policy(
                specification,
                policy_key,
                evidence_only_value,
            )
    return policies


def _contribution_impact_policies(
    specification: PerformanceComparisonSpecification,
) -> dict[tuple[str, str], str]:
    """Return validated YAML-selected contribution impact policies.

    Args:
        specification: Parsed comparison specification.

    Returns:
        Policy labels keyed by ``(dataset, source_column)``. Missing
        configuration returns an empty mapping, which leaves candidate rows as
        evidence-only.

    Raises:
        PpaError: If contribution impact method configuration is malformed or
            names an unsupported method.
    """
    methods_value = specification.values.get(_CONTRIBUTION_IMPACT_METHODS_KEY, {})
    if methods_value is None:
        return {}
    if not isinstance(methods_value, dict):
        raise PpaError(
            (
                f"{specification.path}: {_CONTRIBUTION_IMPACT_METHODS_KEY} "
                "must be a mapping."
            ),
            504,
        )

    supported_keys = {
        _PORTFOLIO_SOURCE_FIELD_KEY,
        _SECURITY_CONTRIBUTION_KEY,
        _SECURITY_RETURN_KEY,
    }
    unsupported_keys = set(methods_value) - supported_keys
    if unsupported_keys:
        unsupported = ", ".join(sorted(str(key) for key in unsupported_keys))
        raise PpaError(
            (
                f"{specification.path}: unsupported "
                f"{_CONTRIBUTION_IMPACT_METHODS_KEY} keys: {unsupported}."
            ),
            504,
        )

    policies: dict[tuple[str, str], str] = {}
    portfolio_source_field_value = methods_value.get(_PORTFOLIO_SOURCE_FIELD_KEY)
    if portfolio_source_field_value is not None:
        policies.update(
            _validated_portfolio_source_field_policy(
                specification,
                portfolio_source_field_value,
            )
        )

    security_contribution_value = methods_value.get(_SECURITY_CONTRIBUTION_KEY)
    if security_contribution_value is not None:
        policies.update(
            _validated_security_contribution_policy(
                specification,
                security_contribution_value,
            )
        )

    security_return_value = methods_value.get(_SECURITY_RETURN_KEY)
    if security_return_value is not None:
        policies.update(
            _validated_security_return_policy(
                specification,
                security_return_value,
            )
        )
    return policies


def _position_impact_policies(
    specification: PerformanceComparisonSpecification,
) -> dict[str, str]:
    """Return validated YAML-selected position impact policies.

    Args:
        specification: Parsed comparison specification.

    Returns:
        Policy labels keyed by position source column. Missing configuration
        returns an empty mapping, which leaves position rows as evidence-only.

    Raises:
        PpaError: If position impact method configuration is malformed or
            names an unsupported method.
    """
    methods_value = specification.values.get(_POSITION_IMPACT_METHODS_KEY, {})
    if methods_value is None:
        return {}
    if not isinstance(methods_value, dict):
        raise PpaError(
            (
                f"{specification.path}: {_POSITION_IMPACT_METHODS_KEY} "
                "must be a mapping."
            ),
            504,
        )

    unsupported_keys = set(methods_value) - {
        _MARKET_VALUE_KEY,
        pc_cols.ACCRUED,
        pc_cols.QUANTITY,
        pc_cols.COST,
    }
    if unsupported_keys:
        unsupported = ", ".join(sorted(str(key) for key in unsupported_keys))
        raise PpaError(
            (
                f"{specification.path}: unsupported "
                f"{_POSITION_IMPACT_METHODS_KEY} keys: {unsupported}."
            ),
            504,
        )

    policies: dict[str, str] = {}
    market_value = methods_value.get(_MARKET_VALUE_KEY)
    if market_value is not None:
        policy = _require_policy_mapping(
            specification,
            _POSITION_IMPACT_METHODS_KEY,
            _MARKET_VALUE_KEY,
            market_value,
        )
        _validate_policy_keys(
            specification,
            _POSITION_IMPACT_METHODS_KEY,
            _MARKET_VALUE_KEY,
            policy,
            _POSITION_MARKET_VALUE_REQUIRED_KEYS,
        )
        _validate_policy_method(
            specification,
            _POSITION_IMPACT_METHODS_KEY,
            _MARKET_VALUE_KEY,
            policy,
            _POSITION_MARKET_VALUE_DELTA_METHOD,
        )
        _validate_allowed_policy_values(
            specification,
            _POSITION_IMPACT_METHODS_KEY,
            _MARKET_VALUE_KEY,
            policy,
            _POSITION_MARKET_VALUE_ALLOWED_VALUES,
        )
        policies[pc_cols.MARKET_VALUE] = IMPACT_POLICY_POSITION_MARKET_VALUE
    accrued = methods_value.get(pc_cols.ACCRUED)
    if accrued is not None:
        policy = _require_policy_mapping(
            specification,
            _POSITION_IMPACT_METHODS_KEY,
            pc_cols.ACCRUED,
            accrued,
        )
        _validate_policy_keys(
            specification,
            _POSITION_IMPACT_METHODS_KEY,
            pc_cols.ACCRUED,
            policy,
            _POSITION_ACCRUED_REQUIRED_KEYS,
        )
        _validate_policy_method(
            specification,
            _POSITION_IMPACT_METHODS_KEY,
            pc_cols.ACCRUED,
            policy,
            _POSITION_ACCRUED_DELTA_METHOD,
        )
        _validate_allowed_policy_values(
            specification,
            _POSITION_IMPACT_METHODS_KEY,
            pc_cols.ACCRUED,
            policy,
            _POSITION_ACCRUED_ALLOWED_VALUES,
        )
        policies[pc_cols.ACCRUED] = IMPACT_POLICY_POSITION_ACCRUED
    quantity_value = methods_value.get(pc_cols.QUANTITY)
    if quantity_value is not None:
        policies[pc_cols.QUANTITY] = _validated_position_quantity_policy(
            specification,
            quantity_value,
        )
    cost_value = methods_value.get(pc_cols.COST)
    if cost_value is not None:
        policies[pc_cols.COST] = _validated_position_evidence_only_policy(
            specification,
            pc_cols.COST,
            cost_value,
        )
    return policies


def _validated_position_quantity_policy(
    specification: PerformanceComparisonSpecification,
    policy_value: object,
) -> str:
    """Validate and return the configured position quantity policy label."""
    policy = _require_policy_mapping(
        specification,
        _POSITION_IMPACT_METHODS_KEY,
        pc_cols.QUANTITY,
        policy_value,
    )
    method = policy.get(_METHOD_KEY)
    if method is None:
        _validate_policy_keys(
            specification,
            _POSITION_IMPACT_METHODS_KEY,
            pc_cols.QUANTITY,
            policy,
            _POSITION_QUANTITY_REQUIRED_KEYS,
        )
    if method == _POSITION_EVIDENCE_ONLY_METHOD:
        _validate_policy_keys(
            specification,
            _POSITION_IMPACT_METHODS_KEY,
            pc_cols.QUANTITY,
            policy,
            _POSITION_EVIDENCE_ONLY_REQUIRED_KEYS,
        )
        return _evidence_only_impact_policy_label(
            pc_cols.POSITIONS,
            pc_cols.QUANTITY,
        )
    if method != _POSITION_QUANTITY_UNIT_MARKET_VALUE_METHOD:
        _validate_policy_method(
            specification,
            _POSITION_IMPACT_METHODS_KEY,
            pc_cols.QUANTITY,
            policy,
            _POSITION_QUANTITY_UNIT_MARKET_VALUE_METHOD,
        )
    _validate_policy_keys(
        specification,
        _POSITION_IMPACT_METHODS_KEY,
        pc_cols.QUANTITY,
        policy,
        _POSITION_QUANTITY_REQUIRED_KEYS,
    )
    _validate_allowed_policy_values(
        specification,
        _POSITION_IMPACT_METHODS_KEY,
        pc_cols.QUANTITY,
        policy,
        _POSITION_QUANTITY_ALLOWED_VALUES,
    )
    return IMPACT_POLICY_POSITION_QUANTITY_UNIT_MARKET_VALUE


def _validated_position_evidence_only_policy(
    specification: PerformanceComparisonSpecification,
    source_column: str,
    policy_value: object,
) -> str:
    """Validate and return an evidence-only position policy label."""
    policy = _require_policy_mapping(
        specification,
        _POSITION_IMPACT_METHODS_KEY,
        source_column,
        policy_value,
    )
    _validate_policy_keys(
        specification,
        _POSITION_IMPACT_METHODS_KEY,
        source_column,
        policy,
        _POSITION_EVIDENCE_ONLY_REQUIRED_KEYS,
    )
    _validate_policy_method(
        specification,
        _POSITION_IMPACT_METHODS_KEY,
        source_column,
        policy,
        _POSITION_EVIDENCE_ONLY_METHOD,
    )
    return _evidence_only_impact_policy_label(
        pc_cols.POSITIONS,
        source_column,
    )

def _price_impact_policies(
    specification: PerformanceComparisonSpecification,
) -> dict[str, str]:
    """Return validated YAML-selected price impact policies.

    Args:
        specification: Parsed comparison specification.

    Returns:
        Policy labels keyed by price source column. Missing configuration
        returns an empty mapping, which leaves price rows as evidence-only.

    Raises:
        PpaError: If price impact method configuration is malformed or names
            an unsupported method.
    """
    methods_value = specification.values.get(_PRICE_IMPACT_METHODS_KEY, {})
    if methods_value is None:
        return {}
    if not isinstance(methods_value, dict):
        raise PpaError(
            (
                f"{specification.path}: {_PRICE_IMPACT_METHODS_KEY} "
                "must be a mapping."
            ),
            504,
        )

    unsupported_keys = set(methods_value) - {pc_cols.PRICE}
    if unsupported_keys:
        unsupported = ", ".join(sorted(str(key) for key in unsupported_keys))
        raise PpaError(
            (
                f"{specification.path}: unsupported "
                f"{_PRICE_IMPACT_METHODS_KEY} keys: {unsupported}."
            ),
            504,
        )

    policies: dict[str, str] = {}
    price_value = methods_value.get(pc_cols.PRICE)
    if price_value is not None:
        policy = _require_policy_mapping(
            specification,
            _PRICE_IMPACT_METHODS_KEY,
            pc_cols.PRICE,
            price_value,
        )
        _validate_policy_keys(
            specification,
            _PRICE_IMPACT_METHODS_KEY,
            pc_cols.PRICE,
            policy,
            _PRICE_REQUIRED_KEYS,
        )
        _validate_policy_method(
            specification,
            _PRICE_IMPACT_METHODS_KEY,
            pc_cols.PRICE,
            policy,
            _PRICE_DELTA_OVER_SNAPSHOT_A_PRICE_TIMES_WEIGHT_METHOD,
        )
        _validate_allowed_policy_values(
            specification,
            _PRICE_IMPACT_METHODS_KEY,
            pc_cols.PRICE,
            policy,
            _PRICE_ALLOWED_VALUES,
        )
        policies[pc_cols.PRICE] = IMPACT_POLICY_PRICE_WEIGHTED
    return policies


def _cash_impact_policies(
    specification: PerformanceComparisonSpecification,
) -> dict[str, str]:
    """Return validated YAML-selected cash impact policies.

    Args:
        specification: Parsed comparison specification.

    Returns:
        Policy labels keyed by cash source column. Missing configuration
        returns an empty mapping, which leaves cash rows as evidence-only.

    Raises:
        PpaError: If cash impact method configuration is malformed or names an
            unsupported method.
    """
    methods_value = specification.values.get(_CASH_IMPACT_METHODS_KEY, {})
    if methods_value is None:
        return {}
    if not isinstance(methods_value, dict):
        raise PpaError(
            (
                f"{specification.path}: {_CASH_IMPACT_METHODS_KEY} "
                "must be a mapping."
            ),
            504,
        )

    supported_keys = {pc_cols.CASH_BALANCE, pc_cols.MARKET_VALUE}
    unsupported_keys = set(methods_value) - supported_keys
    if unsupported_keys:
        unsupported = ", ".join(sorted(str(key) for key in unsupported_keys))
        raise PpaError(
            (
                f"{specification.path}: unsupported "
                f"{_CASH_IMPACT_METHODS_KEY} keys: {unsupported}."
            ),
            504,
        )

    policies: dict[str, str] = {}
    for source_column in (pc_cols.CASH_BALANCE, pc_cols.MARKET_VALUE):
        method_value = methods_value.get(source_column)
        if method_value is None:
            continue
        policy = _require_policy_mapping(
            specification,
            _CASH_IMPACT_METHODS_KEY,
            source_column,
            method_value,
        )
        _validate_policy_keys(
            specification,
            _CASH_IMPACT_METHODS_KEY,
            source_column,
            policy,
            _CASH_REQUIRED_KEYS,
        )
        _validate_policy_method(
            specification,
            _CASH_IMPACT_METHODS_KEY,
            source_column,
            policy,
            _CASH_DELTA_OVER_RETURN_DENOMINATOR_METHOD,
        )
        _validate_allowed_policy_values(
            specification,
            _CASH_IMPACT_METHODS_KEY,
            source_column,
            policy,
            _CASH_ALLOWED_VALUES,
        )
        if source_column == pc_cols.CASH_BALANCE:
            policies[source_column] = IMPACT_POLICY_CASH_BALANCE
        else:
            policies[source_column] = IMPACT_POLICY_CASH_MARKET_VALUE
    return policies


def _fx_rate_impact_policies(
    specification: PerformanceComparisonSpecification,
) -> dict[str, str]:
    """Return validated YAML-selected FX rate impact policies.

    Args:
        specification: Parsed comparison specification.

    Returns:
        Policy labels keyed by FX rate source column. Missing configuration
        returns an empty mapping, which leaves FX rows as ordinary review
        evidence.

    Raises:
        PpaError: If FX rate impact method configuration is malformed or names
            an unsupported method.
    """
    methods_value = specification.values.get(_FX_RATE_IMPACT_METHODS_KEY, {})
    if methods_value is None:
        return {}
    if not isinstance(methods_value, dict):
        raise PpaError(
            (
                f"{specification.path}: {_FX_RATE_IMPACT_METHODS_KEY} "
                "must be a mapping."
            ),
            504,
        )

    unsupported_keys = set(methods_value) - {pc_cols.FX_RATE}
    if unsupported_keys:
        unsupported = ", ".join(sorted(str(key) for key in unsupported_keys))
        raise PpaError(
            (
                f"{specification.path}: unsupported "
                f"{_FX_RATE_IMPACT_METHODS_KEY} keys: {unsupported}."
            ),
            504,
        )

    fx_rate_value = methods_value.get(pc_cols.FX_RATE)
    if fx_rate_value is None:
        return {}
    policy = _require_policy_mapping(
        specification,
        _FX_RATE_IMPACT_METHODS_KEY,
        pc_cols.FX_RATE,
        fx_rate_value,
    )
    _validate_policy_keys(
        specification,
        _FX_RATE_IMPACT_METHODS_KEY,
        pc_cols.FX_RATE,
        policy,
        _FX_RATE_EVIDENCE_ONLY_REQUIRED_KEYS,
    )
    _validate_policy_method(
        specification,
        _FX_RATE_IMPACT_METHODS_KEY,
        pc_cols.FX_RATE,
        policy,
        _FX_RATE_EVIDENCE_ONLY_METHOD,
    )
    return {
        pc_cols.FX_RATE: _evidence_only_impact_policy_label(
            pc_cols.FX_RATES,
            pc_cols.FX_RATE,
        )
    }


def _security_master_impact_policies(
    specification: PerformanceComparisonSpecification,
) -> dict[str, str]:
    """Return validated YAML-selected security master impact policies.

    Args:
        specification: Parsed comparison specification.

    Returns:
        Policy labels keyed by security master source column. Missing
        configuration returns an empty mapping, which leaves security master
        differences as context evidence.

    Raises:
        PpaError: If security master impact method configuration is malformed
            or names an unsupported method.
    """
    methods_value = specification.values.get(_SECURITY_MASTER_IMPACT_METHODS_KEY, {})
    if methods_value is None:
        return {}
    if not isinstance(methods_value, dict):
        raise PpaError(
            (
                f"{specification.path}: {_SECURITY_MASTER_IMPACT_METHODS_KEY} "
                "must be a mapping."
            ),
            504,
        )

    supported_keys = set(_EVIDENCE_ONLY_SUPPORTED_SOURCE_FIELDS[pc_cols.SECURITY_MASTER])
    unsupported_keys = set(methods_value) - supported_keys
    if unsupported_keys:
        unsupported = ", ".join(sorted(str(key) for key in unsupported_keys))
        raise PpaError(
            (
                f"{specification.path}: unsupported "
                f"{_SECURITY_MASTER_IMPACT_METHODS_KEY} keys: {unsupported}."
            ),
            504,
        )

    policies: dict[str, str] = {}
    for source_column in sorted(str(key) for key in methods_value):
        policy = _require_policy_mapping(
            specification,
            _SECURITY_MASTER_IMPACT_METHODS_KEY,
            source_column,
            methods_value[source_column],
        )
        _validate_policy_keys(
            specification,
            _SECURITY_MASTER_IMPACT_METHODS_KEY,
            source_column,
            policy,
            _SECURITY_MASTER_EVIDENCE_ONLY_REQUIRED_KEYS,
        )
        _validate_policy_method(
            specification,
            _SECURITY_MASTER_IMPACT_METHODS_KEY,
            source_column,
            policy,
            _SECURITY_MASTER_EVIDENCE_ONLY_METHOD,
        )
        policies[source_column] = _evidence_only_impact_policy_label(
            pc_cols.SECURITY_MASTER,
            source_column,
        )
    return policies


def _evidence_only_impact_policies(
    specification: PerformanceComparisonSpecification,
) -> dict[tuple[str, str], str]:
    """Return validated YAML-selected evidence-only impact policies.

    Args:
        specification: Parsed comparison specification.

    Returns:
        Policy labels keyed by ``(dataset, source_column)``. These policies
        document known source-data differences that should remain review-only.

    Raises:
        PpaError: If evidence-only impact configuration is malformed or names
            an unsupported dataset or field.
    """
    methods_value = specification.values.get(_EVIDENCE_ONLY_IMPACT_METHODS_KEY, {})
    if methods_value is None:
        return {}
    if not isinstance(methods_value, dict):
        raise PpaError(
            (
                f"{specification.path}: {_EVIDENCE_ONLY_IMPACT_METHODS_KEY} "
                "must be a mapping."
            ),
            504,
        )

    unsupported_keys = set(methods_value) - set(_EVIDENCE_ONLY_SUPPORTED_SOURCE_FIELDS)
    if unsupported_keys:
        unsupported = ", ".join(sorted(str(key) for key in unsupported_keys))
        raise PpaError(
            (
                f"{specification.path}: unsupported "
                f"{_EVIDENCE_ONLY_IMPACT_METHODS_KEY} keys: {unsupported}."
            ),
            504,
        )

    policies: dict[tuple[str, str], str] = {}
    for dataset, policy_value in methods_value.items():
        dataset_name = str(dataset)
        policy = _require_policy_mapping(
            specification,
            _EVIDENCE_ONLY_IMPACT_METHODS_KEY,
            dataset_name,
            policy_value,
        )
        _validate_policy_keys(
            specification,
            _EVIDENCE_ONLY_IMPACT_METHODS_KEY,
            dataset_name,
            policy,
            _EVIDENCE_ONLY_REQUIRED_KEYS,
        )
        _validate_policy_method(
            specification,
            _EVIDENCE_ONLY_IMPACT_METHODS_KEY,
            dataset_name,
            policy,
            _EVIDENCE_ONLY_METHOD,
        )
        policies.update(
            _validated_evidence_only_source_fields(
                specification,
                dataset_name,
                policy,
            )
        )
    return policies


def _validated_portfolio_source_field_policy(
    specification: PerformanceComparisonSpecification,
    policy_value: object,
) -> dict[tuple[str, str], str]:
    """Validate portfolio source-field contribution policy configuration."""
    policy = _require_policy_mapping(
        specification,
        _CONTRIBUTION_IMPACT_METHODS_KEY,
        _PORTFOLIO_SOURCE_FIELD_KEY,
        policy_value,
    )
    _validate_policy_keys(
        specification,
        _CONTRIBUTION_IMPACT_METHODS_KEY,
        _PORTFOLIO_SOURCE_FIELD_KEY,
        policy,
        _PORTFOLIO_SOURCE_FIELD_REQUIRED_KEYS,
    )
    _validate_policy_method(
        specification,
        _CONTRIBUTION_IMPACT_METHODS_KEY,
        _PORTFOLIO_SOURCE_FIELD_KEY,
        policy,
        _SOURCE_FIELD_DELTA_OVER_BEGIN_MV_METHOD,
    )
    _validate_allowed_policy_values(
        specification,
        _CONTRIBUTION_IMPACT_METHODS_KEY,
        _PORTFOLIO_SOURCE_FIELD_KEY,
        policy,
        _PORTFOLIO_SOURCE_FIELD_ALLOWED_VALUES,
    )
    source_fields = policy[_SOURCE_FIELDS_KEY]
    if not isinstance(source_fields, list) or not source_fields:
        raise PpaError(
            (
                f"{specification.path}: "
                f"{_CONTRIBUTION_IMPACT_METHODS_KEY}.{_PORTFOLIO_SOURCE_FIELD_KEY}."
                f"{_SOURCE_FIELDS_KEY} must be a non-empty list."
            ),
            504,
        )
    if any(not isinstance(field, str) for field in source_fields):
        raise PpaError(
            (
                f"{specification.path}: "
                f"{_CONTRIBUTION_IMPACT_METHODS_KEY}.{_PORTFOLIO_SOURCE_FIELD_KEY}."
                f"{_SOURCE_FIELDS_KEY} values must be strings."
            ),
            504,
        )
    unsupported_fields = set(source_fields) - _PORTFOLIO_SOURCE_FIELD_ALLOWED_SOURCE_FIELDS
    if unsupported_fields:
        unsupported = ", ".join(sorted(str(field) for field in unsupported_fields))
        allowed = ", ".join(sorted(_PORTFOLIO_SOURCE_FIELD_ALLOWED_SOURCE_FIELDS))
        raise PpaError(
            (
                f"{specification.path}: "
                f"{_CONTRIBUTION_IMPACT_METHODS_KEY}.{_PORTFOLIO_SOURCE_FIELD_KEY}."
                f"{_SOURCE_FIELDS_KEY} contains unsupported fields: {unsupported}. "
                f"Allowed fields: {allowed}."
            ),
            504,
        )
    return {
        (pc_cols.PORTFOLIO_PERFORMANCE, str(field)): IMPACT_POLICY_PORTFOLIO_SOURCE_FIELD
        for field in source_fields
    }


def _validated_evidence_only_source_fields(
    specification: PerformanceComparisonSpecification,
    dataset: str,
    policy: Mapping[str, object],
) -> dict[tuple[str, str], str]:
    """Validate and return evidence-only policy labels for source fields."""
    source_fields = policy[_SOURCE_FIELDS_KEY]
    if not isinstance(source_fields, list) or not source_fields:
        raise PpaError(
            (
                f"{specification.path}: "
                f"{_EVIDENCE_ONLY_IMPACT_METHODS_KEY}.{dataset}."
                f"{_SOURCE_FIELDS_KEY} must be a non-empty list."
            ),
            504,
        )
    if any(not isinstance(field, str) for field in source_fields):
        raise PpaError(
            (
                f"{specification.path}: "
                f"{_EVIDENCE_ONLY_IMPACT_METHODS_KEY}.{dataset}."
                f"{_SOURCE_FIELDS_KEY} values must be strings."
            ),
            504,
        )

    allowed_fields = _EVIDENCE_ONLY_SUPPORTED_SOURCE_FIELDS[dataset]
    unsupported_fields = set(source_fields) - allowed_fields
    if unsupported_fields:
        unsupported = ", ".join(sorted(str(field) for field in unsupported_fields))
        allowed = ", ".join(sorted(allowed_fields))
        raise PpaError(
            (
                f"{specification.path}: "
                f"{_EVIDENCE_ONLY_IMPACT_METHODS_KEY}.{dataset}."
                f"{_SOURCE_FIELDS_KEY} contains unsupported fields: {unsupported}. "
                f"Allowed fields: {allowed}."
            ),
            504,
        )

    return {
        (dataset, str(field)): _evidence_only_impact_policy_label(dataset, str(field))
        for field in source_fields
    }


def _validated_security_contribution_policy(
    specification: PerformanceComparisonSpecification,
    policy_value: object,
) -> dict[tuple[str, str], str]:
    """Validate vendor contribution-delta policy configuration."""
    policy = _require_policy_mapping(
        specification,
        _CONTRIBUTION_IMPACT_METHODS_KEY,
        _SECURITY_CONTRIBUTION_KEY,
        policy_value,
    )
    _validate_policy_keys(
        specification,
        _CONTRIBUTION_IMPACT_METHODS_KEY,
        _SECURITY_CONTRIBUTION_KEY,
        policy,
        _SECURITY_CONTRIBUTION_REQUIRED_KEYS,
    )
    _validate_policy_method(
        specification,
        _CONTRIBUTION_IMPACT_METHODS_KEY,
        _SECURITY_CONTRIBUTION_KEY,
        policy,
        _VENDOR_CONTRIBUTION_DELTA_METHOD,
    )
    return {
        (pc_cols.SECURITY_PERFORMANCE, pc_cols.CONTRIBUTION): (
            IMPACT_POLICY_SECURITY_CONTRIBUTION
        )
    }


def _evidence_only_impact_policy_label(dataset: str, source_column: str) -> str:
    """Return a stable evidence-only impact policy label."""
    return f"{IMPACT_POLICY_EVIDENCE_ONLY_PREFIX}{dataset}.{source_column}"


def _is_evidence_only_policy_label(value: object) -> bool:
    """Return whether a policy label represents explicit evidence-only setup."""
    return isinstance(value, str) and value.startswith(IMPACT_POLICY_EVIDENCE_ONLY_PREFIX)


def _validated_security_return_policy(
    specification: PerformanceComparisonSpecification,
    policy_value: object,
) -> dict[tuple[str, str], str]:
    """Validate weighted security-return policy configuration."""
    policy = _require_policy_mapping(
        specification,
        _CONTRIBUTION_IMPACT_METHODS_KEY,
        _SECURITY_RETURN_KEY,
        policy_value,
    )
    _validate_policy_keys(
        specification,
        _CONTRIBUTION_IMPACT_METHODS_KEY,
        _SECURITY_RETURN_KEY,
        policy,
        _SECURITY_RETURN_REQUIRED_KEYS,
    )
    _validate_policy_method(
        specification,
        _CONTRIBUTION_IMPACT_METHODS_KEY,
        _SECURITY_RETURN_KEY,
        policy,
        _SECURITY_RETURN_DELTA_TIMES_WEIGHT_METHOD,
    )
    _validate_allowed_policy_values(
        specification,
        _CONTRIBUTION_IMPACT_METHODS_KEY,
        _SECURITY_RETURN_KEY,
        policy,
        _SECURITY_RETURN_ALLOWED_VALUES,
    )
    return {
        (pc_cols.SECURITY_PERFORMANCE, pc_cols.SECURITY_RETURN): (
            IMPACT_POLICY_SECURITY_RETURN_WEIGHTED
        )
    }


def _require_policy_mapping(
    specification: PerformanceComparisonSpecification,
    root_key: str,
    policy_key: str,
    policy_value: object,
) -> Mapping[str, object]:
    """Return a YAML policy mapping or raise a contract error."""
    if isinstance(policy_value, dict):
        return policy_value
    raise PpaError(
        (
            f"{specification.path}: {root_key}.{policy_key} "
            "must be a mapping."
        ),
        504,
    )


def _validate_policy_keys(
    specification: PerformanceComparisonSpecification,
    root_key: str,
    policy_key: str,
    policy: Mapping[str, object],
    required_keys: frozenset[str],
) -> None:
    """Validate one explicit policy has exactly the supported keys."""
    unsupported_keys = set(policy) - required_keys
    if unsupported_keys:
        unsupported = ", ".join(sorted(str(key) for key in unsupported_keys))
        raise PpaError(
            (
                f"{specification.path}: {root_key}.{policy_key} "
                f"has unsupported keys: {unsupported}."
            ),
            504,
        )
    missing_keys = required_keys - set(policy)
    if missing_keys:
        missing = ", ".join(sorted(str(key) for key in missing_keys))
        raise PpaError(
            (
                f"{specification.path}: {root_key}.{policy_key} "
                f"is missing required keys: {missing}."
            ),
            504,
        )


def _validate_policy_method(
    specification: PerformanceComparisonSpecification,
    root_key: str,
    policy_key: str,
    policy: Mapping[str, object],
    expected_method: str,
) -> None:
    """Validate one explicit policy selects the only supported method."""
    if policy.get(_METHOD_KEY) != expected_method:
        raise PpaError(
            (
                f"{specification.path}: {root_key}.{policy_key}."
                f"{_METHOD_KEY} must be {expected_method!r}."
            ),
            504,
        )


def _validate_allowed_policy_values(
    specification: PerformanceComparisonSpecification,
    root_key: str,
    policy_key: str,
    policy: Mapping[str, object],
    allowed_values_by_key: Mapping[str, frozenset[str]],
) -> None:
    """Validate one explicit policy's constrained option values."""
    for key, allowed_values in allowed_values_by_key.items():
        value = policy.get(key)
        if value not in allowed_values:
            allowed = ", ".join(sorted(allowed_values))
            raise PpaError(
                (
                    f"{specification.path}: {root_key}.{policy_key}."
                    f"{key} must be one of: {allowed}."
                ),
                504,
            )


def _validated_external_flow_policy(
    specification: PerformanceComparisonSpecification,
    external_flow_value: Mapping[str, object],
) -> _TransactionImpactPolicy:
    """Validate and preserve the external-flow YAML policy."""
    method = external_flow_value.get(_METHOD_KEY)
    if method is None:
        raise PpaError(
            (
                f"{specification.path}: "
                f"{_TRANSACTION_IMPACT_METHODS_KEY}.{_EXTERNAL_FLOW_KEY}."
                f"{_METHOD_KEY} is required."
            ),
            504,
        )
    if method == _MODIFIED_DIETZ_METHOD:
        return _validated_modified_dietz_policy(
            specification,
            external_flow_value,
        )
    if method != _EVIDENCE_ONLY_METHOD:
        _raise_unsupported_external_flow_method(specification, method)

    return _TransactionImpactPolicy(
        method=_EVIDENCE_ONLY_METHOD,
        finding_label=TRANSACTION_IMPACT_POLICY_EXTERNAL_FLOW_EVIDENCE_ONLY,
    )


def _validated_performance_amount_policy(
    specification: PerformanceComparisonSpecification,
    performance_value: Mapping[str, object],
) -> _TransactionImpactPolicy:
    """Validate the performance transaction-amount impact YAML policy."""
    unsupported_keys = set(performance_value) - _PERFORMANCE_AMOUNT_REQUIRED_KEYS
    if unsupported_keys:
        unsupported = ", ".join(sorted(str(key) for key in unsupported_keys))
        raise PpaError(
            (
                f"{specification.path}: "
                f"{_TRANSACTION_IMPACT_METHODS_KEY}.{_PERFORMANCE_KEY} "
                f"has unsupported keys: {unsupported}."
            ),
            504,
        )

    missing_keys = _PERFORMANCE_AMOUNT_REQUIRED_KEYS - set(performance_value)
    if missing_keys:
        missing = ", ".join(sorted(str(key) for key in missing_keys))
        raise PpaError(
            (
                f"{specification.path}: "
                f"{_TRANSACTION_IMPACT_METHODS_KEY}.{_PERFORMANCE_KEY} "
                f"is missing required keys: {missing}."
            ),
            504,
        )

    method = performance_value.get(_METHOD_KEY)
    if method != _TRANSACTION_AMOUNT_DELTA_METHOD:
        raise PpaError(
            (
                f"{specification.path}: "
                f"{_TRANSACTION_IMPACT_METHODS_KEY}.{_PERFORMANCE_KEY}."
                f"{_METHOD_KEY} must be {_TRANSACTION_AMOUNT_DELTA_METHOD!r}."
            ),
            504,
        )
    for key, allowed_values in _PERFORMANCE_AMOUNT_ALLOWED_VALUES.items():
        value = performance_value.get(key)
        if value not in allowed_values:
            allowed = ", ".join(sorted(allowed_values))
            raise PpaError(
                (
                    f"{specification.path}: "
                    f"{_TRANSACTION_IMPACT_METHODS_KEY}.{_PERFORMANCE_KEY}."
                    f"{key} must be one of: {allowed}."
                ),
                504,
            )
    return _TransactionImpactPolicy(
        method=_TRANSACTION_AMOUNT_DELTA_METHOD,
        finding_label=TRANSACTION_IMPACT_POLICY_PERFORMANCE_AMOUNT_DELTA,
        denominator_source=str(performance_value[_DENOMINATOR_SOURCE_KEY]),
    )


def _validated_transaction_evidence_only_policy(
    specification: PerformanceComparisonSpecification,
    policy_key: str,
    policy_value: Mapping[str, object],
) -> _TransactionImpactPolicy:
    """Validate a transaction source-field evidence-only YAML policy."""
    _validate_policy_keys(
        specification,
        _TRANSACTION_IMPACT_METHODS_KEY,
        policy_key,
        policy_value,
        _TRANSACTION_EVIDENCE_ONLY_REQUIRED_KEYS,
    )
    _validate_policy_method(
        specification,
        _TRANSACTION_IMPACT_METHODS_KEY,
        policy_key,
        policy_value,
        _EVIDENCE_ONLY_METHOD,
    )
    return _TransactionImpactPolicy(
        method=_EVIDENCE_ONLY_METHOD,
        finding_label=_evidence_only_impact_policy_label(
            pc_cols.TRANSACTIONS,
            policy_key,
        ),
    )


def _validated_modified_dietz_policy(
    specification: PerformanceComparisonSpecification,
    external_flow_value: Mapping[str, object],
) -> _TransactionImpactPolicy:
    """Validate and preserve the Modified Dietz YAML policy shape."""
    unsupported_keys = set(external_flow_value) - _MODIFIED_DIETZ_REQUIRED_KEYS
    if unsupported_keys:
        unsupported = ", ".join(sorted(str(key) for key in unsupported_keys))
        raise PpaError(
            (
                f"{specification.path}: "
                f"{_TRANSACTION_IMPACT_METHODS_KEY}.{_EXTERNAL_FLOW_KEY} "
                f"has unsupported modified_dietz keys: {unsupported}."
            ),
            504,
        )

    missing_keys = _MODIFIED_DIETZ_REQUIRED_KEYS - set(external_flow_value)
    if missing_keys:
        missing = ", ".join(sorted(str(key) for key in missing_keys))
        raise PpaError(
            (
                f"{specification.path}: "
                f"{_TRANSACTION_IMPACT_METHODS_KEY}.{_EXTERNAL_FLOW_KEY} "
                f"is missing required modified_dietz keys: {missing}."
            ),
            504,
        )

    for key, allowed_values in _MODIFIED_DIETZ_ALLOWED_VALUES.items():
        value = external_flow_value.get(key)
        if value not in allowed_values:
            allowed = ", ".join(sorted(allowed_values))
            raise PpaError(
                (
                    f"{specification.path}: "
                    f"{_TRANSACTION_IMPACT_METHODS_KEY}.{_EXTERNAL_FLOW_KEY}."
                    f"{key} must be one of: {allowed}."
                ),
                504,
            )

    return _TransactionImpactPolicy(
        method=_MODIFIED_DIETZ_METHOD,
        finding_label="external_flow:modified_dietz",
        flow_timing=cast(str, external_flow_value[_FLOW_TIMING_KEY]),
        day_count=cast(str, external_flow_value[_DAY_COUNT_KEY]),
        inclusion_rule=cast(str, external_flow_value[_INCLUSION_RULE_KEY]),
        denominator_source=cast(str, external_flow_value[_DENOMINATOR_SOURCE_KEY]),
        double_count_policy=cast(str, external_flow_value[_DOUBLE_COUNT_POLICY_KEY]),
    )


def _modified_dietz_external_flow_eligibility(
    *,
    row: Mapping[str, object],
    policy: _TransactionImpactPolicy | None,
    portfolio_id: object | None,
    from_date: object | None,
    thru_date: object | None,
    denominator: object | None,
) -> _ModifiedDietzEligibility:
    """Return whether a transaction row has explicit Modified Dietz inputs.

    This is a guardrail for the diagnostic-only Modified Dietz path. It
    deliberately does not make the estimate part of regular contribution
    totals.
    """
    missing_inputs: list[str] = []
    flow_date = _modified_dietz_flow_date(row, policy)

    if row.get(pc_cols.PERFORMANCE_FLOW_SIGN) != TRANSACTION_PERFORMANCE_FLOW_SIGN_EXTERNAL:
        missing_inputs.append("external performance-flow semantics")
    if policy is None or policy.method != _MODIFIED_DIETZ_METHOD:
        missing_inputs.append("modified_dietz policy")
    elif (
        policy.double_count_policy
        != ModifiedDietzDoubleCountPolicy.CROSS_CHECK_ONLY.value
    ):
        missing_inputs.append("cross_check_only double-count policy")
    if flow_date is None:
        missing_inputs.append("flow date")
    if portfolio_id is None:
        missing_inputs.append("portfolio")
    if not isinstance(from_date, dt.date) or not isinstance(thru_date, dt.date):
        missing_inputs.append("portfolio period")
    if not _usable_modified_dietz_denominator(denominator):
        missing_inputs.append("nonzero begin_market_value denominator")

    if (
        flow_date is not None
        and isinstance(from_date, dt.date)
        and isinstance(thru_date, dt.date)
        and not from_date <= flow_date <= thru_date
    ):
        missing_inputs.append("in-period flow date")

    return _ModifiedDietzEligibility(
        eligible=not missing_inputs,
        missing_inputs=tuple(missing_inputs),
        flow_date=flow_date,
    )


def _modified_dietz_external_flow_impact(
    *,
    flow_delta: float,
    denominator: float,
    from_date: dt.date,
    thru_date: dt.date,
    flow_date: dt.date,
    inclusion_rule: str,
) -> float:
    """Return a Modified Dietz cross-check estimate for one external flow."""
    flow_weight = _modified_dietz_flow_weight(
        from_date=from_date,
        thru_date=thru_date,
        flow_date=flow_date,
        inclusion_rule=inclusion_rule,
    )
    return flow_delta * flow_weight / denominator


def _modified_dietz_flow_weight(
    *,
    from_date: dt.date,
    thru_date: dt.date,
    flow_date: dt.date,
    inclusion_rule: str,
) -> float:
    """Return the actual-days Modified Dietz flow weight."""
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


def _modified_dietz_flow_date(
    row: Mapping[str, object],
    policy: _TransactionImpactPolicy | None,
) -> dt.date | None:
    """Return the YAML-selected transaction flow date for Modified Dietz."""
    if policy is None:
        return None
    flow_date_column_by_timing = {
        ModifiedDietzFlowTiming.TRADE_DATE.value: pc_cols.TRANSACTION_DATE,
        ModifiedDietzFlowTiming.SETTLEMENT_DATE.value: pc_cols.SETTLEMENT_DATE,
    }
    if policy.flow_timing is None:
        return None
    date_column = flow_date_column_by_timing.get(policy.flow_timing)
    if date_column is None:
        return None
    value = row.get(date_column)
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return None


def _usable_modified_dietz_denominator(value: object) -> bool:
    """Return whether a configured Modified Dietz denominator is usable."""
    number = _modified_dietz_float(value)
    return number is not None and number != 0


def _usable_modified_dietz_number(value: object) -> bool:
    """Return whether a value can be used in Modified Dietz arithmetic."""
    return _modified_dietz_float(value) is not None


def _modified_dietz_float(value: object) -> float | None:
    """Return a float for non-boolean numeric Modified Dietz values."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _raise_unsupported_external_flow_method(
    specification: PerformanceComparisonSpecification,
    method: object,
) -> None:
    """Raise for external-flow methods that are not implemented yet."""
    method_text = str(method)
    reserved_note = ""
    if method_text in _RESERVED_EXTERNAL_FLOW_METHODS:
        reserved_note = " The method name is reserved but not implemented."
    raise PpaError(
        (
            f"{specification.path}: "
            f"{_TRANSACTION_IMPACT_METHODS_KEY}.{_EXTERNAL_FLOW_KEY}."
            f"{_METHOD_KEY} must be {_EVIDENCE_ONLY_METHOD!r} until an "
            "external-flow impact formula is explicitly supported."
            f"{reserved_note}"
        ),
        504,
    )
