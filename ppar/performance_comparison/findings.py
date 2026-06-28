"""Define public performance comparison finding records and codes.

These constants are intentionally public. They describe stable finding-table
columns, evidence roles, severities, and finding codes used in audit outputs
and report helper tables.
"""

from __future__ import annotations

# Python imports
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Sequence

# Third-party imports
import polars as pl

# Project imports
from ppar.performance_comparison.methods import (
    CashImpactMethod,
    ContributionImpactMethod,
    HoldingImpactMethod,
    PriceImpactMethod,
    TransactionImpactMethod,
)

__all__ = [
    "FindingSeverity",
    "FindingConfidence",
    "EvidenceRole",
    "TransactionMatchStatus",
    "FINDING_CODE",
    "SEVERITY",
    "CONFIDENCE",
    "DATASET",
    "EVIDENCE_ROLE",
    "SNAPSHOT_A_VALUE",
    "SNAPSHOT_B_VALUE",
    "DELTA_B_MINUS_A",
    "RETURN_DENOMINATOR",
    "RETURN_WEIGHT",
    "IMPACT_INPUT_VALUE",
    "PORTFOLIO_ID",
    "SECURITY_ID",
    "FROM_DATE",
    "THRU_DATE",
    "INPUT_DATE",
    "SOURCE_FILE",
    "SOURCE_COLUMN",
    "TRANSACTION_CODE",
    "TRANSACTION_CATEGORY",
    "CASH_FLOW_SIGN",
    "PERFORMANCE_FLOW_SIGN",
    "TRANSACTION_SEMANTICS_SOURCE",
    "TRANSACTION_MATCH_STATUS",
    "TRANSACTION_MATCH_STATUS_ID_MATCH",
    "TRANSACTION_MATCH_STATUS_ID_UNMATCHED",
    "TRANSACTION_MATCH_STATUS_STRICT_FALLBACK_UNMATCHED",
    "IMPACT_POLICY",
    "IMPACT_POLICY_EVIDENCE_ONLY_PREFIX",
    "IMPACT_POLICY_CASH_BALANCE",
    "IMPACT_POLICY_CASH_MARKET_VALUE",
    "IMPACT_POLICY_HOLDING_ACCRUED",
    "IMPACT_POLICY_PORTFOLIO_SOURCE_FIELD",
    "IMPACT_POLICY_HOLDING_MARKET_VALUE",
    "IMPACT_POLICY_HOLDING_QUANTITY_UNIT_MARKET_VALUE",
    "IMPACT_POLICY_PRICE_WEIGHTED",
    "IMPACT_POLICY_SECURITY_CONTRIBUTION",
    "IMPACT_POLICY_SECURITY_RETURN_WEIGHTED",
    "TRANSACTION_IMPACT_POLICY",
    "TRANSACTION_IMPACT_POLICY_EXTERNAL_FLOW_EVIDENCE_ONLY",
    "TRANSACTION_IMPACT_POLICY_PERFORMANCE_AMOUNT_DELTA",
    "TRANSACTION_IMPACT_POLICY_SECURITY_FLOW_MODIFIED_DIETZ",
    "TRANSACTION_IMPACT_DIAGNOSTIC",
    "TRANSACTION_IMPACT_DIAGNOSTIC_ESTIMATE",
    "MESSAGE",
    "SUPPRESSED",
    "PC_ROW_ADD",
    "PC_ROW_DROP",
    "PC_PORT_RET",
    "PC_PORT_MV",
    "PC_PORT_FLOW",
    "PC_SEC_RET",
    "PC_SEC_WGT",
    "PC_SEC_CONTR",
    "PC_SEC_ADD",
    "PC_SEC_DROP",
    "PC_REF_ID",
    "PC_REF_CLASS",
    "PC_HOLD_QTY",
    "PC_HOLD_MV",
    "PC_HOLD_COST",
    "PC_HOLD_ACCR",
    "PC_CASH_MV",
    "PC_PRICE",
    "PC_FX_RATE",
    "PC_TXN_ADD",
    "PC_TXN_DROP",
    "PC_TXN_AMT",
    "PC_TXN_QTY",
    "PC_TXN_PRICE",
    "PC_TXN_COMM",
    "SEVERITY_INFORMATIONAL",
    "SEVERITY_MATERIAL",
    "CONFIDENCE_HIGH",
    "TARGET_OUTPUT",
    "DIRECT_INPUT",
    "RELATED_OUTPUT",
    "CONTEXT",
    "FINDING_COLUMNS",
    "Finding",
    "findings_to_polars",
]

FINDING_CODE = "code"
SEVERITY = "severity"
CONFIDENCE = "confidence"
DATASET = "dataset"
EVIDENCE_ROLE = "evidence_role"
SNAPSHOT_A_VALUE = "snapshot_a_value"
SNAPSHOT_B_VALUE = "snapshot_b_value"
DELTA_B_MINUS_A = "delta_b_minus_a"
RETURN_DENOMINATOR = "return_denominator"
RETURN_WEIGHT = "return_weight"
IMPACT_INPUT_VALUE = "impact_input_value"
PORTFOLIO_ID = "portfolio_id"
SECURITY_ID = "security_id"
FROM_DATE = "from_date"
THRU_DATE = "thru_date"
INPUT_DATE = "input_date"
SOURCE_FILE = "source_file"
SOURCE_COLUMN = "source_column"
TRANSACTION_CODE = "transaction_code"
TRANSACTION_CATEGORY = "transaction_category"
CASH_FLOW_SIGN = "cash_flow_sign"
PERFORMANCE_FLOW_SIGN = "performance_flow_sign"
TRANSACTION_SEMANTICS_SOURCE = "transaction_semantics_source"
TRANSACTION_MATCH_STATUS = "transaction_match_status"
IMPACT_POLICY = "impact_policy"
IMPACT_POLICY_EVIDENCE_ONLY_PREFIX = "evidence_only:"
IMPACT_POLICY_CASH_BALANCE = (
    f"cash_balance:{CashImpactMethod.CASH_DELTA_OVER_RETURN_DENOMINATOR.value}"
)
IMPACT_POLICY_CASH_MARKET_VALUE = (
    f"cash_market_value:{CashImpactMethod.CASH_DELTA_OVER_RETURN_DENOMINATOR.value}"
)
IMPACT_POLICY_PORTFOLIO_SOURCE_FIELD = (
    "portfolio_source_field:"
    f"{ContributionImpactMethod.SOURCE_FIELD_DELTA_OVER_BEGIN_MARKET_VALUE.value}"
)
IMPACT_POLICY_HOLDING_MARKET_VALUE = (
    "holding_market_value:"
    f"{HoldingImpactMethod.MARKET_VALUE_DELTA_OVER_RETURN_DENOMINATOR.value}"
)
IMPACT_POLICY_HOLDING_ACCRUED = (
    "holding_accrued:"
    f"{HoldingImpactMethod.ACCRUED_DELTA_OVER_RETURN_DENOMINATOR.value}"
)
IMPACT_POLICY_HOLDING_QUANTITY_UNIT_MARKET_VALUE = (
    "holding_quantity:"
    f"{HoldingImpactMethod.QUANTITY_DELTA_TIMES_SNAPSHOT_A_UNIT_MARKET_VALUE_OVER_RETURN_DENOMINATOR.value}"
)
IMPACT_POLICY_PRICE_WEIGHTED = (
    "price_weighted:"
    f"{PriceImpactMethod.PRICE_DELTA_OVER_SNAPSHOT_A_PRICE_TIMES_WEIGHT.value}"
)
IMPACT_POLICY_SECURITY_CONTRIBUTION = (
    f"security_contribution:{ContributionImpactMethod.VENDOR_CONTRIBUTION_DELTA.value}"
)
IMPACT_POLICY_SECURITY_RETURN_WEIGHTED = (
    "security_return:"
    f"{ContributionImpactMethod.SECURITY_RETURN_DELTA_TIMES_WEIGHT.value}"
)
TRANSACTION_IMPACT_POLICY = "transaction_impact_policy"
TRANSACTION_IMPACT_POLICY_EXTERNAL_FLOW_EVIDENCE_ONLY = (
    f"external_flow:{TransactionImpactMethod.EVIDENCE_ONLY.value}"
)
TRANSACTION_IMPACT_POLICY_PERFORMANCE_AMOUNT_DELTA = (
    "performance:"
    f"{TransactionImpactMethod.TRANSACTION_AMOUNT_DELTA_OVER_RETURN_DENOMINATOR.value}"
)
TRANSACTION_IMPACT_POLICY_SECURITY_FLOW_MODIFIED_DIETZ = (
    f"security_flow:{TransactionImpactMethod.MODIFIED_DIETZ.value}"
)
TRANSACTION_IMPACT_DIAGNOSTIC = "transaction_impact_diagnostic"
TRANSACTION_IMPACT_DIAGNOSTIC_ESTIMATE = "transaction_impact_diagnostic_estimate"
MESSAGE = "message"
SUPPRESSED = "suppressed"

PC_ROW_ADD: Final[str] = "PC-ROW-ADD"
PC_ROW_DROP: Final[str] = "PC-ROW-DROP"
PC_PORT_RET: Final[str] = "PC-PORT-RET"
PC_PORT_MV: Final[str] = "PC-PORT-MV"
PC_PORT_FLOW: Final[str] = "PC-PORT-FLOW"
PC_SEC_RET: Final[str] = "PC-SEC-RET"
PC_SEC_WGT: Final[str] = "PC-SEC-WGT"
PC_SEC_CONTR: Final[str] = "PC-SEC-CONTR"
PC_SEC_ADD: Final[str] = "PC-SEC-ADD"
PC_SEC_DROP: Final[str] = "PC-SEC-DROP"
PC_REF_ID: Final[str] = "PC-REF-ID"
PC_REF_CLASS: Final[str] = "PC-REF-CLASS"
PC_HOLD_QTY: Final[str] = "PC-HOLD-QTY"
PC_HOLD_MV: Final[str] = "PC-HOLD-MV"
PC_HOLD_COST: Final[str] = "PC-HOLD-COST"
PC_HOLD_ACCR: Final[str] = "PC-HOLD-ACCR"
PC_CASH_MV: Final[str] = "PC-CASH-MV"
PC_PRICE: Final[str] = "PC-PRICE"
PC_FX_RATE: Final[str] = "PC-FX-RATE"
PC_TXN_ADD: Final[str] = "PC-TXN-ADD"
PC_TXN_DROP: Final[str] = "PC-TXN-DROP"
PC_TXN_AMT: Final[str] = "PC-TXN-AMT"
PC_TXN_QTY: Final[str] = "PC-TXN-QTY"
PC_TXN_PRICE: Final[str] = "PC-TXN-PRICE"
PC_TXN_COMM: Final[str] = "PC-TXN-COMM"


class FindingSeverity(StrEnum):
    """Supported finding materiality/severity labels."""

    INFORMATIONAL = "informational"
    MATERIAL = "material"


class FindingConfidence(StrEnum):
    """Supported finding explanation confidence labels."""

    HIGH = "high"


class EvidenceRole(StrEnum):
    """Supported evidence roles in the explanation model."""

    TARGET_OUTPUT = "target_output"
    DIRECT_INPUT = "direct_input"
    RELATED_OUTPUT = "related_output"
    CONTEXT = "context"


class TransactionMatchStatus(StrEnum):
    """Supported transaction matching diagnostic labels."""

    ID_MATCH = "transaction_id_match"
    ID_UNMATCHED = "transaction_id_unmatched"
    STRICT_FALLBACK_UNMATCHED = "strict_fallback_unmatched"


TRANSACTION_MATCH_STATUS_ID_MATCH: Final[TransactionMatchStatus] = (
    TransactionMatchStatus.ID_MATCH
)
TRANSACTION_MATCH_STATUS_ID_UNMATCHED: Final[TransactionMatchStatus] = (
    TransactionMatchStatus.ID_UNMATCHED
)
TRANSACTION_MATCH_STATUS_STRICT_FALLBACK_UNMATCHED: Final[TransactionMatchStatus] = (
    TransactionMatchStatus.STRICT_FALLBACK_UNMATCHED
)

SEVERITY_INFORMATIONAL: Final[FindingSeverity] = FindingSeverity.INFORMATIONAL
SEVERITY_MATERIAL: Final[FindingSeverity] = FindingSeverity.MATERIAL
CONFIDENCE_HIGH: Final[FindingConfidence] = FindingConfidence.HIGH
TARGET_OUTPUT: Final[EvidenceRole] = EvidenceRole.TARGET_OUTPUT
DIRECT_INPUT: Final[EvidenceRole] = EvidenceRole.DIRECT_INPUT
RELATED_OUTPUT: Final[EvidenceRole] = EvidenceRole.RELATED_OUTPUT
CONTEXT: Final[EvidenceRole] = EvidenceRole.CONTEXT

FINDING_COLUMNS: Final[tuple[str, ...]] = (
    FINDING_CODE,
    SEVERITY,
    CONFIDENCE,
    DATASET,
    EVIDENCE_ROLE,
    PORTFOLIO_ID,
    SECURITY_ID,
    FROM_DATE,
    THRU_DATE,
    INPUT_DATE,
    SOURCE_FILE,
    SOURCE_COLUMN,
    TRANSACTION_CODE,
    TRANSACTION_CATEGORY,
    CASH_FLOW_SIGN,
    PERFORMANCE_FLOW_SIGN,
    TRANSACTION_SEMANTICS_SOURCE,
    TRANSACTION_MATCH_STATUS,
    IMPACT_POLICY,
    TRANSACTION_IMPACT_POLICY,
    TRANSACTION_IMPACT_DIAGNOSTIC,
    TRANSACTION_IMPACT_DIAGNOSTIC_ESTIMATE,
    SNAPSHOT_A_VALUE,
    SNAPSHOT_B_VALUE,
    DELTA_B_MINUS_A,
    RETURN_DENOMINATOR,
    RETURN_WEIGHT,
    IMPACT_INPUT_VALUE,
    MESSAGE,
    SUPPRESSED,
)


@dataclass(frozen=True)
class Finding:
    """Represent one performance comparison finding.

    Attributes:
        code: Stable mnemonic finding code.
        severity: Finding materiality/severity label.
        confidence: Explanation confidence label.
        dataset: Normalized dataset associated with the finding.
        evidence_role: Role this finding plays in the explanation model.
        portfolio_id: Optional portfolio identifier.
        security_id: Optional security identifier.
        from_date: Optional period start date.
        thru_date: Optional period end date.
        input_date: Optional date represented by the changed input row.
        source_file: Optional configured source file associated with the
            finding.
        source_column: Optional normalized column associated with the finding.
        transaction_code: Optional source transaction code for transaction
            findings.
        transaction_category: Optional normalized transaction category for
            transaction findings.
        cash_flow_sign: Optional normalized source-supplied cash-flow sign for
            transaction findings.
        performance_flow_sign: Optional normalized source-supplied performance
            flow treatment for transaction findings.
        transaction_semantics_source: Optional provenance label for normalized
            transaction category/sign/flow semantics.
        transaction_match_status: Optional diagnostic label describing how a
            transaction finding was matched or why it remains unmatched.
        impact_policy: Optional YAML-configured policy controlling
            non-transaction contribution-ranking estimates.
        transaction_impact_policy: Optional YAML-configured policy controlling
            transaction impact treatment.
        transaction_impact_diagnostic: Optional review diagnostic explaining
            inactive or ineligible transaction impact treatment.
        transaction_impact_diagnostic_estimate: Optional review-only impact
            estimate excluded from contribution totals.
        snapshot_a_value: Value from snapshot A.
        snapshot_b_value: Value from snapshot B.
        delta_b_minus_a: Numeric difference calculated as B minus A.
        return_denominator: Optional denominator for approximate
            return-impact estimates.
        return_weight: Optional weight for approximate security return-impact
            estimates.
        message: Human-readable summary.
        suppressed: Whether a suppression rule hid the finding.
    """

    code: str
    severity: FindingSeverity
    confidence: FindingConfidence
    dataset: str
    evidence_role: EvidenceRole
    portfolio_id: object | None = None
    security_id: object | None = None
    from_date: object | None = None
    thru_date: object | None = None
    input_date: object | None = None
    source_file: str | None = None
    source_column: str | None = None
    transaction_code: object | None = None
    transaction_category: object | None = None
    cash_flow_sign: object | None = None
    performance_flow_sign: object | None = None
    transaction_semantics_source: object | None = None
    transaction_match_status: TransactionMatchStatus | None = None
    impact_policy: object | None = None
    transaction_impact_policy: object | None = None
    transaction_impact_diagnostic: object | None = None
    transaction_impact_diagnostic_estimate: float | None = None
    snapshot_a_value: object | None = None
    snapshot_b_value: object | None = None
    delta_b_minus_a: float | None = None
    return_denominator: float | None = None
    return_weight: float | None = None
    impact_input_value: float | None = None
    message: str = ""
    suppressed: bool = False

    def to_dict(self) -> dict[str, object | None]:
        """Return this finding as a column-aligned dictionary."""
        return {
            FINDING_CODE: self.code,
            SEVERITY: _string_value(self.severity),
            CONFIDENCE: _string_value(self.confidence),
            DATASET: self.dataset,
            EVIDENCE_ROLE: _string_value(self.evidence_role),
            PORTFOLIO_ID: self.portfolio_id,
            SECURITY_ID: self.security_id,
            FROM_DATE: self.from_date,
            THRU_DATE: self.thru_date,
            INPUT_DATE: self.input_date,
            SOURCE_FILE: self.source_file,
            SOURCE_COLUMN: self.source_column,
            TRANSACTION_CODE: _string_value_or_none(self.transaction_code),
            TRANSACTION_CATEGORY: _string_value_or_none(self.transaction_category),
            CASH_FLOW_SIGN: _string_value_or_none(self.cash_flow_sign),
            PERFORMANCE_FLOW_SIGN: _string_value_or_none(self.performance_flow_sign),
            TRANSACTION_SEMANTICS_SOURCE: _string_value_or_none(
                self.transaction_semantics_source
            ),
            TRANSACTION_MATCH_STATUS: _string_value_or_none(
                self.transaction_match_status
            ),
            IMPACT_POLICY: self.impact_policy,
            TRANSACTION_IMPACT_POLICY: self.transaction_impact_policy,
            TRANSACTION_IMPACT_DIAGNOSTIC: self.transaction_impact_diagnostic,
            TRANSACTION_IMPACT_DIAGNOSTIC_ESTIMATE: (
                self.transaction_impact_diagnostic_estimate
            ),
            SNAPSHOT_A_VALUE: self.snapshot_a_value,
            SNAPSHOT_B_VALUE: self.snapshot_b_value,
            DELTA_B_MINUS_A: self.delta_b_minus_a,
            RETURN_DENOMINATOR: self.return_denominator,
            RETURN_WEIGHT: self.return_weight,
            IMPACT_INPUT_VALUE: self.impact_input_value,
            MESSAGE: self.message,
            SUPPRESSED: self.suppressed,
        }


def _string_value(value: StrEnum | str) -> str:
    """Return a plain string for enum-backed output values."""
    return str(value)


def _string_value_or_none(value: object | None) -> object | None:
    """Return a plain string for enum-backed optional output values."""
    if isinstance(value, StrEnum):
        return str(value)
    return value


def findings_to_polars(findings: Sequence[Finding]) -> pl.DataFrame:
    """Return findings as a Polars DataFrame with stable output columns.

    Args:
        findings: Finding records to convert.

    Returns:
        DataFrame containing one row per finding. Empty inputs return an empty
        DataFrame with the standard finding columns.
    """
    if not findings:
        return pl.DataFrame(schema={column: pl.Null for column in FINDING_COLUMNS})
    return pl.DataFrame(
        [finding.to_dict() for finding in findings],
        infer_schema_length=None,
    ).select(FINDING_COLUMNS)
