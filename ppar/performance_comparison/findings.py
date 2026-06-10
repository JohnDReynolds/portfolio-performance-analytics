"""Define performance comparison finding records and codes."""

from __future__ import annotations

# Python imports
from dataclasses import dataclass
from typing import Final, Sequence

# Third-party imports
import polars as pl

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
PORTFOLIO_ID = "portfolio_id"
SECURITY_ID = "security_id"
FROM_DATE = "from_date"
THRU_DATE = "thru_date"
SOURCE_FILE = "source_file"
SOURCE_COLUMN = "source_column"
TRANSACTION_CATEGORY = "transaction_category"
CASH_FLOW_SIGN = "cash_flow_sign"
PERFORMANCE_FLOW_SIGN = "performance_flow_sign"
TRANSACTION_SEMANTICS_SOURCE = "transaction_semantics_source"
TRANSACTION_IMPACT_POLICY = "transaction_impact_policy"
TRANSACTION_IMPACT_POLICY_EXTERNAL_FLOW_EVIDENCE_ONLY = "external_flow:evidence_only"
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
PC_POS_QTY: Final[str] = "PC-POS-QTY"
PC_POS_MV: Final[str] = "PC-POS-MV"
PC_POS_ACCR: Final[str] = "PC-POS-ACCR"
PC_CASH_MV: Final[str] = "PC-CASH-MV"
PC_PRICE: Final[str] = "PC-PRICE"
PC_FX_RATE: Final[str] = "PC-FX-RATE"
PC_TXN_ADD: Final[str] = "PC-TXN-ADD"
PC_TXN_DROP: Final[str] = "PC-TXN-DROP"
PC_TXN_AMT: Final[str] = "PC-TXN-AMT"
PC_TXN_QTY: Final[str] = "PC-TXN-QTY"
PC_TXN_PRICE: Final[str] = "PC-TXN-PRICE"

SEVERITY_INFORMATIONAL: Final[str] = "informational"
SEVERITY_MATERIAL: Final[str] = "material"
CONFIDENCE_HIGH: Final[str] = "high"
TARGET_OUTPUT: Final[str] = "target_output"
DIRECT_INPUT: Final[str] = "direct_input"
RELATED_OUTPUT: Final[str] = "related_output"
CONTEXT: Final[str] = "context"

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
    SOURCE_FILE,
    SOURCE_COLUMN,
    TRANSACTION_CATEGORY,
    CASH_FLOW_SIGN,
    PERFORMANCE_FLOW_SIGN,
    TRANSACTION_SEMANTICS_SOURCE,
    TRANSACTION_IMPACT_POLICY,
    TRANSACTION_IMPACT_DIAGNOSTIC,
    TRANSACTION_IMPACT_DIAGNOSTIC_ESTIMATE,
    SNAPSHOT_A_VALUE,
    SNAPSHOT_B_VALUE,
    DELTA_B_MINUS_A,
    RETURN_DENOMINATOR,
    RETURN_WEIGHT,
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
        source_file: Optional configured source file associated with the
            finding.
        source_column: Optional normalized column associated with the finding.
        transaction_category: Optional normalized transaction category for
            transaction findings.
        cash_flow_sign: Optional normalized source-supplied cash-flow sign for
            transaction findings.
        performance_flow_sign: Optional normalized source-supplied performance
            flow treatment for transaction findings.
        transaction_semantics_source: Optional provenance label for normalized
            transaction category/sign/flow semantics.
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
    severity: str
    confidence: str
    dataset: str
    evidence_role: str
    portfolio_id: object | None = None
    security_id: object | None = None
    from_date: object | None = None
    thru_date: object | None = None
    source_file: str | None = None
    source_column: str | None = None
    transaction_category: object | None = None
    cash_flow_sign: object | None = None
    performance_flow_sign: object | None = None
    transaction_semantics_source: object | None = None
    transaction_impact_policy: object | None = None
    transaction_impact_diagnostic: object | None = None
    transaction_impact_diagnostic_estimate: float | None = None
    snapshot_a_value: object | None = None
    snapshot_b_value: object | None = None
    delta_b_minus_a: float | None = None
    return_denominator: float | None = None
    return_weight: float | None = None
    message: str = ""
    suppressed: bool = False

    def to_dict(self) -> dict[str, object | None]:
        """Return this finding as a column-aligned dictionary."""
        return {
            FINDING_CODE: self.code,
            SEVERITY: self.severity,
            CONFIDENCE: self.confidence,
            DATASET: self.dataset,
            EVIDENCE_ROLE: self.evidence_role,
            PORTFOLIO_ID: self.portfolio_id,
            SECURITY_ID: self.security_id,
            FROM_DATE: self.from_date,
            THRU_DATE: self.thru_date,
            SOURCE_FILE: self.source_file,
            SOURCE_COLUMN: self.source_column,
            TRANSACTION_CATEGORY: self.transaction_category,
            CASH_FLOW_SIGN: self.cash_flow_sign,
            PERFORMANCE_FLOW_SIGN: self.performance_flow_sign,
            TRANSACTION_SEMANTICS_SOURCE: self.transaction_semantics_source,
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
            MESSAGE: self.message,
            SUPPRESSED: self.suppressed,
        }


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
    return pl.DataFrame([finding.to_dict() for finding in findings]).select(
        FINDING_COLUMNS
    )
