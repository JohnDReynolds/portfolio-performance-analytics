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
SNAPSHOT_A_VALUE = "snapshot_a_value"
SNAPSHOT_B_VALUE = "snapshot_b_value"
DELTA_B_MINUS_A = "delta_b_minus_a"
PORTFOLIO_ID = "portfolio_id"
SECURITY_ID = "security_id"
FROM_DATE = "from_date"
THRU_DATE = "thru_date"
SOURCE_COLUMN = "source_column"
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

SEVERITY_INFORMATIONAL: Final[str] = "informational"
SEVERITY_MATERIAL: Final[str] = "material"
CONFIDENCE_HIGH: Final[str] = "high"

FINDING_COLUMNS: Final[tuple[str, ...]] = (
    FINDING_CODE,
    SEVERITY,
    CONFIDENCE,
    DATASET,
    PORTFOLIO_ID,
    SECURITY_ID,
    FROM_DATE,
    THRU_DATE,
    SOURCE_COLUMN,
    SNAPSHOT_A_VALUE,
    SNAPSHOT_B_VALUE,
    DELTA_B_MINUS_A,
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
        portfolio_id: Optional portfolio identifier.
        security_id: Optional security identifier.
        from_date: Optional period start date.
        thru_date: Optional period end date.
        source_column: Optional normalized column associated with the finding.
        snapshot_a_value: Value from snapshot A.
        snapshot_b_value: Value from snapshot B.
        delta_b_minus_a: Numeric difference calculated as B minus A.
        message: Human-readable summary.
        suppressed: Whether a suppression rule hid the finding.
    """

    code: str
    severity: str
    confidence: str
    dataset: str
    portfolio_id: object | None = None
    security_id: object | None = None
    from_date: object | None = None
    thru_date: object | None = None
    source_column: str | None = None
    snapshot_a_value: object | None = None
    snapshot_b_value: object | None = None
    delta_b_minus_a: float | None = None
    message: str = ""
    suppressed: bool = False

    def to_dict(self) -> dict[str, object | None]:
        """Return this finding as a column-aligned dictionary."""
        return {
            FINDING_CODE: self.code,
            SEVERITY: self.severity,
            CONFIDENCE: self.confidence,
            DATASET: self.dataset,
            PORTFOLIO_ID: self.portfolio_id,
            SECURITY_ID: self.security_id,
            FROM_DATE: self.from_date,
            THRU_DATE: self.thru_date,
            SOURCE_COLUMN: self.source_column,
            SNAPSHOT_A_VALUE: self.snapshot_a_value,
            SNAPSHOT_B_VALUE: self.snapshot_b_value,
            DELTA_B_MINUS_A: self.delta_b_minus_a,
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
