"""Apply comparison finding rules such as suppressions."""

from __future__ import annotations

# Python imports
from dataclasses import dataclass, replace
from typing import Final, Sequence

# Project imports
from ppar.errors import PpaError
from ppar.performance_comparison.findings import Finding
from ppar.performance_comparison.specification import PerformanceComparisonSpecification

_SUPPRESSIONS_KEY: Final[str] = "suppressions"
_CODE_KEY: Final[str] = "code"
_DATASET_KEY: Final[str] = "dataset"
_PORTFOLIO_ID_KEY: Final[str] = "portfolio_id"
_SECURITY_ID_KEY: Final[str] = "security_id"
_FROM_DATE_KEY: Final[str] = "from_date"
_THRU_DATE_KEY: Final[str] = "thru_date"
_SOURCE_COLUMN_KEY: Final[str] = "source_column"
_SUPPORTED_OPTIONAL_KEYS: Final[frozenset[str]] = frozenset(
    {
        _DATASET_KEY,
        _PORTFOLIO_ID_KEY,
        _SECURITY_ID_KEY,
        _FROM_DATE_KEY,
        _THRU_DATE_KEY,
        _SOURCE_COLUMN_KEY,
        "reason",
    }
)


@dataclass(frozen=True)
class SuppressionRule:
    """Define one exact-match finding suppression rule.

    Attributes:
        code: Uppercase finding code to suppress.
        dataset: Optional normalized dataset name.
        portfolio_id: Optional portfolio identifier.
        security_id: Optional security identifier.
        from_date: Optional period start date.
        thru_date: Optional period end date.
        source_column: Optional normalized source column name.
    """

    code: str
    dataset: object | None = None
    portfolio_id: object | None = None
    security_id: object | None = None
    from_date: object | None = None
    thru_date: object | None = None
    source_column: object | None = None

    @classmethod
    def from_mapping(cls, values: object, index: int) -> "SuppressionRule":
        """Return a suppression rule parsed from YAML values.

        Args:
            values: One item from the ``suppressions`` YAML list.
            index: Zero-based list holding used in validation messages.

        Returns:
            Parsed suppression rule.

        Raises:
            PpaError: If the suppression entry is not a mapping or does not
                contain a string ``code``.
        """
        if not isinstance(values, dict):
            raise PpaError(f"suppressions[{index}] must be a mapping.", 504)
        code = values.get(_CODE_KEY)
        if not isinstance(code, str) or not code:
            raise PpaError(f"suppressions[{index}].code must be a string.", 504)

        unsupported_keys = set(values) - _SUPPORTED_OPTIONAL_KEYS - {_CODE_KEY}
        if unsupported_keys:
            unsupported = ", ".join(sorted(unsupported_keys))
            raise PpaError(
                f"suppressions[{index}] has unsupported keys: {unsupported}.",
                504,
            )

        return cls(
            code=code.upper(),
            dataset=values.get(_DATASET_KEY),
            portfolio_id=values.get(_PORTFOLIO_ID_KEY),
            security_id=values.get(_SECURITY_ID_KEY),
            from_date=values.get(_FROM_DATE_KEY),
            thru_date=values.get(_THRU_DATE_KEY),
            source_column=values.get(_SOURCE_COLUMN_KEY),
        )

    def matches(self, finding: Finding) -> bool:
        """Return whether this rule suppresses a finding."""
        if finding.code.upper() != self.code:
            return False
        return (
            self._matches_optional(self.dataset, finding.dataset)
            and self._matches_optional(self.portfolio_id, finding.portfolio_id)
            and self._matches_optional(self.security_id, finding.security_id)
            and self._matches_optional(self.from_date, finding.from_date)
            and self._matches_optional(self.thru_date, finding.thru_date)
            and self._matches_optional(self.source_column, finding.source_column)
        )

    @staticmethod
    def _matches_optional(expected: object | None, actual: object | None) -> bool:
        """Return whether an optional rule field matches an actual value."""
        if expected is None:
            return True
        if actual is None:
            return False
        return str(expected) == str(actual)


def suppression_rules(
    specification: PerformanceComparisonSpecification,
) -> list[SuppressionRule]:
    """Return suppression rules from a comparison specification.

    Args:
        specification: Parsed comparison specification.

    Returns:
        Parsed suppression rules. Missing ``suppressions`` returns an empty
        list.

    Raises:
        PpaError: If the ``suppressions`` section has an invalid shape.
    """
    values = specification.values.get(_SUPPRESSIONS_KEY, [])
    if values is None:
        return []
    if not isinstance(values, list):
        raise PpaError("suppressions must be a list.", 504)
    return [
        SuppressionRule.from_mapping(suppression, index)
        for index, suppression in enumerate(values)
    ]


def apply_suppressions(
    findings: Sequence[Finding],
    specification: PerformanceComparisonSpecification,
) -> list[Finding]:
    """Mark findings suppressed when they match configured suppression rules.

    Args:
        findings: Findings produced by a comparison.
        specification: Parsed comparison specification containing optional
            suppression rules.

    Returns:
        New list of findings with matching records marked ``suppressed=True``.
    """
    rules = suppression_rules(specification)
    if not rules:
        return list(findings)

    suppressed_findings: list[Finding] = []
    for finding in findings:
        if finding.suppressed or any(rule.matches(finding) for rule in rules):
            suppressed_findings.append(replace(finding, suppressed=True))
        else:
            suppressed_findings.append(finding)
    return suppressed_findings
