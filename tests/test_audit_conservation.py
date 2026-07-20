"""Tests for lossless evidence and single-owner explanation invariants."""

from __future__ import annotations

# Python imports
import datetime as dt
from pathlib import Path
import unittest

# Third-party imports
import polars as pl

# Project imports
from ppar.errors import PpaError
from ppar.audit import bundle as _pc_bundle
from ppar.audit import conservation as _conservation
from ppar.audit import schema as pc_cols
from ppar.audit import workbook_tables as _workbook_tables
from ppar.audit.runner import compare_snapshots
from ppar.audit.safety_invariants import DifferenceDisposition

_CASH_HOLDING_COMPARISON_PATH = Path(
    "ppar/setup_templates/axys_apx_audit/"
    "axys_apx_audit.yaml"
)


def _cause_row(
    *,
    source_column: str,
    estimated_impact: float | None,
    dataset: str = pc_cols.HOLDINGS,
) -> dict[str, object]:
    """Return one compact cause row for invariant tests."""
    return {
        "portfolio_id": "TEST",
        "from_date": dt.date(2026, 1, 1),
        "thru_date": dt.date(2026, 1, 31),
        "security_id": "SEC1",
        "as_of_date": dt.date(2026, 1, 31),
        "dataset": dataset,
        "source_column": source_column,
        "code": "PC-HOLD-MV",
        "transaction_code": None,
        "transaction_category": None,
        "snapshot_a_value": 100.0,
        "snapshot_b_value": 110.0,
        "estimated_impact": estimated_impact,
    }


class TestAuditConservation(unittest.TestCase):
    """Verify Phase 2 safety nets fail closed on broken internal states."""

    def test_finding_audit_trail_is_lossless_and_includes_suppressed_rows(self) -> None:
        """SN-01 assigns every source finding a visible audit disposition."""
        findings = pl.DataFrame(
            {
                "code": ["PC-ONE", "PC-TWO"],
                "dataset": ["holdings", "transactions"],
                "source_record_locator": ["source:holdings:one", "source:transactions:two"],
                "suppressed": [False, True],
            }
        )

        trail = _conservation.finding_audit_trail(findings)
        _conservation.assert_complete_finding_audit_trail(findings, trail)

        self.assertEqual(trail.height, 2)
        self.assertEqual(trail["suppressed"].to_list(), [False, True])
        self.assertEqual(
            set(trail[_conservation.SAFETY_DISPOSITION].to_list()),
            {DifferenceDisposition.REVIEW_EVIDENCE.value},
        )
        self.assertEqual(
            _pc_bundle._finding_audit_trail_issues(trail),  # pylint: disable=protected-access
            [],
        )

    def test_finding_fingerprint_bytes_and_occurrences_are_stable(self) -> None:
        """Column-wise metadata construction preserves persisted SHA-256 identities."""
        findings = pl.DataFrame(
            {
                "code": ["PC-ONE", "PC-ONE"],
                "dataset": ["holdings", "holdings"],
                "source_record_locator": [
                    "source:holdings:one",
                    "source:holdings:one",
                ],
                "suppressed": [False, False],
            }
        )

        trail = _conservation.finding_audit_trail(findings)

        self.assertEqual(
            trail[_conservation.FINDING_FINGERPRINT].to_list(),
            [
                "9193989520a945810c5e35861a95bd35edc2bc489563cccef35053b68c0a22ef:1",
                "9193989520a945810c5e35861a95bd35edc2bc489563cccef35053b68c0a22ef:2",
            ],
        )
        self.assertEqual(trail.schema[_conservation.FINDING_SEQUENCE], pl.Int64)

    def test_finding_fingerprint_scalar_encoding_remains_compatible(self) -> None:
        """Fingerprint bytes preserve JSON edge-case representations."""
        findings = pl.DataFrame(
            {
                "text": ["caf\u00e9\n", 'quote " slash \\', "emoji \U0001f600", "plain"],
                "date": [dt.date(2026, 1, 2)] * 4,
                "timestamp": [dt.datetime(2026, 1, 2, 3, 4, 5, 6789)] * 4,
                "number": [0.0, -0.0, float("nan"), float("inf")],
                "count": [1, -2, 0, 42],
                "flag": [True, False, True, False],
                "empty": [None] * 4,
                "source_record_locator": ["source:test:1"] * 4,
            },
            schema_overrides={
                "date": pl.Date,
                "timestamp": pl.Datetime,
                "empty": pl.Null,
            },
        )

        trail = _conservation.finding_audit_trail(findings)

        self.assertEqual(
            trail[_conservation.FINDING_FINGERPRINT].to_list(),
            [
                "68afb1df199d4802afe9e05a6d93ae8062ad2e8e2a7adc52f0d5ba7db8e1f72f:1",
                "f423595cd834e88e0139f52d25a68c8948010191dcedda904645986a5d72f258:1",
                "90aed75d890f47950de0ba1b4b6a38028cd831faa84ca9ac00457903498db17f:1",
                "9188e6f80092b0464ba808e3c7a6e7cf1d7ce65fb9fc015616e41dfa5efaeeae:1",
            ],
        )

    def test_no_lost_findings_rejects_a_removed_row(self) -> None:
        """SN-01 raises if disposition processing removes a source finding."""
        findings = pl.DataFrame({"code": ["PC-ONE", "PC-TWO"]})
        broken_trail = _conservation.finding_audit_trail(findings).head(1)

        with self.assertRaisesRegex(PpaError, "SN-01 no-lost-differences"):
            _conservation.assert_complete_finding_audit_trail(
                findings,
                broken_trail,
            )

    def test_persisted_audit_trail_rejects_invalid_disposition(self) -> None:
        """Bundle validation rejects an invalid persisted safety disposition."""
        findings = pl.DataFrame(
            {
                "code": ["PC-ONE"],
                "source_record_locator": ["source:holdings:one"],
            }
        )
        trail = _conservation.finding_audit_trail(findings).with_columns(
            pl.lit("hidden").alias(_conservation.SAFETY_DISPOSITION)
        )

        issues = _pc_bundle._finding_audit_trail_issues(  # pylint: disable=protected-access
            trail
        )

        self.assertIn(
            "findings audit trail contains an invalid safety disposition",
            issues,
        )

    def test_cause_dispositions_preserve_counted_and_review_rows(self) -> None:
        """Cause rows explicitly distinguish counted ownership from review evidence."""
        original = pl.DataFrame(
            [
                _cause_row(
                    source_column=pc_cols.MARKET_VALUE,
                    estimated_impact=0.01,
                ),
                _cause_row(
                    source_column=pc_cols.PRICE,
                    estimated_impact=None,
                ),
            ],
            infer_schema_length=None,
        )

        causes = _conservation.cause_conservation_table(
            original,
            comparison_level="portfolio",
        )
        _conservation.assert_cause_conservation(
            original,
            causes,
            comparison_level="portfolio",
        )

        self.assertEqual(
            causes[_conservation.SAFETY_DISPOSITION].to_list(),
            [
                DifferenceDisposition.COUNTED_CAUSE.value,
                DifferenceDisposition.REVIEW_EVIDENCE.value,
            ],
        )
        self.assertEqual(
            causes[_conservation.ECONOMIC_EFFECT_ID].n_unique(),
            1,
        )
        self.assertIsNotNone(causes[_conservation.COUNTED_CAUSE_OWNER][0])
        self.assertIsNone(causes[_conservation.COUNTED_CAUSE_OWNER][1])

    def test_nonfinite_impacts_remain_review_evidence(self) -> None:
        """NaN and infinite estimates cannot become counted cause owners."""
        original = pl.DataFrame(
            [
                _cause_row(
                    source_column=pc_cols.MARKET_VALUE,
                    estimated_impact=float("nan"),
                ),
                _cause_row(
                    source_column=pc_cols.PRICE,
                    estimated_impact=float("inf"),
                ),
            ],
            infer_schema_length=None,
        )

        causes = _conservation.cause_conservation_table(
            original,
            comparison_level="portfolio",
        )
        _conservation.assert_cause_conservation(
            original,
            causes,
            comparison_level="portfolio",
        )

        self.assertEqual(
            causes[_conservation.SAFETY_DISPOSITION].to_list(),
            [DifferenceDisposition.REVIEW_EVIDENCE.value] * 2,
        )
        self.assertEqual(
            causes[_conservation.COUNTED_CAUSE_OWNER].null_count(),
            2,
        )

    def test_no_double_counting_rejects_two_owners_for_one_effect(self) -> None:
        """SN-02 raises when market value and price both own one holding effect."""
        original = pl.DataFrame(
            [
                _cause_row(
                    source_column=pc_cols.MARKET_VALUE,
                    estimated_impact=0.01,
                ),
                _cause_row(
                    source_column=pc_cols.PRICE,
                    estimated_impact=0.01,
                ),
            ],
            infer_schema_length=None,
        )
        causes = _conservation.cause_conservation_table(
            original,
            comparison_level="portfolio",
        )

        with self.assertRaisesRegex(PpaError, "SN-02 no-double-counting"):
            _conservation.assert_cause_conservation(
                original,
                causes,
                comparison_level="portfolio",
            )

    def test_transaction_component_cannot_own_explained_performance(self) -> None:
        """Transaction price remains supporting evidence rather than a second cause."""
        original = pl.DataFrame(
            [
                _cause_row(
                    dataset=pc_cols.TRANSACTIONS,
                    source_column=pc_cols.PRICE,
                    estimated_impact=0.01,
                )
            ],
            infer_schema_length=None,
        )
        causes = _conservation.cause_conservation_table(
            original,
            comparison_level="security",
        )

        with self.assertRaisesRegex(PpaError, "support-only field transactions.price"):
            _conservation.assert_cause_conservation(
                original,
                causes,
                comparison_level="security",
            )

    def test_cash_holding_value_has_one_counted_owner(self) -> None:
        """Cash holding market value owns the effect while quantity stays visible."""
        findings = compare_snapshots(
            _CASH_HOLDING_COMPARISON_PATH,
            comparison_level="portfolio",
        )
        # pylint: disable=protected-access
        causes = _workbook_tables._workbook_underlying_causes_table(
            findings,
            comparison_path=_CASH_HOLDING_COMPARISON_PATH,
        ).filter(
            (pl.col("dataset") == pc_cols.HOLDINGS)
            & (pl.col("portfolio_id") == "ALPHA")
            & (pl.col("security_id") == "causCASHUSD")
            & (pl.col("from_date") == pl.date(2026, 1, 31))
            & (pl.col("thru_date") == pl.date(2026, 2, 27))
        )

        dispositions = {
            row["source_column"]: row[_conservation.SAFETY_DISPOSITION]
            for row in causes.iter_rows(named=True)
        }

        self.assertEqual(
            dispositions,
            {
                pc_cols.MARKET_VALUE: DifferenceDisposition.COUNTED_CAUSE.value,
                pc_cols.QUANTITY: DifferenceDisposition.REVIEW_EVIDENCE.value,
            },
        )

    def test_security_fully_explained_arithmetic_is_a_hard_invariant(self) -> None:
        """SN-03 applies the same arithmetic guarantee at security grain."""
        period = {
            "portfolio_id": "TEST",
            "security_id": "SEC1",
            "from_date": dt.date(2026, 1, 1),
            "thru_date": dt.date(2026, 1, 31),
        }
        primary = pl.DataFrame(
            [
                {
                    **period,
                    "performance_change": 0.01,
                    "estimated_cause_total": 0.01,
                    "unexplained_change": 0.0,
                    "review_status": "Fully Explained",
                }
            ]
        )
        causes = pl.DataFrame([{**period, "estimated_impact": 0.009}])

        with self.assertRaisesRegex(PpaError, "SN-03 explanation invariant"):
            # pylint: disable=protected-access
            _workbook_tables._assert_explanation_invariants(
                primary,
                causes,
                (),
                comparison_level="security",
            )


if __name__ == "__main__":
    unittest.main()
