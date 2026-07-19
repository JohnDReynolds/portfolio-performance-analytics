"""Tests for performance-comparison Data Issues checks."""

from __future__ import annotations

# Python imports
import datetime as dt
from pathlib import Path
import tempfile
import textwrap
import unittest

# Third-party imports
import polars as pl

# Project imports
from ppar.audit import AuditSpecification, compare_snapshots, schema as pc_cols
from ppar.audit.config_validation import validate_config
from ppar.audit.data_issues import checks as data_issues
from ppar.errors import PpaError


class TestDataIssues(unittest.TestCase):
    """Validate source-data consistency checks used by the Data Issues sheet."""

    def test_continuity_respects_adjacency_and_combined_tolerance(self) -> None:
        """Continuity flags only adjacent periods beyond absolute and percent limits."""
        frame = pl.DataFrame(
            {
                data_issues.SNAPSHOT: ["Snapshot A"] * 6,
                pc_cols.PORTFOLIO_ID: ["P1", "P1", "P2", "P2", "P3", "P3"],
                pc_cols.FROM_DATE: [
                    dt.date(2026, 1, 1),
                    dt.date(2026, 2, 2),
                    dt.date(2026, 1, 1),
                    dt.date(2026, 2, 1),
                    dt.date(2026, 1, 1),
                    dt.date(2026, 2, 1),
                ],
                pc_cols.THRU_DATE: [
                    dt.date(2026, 1, 31),
                    dt.date(2026, 2, 28),
                    dt.date(2026, 1, 31),
                    dt.date(2026, 2, 28),
                    dt.date(2026, 1, 31),
                    dt.date(2026, 2, 28),
                ],
                pc_cols.BEGIN_MARKET_VALUE: [900.0, 500.0, 900.0, 989.0, 1900.0, 1985.0],
                pc_cols.END_MARKET_VALUE: [1000.0, 600.0, 1000.0, 1100.0, 2000.0, 2100.0],
            }
        )
        config = {
            data_issues.ISSUE_PORTFOLIO_MV_CONTINUITY: {
                "absolute_tolerance": 10.0,
                "percent_tolerance": 1.0,
            }
        }

        issues = data_issues._market_value_continuity_issues(
            (frame,),
            config,
            dataset_name=pc_cols.PORTFOLIO_PERFORMANCE,
        )

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0][pc_cols.PORTFOLIO_ID], "P2")
        self.assertEqual(issues[0][data_issues.DIFFERENCE], -11.0)

    def test_packaged_demo_includes_every_enabled_data_issues_type(self) -> None:
        """The Axys/APX demo keeps examples of every enabled X-Ref issue type."""
        comparison_path = (
            Path(__file__).resolve().parents[1]
            / "ppar"
            / "setup_templates"
            / "axys_apx_audit"
            / "axys_apx_audit.yaml"
        )

        issues = data_issues.data_issues_table(comparison_path)
        issue_types = set(issues.get_column(data_issues.ISSUE_TYPE).to_list())

        self.assertEqual(
            issue_types,
            {
                "duplicate_transactions",
                "dividend_rate",
                "holdings_nonpositive_price",
                "holdings_price_range",
                "holdings_stale_price",
                "large_price_variation",
                "missing_dividend",
                "transaction_security_type_mismatch",
                "transactions_nonpositive_price",
                "transactions_price_range",
                "holdings_accrued_rate",
                "pa_sa_rate",
            },
        )

    def test_packaged_demo_includes_dividend_rate_data_issues_issue(self) -> None:
        """The Axys/APX demo includes a visible dividend-rate mismatch example."""
        comparison_path = (
            Path(__file__).resolve().parents[1]
            / "ppar"
            / "setup_templates"
            / "axys_apx_audit"
            / "axys_apx_audit.yaml"
        )

        issues = data_issues.data_issues_table(comparison_path)
        dividend_rate_issues = issues.filter(
            (issues[data_issues.ISSUE_TYPE] == "dividend_rate")
            & (issues["security_id"] == "csusJPM")
        )

        self.assertEqual(dividend_rate_issues.height, 2)
        self.assertEqual(
            set(dividend_rate_issues.get_column("portfolio_id").to_list()),
            {"ALPHA", "BALANCED"},
        )

    def test_packaged_demo_includes_pa_sa_rate_data_issues_issue(self) -> None:
        """The Axys/APX demo includes a visible accrued-interest rate mismatch."""
        comparison_path = (
            Path(__file__).resolve().parents[1]
            / "ppar"
            / "setup_templates"
            / "axys_apx_audit"
            / "axys_apx_audit.yaml"
        )

        issues = data_issues.data_issues_table(comparison_path)
        pa_rate_issues = issues.filter(
            (issues[data_issues.ISSUE_TYPE] == "pa_sa_rate")
            & (issues["security_id"] == "fius91282Y5Y1")
        )

        self.assertEqual(pa_rate_issues.height, 2)
        self.assertEqual(
            set(pa_rate_issues.get_column("portfolio_id").to_list()),
            {"ALPHA", "INCOME"},
        )

    def test_packaged_demo_scopes_nonpositive_holding_price_example(self) -> None:
        """The packaged opt-in price check flags only its deliberate population."""
        comparison_path = (
            Path(__file__).resolve().parents[1]
            / "ppar"
            / "setup_templates"
            / "axys_apx_audit"
            / "axys_apx_audit.yaml"
        )

        issues = data_issues.data_issues_table(comparison_path).filter(
            pl.col(data_issues.ISSUE_TYPE)
            == data_issues.ISSUE_HOLDINGS_NONPOSITIVE_PRICE
        )

        self.assertEqual(issues.height, 2)
        self.assertEqual(
            set(issues.get_column(data_issues.SNAPSHOT).to_list()),
            {"Snapshot A", "Snapshot B"},
        )
        self.assertEqual(
            set(issues.get_column(pc_cols.PORTFOLIO_ID).to_list()),
            {"ALPHA"},
        )
        self.assertEqual(
            set(issues.get_column(pc_cols.SECURITY_ID).to_list()),
            {"csusMSFT"},
        )

    def test_packaged_demo_scopes_nonpositive_transaction_price_example(self) -> None:
        """The packaged trade-price check ignores zero-quantity duplicate rows."""
        comparison_path = (
            Path(__file__).resolve().parents[1]
            / "ppar"
            / "setup_templates"
            / "axys_apx_audit"
            / "axys_apx_audit.yaml"
        )

        issues = data_issues.data_issues_table(comparison_path).filter(
            pl.col(data_issues.ISSUE_TYPE)
            == data_issues.ISSUE_TRANSACTIONS_NONPOSITIVE_PRICE
        )

        self.assertEqual(issues.height, 2)
        self.assertEqual(
            set(issues.get_column(data_issues.SNAPSHOT).to_list()),
            {"Snapshot A", "Snapshot B"},
        )
        self.assertEqual(
            set(issues.get_column(pc_cols.PORTFOLIO_ID).to_list()),
            {"DATAAUDIT"},
        )
        self.assertEqual(
            set(issues.get_column(pc_cols.SECURITY_ID).to_list()),
            {"csusKO"},
        )

    def test_packaged_demo_scopes_stale_holding_price_example(self) -> None:
        """The packaged stale-price example is one observed run per snapshot."""
        comparison_path = (
            Path(__file__).resolve().parents[1]
            / "ppar"
            / "setup_templates"
            / "axys_apx_audit"
            / "axys_apx_audit.yaml"
        )

        issues = data_issues.data_issues_table(comparison_path).filter(
            pl.col(data_issues.ISSUE_TYPE)
            == data_issues.ISSUE_HOLDINGS_STALE_PRICE
        )

        self.assertEqual(issues.height, 2)
        self.assertEqual(
            set(issues.get_column(data_issues.SNAPSHOT).to_list()),
            {"Snapshot A", "Snapshot B"},
        )
        self.assertEqual(
            set(issues.get_column(pc_cols.PORTFOLIO_ID).to_list()),
            {"ALPHA"},
        )
        self.assertEqual(
            set(issues.get_column(pc_cols.SECURITY_ID).to_list()),
            {"csusGOOGL"},
        )
        self.assertEqual(
            set(issues.get_column(data_issues.VALUE_A).to_list()),
            {313.0},
        )
        self.assertEqual(
            set(issues.get_column(data_issues.AS_OF_DATE).to_list()),
            {dt.date(2026, 1, 30)},
        )

    def test_packaged_demo_reports_real_avgo_large_price_variations(self) -> None:
        """The packaged named rule reports real AVGO period observations."""
        comparison_path = (
            Path(__file__).resolve().parents[1]
            / "ppar"
            / "setup_templates"
            / "axys_apx_audit"
            / "axys_apx_audit.yaml"
        )

        issues = data_issues.data_issues_table(comparison_path).filter(
            pl.col(data_issues.ISSUE_TYPE)
            == data_issues.ISSUE_LARGE_PRICE_VARIATION
        )

        self.assertEqual(issues.height, 6)
        self.assertEqual(
            set(issues.get_column(data_issues.SNAPSHOT).to_list()),
            {"Snapshot A", "Snapshot B"},
        )
        self.assertEqual(
            set(issues.get_column(pc_cols.SECURITY_ID).to_list()),
            {"csusAVGO"},
        )
        balanced = issues.filter(
            pl.col(pc_cols.PORTFOLIO_ID) == "BALANCED"
        )
        self.assertEqual(balanced.height, 2)
        self.assertEqual(
            set(balanced.get_column(data_issues.VALUE_A).to_list()),
            {309.51},
        )
        self.assertEqual(
            set(balanced.get_column(data_issues.VALUE_B).to_list()),
            {371.55},
        )
        self.assertTrue(
            all(
                "20.04% maximum price variation" in explanation
                and "10 calendar days" in explanation
                for explanation in balanced.get_column(data_issues.EXPLANATION)
            )
        )

    def test_packaged_demo_keeps_original_cost_review_optional(self) -> None:
        """The primary demo omits optional cost inputs and disables their check."""
        comparison_path = (
            Path(__file__).resolve().parents[1]
            / "ppar"
            / "setup_templates"
            / "axys_apx_audit"
            / "axys_apx_audit.yaml"
        )

        specification = AuditSpecification(comparison_path)
        check_config = specification.values["data_issues"][
            data_issues.ISSUE_DELIVER_IN_ORIGINAL_COST_INCOMPLETE
        ]
        issue_types = set(
            data_issues.data_issues_table(comparison_path)
            .get_column(data_issues.ISSUE_TYPE)
            .to_list()
        )
        for snapshot_name in ("snapshot_a", "snapshot_b"):
            header = (
                comparison_path.parent / snapshot_name / "transactions.csv"
            ).read_text(encoding="utf-8").splitlines()[0].split(",")
            self.assertNotIn("ORIGINAL_COST_DATE", header)
            self.assertNotIn("ORIGINAL_COST", header)

        self.assertFalse(check_config["enabled"])
        self.assertNotIn(
            data_issues.ISSUE_DELIVER_IN_ORIGINAL_COST_INCOMPLETE,
            issue_types,
        )

    def test_packaged_demo_scopes_security_type_mismatch_example(self) -> None:
        """The packaged classification example is an isolated case-only mismatch."""
        comparison_path = (
            Path(__file__).resolve().parents[1]
            / "ppar"
            / "setup_templates"
            / "axys_apx_audit"
            / "axys_apx_audit.yaml"
        )

        issues = data_issues.data_issues_table(comparison_path).filter(
            pl.col(data_issues.ISSUE_TYPE)
            == data_issues.ISSUE_TRANSACTION_SECURITY_TYPE_MISMATCH
        )

        self.assertEqual(issues.height, 2)
        self.assertEqual(
            set(issues.get_column(data_issues.SNAPSHOT).to_list()),
            {"Snapshot A", "Snapshot B"},
        )
        self.assertEqual(
            set(issues.get_column(pc_cols.PORTFOLIO_ID).to_list()),
            {"DATAAUDIT"},
        )
        self.assertEqual(
            set(issues.get_column(pc_cols.SECURITY_ID).to_list()),
            {"csusKO"},
        )
        self.assertEqual(
            set(issues.get_column(data_issues.CATEGORY).to_list()),
            {"classification"},
        )
        self.assertTrue(
            all(
                "differs only by case" in explanation
                for explanation in issues.get_column(data_issues.EXPLANATION)
            )
        )

    def test_data_issues_detect_rate_and_missing_dividend_issues(self) -> None:
        """Dividend and pa/sa checks compare same-day same-security rates."""
        with tempfile.TemporaryDirectory() as directory:
            comparison_path = _write_site(
                Path(directory),
                holdings_rows=[
                    "P1,ABC,2026-01-31,100,10,1000,0",
                    "P1,ABC,2026-02-28,100,11,1100,0",
                    "P2,ABC,2026-01-31,100,10,1000,0",
                    "P2,ABC,2026-02-28,100,11,1100,0",
                    "P2,XYZ,2026-01-31,100,20,2000,0",
                    "P2,XYZ,2026-02-28,100,21,2100,0",
                    "P3,XYZ,2026-01-31,0,20,0,0",
                    "P3,XYZ,2026-02-28,50,21,1050,0",
                    "P4,XYZ,2026-01-31,100,20,2000,0",
                    "P4,XYZ,2026-02-28,90,21,1890,0",
                    "P1,BOND,2026-02-28,10000,99,9900,50",
                    "P2,BOND,2026-02-28,10000,99,9900,60",
                ],
                transaction_rows=[
                    "P1,2026-02-15,,ABC,dv,stock,,,,,0,0,50,0",
                    "P2,2026-02-15,,ABC,dv,stock,,,,,0,0,60,0",
                    "P1,2026-02-16,,BOND,pa,fius,,,,,0,0,40,0",
                    "P2,2026-02-16,,BOND,pa,fius,,,,,0,0,45,0",
                    "P1,2026-02-20,,XYZ,dv,stock,,,,,100,0,25,0",
                    "P3,2026-02-10,,XYZ,by,stock,,,,,50,20,-1000,0",
                    "P4,2026-02-10,,XYZ,sl,stock,,,,,10,20,200,0",
                    "P5,2026-02-20,,XYZ,dv,stock,,,,,100,0,25,0",
                ],
                transaction_rules="""
                transaction_rules:
                  pa:
                    - when:
                        security_type: fius
                      transaction_category: fee_expense
                      cash_flow_sign: negative
                      performance_flow_sign: performance
                """,
            )

            issues = data_issues.data_issues_table(comparison_path)
            issue_types = set(issues.get_column(data_issues.ISSUE_TYPE).to_list())
            missing_dividends = issues.filter(
                issues[data_issues.ISSUE_TYPE] == "missing_dividend"
            )
            missing_portfolios = set(
                missing_dividends.get_column("portfolio_id").to_list()
            )
            missing_explanations = set(
                missing_dividends.get_column(data_issues.EXPLANATION).to_list()
            )

        self.assertIn("dividend_rate", issue_types)
        self.assertIn("missing_dividend", issue_types)
        self.assertIn("pa_sa_rate", issue_types)
        self.assertIn("holdings_accrued_rate", issue_types)
        self.assertEqual(missing_dividends.height, 4)
        self.assertEqual(missing_portfolios, {"P2", "P3"})
        self.assertEqual(
            missing_explanations,
            {
                "Missing a dividend for XYZ on 2026-02-20 that is in "
                "portfolio P1 and other portfolios."
            },
        )

    def test_data_issues_detect_holdings_and_transaction_price_ranges(self) -> None:
        """Price-range checks compare same-day same-security prices."""
        with tempfile.TemporaryDirectory() as directory:
            comparison_path = _write_site(
                Path(directory),
                holdings_rows=[
                    "P1,ABC,2026-02-28,100,10.00,1000,0",
                    "P2,ABC,2026-02-28,100,10.25,1025,0",
                ],
                transaction_rows=[
                    "P1,2026-02-15,,ABC,by,stock,$cash,CASHUSD,,,10,10.00,-100,0",
                    "P2,2026-02-15,,ABC,by,stock,$cash,CASHUSD,,,10,10.50,-105,0",
                ],
                data_issues_config="""
                data_issues:
                  holdings_price_range:
                    percent_tolerance: 1.0
                  transactions_price_range:
                    percent_tolerance: 1.0
                """,
            )

            issues = data_issues.data_issues_table(comparison_path)
            issue_types = set(issues.get_column(data_issues.ISSUE_TYPE).to_list())

        self.assertIn("holdings_price_range", issue_types)
        self.assertIn("transactions_price_range", issue_types)

    def test_holdings_nonpositive_price_is_scoped_and_independent(self) -> None:
        """The opt-in check respects its population without changing findings."""
        with tempfile.TemporaryDirectory() as directory:
            comparison_path = _write_site(
                Path(directory),
                holdings_rows=[
                    "P1,ZERO,2026-02-28,100,0,1000,0",
                    "P2,NEGATIVE,2026-02-28,-50,-2,-100,0",
                    "P3,ZERO_QUANTITY,2026-02-28,0,0,0,0",
                    "P4,EXCLUDED,2026-02-28,100,0,1000,0",
                ],
                transaction_rows=[],
                data_issues_config="""
                data_issues:
                  holdings_nonpositive_price:
                    enabled: true
                    only:
                      security_id:
                        - ZERO
                        - NEGATIVE
                        - ZERO_QUANTITY
                        - EXCLUDED
                    exclude:
                      portfolio_id: P4
                """,
            )

            issues = data_issues.data_issues_table(comparison_path).filter(
                pl.col(data_issues.ISSUE_TYPE)
                == data_issues.ISSUE_HOLDINGS_NONPOSITIVE_PRICE
            )
            performance_findings = compare_snapshots(comparison_path)

        self.assertEqual(issues.height, 4)
        self.assertEqual(
            set(issues.get_column(pc_cols.SECURITY_ID).to_list()),
            {"ZERO", "NEGATIVE"},
        )
        self.assertEqual(
            set(issues.get_column(data_issues.VALUE_B).to_list()),
            {0.0, -2.0},
        )
        self.assertTrue(issues.get_column(data_issues.VALUE_A).is_null().all())
        self.assertTrue(issues.get_column(data_issues.DIFFERENCE).is_null().all())
        self.assertEqual(
            set(issues.get_column(data_issues.TOLERANCE).to_list()),
            {"price must be greater than 0"},
        )
        self.assertEqual(
            set(issues.get_column(data_issues.CATEGORY).to_list()),
            {"price"},
        )
        self.assertEqual(issues.get_column(data_issues.REVIEW_KEY).n_unique(), 4)
        self.assertTrue(performance_findings.is_empty())

    def test_holdings_nonpositive_price_is_off_by_default(self) -> None:
        """Existing configurations do not run the conservative new check."""
        with tempfile.TemporaryDirectory() as directory:
            comparison_path = _write_site(
                Path(directory),
                holdings_rows=["P1,ZERO,2026-02-28,100,0,1000,0"],
                transaction_rows=[],
            )

            issue_types = set(
                data_issues.data_issues_table(comparison_path)
                .get_column(data_issues.ISSUE_TYPE)
                .to_list()
            )

        self.assertNotIn(data_issues.ISSUE_HOLDINGS_NONPOSITIVE_PRICE, issue_types)

    def test_holdings_stale_price_tracks_observed_unchanged_run(self) -> None:
        """Stale-price review uses supplied dates and resets when price changes."""
        with tempfile.TemporaryDirectory() as directory:
            comparison_path = _write_site(
                Path(directory),
                holdings_rows=[
                    "P1,ABC,2026-01-01,100,10,1000,0",
                    "P1,ABC,2026-01-15,100,10,1000,0",
                    "P1,ABC,2026-01-29,100,10,1000,0",
                    "P1,ABC,2026-02-01,100,11,1100,0",
                    "P1,ABC,2026-03-01,100,11,1100,0",
                    "P2,ABC,2026-01-01,0,10,0,0",
                    "P2,ABC,2026-03-01,0,10,0,0",
                ],
                transaction_rows=[],
                security_reference_rows=["ABC,csus,EQ"],
                data_issues_config="""
                data_issues:
                  holdings_stale_price:
                    enabled: true
                    only:
                      security_reference.security_type: csus
                    minimum_calendar_days: 28
                """,
            )

            issues = data_issues.data_issues_table(comparison_path).filter(
                pl.col(data_issues.ISSUE_TYPE)
                == data_issues.ISSUE_HOLDINGS_STALE_PRICE
            )
            performance_findings = compare_snapshots(comparison_path)

        self.assertEqual(issues.height, 4)
        self.assertEqual(
            set(issues.get_column(data_issues.AS_OF_DATE).to_list()),
            {dt.date(2026, 1, 29), dt.date(2026, 3, 1)},
        )
        self.assertEqual(
            set(issues.get_column(pc_cols.PORTFOLIO_ID).to_list()),
            {"P1"},
        )
        self.assertEqual(
            set(issues.get_column(data_issues.VALUE_A).to_list()),
            {10.0, 11.0},
        )
        self.assertTrue(
            all(
                "did not observe every intervening day" in explanation
                for explanation in issues.get_column(data_issues.EXPLANATION)
            )
        )
        self.assertEqual(
            set(issues.get_column(data_issues.CATEGORY).to_list()),
            {"price"},
        )
        self.assertTrue(performance_findings.is_empty())

    def test_holdings_stale_price_is_off_by_default(self) -> None:
        """Existing configurations do not activate observed-price enrichment."""
        with tempfile.TemporaryDirectory() as directory:
            comparison_path = _write_site(
                Path(directory),
                holdings_rows=[
                    "P1,ABC,2026-01-01,100,10,1000,0",
                    "P1,ABC,2026-02-01,100,10,1000,0",
                ],
                transaction_rows=[],
            )

            issue_types = set(
                data_issues.data_issues_table(comparison_path)
                .get_column(data_issues.ISSUE_TYPE)
                .to_list()
            )

        self.assertNotIn(data_issues.ISSUE_HOLDINGS_STALE_PRICE, issue_types)

    def test_large_price_variation_uses_named_rules_and_inclusive_periods(
        self,
    ) -> None:
        """Named rules retain holdings while filtering inclusive trade prices."""
        with tempfile.TemporaryDirectory() as directory:
            comparison_path = _write_site(
                Path(directory),
                portfolio_performance_rows=[
                    "P1,2026-01-01,2026-01-31,0.01",
                    "P1,2026-02-01,2026-02-01,0.01",
                ],
                holdings_rows=[
                    "P1,ABC,2026-01-31,100,100,10000,0",
                    "P1,ABC,2026-02-01,100,115,11500,0",
                ],
                transaction_rows=[
                    "P1,2026-02-01,2026-02-01,ABC,by,csus,,,,,10,130,1300,0",
                    "P1,2026-02-01,2026-02-01,ABC,dv,csus,,,,,0,500,100,0",
                ],
                security_reference_rows=["ABC,csus,EQ"],
                data_issues_config="""
                data_issues:
                  large_price_variation:
                    enabled: true
                    rules:
                      - rule_id: common_stock_default
                        minimum_calendar_days: 1
                        minimum_tolerance: 0.20
                        only:
                          transactions.transaction_code: [by, sl]
                          security_reference.security_type: csus
                      - rule_id: common_stock_25_percent
                        only:
                          transactions.transaction_code: [by, sl]
                          security_reference.security_type: csus
                        minimum_calendar_days: 1
                        minimum_tolerance: 0.25
                      - rule_id: requires_two_days
                        minimum_calendar_days: 2
                        minimum_tolerance: 0.20
                """,
            )

            issues = data_issues.data_issues_table(comparison_path).filter(
                pl.col(data_issues.ISSUE_TYPE)
                == data_issues.ISSUE_LARGE_PRICE_VARIATION
            )
            performance_findings = compare_snapshots(comparison_path)

        self.assertEqual(issues.height, 4)
        self.assertEqual(
            set(issues.get_column(data_issues.SNAPSHOT).to_list()),
            {"Snapshot A", "Snapshot B"},
        )
        self.assertEqual(set(issues.get_column(data_issues.VALUE_A)), {100.0})
        self.assertEqual(set(issues.get_column(data_issues.VALUE_B)), {130.0})
        self.assertEqual(set(issues.get_column(data_issues.DIFFERENCE)), {30.0})
        self.assertEqual(issues.get_column(data_issues.REVIEW_KEY).n_unique(), 4)
        self.assertTrue(
            all(
                "1 calendar days" in explanation
                and "beginning-period holdings.price" in explanation
                and "transactions.price" in explanation
                for explanation in issues.get_column(data_issues.EXPLANATION)
            )
        )
        self.assertFalse(
            any(
                "500" in explanation
                for explanation in issues.get_column(data_issues.EXPLANATION)
            )
        )
        self.assertTrue(performance_findings.is_empty())

    def test_large_price_variation_split_normalizes_to_period_end_basis(self) -> None:
        """A supplied split removes its mechanical raw-price discontinuity."""
        config = """
        data_issues:
          large_price_variation:
            enabled: true
            rules:
              - rule_id: common_stock_default
                minimum_calendar_days: 1
                minimum_tolerance: 0.20
        """
        periods = [
            "P1,2026-01-01,2026-01-31,0.01",
            "P1,2026-02-01,2026-02-01,0.01",
        ]
        holdings = [
            "P1,ABC,2026-01-31,100,100,10000,0",
            "P1,ABC,2026-02-01,200,50,10000,0",
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            normalized_path = _write_site(
                root / "normalized",
                portfolio_performance_rows=periods,
                holdings_rows=holdings,
                transaction_rows=[],
                split_rows=["ABC,2026-02-01,2"],
                data_issues_config=config,
            )
            missing_split_path = _write_site(
                root / "missing_split",
                portfolio_performance_rows=periods,
                holdings_rows=holdings,
                transaction_rows=[],
                data_issues_config=config,
            )

            normalized_issues = data_issues.data_issues_table(
                normalized_path
            ).filter(
                pl.col(data_issues.ISSUE_TYPE)
                == data_issues.ISSUE_LARGE_PRICE_VARIATION
            )
            missing_split_issues = data_issues.data_issues_table(
                missing_split_path
            ).filter(
                pl.col(data_issues.ISSUE_TYPE)
                == data_issues.ISSUE_LARGE_PRICE_VARIATION
            )

        self.assertTrue(normalized_issues.is_empty())
        self.assertEqual(missing_split_issues.height, 2)
        self.assertEqual(
            set(missing_split_issues.get_column(data_issues.VALUE_A)),
            {50.0},
        )
        self.assertEqual(
            set(missing_split_issues.get_column(data_issues.VALUE_B)),
            {100.0},
        )
        self.assertTrue(
            all(
                "split evidence is missing" in explanation
                for explanation in missing_split_issues.get_column(
                    data_issues.EXPLANATION
                )
            )
        )

    def test_large_price_variation_allows_missing_boundary_and_exceeds_strictly(
        self,
    ) -> None:
        """Transactions can replace a boundary, and equality does not report."""
        with tempfile.TemporaryDirectory() as directory:
            comparison_path = _write_site(
                Path(directory),
                portfolio_performance_rows=[
                    "P1,2026-01-01,2026-01-31,0.01",
                    "P1,2026-02-01,2026-02-01,0.01",
                ],
                holdings_rows=[
                    "P1,ABC,2026-02-01,100,120,12000,0",
                ],
                transaction_rows=[
                    "P1,2026-02-01,2026-02-01,ABC,by,csus,,,,,10,100,1000,0",
                ],
                data_issues_config="""
                data_issues:
                  large_price_variation:
                    enabled: true
                    rules:
                      - rule_id: exact_20_percent
                        minimum_calendar_days: 1
                        minimum_tolerance: 0.20
                      - rule_id: below_observed_variation
                        minimum_calendar_days: 1
                        minimum_tolerance: 0.199
                """,
            )

            issues = data_issues.data_issues_table(comparison_path).filter(
                pl.col(data_issues.ISSUE_TYPE)
                == data_issues.ISSUE_LARGE_PRICE_VARIATION
            )

        self.assertEqual(issues.height, 2)
        self.assertTrue(
            all(
                "below_observed_variation" in tolerance
                for tolerance in issues.get_column(data_issues.TOLERANCE)
            )
        )
        self.assertEqual(set(issues.get_column(data_issues.VALUE_A)), {100.0})
        self.assertEqual(set(issues.get_column(data_issues.VALUE_B)), {120.0})
        self.assertEqual(
            set(issues.get_column(data_issues.DATASET_FIELD)),
            {"holdings.price + transactions.price"},
        )

    def test_large_price_variation_is_off_by_default(self) -> None:
        """Existing configurations do not activate period price variation."""
        with tempfile.TemporaryDirectory() as directory:
            comparison_path = _write_site(
                Path(directory),
                portfolio_performance_rows=[
                    "P1,2026-01-01,2026-01-31,0.01",
                    "P1,2026-02-01,2026-02-01,0.01",
                ],
                holdings_rows=[
                    "P1,ABC,2026-01-31,100,100,10000,0",
                    "P1,ABC,2026-02-01,100,130,13000,0",
                ],
                transaction_rows=[],
            )

            issue_types = set(
                data_issues.data_issues_table(comparison_path)
                .get_column(data_issues.ISSUE_TYPE)
                .to_list()
            )

        self.assertNotIn(data_issues.ISSUE_LARGE_PRICE_VARIATION, issue_types)

    def test_transactions_nonpositive_price_is_scoped_and_independent(self) -> None:
        """The opt-in trade check requires quantity, code, and reference scope."""
        with tempfile.TemporaryDirectory() as directory:
            comparison_path = _write_site(
                Path(directory),
                holdings_rows=[],
                transaction_rows=[
                    "P1,2026-02-15,2026-02-15,ABC,by,csus,$cash,CASHUSD,,,"
                    "10,0,-100,0",
                    "P2,2026-02-16,2026-02-16,ABC,sl,csus,$cash,CASHUSD,,,"
                    "-5,-2,10,0",
                    "P3,2026-02-17,2026-02-17,ABC,dv,csus,$income,$cash,,,"
                    "10,0,20,0",
                    "P4,2026-02-18,2026-02-18,ABC,by,csus,$cash,CASHUSD,,,"
                    "0,0,0,0",
                    "P5,2026-02-19,2026-02-19,CASH,by,caus,$cash,CASHUSD,,,"
                    "10,0,0,0",
                ],
                security_reference_rows=["ABC,csus,EQ", "CASH,caus,CASH"],
                data_issues_config="""
                data_issues:
                  transactions_nonpositive_price:
                    enabled: true
                    only:
                      transactions.transaction_code:
                        - by
                        - sl
                      security_reference.security_type: csus
                """,
            )

            issues = data_issues.data_issues_table(comparison_path).filter(
                pl.col(data_issues.ISSUE_TYPE)
                == data_issues.ISSUE_TRANSACTIONS_NONPOSITIVE_PRICE
            )
            performance_findings = compare_snapshots(comparison_path)

        self.assertEqual(issues.height, 4)
        self.assertEqual(
            set(issues.get_column(pc_cols.PORTFOLIO_ID).to_list()),
            {"P1", "P2"},
        )
        self.assertEqual(
            set(issues.get_column(data_issues.VALUE_B).to_list()),
            {0.0, -2.0},
        )
        self.assertTrue(issues.get_column(data_issues.VALUE_A).is_null().all())
        self.assertTrue(issues.get_column(data_issues.DIFFERENCE).is_null().all())
        self.assertEqual(
            set(issues.get_column(data_issues.CATEGORY).to_list()),
            {"price"},
        )
        self.assertEqual(issues.get_column(data_issues.REVIEW_KEY).n_unique(), 4)
        self.assertTrue(performance_findings.is_empty())

    def test_deliver_in_original_cost_incomplete_is_scoped_and_independent(
        self,
    ) -> None:
        """The opt-in check reports one row per incomplete configured deliver-in."""
        with tempfile.TemporaryDirectory() as directory:
            comparison_path = _write_site(
                Path(directory),
                holdings_rows=[],
                transaction_header=(
                    "PORT,TRANSACTION_DATE,SETTLE_DATE,SEC,TRAN,SEC_TYPE,"
                    "SRC_DEST_TYPE,SRC_DEST_SYMBOL,SPECIAL_SEC_TYPE,"
                    "SPECIAL_SEC_SYMBOL,QTY,PRICE,AMOUNT,COMMISSION,"
                    "ORIGINAL_COST_DATE,ORIGINAL_COST"
                ),
                transaction_rows=[
                    "P1,2026-02-15,2026-02-15,ABC,ti,csus,$pty,external_delivery,,"
                    ",5,10,50,0,,",
                    "P2,2026-02-16,2026-02-16,ABC,ti,csus,$pty,external_delivery,,"
                    ",5,10,50,0,2020-01-01,",
                    "P3,2026-02-17,2026-02-17,ABC,ti,csus,$pty,external_delivery,,"
                    ",5,10,50,0,,100",
                    "P4,2026-02-18,2026-02-18,ABC,ti,csus,$pty,external_delivery,,"
                    ",5,10,50,0,2020-01-01,0",
                    "P5,2026-02-19,2026-02-19,ABC,by,csus,$pty,external_delivery,,"
                    ",5,10,-50,0,,",
                ],
                transaction_rules="""
                transaction_rules:
                  ti:
                    when:
                      security_type: csus
                      source_destination_type: $pty
                      source_destination_symbol: external_delivery
                    transaction_category: external_flow
                    cash_flow_sign: positive
                    performance_flow_sign: external
                """,
                data_issues_config="""
                data_issues:
                  deliver_in_original_cost_incomplete:
                    enabled: true
                    only:
                      transaction_code: ti
                      security_type: csus
                      source_destination_type: $pty
                      source_destination_symbol: external_delivery
                """,
            )

            issues = data_issues.data_issues_table(comparison_path).filter(
                pl.col(data_issues.ISSUE_TYPE)
                == data_issues.ISSUE_DELIVER_IN_ORIGINAL_COST_INCOMPLETE
            )
            performance_findings = compare_snapshots(comparison_path)

        self.assertEqual(issues.height, 6)
        self.assertEqual(
            set(issues.get_column(pc_cols.PORTFOLIO_ID).to_list()),
            {"P1", "P2", "P3"},
        )
        self.assertEqual(
            set(issues.get_column(data_issues.DATASET_FIELD).to_list()),
            {
                "transactions.original_cost",
                "transactions.original_cost_date",
                "transactions.original_cost + transactions.original_cost_date",
            },
        )
        self.assertEqual(
            set(issues.get_column(data_issues.CATEGORY).to_list()),
            {"position_value"},
        )
        self.assertTrue(issues.get_column(data_issues.VALUE_A).is_null().all())
        self.assertTrue(issues.get_column(data_issues.VALUE_B).is_null().all())
        self.assertTrue(issues.get_column(data_issues.DIFFERENCE).is_null().all())
        self.assertEqual(issues.get_column(data_issues.REVIEW_KEY).n_unique(), 6)
        self.assertTrue(
            all(
                "may fall back to trade-date market value" in explanation
                for explanation in issues.get_column(data_issues.EXPLANATION)
            )
        )
        self.assertTrue(performance_findings.is_empty())

    def test_deliver_in_original_cost_incomplete_is_off_by_default(self) -> None:
        """Existing configurations do not require original-cost columns."""
        with tempfile.TemporaryDirectory() as directory:
            comparison_path = _write_site(
                Path(directory),
                holdings_rows=[],
                transaction_rows=[
                    "P1,2026-02-15,2026-02-15,ABC,by,csus,$cash,CASHUSD,,,"
                    "5,10,-50,0"
                ],
            )

            issue_types = set(
                data_issues.data_issues_table(comparison_path)
                .get_column(data_issues.ISSUE_TYPE)
                .to_list()
            )

        self.assertNotIn(
            data_issues.ISSUE_DELIVER_IN_ORIGINAL_COST_INCOMPLETE,
            issue_types,
        )

    def test_deliver_in_original_cost_requires_source_columns(self) -> None:
        """Enabled completeness review fails when source columns are absent."""
        with tempfile.TemporaryDirectory() as directory:
            comparison_path = _write_site(
                Path(directory),
                holdings_rows=[],
                transaction_rows=[
                    "P1,2026-02-15,2026-02-15,ABC,ti,csus,$pty,"
                    "external_delivery,,,5,10,50,0"
                ],
                transaction_rules="""
                transaction_rules:
                  ti:
                    when:
                      security_type: csus
                      source_destination_type: $pty
                      source_destination_symbol: external_delivery
                    transaction_category: external_flow
                    cash_flow_sign: positive
                    performance_flow_sign: external
                """,
                data_issues_config="""
                data_issues:
                  deliver_in_original_cost_incomplete:
                    enabled: true
                    only:
                      transaction_code: ti
                      security_type: csus
                      source_destination_type: $pty
                      source_destination_symbol: external_delivery
                """,
            )

            with self.assertRaisesRegex(
                PpaError,
                "original_cost, original_cost_date",
            ):
                validate_config(comparison_path, require_complete_yaml_setup=False)

            with self.assertRaisesRegex(
                PpaError,
                "original_cost, original_cost_date",
            ):
                data_issues.data_issues_table(comparison_path)

    def test_deliver_in_original_cost_filter_honors_exact_case(self) -> None:
        """Exact source contracts do not fold deliver-in filter context case."""
        config = {
            data_issues.ISSUE_DELIVER_IN_ORIGINAL_COST_INCOMPLETE: {
                "enabled": True,
                "only": {
                    "transaction_code": "ti",
                    "security_type": "csus",
                    "source_destination_type": "$pty",
                    "source_destination_symbol": "external_delivery",
                },
            }
        }
        base_row = {
            data_issues.SNAPSHOT: "Snapshot A",
            pc_cols.PORTFOLIO_ID: "P1",
            pc_cols.TRANSACTION_DATE: dt.date(2026, 2, 15),
            pc_cols.SECURITY_ID: "ABC",
            pc_cols.TRANSACTION_CODE: "ti",
            pc_cols.SECURITY_TYPE: "csus",
            pc_cols.SOURCE_DESTINATION_TYPE: "$pty",
            pc_cols.SOURCE_DESTINATION_SYMBOL: "external_delivery",
            pc_cols.ORIGINAL_COST: None,
            pc_cols.ORIGINAL_COST_DATE: None,
        }
        rows = [
            base_row,
            {**base_row, pc_cols.TRANSACTION_CODE: "TI"},
            {**base_row, pc_cols.SOURCE_DESTINATION_SYMBOL: "EXTERNAL_DELIVERY"},
        ]

        issues = data_issues._deliver_in_original_cost_incomplete_issues(
            rows,
            config,
            exact_case=True,
        )

        self.assertEqual(len(issues), 1)

    def test_deliver_in_review_identity_is_invariant_across_scale_copies(
        self,
    ) -> None:
        """Synthetic portfolio copies retain the same normalized review key."""
        config = {
            data_issues.ISSUE_DELIVER_IN_ORIGINAL_COST_INCOMPLETE: {
                "enabled": True,
                "only": {
                    "transaction_code": "ti",
                    "security_type": "csus",
                    "source_destination_type": "$pty",
                    "source_destination_symbol": "external_delivery",
                },
            }
        }
        base_row = {
            data_issues.SNAPSHOT: "Snapshot A",
            pc_cols.PORTFOLIO_ID: "P1",
            pc_cols.TRANSACTION_DATE: dt.date(2026, 2, 15),
            pc_cols.SECURITY_ID: "ABC",
            pc_cols.TRANSACTION_CODE: "ti",
            pc_cols.SECURITY_TYPE: "csus",
            pc_cols.SOURCE_DESTINATION_TYPE: "$pty",
            pc_cols.SOURCE_DESTINATION_SYMBOL: "external_delivery",
            pc_cols.ORIGINAL_COST: None,
            pc_cols.ORIGINAL_COST_DATE: None,
        }
        rows = [
            base_row,
            {**base_row, pc_cols.PORTFOLIO_ID: "P1_SCALE_001"},
        ]

        issues = data_issues._deliver_in_original_cost_incomplete_issues(
            rows,
            config,
            exact_case=True,
        )
        normalized_review_keys = {
            str(issue[data_issues.REVIEW_KEY]).replace("_SCALE_001", "")
            for issue in issues
        }

        self.assertEqual(len(issues), 2)
        self.assertEqual(len(normalized_review_keys), 1)

    def test_transactions_nonpositive_price_is_off_by_default(self) -> None:
        """Existing configurations do not require reference data for the new check."""
        with tempfile.TemporaryDirectory() as directory:
            comparison_path = _write_site(
                Path(directory),
                holdings_rows=[],
                transaction_rows=[
                    "P1,2026-02-15,,ABC,by,csus,$cash,CASHUSD,,,10,0,-100,0"
                ],
            )

            issue_types = set(
                data_issues.data_issues_table(comparison_path)
                .get_column(data_issues.ISSUE_TYPE)
                .to_list()
            )

        self.assertNotIn(
            data_issues.ISSUE_TRANSACTIONS_NONPOSITIVE_PRICE,
            issue_types,
        )

    def test_transaction_security_type_mismatch_is_exact_case_and_neutral(
        self,
    ) -> None:
        """Classification mismatches report both values without choosing one."""
        with tempfile.TemporaryDirectory() as directory:
            comparison_path = _write_site(
                Path(directory),
                holdings_rows=[],
                transaction_rows=[
                    "P1,2026-02-15,2026-02-15,ABC,by,csus,$cash,CASHUSD,,,"
                    "10,10,-100,0",
                    "P2,2026-02-16,2026-02-16,ABC,by,CSUS,$cash,CASHUSD,,,"
                    "10,10,-100,0",
                    "P3,2026-02-17,2026-02-17,ABC,by,fius,$cash,CASHUSD,,,"
                    "10,10,-100,0",
                    "P4,2026-02-18,2026-02-18,ABC,by,,$cash,CASHUSD,,,"
                    "10,10,-100,0",
                ],
                security_reference_rows=["ABC,csus,EQ"],
                data_issues_config="""
                data_issues:
                  transaction_security_type_mismatch:
                    enabled: true
                    only:
                      security_reference.security_type: csus
                """,
            )

            issues = data_issues.data_issues_table(comparison_path).filter(
                pl.col(data_issues.ISSUE_TYPE)
                == data_issues.ISSUE_TRANSACTION_SECURITY_TYPE_MISMATCH
            )
            performance_findings = compare_snapshots(comparison_path)

        self.assertEqual(issues.height, 6)
        self.assertEqual(
            set(issues.get_column(pc_cols.PORTFOLIO_ID).to_list()),
            {"P2", "P3", "P4"},
        )
        explanations = issues.get_column(data_issues.EXPLANATION).to_list()
        self.assertTrue(any("differs only by case" in text for text in explanations))
        self.assertTrue(
            any("'fius'" in text and "'csus'" in text for text in explanations)
        )
        self.assertTrue(any("blank" in text for text in explanations))
        self.assertTrue(all("does not choose" in text for text in explanations))
        self.assertEqual(
            set(issues.get_column(data_issues.CATEGORY).to_list()),
            {"classification"},
        )
        self.assertTrue(issues.get_column(data_issues.VALUE_A).is_null().all())
        self.assertTrue(issues.get_column(data_issues.VALUE_B).is_null().all())
        self.assertTrue(issues.get_column(data_issues.DIFFERENCE).is_null().all())
        self.assertTrue(performance_findings.is_empty())

    def test_transaction_security_type_mismatch_is_off_by_default(self) -> None:
        """Existing configurations do not require a reference dataset for it."""
        with tempfile.TemporaryDirectory() as directory:
            comparison_path = _write_site(
                Path(directory),
                holdings_rows=[],
                transaction_rows=[
                    "P1,2026-02-15,2026-02-15,ABC,by,CSUS,$cash,CASHUSD,,,"
                    "10,10,-100,0"
                ],
            )

            issue_types = set(
                data_issues.data_issues_table(comparison_path)
                .get_column(data_issues.ISSUE_TYPE)
                .to_list()
            )

        self.assertNotIn(
            data_issues.ISSUE_TRANSACTION_SECURITY_TYPE_MISMATCH,
            issue_types,
        )

    def test_transaction_security_type_mismatch_rejects_ambiguous_reference(
        self,
    ) -> None:
        """The comparison fails closed when a reference ID is duplicated."""
        with tempfile.TemporaryDirectory() as directory:
            comparison_path = _write_site(
                Path(directory),
                holdings_rows=[],
                transaction_rows=[
                    "P1,2026-02-15,2026-02-15,ABC,by,CSUS,$cash,CASHUSD,,,"
                    "10,10,-100,0"
                ],
                security_reference_rows=["ABC,csus,EQ", "ABC,fius,FI"],
                data_issues_config="""
                data_issues:
                  transaction_security_type_mismatch:
                    enabled: true
                    only:
                      security_reference.security_type: csus
                """,
            )

            with self.assertRaisesRegex(PpaError, "one exact-case row"):
                data_issues.data_issues_table(comparison_path)

    def test_data_issues_detect_duplicate_transactions(self) -> None:
        """Duplicate-transaction checks flag exact repeated activity rows."""
        with tempfile.TemporaryDirectory() as directory:
            comparison_path = _write_site(
                Path(directory),
                holdings_rows=[],
                transaction_rows=[
                    "P1,2026-02-15,,ABC,by,stock,$cash,$cash,,,0,0,0,0",
                    "P1,2026-02-15,,ABC,by,stock,$cash,$cash,,,0,0,0,0",
                ],
            )

            issues = data_issues.data_issues_table(comparison_path)
            duplicate_issues = issues.filter(
                issues[data_issues.ISSUE_TYPE] == "duplicate_transactions"
            )

        self.assertEqual(duplicate_issues.height, 4)
        self.assertEqual(
            set(duplicate_issues.get_column(data_issues.EXPLANATION).to_list()),
            {
                "Duplicate transaction rows have the same portfolio, date, "
                "security, code, amount, quantity, and price."
            },
        )

    def test_data_issues_issue_type_can_be_disabled(self) -> None:
        """Each issue type is on by default but can be opted out in YAML."""
        with tempfile.TemporaryDirectory() as directory:
            comparison_path = _write_site(
                Path(directory),
                holdings_rows=[
                    "P1,ABC,2026-01-31,100,10,1000,0",
                    "P1,ABC,2026-02-28,100,11,1100,0",
                    "P2,ABC,2026-01-31,100,10,1000,0",
                    "P2,ABC,2026-02-28,100,11,1100,0",
                    "P1,BOND,2026-02-28,10000,99,9900,50",
                    "P2,BOND,2026-02-28,10000,99,9900,60",
                ],
                transaction_rows=[
                    "P1,2026-02-15,,ABC,dv,stock,,,,,0,0,50,0",
                    "P2,2026-02-15,,ABC,dv,stock,,,,,0,0,60,0",
                    "P1,2026-02-16,,BOND,pa,fius,,,,,0,0,40,0",
                    "P2,2026-02-16,,BOND,pa,fius,,,,,0,0,45,0",
                ],
                data_issues_config="""
                data_issues:
                  dividend_rate:
                    enabled: false
                """,
                transaction_rules="""
                transaction_rules:
                  pa:
                    - when:
                        security_type: fius
                      transaction_category: fee_expense
                      cash_flow_sign: negative
                      performance_flow_sign: performance
                """,
            )

            issues = data_issues.data_issues_table(comparison_path)
            issue_types = set(issues.get_column(data_issues.ISSUE_TYPE).to_list())

        self.assertNotIn("dividend_rate", issue_types)
        self.assertIn("pa_sa_rate", issue_types)

    def test_data_issues_issue_filters_support_only_and_exclude(self) -> None:
        """Issue filters support exact-match dataset.field and common field names."""
        with tempfile.TemporaryDirectory() as directory:
            comparison_path = _write_site(
                Path(directory),
                holdings_rows=[
                    "P1,ABC,2026-01-31,100,10,1000,0",
                    "P1,ABC,2026-02-28,100,11,1100,0",
                    "P2,ABC,2026-01-31,100,10,1000,0",
                    "P2,ABC,2026-02-28,100,11,1100,0",
                ],
                transaction_rows=[
                    "P1,2026-02-15,,ABC,dv,stock,,,,,0,0,50,0",
                    "P2,2026-02-15,,ABC,dv,stock,,,,,0,0,60,0",
                ],
                data_issues_config="""
                data_issues:
                  dividend_rate:
                    only:
                      transactions.security_type: stock
                      security_id: ABC
                    exclude:
                      portfolio_id: P2
                """,
            )

            issues = data_issues.data_issues_table(comparison_path)
            dividend_issues = issues.filter(
                issues[data_issues.ISSUE_TYPE] == "dividend_rate"
            )

        self.assertEqual(dividend_issues.height, 0)

    def test_security_reference_filter_qualifies_rows_with_exact_case(self) -> None:
        """Reference qualifiers join by security ID and preserve source case."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matching_path = _write_site(
                root / "matching",
                holdings_rows=[
                    "P1,ABC,2026-02-28,100,10.00,1000,0",
                    "P2,ABC,2026-02-28,100,10.25,1025,0",
                ],
                transaction_rows=[],
                security_reference_rows=["ABC,csus,EQ"],
                data_issues_config="""
                data_issues:
                  holdings_price_range:
                    only:
                      security_reference.asset_class_code: EQ
                """,
            )
            wrong_case_path = _write_site(
                root / "wrong_case",
                holdings_rows=[
                    "P1,ABC,2026-02-28,100,10.00,1000,0",
                    "P2,ABC,2026-02-28,100,10.25,1025,0",
                ],
                transaction_rows=[],
                security_reference_rows=["ABC,csus,EQ"],
                data_issues_config="""
                data_issues:
                  holdings_price_range:
                    only:
                      security_reference.asset_class_code: eq
                """,
            )

            matching_issues = data_issues.data_issues_table(matching_path).filter(
                pl.col(data_issues.ISSUE_TYPE) == data_issues.ISSUE_HOLDINGS_PRICE_RANGE
            )
            wrong_case_issues = data_issues.data_issues_table(wrong_case_path).filter(
                pl.col(data_issues.ISSUE_TYPE) == data_issues.ISSUE_HOLDINGS_PRICE_RANGE
            )

        self.assertEqual(matching_issues.height, 4)
        self.assertTrue(wrong_case_issues.is_empty())

    def test_security_reference_filter_requires_dataset(self) -> None:
        """A reference-dependent filter fails closed without the dataset."""
        with tempfile.TemporaryDirectory() as directory:
            comparison_path = _write_site(
                Path(directory),
                holdings_rows=["P1,ABC,2026-02-28,100,10,1000,0"],
                transaction_rows=[],
                data_issues_config="""
                data_issues:
                  holdings_price_range:
                    only:
                      security_reference.asset_class_code: EQ
                """,
            )

            with self.assertRaisesRegex(PpaError, "files.security_reference"):
                data_issues.data_issues_table(comparison_path)

    def test_disabled_security_reference_filter_does_not_require_dataset(self) -> None:
        """A retained filter under a disabled check does not activate enrichment."""
        with tempfile.TemporaryDirectory() as directory:
            comparison_path = _write_site(
                Path(directory),
                holdings_rows=["P1,ABC,2026-02-28,100,10,1000,0"],
                transaction_rows=[],
                data_issues_config="""
                data_issues:
                  holdings_price_range:
                    enabled: false
                    only:
                      security_reference.asset_class_code: EQ
                """,
            )

            issues = data_issues.data_issues_table(comparison_path)

        self.assertNotIn(
            data_issues.ISSUE_HOLDINGS_PRICE_RANGE,
            set(issues.get_column(data_issues.ISSUE_TYPE).to_list()),
        )

    def test_security_reference_filter_requires_referenced_column(self) -> None:
        """A qualifier cannot silently evaluate against an absent reference field."""
        with tempfile.TemporaryDirectory() as directory:
            comparison_path = _write_site(
                Path(directory),
                holdings_rows=["P1,ABC,2026-02-28,100,10,1000,0"],
                transaction_rows=[],
                security_reference_rows=["ABC,csus,EQ"],
                data_issues_config="""
                data_issues:
                  holdings_price_range:
                    only:
                      security_reference.ticker: ABC
                """,
            )

            with self.assertRaisesRegex(PpaError, "missing filter columns: ticker"):
                data_issues.data_issues_table(comparison_path)

    def test_security_reference_join_requires_exact_case_identifier(self) -> None:
        """A differently cased reference identifier cannot qualify a source row."""
        with tempfile.TemporaryDirectory() as directory:
            comparison_path = _write_site(
                Path(directory),
                holdings_rows=["P1,ABC,2026-02-28,100,10,1000,0"],
                transaction_rows=[],
                security_reference_rows=["abc,csus,EQ"],
                data_issues_config="""
                data_issues:
                  holdings_price_range:
                    only:
                      security_reference.asset_class_code: EQ
                """,
            )

            with self.assertRaisesRegex(PpaError, "no exact-case security_reference"):
                data_issues.data_issues_table(comparison_path)

    def test_security_reference_rejects_duplicate_identifier(self) -> None:
        """Reference data must contain one exact-case row per security ID."""
        with tempfile.TemporaryDirectory() as directory:
            comparison_path = _write_site(
                Path(directory),
                holdings_rows=["P1,ABC,2026-02-28,100,10,1000,0"],
                transaction_rows=[],
                security_reference_rows=["ABC,csus,EQ", "ABC,fius,FI"],
                data_issues_config="""
                data_issues:
                  holdings_price_range:
                    only:
                      security_reference.asset_class_code: EQ
                """,
            )

            with self.assertRaisesRegex(PpaError, "one exact-case row"):
                data_issues.data_issues_table(comparison_path)


def _write_site(
    root: Path,
    *,
    holdings_rows: list[str],
    transaction_rows: list[str],
    transaction_header: str | None = None,
    portfolio_performance_rows: list[str] | None = None,
    security_reference_rows: list[str] | None = None,
    split_rows: list[str] | None = None,
    data_issues_config: str = "data_issues: {}",
    transaction_rules: str = "",
) -> Path:
    """Write a minimal performance-comparison site and return its YAML path."""
    for snapshot in ("snapshot_a", "snapshot_b"):
        snapshot_directory = root / snapshot
        snapshot_directory.mkdir(parents=True)
        _write_csv(
            snapshot_directory / "portperf.csv",
            "PORTFOLIO_CODE,FROM_DATE,THRU_DATE,PORT_RETURN",
            portfolio_performance_rows
            or ["P1,2026-01-31,2026-02-28,0.01"],
        )
        _write_csv(
            snapshot_directory / "holdings.csv",
            "PORT,SEC,HOLDING_DATE,QTY,PRICE,MKT_VAL,ACCRUED",
            holdings_rows,
        )
        if transaction_rows:
            _write_csv(
                snapshot_directory / "transactions.csv",
                (
                    transaction_header
                    or "PORT,TRANSACTION_DATE,SETTLE_DATE,SEC,TRAN,SEC_TYPE,"
                    "SRC_DEST_TYPE,SRC_DEST_SYMBOL,SPECIAL_SEC_TYPE,"
                    "SPECIAL_SEC_SYMBOL,QTY,PRICE,AMOUNT,COMMISSION"
                ),
                transaction_rows,
            )
        if security_reference_rows is not None:
            _write_csv(
                snapshot_directory / "secref.csv",
                "SECURITY_ID,SECURITY_TYPE,ASSET_CLASS_CODE",
                security_reference_rows,
            )
        if split_rows is not None:
            _write_csv(
                snapshot_directory / "splits.csv",
                "SEC,SPLIT_DATE,SPLIT_FACTOR",
                split_rows,
            )

    comparison_path = root / "ppar.yaml"
    optional_blocks = "\n".join(
        block
        for block in (
            textwrap.dedent(transaction_rules).strip(),
            textwrap.dedent(data_issues_config).strip(),
        )
        if block
    )
    security_reference_file = (
        "\n  security_reference: secref.csv"
        if security_reference_rows is not None
        else ""
    )
    splits_file = "\n  splits: splits.csv" if split_rows is not None else ""
    files_yaml = textwrap.dedent(
        """
        comparison:
          level: portfolio
        snapshots:
          a:
            path: snapshot_a
          b:
            path: snapshot_b
        files:
          portfolio_performance: portperf.csv
          holdings: holdings.csv
          transactions: transactions.csv
        """
    ).strip() + security_reference_file + splits_file
    policy_yaml = textwrap.dedent(
        """
        extract_contract:
          enforce_ambiguous_axys_flows: true
          transaction_semantics_case: legacy_case_insensitive
        transaction_impact_methods:
          external_flow:
            method: evidence_only
          performance:
            method: transaction_amount_delta_over_return_denominator
            denominator_source: begin_market_value
          quantity:
            method: evidence_only
          price:
            method: evidence_only
          commission:
            method: evidence_only
        holding_impact_methods:
          market_value:
            method: market_value_delta_over_return_denominator
            denominator_source: begin_market_value
          accrued:
            method: accrued_delta_over_return_denominator
            denominator_source: begin_market_value
          quantity:
            method: quantity_delta_times_snapshot_a_unit_market_value_over_return_denominator
            denominator_source: begin_market_value
          cost:
            method: evidence_only
        price_impact_methods:
          price:
            method: price_delta_over_snapshot_a_price_times_weight
            weight_source: snapshot_a_weight
        tolerances:
          return: 0.000001
          contribution: 0.000001
          weight: 0.000001
          market_value: 0.01
          quantity: 0.000001
          price: 0.000001
          split_factor: 0.00000001
          fx_rate: 0.00000001
        """
    ).strip()
    base_yaml = files_yaml + "\n" + policy_yaml
    comparison_path.write_text(
        "\n".join([base_yaml, optional_blocks]).strip() + "\n",
        encoding="utf-8",
    )
    return comparison_path


def _write_csv(path: Path, header: str, rows: list[str]) -> None:
    """Write a CSV file from a header and raw row strings."""
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
