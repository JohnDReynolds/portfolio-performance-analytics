"""Tests for performance-comparison Data Audit Issues checks."""

from __future__ import annotations

# Python imports
from pathlib import Path
import tempfile
import textwrap
import unittest

# Project imports
from ppar.performance_comparison import x_ref


class TestPerformanceComparisonXRefIssues(unittest.TestCase):
    """Validate source-data consistency checks used by the Data Audit Issues sheet."""

    def test_packaged_demo_includes_every_x_ref_issue_type(self) -> None:
        """The Axys/APX demo keeps one visible example of every X-Ref issue type."""
        comparison_path = (
            Path(__file__).resolve().parents[1]
            / "ppar"
            / "setup_templates"
            / "axysapx_performance_comparison"
            / "axysapx_performance_comparison.yaml"
        )

        issues = x_ref.x_ref_issues_table(comparison_path)
        issue_types = set(issues.get_column(x_ref.ISSUE_TYPE).to_list())

        self.assertEqual(
            issue_types,
            {
                "duplicate_transactions",
                "dividend_rate",
                "holding_market_value",
                "holdings_price_range",
                "missing_dividend",
                "transaction_amount_rate",
                "transactions_price_range",
                "holdings_accrued_rate",
                "pa_sa_rate",
            },
        )

    def test_packaged_demo_includes_dividend_rate_x_ref_issue(self) -> None:
        """The Axys/APX demo includes a visible dividend-rate mismatch example."""
        comparison_path = (
            Path(__file__).resolve().parents[1]
            / "ppar"
            / "setup_templates"
            / "axysapx_performance_comparison"
            / "axysapx_performance_comparison.yaml"
        )

        issues = x_ref.x_ref_issues_table(comparison_path)
        dividend_rate_issues = issues.filter(
            (issues[x_ref.ISSUE_TYPE] == "dividend_rate")
            & (issues["security_id"] == "JPM")
        )

        self.assertEqual(dividend_rate_issues.height, 2)
        self.assertEqual(
            set(dividend_rate_issues.get_column("portfolio_id").to_list()),
            {"ALPHA", "BALANCED"},
        )

    def test_packaged_demo_includes_pa_sa_rate_x_ref_issue(self) -> None:
        """The Axys/APX demo includes a visible accrued-interest rate mismatch."""
        comparison_path = (
            Path(__file__).resolve().parents[1]
            / "ppar"
            / "setup_templates"
            / "axysapx_performance_comparison"
            / "axysapx_performance_comparison.yaml"
        )

        issues = x_ref.x_ref_issues_table(comparison_path)
        pa_rate_issues = issues.filter(
            (issues[x_ref.ISSUE_TYPE] == "pa_sa_rate")
            & (issues["security_id"] == "TNOTE5Y")
        )

        self.assertEqual(pa_rate_issues.height, 2)
        self.assertEqual(
            set(pa_rate_issues.get_column("portfolio_id").to_list()),
            {"ALPHA", "INCOME"},
        )

    def test_x_ref_issues_detect_rate_and_missing_dividend_issues(self) -> None:
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

            issues = x_ref.x_ref_issues_table(comparison_path)
            issue_types = set(issues.get_column(x_ref.ISSUE_TYPE).to_list())
            missing_dividends = issues.filter(
                issues[x_ref.ISSUE_TYPE] == "missing_dividend"
            )
            missing_portfolios = set(
                missing_dividends.get_column("portfolio_id").to_list()
            )
            missing_explanations = set(
                missing_dividends.get_column(x_ref.EXPLANATION).to_list()
            )

        self.assertIn("dividend_rate", issue_types)
        self.assertIn("missing_dividend", issue_types)
        self.assertIn("pa_sa_rate", issue_types)
        self.assertIn("holdings_accrued_rate", issue_types)
        self.assertEqual(missing_portfolios, {"P2", "P3"})
        self.assertEqual(
            missing_explanations,
            {
                "Missing a dividend for XYZ on 2026-02-20 that is in "
                "portfolio P1 and other portfolios."
            },
        )

    def test_x_ref_issues_detect_holdings_and_transaction_price_ranges(self) -> None:
        """Price-range checks compare same-day same-security prices."""
        with tempfile.TemporaryDirectory() as directory:
            comparison_path = _write_site(
                Path(directory),
                holdings_rows=[
                    "P1,ABC,2026-02-28,100,10.00,1000,0",
                    "P2,ABC,2026-02-28,100,10.25,1025,0",
                ],
                transaction_rows=[
                    "P1,2026-02-15,,ABC,by,stock,$cash,CASH_USD,,,10,10.00,-100,0",
                    "P2,2026-02-15,,ABC,by,stock,$cash,CASH_USD,,,10,10.50,-105,0",
                ],
                x_ref_config="""
                data_audit_checks:
                  holdings_price_range:
                    percent_tolerance: 1.0
                  transactions_price_range:
                    percent_tolerance: 1.0
                """,
            )

            issues = x_ref.x_ref_issues_table(comparison_path)
            issue_types = set(issues.get_column(x_ref.ISSUE_TYPE).to_list())

        self.assertIn("holdings_price_range", issue_types)
        self.assertIn("transactions_price_range", issue_types)

    def test_x_ref_issues_detect_duplicate_transactions(self) -> None:
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

            issues = x_ref.x_ref_issues_table(comparison_path)
            duplicate_issues = issues.filter(
                issues[x_ref.ISSUE_TYPE] == "duplicate_transactions"
            )

        self.assertEqual(duplicate_issues.height, 4)
        self.assertEqual(
            set(duplicate_issues.get_column(x_ref.EXPLANATION).to_list()),
            {
                "Duplicate transaction rows have the same portfolio, date, "
                "security, code, amount, quantity, and price."
            },
        )

    def test_x_ref_issues_detect_transaction_amount_rates(self) -> None:
        """Transaction amount-rate checks compare amount per unit by code."""
        with tempfile.TemporaryDirectory() as directory:
            comparison_path = _write_site(
                Path(directory),
                holdings_rows=[],
                transaction_rows=[
                    "P1,2026-02-15,,ABC,by,stock,$cash,CASH_USD,,,10,10,-100,0",
                    "P2,2026-02-15,,ABC,by,stock,$cash,CASH_USD,,,10,10,-110,0",
                ],
                x_ref_config="""
                data_audit_checks:
                  transaction_amount_rate:
                    percent_tolerance: 1.0
                """,
            )

            issues = x_ref.x_ref_issues_table(comparison_path)
            amount_rate_issues = issues.filter(
                issues[x_ref.ISSUE_TYPE] == "transaction_amount_rate"
            )

        self.assertEqual(amount_rate_issues.height, 4)
        self.assertEqual(
            set(amount_rate_issues.get_column("portfolio_id").to_list()),
            {"P1", "P2"},
        )

    def test_x_ref_issues_honor_holding_market_value_multiplier(self) -> None:
        """Configured multipliers avoid false issues for bond price conventions."""
        with tempfile.TemporaryDirectory() as directory:
            comparison_path = _write_site(
                Path(directory),
                holdings_rows=[
                    "P1,BOND,2026-02-28,100000,99,99000,0",
                    "P2,BOND,2026-02-28,100000,99,990,0",
                ],
                transaction_rows=[],
                x_ref_config="""
                data_audit_checks:
                  holding_market_value:
                    multipliers:
                      default: 1.0
                      by_security_id:
                        BOND: 0.01
                """,
            )

            issues = x_ref.x_ref_issues_table(comparison_path)
            market_value_issues = issues.filter(
                issues[x_ref.ISSUE_TYPE] == "holding_market_value"
            )

        self.assertEqual(market_value_issues.height, 2)
        self.assertEqual(
            set(market_value_issues.get_column("portfolio_id").to_list()),
            {"P2"},
        )

    def test_x_ref_issue_type_can_be_disabled(self) -> None:
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
                x_ref_config="""
                data_audit_checks:
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

            issues = x_ref.x_ref_issues_table(comparison_path)
            issue_types = set(issues.get_column(x_ref.ISSUE_TYPE).to_list())

        self.assertNotIn("dividend_rate", issue_types)
        self.assertIn("pa_sa_rate", issue_types)

    def test_x_ref_issue_filters_support_only_and_exclude(self) -> None:
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
                x_ref_config="""
                data_audit_checks:
                  dividend_rate:
                    only:
                      transactions.security_type: stock
                      security_id: ABC
                    exclude:
                      portfolio_id: P2
                """,
            )

            issues = x_ref.x_ref_issues_table(comparison_path)
            dividend_issues = issues.filter(
                issues[x_ref.ISSUE_TYPE] == "dividend_rate"
            )

        self.assertEqual(dividend_issues.height, 0)


def _write_site(
    root: Path,
    *,
    holdings_rows: list[str],
    transaction_rows: list[str],
    x_ref_config: str = "data_audit_checks: {}",
    transaction_rules: str = "",
) -> Path:
    """Write a minimal performance-comparison site and return its YAML path."""
    for snapshot in ("snapshot_a", "snapshot_b"):
        snapshot_directory = root / snapshot
        snapshot_directory.mkdir(parents=True)
        _write_csv(
            snapshot_directory / "portperf.csv",
            "PORTFOLIO_CODE,FROM_DATE,THRU_DATE,PORT_RETURN",
            ["P1,2026-01-31,2026-02-28,0.01"],
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
                    "PORT,TRANSACTION_DATE,SETTLE_DATE,SEC,TRAN,SEC_TYPE,"
                    "SRC_DEST_TYPE,SRC_DEST_SYMBOL,SPECIAL_SEC_TYPE,"
                    "SPECIAL_SEC_SYMBOL,QTY,PRICE,AMOUNT,COMMISSION"
                ),
                transaction_rows,
            )

    comparison_path = root / "ppar.yaml"
    optional_blocks = "\n".join(
        block
        for block in (
            textwrap.dedent(transaction_rules).strip(),
            textwrap.dedent(x_ref_config).strip(),
        )
        if block
    )
    base_yaml = textwrap.dedent(
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
    ).strip()
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
