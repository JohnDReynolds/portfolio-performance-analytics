"""Tests for portfolio-level performance comparison findings."""

# Python imports
import datetime as dt
from pathlib import Path
import tempfile
from typing import cast
import unittest
from unittest import mock

# Third-party imports
import polars as pl
import yaml

# Project imports
from ppar.errors import PpaError
from ppar.audit import AuditSpecification
from ppar.audit.performance_comparison import (
    PerformanceComparison,
    findings_to_polars,
)
from ppar.audit import schema as pc_cols
from ppar.audit.performance_comparison.policies import (
    _fx_rate_impact_policies,
    _modified_dietz_external_flow_eligibility,
    _holding_impact_policies,
    _price_impact_policies,
    _transaction_impact_policies,
    _validated_modified_dietz_policy,
)
from ppar.audit.performance_comparison.explain import (
    ESTIMATED_RETURN_IMPACT,
    portfolio_period_contribution_candidates,
)
from ppar.audit.performance_comparison.findings import (
    CASH_FLOW_SIGN,
    CONTEXT,
    DELTA_B_MINUS_A,
    DIRECT_INPUT,
    EVIDENCE_ROLE,
    FINDING_CODE,
    FROM_DATE,
    IMPACT_POLICY,
    IMPACT_POLICY_EVIDENCE_ONLY_PREFIX,
    IMPACT_POLICY_HOLDING_ACCRUED,
    IMPACT_POLICY_HOLDING_MARKET_VALUE,
    IMPACT_POLICY_HOLDING_QUANTITY_UNIT_MARKET_VALUE,
    IMPACT_POLICY_PRICE_WEIGHTED,
    IMPACT_INPUT_VALUE,
    PORTFOLIO_ID,
    PC_FX_RATE,
    PC_PORT_MV,
    PC_HOLD_ACCR,
    PC_HOLD_COST,
    PC_PORT_RET,
    PC_HOLD_MV,
    PC_HOLD_QTY,
    PC_PRICE,
    PC_SEC_ADD,
    PC_SEC_CONTR,
    PC_SEC_DROP,
    PC_SEC_FLOW,
    PC_SEC_MV,
    PC_SEC_RET,
    PC_SEC_WGT,
    PC_TXN_ADD,
    PC_TXN_AMBIG,
    PC_TXN_AMT,
    PC_TXN_COMM,
    PC_TXN_DROP,
    PC_TXN_PRICE,
    PC_TXN_QTY,
    RETURN_DENOMINATOR,
    RETURN_WEIGHT,
    SECURITY_ID,
    SOURCE_FILE,
    SOURCE_COLUMN,
    TARGET_OUTPUT,
    THRU_DATE,
    TRANSACTION_IMPACT_DIAGNOSTIC,
    TRANSACTION_IMPACT_DIAGNOSTIC_ESTIMATE,
    TRANSACTION_IMPACT_POLICY,
    TRANSACTION_IMPACT_POLICY_EXTERNAL_FLOW_EVIDENCE_ONLY,
    TRANSACTION_IMPACT_POLICY_PERFORMANCE_AMOUNT_DELTA,
    TRANSACTION_CATEGORY,
    TRANSACTION_MATCH_STATUS_ADDED_IN_SNAPSHOT_B,
    TRANSACTION_MATCH_STATUS_AMBIGUOUS_FALLBACK_MATCH,
    TRANSACTION_MATCH_STATUS,
    TRANSACTION_MATCH_STATUS_ID_MATCH,
    TRANSACTION_MATCH_STATUS_MISSING_FROM_SNAPSHOT_B,
    TRANSACTION_MATCH_STATUS_SINGLETON_FALLBACK_MATCH,
    TRANSACTION_SEMANTICS_SOURCE,
    PERFORMANCE_FLOW_SIGN,
    Finding,
)
from ppar.audit.performance_comparison.methods import (
    ModifiedDietzDayCount,
    ModifiedDietzDoubleCountPolicy,
    ModifiedDietzFlowTiming,
    ModifiedDietzInclusionRule,
)

_BASELINE_COMPARISON_PATH = Path("tests/data/axys/validation/ppar_audit.yaml")
_RESTATEMENT_COMPARISON_PATH = Path(
    "tests/data/axys/validation/ppar_audit_restatement.yaml"
)
_RESTATEMENT_TRANSACTION_RULES_PATH = Path(
    "tests/data/axys/validation/ppar_audit_restatement_transaction_rules.yaml"
)


def _write_transaction_fallback_specification(directory: Path) -> Path:
    """Write a minimal transaction comparison fixture without transaction ids."""
    for snapshot_name, amount in (("snapshot_a", "100.00"), ("snapshot_b", "110.00")):
        snapshot_path = directory / snapshot_name
        snapshot_path.mkdir()
        (snapshot_path / "portperf.csv").write_text(
            "PORTFOLIO_CODE,FROM_DATE,THRU_DATE,BEGIN_MV,PORT_RETURN\n"
            "PORT_A,2025-05-01,2025-05-31,1000.00,0.01\n",
            encoding="utf-8",
        )
        (snapshot_path / "transactions.csv").write_text(
            "PORT,SEC,TRADE_DATE,SETTLE_DATE,TRAN,QTY,PRICE,AMOUNT\n"
            f"PORT_A,AAPL,2025-05-15,2025-05-16,BUY,1,100.00,{amount}\n",
            encoding="utf-8",
        )

    specification = {
        "snapshots": {
            "a": {"path": "snapshot_a"},
            "b": {"path": "snapshot_b"},
        },
        "files": {
            "portfolio_performance": "portperf.csv",
            "transactions": "transactions.csv",
        },
    }
    specification_path = directory / "ppar_audit.yaml"
    specification_path.write_text(yaml.safe_dump(specification), encoding="utf-8")
    return specification_path


def _write_no_id_transaction_date_move_specification(directory: Path) -> Path:
    """Write a no-ID fixture where a transaction moves across periods."""
    for snapshot_name, transaction_date in (
        ("snapshot_a", "2025-05-31"),
        ("snapshot_b", "2025-06-01"),
    ):
        snapshot_path = directory / snapshot_name
        snapshot_path.mkdir()
        (snapshot_path / "portperf.csv").write_text(
            "PORTFOLIO_CODE,FROM_DATE,THRU_DATE,BEGIN_MV,PORT_RETURN\n"
            "PORT_A,2025-05-01,2025-05-31,10000.00,0.01\n"
            "PORT_A,2025-06-01,2025-06-30,10000.00,0.01\n",
            encoding="utf-8",
        )
        (snapshot_path / "transactions.csv").write_text(
            "PORT,SEC,TRADE_DATE,SETTLE_DATE,TRAN,QTY,PRICE,AMOUNT,"
            "CASH_FLOW_SIGN,PERFORMANCE_FLOW_SIGN\n"
            f"PORT_A,CASHUSD,{transaction_date},{transaction_date},cf,0,0,"
            "-1000.00,negative,external\n",
            encoding="utf-8",
        )

    specification = {
        "snapshots": {
            "a": {"path": "snapshot_a"},
            "b": {"path": "snapshot_b"},
        },
        "files": {
            "portfolio_performance": "portperf.csv",
            "transactions": "transactions.csv",
        },
        "transaction_rules": {
            "cf": {
                "transaction_category": "external_flow",
                "cash_flow_sign": "negative",
                "performance_flow_sign": "external",
            },
        },
        "transaction_impact_methods": {
            "external_flow": {
                "method": "modified_dietz",
                "flow_timing": "trade_date",
                "day_count": "actual_days",
                "inclusion_rule": "beginning_of_day",
                "denominator_source": "begin_market_value",
                "double_count_policy": "cross_check_only",
            },
        },
    }
    specification_path = directory / "ppar_audit.yaml"
    specification_path.write_text(yaml.safe_dump(specification), encoding="utf-8")
    return specification_path


def _write_duplicate_transaction_fallback_specification(directory: Path) -> Path:
    """Write a fixture with legitimate duplicate same-day fallback keys."""
    for snapshot_name in ("snapshot_a", "snapshot_b"):
        snapshot_path = directory / snapshot_name
        snapshot_path.mkdir()
        (snapshot_path / "portperf.csv").write_text(
            "PORTFOLIO_CODE,FROM_DATE,THRU_DATE,BEGIN_MV,PORT_RETURN\n"
            "PORT_A,2025-05-01,2025-05-31,1000.00,0.01\n",
            encoding="utf-8",
        )
        (snapshot_path / "transactions.csv").write_text(
            "PORT,SEC,TRADE_DATE,SETTLE_DATE,TRAN,QTY,PRICE,AMOUNT\n"
            "PORT_A,AAPL,2025-05-15,2025-05-16,BUY,1,100.00,100.00\n"
            "PORT_A,AAPL,2025-05-15,2025-05-16,BUY,1,100.00,100.00\n",
            encoding="utf-8",
        )

    specification = {
        "snapshots": {
            "a": {"path": "snapshot_a"},
            "b": {"path": "snapshot_b"},
        },
        "files": {
            "portfolio_performance": "portperf.csv",
            "transactions": "transactions.csv",
        },
    }
    specification_path = directory / "ppar_audit.yaml"
    specification_path.write_text(yaml.safe_dump(specification), encoding="utf-8")
    return specification_path


def _write_transaction_singleton_duplicate_specification(
    directory: Path,
    *,
    snapshot_a_count: int,
    snapshot_b_count: int,
) -> Path:
    """Write a no-ID fixture with duplicate singleton fallback candidates."""
    for snapshot_name, row_count in (
        ("snapshot_a", snapshot_a_count),
        ("snapshot_b", snapshot_b_count),
    ):
        snapshot_path = directory / snapshot_name
        snapshot_path.mkdir()
        (snapshot_path / "portperf.csv").write_text(
            "PORTFOLIO_CODE,FROM_DATE,THRU_DATE,BEGIN_MV,PORT_RETURN\n"
            "PORT_A,2025-05-01,2025-05-31,1000.00,0.01\n",
            encoding="utf-8",
        )
        transaction_rows = [
            f"PORT_A,AAPL,2025-05-15,2025-05-16,BUY,{index + 1},100.00,"
            f"{100.00 + index:.2f}\n"
            for index in range(row_count)
        ]
        (snapshot_path / "transactions.csv").write_text(
            "PORT,SEC,TRADE_DATE,SETTLE_DATE,TRAN,QTY,PRICE,AMOUNT\n"
            + "".join(transaction_rows),
            encoding="utf-8",
        )

    specification = {
        "snapshots": {
            "a": {"path": "snapshot_a"},
            "b": {"path": "snapshot_b"},
        },
        "files": {
            "portfolio_performance": "portperf.csv",
            "transactions": "transactions.csv",
        },
    }
    specification_path = directory / "ppar_audit.yaml"
    specification_path.write_text(yaml.safe_dump(specification), encoding="utf-8")
    return specification_path


def _write_transaction_case_sensitive_singleton_specification(
    directory: Path,
    *,
    snapshot_a_security: str = "AAPL",
    snapshot_b_security: str = "AAPL",
    snapshot_a_code: str = "BUY",
    snapshot_b_code: str = "BUY",
) -> Path:
    """Write a no-ID singleton fixture with configurable identifier casing."""
    for snapshot_name, security_id, transaction_code, amount in (
        ("snapshot_a", snapshot_a_security, snapshot_a_code, "100.00"),
        ("snapshot_b", snapshot_b_security, snapshot_b_code, "110.00"),
    ):
        snapshot_path = directory / snapshot_name
        snapshot_path.mkdir()
        (snapshot_path / "portperf.csv").write_text(
            "PORTFOLIO_CODE,FROM_DATE,THRU_DATE,BEGIN_MV,PORT_RETURN\n"
            "PORT_A,2025-05-01,2025-05-31,1000.00,0.01\n",
            encoding="utf-8",
        )
        (snapshot_path / "transactions.csv").write_text(
            "PORT,SEC,TRADE_DATE,SETTLE_DATE,TRAN,QTY,PRICE,AMOUNT\n"
            f"PORT_A,{security_id},2025-05-15,2025-05-16,{transaction_code},"
            f"1,100.00,{amount}\n",
            encoding="utf-8",
        )

    specification = {
        "snapshots": {
            "a": {"path": "snapshot_a"},
            "b": {"path": "snapshot_b"},
        },
        "files": {
            "portfolio_performance": "portperf.csv",
            "transactions": "transactions.csv",
        },
    }
    specification_path = directory / "ppar_audit.yaml"
    specification_path.write_text(yaml.safe_dump(specification), encoding="utf-8")
    return specification_path


def _write_transaction_period_specification(directory: Path) -> Path:
    """Write a minimal transaction comparison fixture with transaction ids."""
    for snapshot_name, amount in (("snapshot_a", "100.00"), ("snapshot_b", "110.00")):
        snapshot_path = directory / snapshot_name
        snapshot_path.mkdir()
        (snapshot_path / "portperf.csv").write_text(
            "PORTFOLIO_CODE,FROM_DATE,THRU_DATE,BEGIN_MV,PORT_RETURN\n"
            "PORT_A,2025-05-01,2025-05-31,1000.00,0.01\n",
            encoding="utf-8",
        )
        (snapshot_path / "transactions.csv").write_text(
            "TRANSACTION_ID,PORT,SEC,TRADE_DATE,SETTLE_DATE,TRAN,QTY,PRICE,"
            "AMOUNT,CASH_FLOW_SIGN,PERFORMANCE_FLOW_SIGN\n"
            f"TXN1,PORT_A,AAPL,2025-05-15,2025-05-16,BUY,1,100.00,{amount},"
            "cash out,external\n",
            encoding="utf-8",
        )

    specification = {
        "snapshots": {
            "a": {"path": "snapshot_a"},
            "b": {"path": "snapshot_b"},
        },
        "files": {
            "portfolio_performance": "portperf.csv",
            "transactions": "transactions.csv",
        },
    }
    specification_path = directory / "ppar_audit.yaml"
    specification_path.write_text(yaml.safe_dump(specification), encoding="utf-8")
    return specification_path


def _write_security_transaction_period_specification(directory: Path) -> Path:
    """Write a minimal security comparison fixture with transaction ids."""
    for snapshot_name, amount, security_return in (
        ("snapshot_a", "-100.00", "0.0100"),
        ("snapshot_b", "-110.00", "0.0110"),
    ):
        snapshot_path = directory / snapshot_name
        snapshot_path.mkdir()
        (snapshot_path / "secperf.csv").write_text(
            "PORT,SEC,FROM_DATE,THRU_DATE,BEG_MV,SEC_RET,WEIGHT,CONTRIB\n"
            f"PORT_A,AAPL,2025-05-01,2025-05-31,1000.00,{security_return},"
            "1.0000,0.0100\n",
            encoding="utf-8",
        )
        (snapshot_path / "transactions.csv").write_text(
            "TRANSACTION_ID,PORT,SEC,TRADE_DATE,SETTLE_DATE,TRAN,QTY,PRICE,"
            "AMOUNT,CASH_FLOW_SIGN,PERFORMANCE_FLOW_SIGN\n"
            f"TXN1,PORT_A,AAPL,2025-05-15,2025-05-16,BUY,1,100.00,{amount},"
            "cash out,performance\n",
            encoding="utf-8",
        )

    specification = {
        "comparison": {
            "level": "security",
        },
        "snapshots": {
            "a": {"path": "snapshot_a"},
            "b": {"path": "snapshot_b"},
        },
        "files": {
            "security_performance": "secperf.csv",
            "transactions": "transactions.csv",
        },
        "transaction_impact_methods": {
            "performance": {
                "method": "transaction_amount_delta_over_return_denominator",
                "denominator_source": "begin_market_value",
            },
        },
    }
    specification_path = directory / "ppar_audit.yaml"
    specification_path.write_text(yaml.safe_dump(specification), encoding="utf-8")
    return specification_path


def _write_security_case_sensitive_specification(directory: Path) -> Path:
    """Write a fixture whose security identifiers differ only by case."""
    for snapshot_name, security_id in (("snapshot_a", "AAPL"), ("snapshot_b", "aapl")):
        snapshot_path = directory / snapshot_name
        snapshot_path.mkdir()
        (snapshot_path / "secperf.csv").write_text(
            "PORT,SEC,FROM_DATE,THRU_DATE,BEG_MV,SEC_RETURN,WEIGHT,CONTRIB\n"
            f"PORT_A,{security_id},2025-05-01,2025-05-31,1000.00,0.0100,"
            "1.0000,0.0100\n",
            encoding="utf-8",
        )

    specification = {
        "comparison": {
            "level": "security",
        },
        "snapshots": {
            "a": {"path": "snapshot_a"},
            "b": {"path": "snapshot_b"},
        },
        "files": {
            "security_performance": "secperf.csv",
        },
        "security_return_impact_methods": {
            "transactions": {
                "method": "modified_dietz",
                "flow_timing": "transaction_date",
                "day_count": "actual_days",
                "inclusion_rule": "beginning_of_day",
            },
        },
    }
    specification_path = directory / "ppar_audit.yaml"
    specification_path.write_text(yaml.safe_dump(specification), encoding="utf-8")
    return specification_path


def _write_transaction_outside_period_specification(directory: Path) -> Path:
    """Write a minimal transaction fixture whose trade date is outside period."""
    for snapshot_name, amount in (("snapshot_a", "100.00"), ("snapshot_b", "110.00")):
        snapshot_path = directory / snapshot_name
        snapshot_path.mkdir()
        (snapshot_path / "portperf.csv").write_text(
            "PORTFOLIO_CODE,FROM_DATE,THRU_DATE,BEGIN_MV,PORT_RETURN\n"
            "PORT_A,2025-05-01,2025-05-31,1000.00,0.01\n",
            encoding="utf-8",
        )
        (snapshot_path / "transactions.csv").write_text(
            "TRANSACTION_ID,PORT,SEC,TRADE_DATE,SETTLE_DATE,TRAN,QTY,PRICE,"
            "AMOUNT,CASH_FLOW_SIGN,PERFORMANCE_FLOW_SIGN\n"
            f"TXN1,PORT_A,AAPL,2025-06-15,2025-06-16,BUY,1,100.00,{amount},"
            "cash out,performance\n",
            encoding="utf-8",
        )

    specification = {
        "snapshots": {
            "a": {"path": "snapshot_a"},
            "b": {"path": "snapshot_b"},
        },
        "files": {
            "portfolio_performance": "portperf.csv",
            "transactions": "transactions.csv",
        },
    }
    specification_path = directory / "ppar_audit.yaml"
    specification_path.write_text(yaml.safe_dump(specification), encoding="utf-8")
    return specification_path


def _write_transaction_changed_period_fallback_specification(
    directory: Path,
    *,
    ambiguous: bool = False,
) -> Path:
    """Write a transaction fixture that requires changed-period fallback."""
    portfolio_rows = [
        ("2025-05-30", "2025-05-30", "1000.00", "0.0100", "0.0110"),
    ]
    if ambiguous:
        portfolio_rows.append(
            ("2025-06-30", "2025-06-30", "1100.00", "0.0200", "0.0210")
        )

    for snapshot_name, amount in (("snapshot_a", "100.00"), ("snapshot_b", "110.00")):
        snapshot_path = directory / snapshot_name
        snapshot_path.mkdir()
        portfolio_lines = [
            "PORTFOLIO_CODE,FROM_DATE,THRU_DATE,BEGIN_MV,PORT_RETURN",
        ]
        return_index = 3 if snapshot_name == "snapshot_a" else 4
        for from_date, thru_date, begin_mv, return_a, return_b in portfolio_rows:
            portfolio_return = (return_a, return_b)[return_index - 3]
            portfolio_lines.append(
                f"PORT_A,{from_date},{thru_date},{begin_mv},{portfolio_return}"
            )
        (snapshot_path / "portperf.csv").write_text(
            "\n".join(portfolio_lines) + "\n",
            encoding="utf-8",
        )
        (snapshot_path / "transactions.csv").write_text(
            "TRANSACTION_ID,PORT,SEC,TRADE_DATE,SETTLE_DATE,TRAN,QTY,PRICE,"
            "AMOUNT,CASH_FLOW_SIGN,PERFORMANCE_FLOW_SIGN\n"
            f"TXN1,PORT_A,AAPL,2025-05-01,2025-05-02,BUY,1,100.00,{amount},"
            "cash out,performance\n",
            encoding="utf-8",
        )

    specification = {
        "snapshots": {
            "a": {"path": "snapshot_a"},
            "b": {"path": "snapshot_b"},
        },
        "files": {
            "portfolio_performance": "portperf.csv",
            "transactions": "transactions.csv",
        },
        "transaction_impact_methods": {
            "performance": {
                "method": "transaction_amount_delta_over_return_denominator",
                "denominator_source": "begin_market_value",
            },
        },
    }
    specification_path = directory / "ppar_audit.yaml"
    specification_path.write_text(yaml.safe_dump(specification), encoding="utf-8")
    return specification_path


def _write_holding_period_specification(directory: Path) -> Path:
    """Write a minimal holding comparison fixture with a containing period."""
    for snapshot_name, market_value, portfolio_return in (
        ("snapshot_a", "1000.00", "0.0100"),
        ("snapshot_b", "1010.00", "0.0110"),
    ):
        snapshot_path = directory / snapshot_name
        snapshot_path.mkdir()
        (snapshot_path / "portperf.csv").write_text(
            "PORTFOLIO_CODE,FROM_DATE,THRU_DATE,BEGIN_MV,PORT_RETURN\n"
            f"PORT_A,2025-05-01,2025-05-31,1000.00,{portfolio_return}\n",
            encoding="utf-8",
        )
        (snapshot_path / "holdings.csv").write_text(
            "PORT,SEC,HOLDING_DATE,QTY,MKT_VAL,ACCRUED\n"
            f"PORT_A,AAPL,2025-05-31,10,{market_value},25.00\n",
            encoding="utf-8",
        )

    specification = {
        "snapshots": {
            "a": {"path": "snapshot_a"},
            "b": {"path": "snapshot_b"},
        },
        "files": {
            "portfolio_performance": "portperf.csv",
            "holdings": "holdings.csv",
        },
    }
    specification_path = directory / "ppar_audit.yaml"
    specification_path.write_text(yaml.safe_dump(specification), encoding="utf-8")
    return specification_path


def _write_multi_portfolio_holding_price_specification(
    directory: Path,
    *,
    include_price_impact_methods: bool = False,
) -> Path:
    """Write a fixture where holding price changes affect two portfolio periods."""
    for snapshot_name, price, port_a_return, port_b_return in (
        ("snapshot_a", "100.00", "0.0100", "0.0200"),
        ("snapshot_b", "101.00", "0.0110", "0.0210"),
    ):
        snapshot_path = directory / snapshot_name
        snapshot_path.mkdir()
        (snapshot_path / "portperf.csv").write_text(
            "PORTFOLIO_CODE,FROM_DATE,THRU_DATE,PORT_RETURN\n"
            f"PORT_A,2025-05-01,2025-05-31,{port_a_return}\n"
            f"PORT_B,2025-05-01,2025-05-31,{port_b_return}\n",
            encoding="utf-8",
        )
        (snapshot_path / "secperf.csv").write_text(
            "PORTFOLIO_CODE,SEC,FROM_DATE,THRU_DATE,SEC_RETURN,WEIGHT\n"
            "PORT_A,AAPL,2025-05-01,2025-05-31,0.01,0.20\n"
            "PORT_B,AAPL,2025-05-01,2025-05-31,0.02,0.50\n",
            encoding="utf-8",
        )
        (snapshot_path / "holdings.csv").write_text(
            "PORT,SEC,HOLDING_DATE,QTY,PRICE,MKT_VAL\n"
            f"PORT_A,AAPL,2025-05-31,20,{price},{float(price) * 20:.2f}\n"
            f"PORT_B,AAPL,2025-05-31,50,{price},{float(price) * 50:.2f}\n",
            encoding="utf-8",
        )

    specification = {
        "snapshots": {
            "a": {"path": "snapshot_a"},
            "b": {"path": "snapshot_b"},
        },
        "files": {
            "portfolio_performance": "portperf.csv",
            "security_performance": "secperf.csv",
            "holdings": "holdings.csv",
        },
    }
    if include_price_impact_methods:
        specification["price_impact_methods"] = {
            "price": {
                "method": "price_delta_over_snapshot_a_price_times_weight",
                "weight_source": "snapshot_a_weight",
            },
        }
    specification_path = directory / "ppar_audit.yaml"
    specification_path.write_text(yaml.safe_dump(specification), encoding="utf-8")
    return specification_path


def _write_duplicate_portfolio_specification(directory: Path) -> Path:
    """Write a minimal comparison fixture with duplicate portfolio keys."""
    for snapshot_name in ("snapshot_a", "snapshot_b"):
        snapshot_path = directory / snapshot_name
        snapshot_path.mkdir()
        rows = [
            "PORTFOLIO_CODE,FROM_DATE,THRU_DATE,PORT_RETURN",
            "PORT_A,2025-05-01,2025-05-31,0.01",
        ]
        if snapshot_name == "snapshot_a":
            rows.append("PORT_A,2025-05-01,2025-05-31,0.02")
        (snapshot_path / "portperf.csv").write_text(
            "\n".join(rows) + "\n",
            encoding="utf-8",
        )

    specification = {
        "snapshots": {
            "a": {"path": "snapshot_a"},
            "b": {"path": "snapshot_b"},
        },
        "files": {"portfolio_performance": "portperf.csv"},
    }
    specification_path = directory / "ppar_audit.yaml"
    specification_path.write_text(yaml.safe_dump(specification), encoding="utf-8")
    return specification_path


def _write_blank_portfolio_key_specification(directory: Path) -> Path:
    """Write a fixture with a blank portfolio comparison key value."""
    for snapshot_name in ("snapshot_a", "snapshot_b"):
        snapshot_path = directory / snapshot_name
        snapshot_path.mkdir()
        portfolio_id = "" if snapshot_name == "snapshot_a" else "PORT_A"
        (snapshot_path / "portperf.csv").write_text(
            "PORTFOLIO_CODE,FROM_DATE,THRU_DATE,PORT_RETURN\n"
            f"{portfolio_id},2025-05-01,2025-05-31,0.01\n",
            encoding="utf-8",
        )

    specification = {
        "snapshots": {
            "a": {"path": "snapshot_a"},
            "b": {"path": "snapshot_b"},
        },
        "files": {"portfolio_performance": "portperf.csv"},
    }
    specification_path = directory / "ppar_audit.yaml"
    specification_path.write_text(yaml.safe_dump(specification), encoding="utf-8")
    return specification_path


def _write_blank_holding_key_specification(directory: Path) -> Path:
    """Write a fixture with a blank optional holding comparison key value."""
    for snapshot_name in ("snapshot_a", "snapshot_b"):
        snapshot_path = directory / snapshot_name
        snapshot_path.mkdir()
        (snapshot_path / "portperf.csv").write_text(
            "PORTFOLIO_CODE,FROM_DATE,THRU_DATE,PORT_RETURN\n"
            "PORT_A,2025-05-01,2025-05-31,0.01\n",
            encoding="utf-8",
        )
        security_id = "" if snapshot_name == "snapshot_a" else "AAPL"
        (snapshot_path / "holdings.csv").write_text(
            "PORT,SEC,HOLDING_DATE,QTY,MKT_VAL\n"
            f"PORT_A,{security_id},2025-05-31,10,1000.00\n",
            encoding="utf-8",
        )

    specification = {
        "snapshots": {
            "a": {"path": "snapshot_a"},
            "b": {"path": "snapshot_b"},
        },
        "files": {
            "portfolio_performance": "portperf.csv",
            "holdings": "holdings.csv",
        },
    }
    specification_path = directory / "ppar_audit.yaml"
    specification_path.write_text(yaml.safe_dump(specification), encoding="utf-8")
    return specification_path


class TestPerformanceComparison(unittest.TestCase):
    """Verify portfolio performance comparison findings."""

    _baseline_specification: AuditSpecification
    _restatement_specification: AuditSpecification
    _baseline_combined_findings: list[Finding]
    _restatement_combined_findings: list[Finding]
    _baseline_portfolio_findings: list[Finding]
    _restatement_portfolio_findings: list[Finding]
    _baseline_security_findings: list[Finding]
    _restatement_security_findings: list[Finding]
    _baseline_holding_findings: list[Finding]
    _restatement_holding_findings: list[Finding]
    _baseline_fx_rate_findings: list[Finding]
    _restatement_fx_rate_findings: list[Finding]
    _baseline_transaction_findings: list[Finding]
    _restatement_transaction_findings: list[Finding]

    @classmethod
    def setUpClass(cls) -> None:
        """Cache shared fixture comparisons for the class."""
        cls._baseline_specification = AuditSpecification(
            _BASELINE_COMPARISON_PATH
        )
        cls._restatement_specification = AuditSpecification(
            _RESTATEMENT_COMPARISON_PATH
        )
        baseline = PerformanceComparison(cls._baseline_specification)
        restatement = PerformanceComparison(cls._restatement_specification)

        cls._baseline_combined_findings = baseline.compare()
        cls._restatement_combined_findings = restatement.compare()
        cls._baseline_portfolio_findings = baseline.compare_portfolio_performance()
        cls._restatement_portfolio_findings = restatement.compare_portfolio_performance()
        cls._baseline_security_findings = baseline.compare_security_performance()
        cls._restatement_security_findings = restatement.compare_security_performance()
        cls._baseline_holding_findings = baseline.compare_holdings()
        cls._restatement_holding_findings = restatement.compare_holdings()
        cls._baseline_fx_rate_findings = baseline.compare_fx_rates()
        cls._restatement_fx_rate_findings = restatement.compare_fx_rates()
        cls._baseline_transaction_findings = baseline.compare_transactions()
        cls._restatement_transaction_findings = restatement.compare_transactions()

    def test_identical_baseline_snapshots_have_no_portfolio_findings(self) -> None:
        """The baseline fixture compares identical A/B snapshots."""
        findings = list(self._baseline_portfolio_findings)

        self.assertEqual(findings, [])

    def test_restatement_fixture_reports_portfolio_return_change(self) -> None:
        """The restatement fixture reports controlled portfolio-level changes."""
        findings = list(self._restatement_portfolio_findings)
        finding_dicts = [finding.to_dict() for finding in findings]
        return_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_PORT_RET
            and finding[SOURCE_COLUMN] == pc_cols.PORTFOLIO_RETURN
        ]
        end_mv_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_PORT_MV
            and finding[SOURCE_COLUMN] == pc_cols.END_MARKET_VALUE
        ]

        self.assertEqual(len(return_findings), 1)
        self.assertEqual(return_findings[0][SOURCE_FILE], "portperf.csv")
        self.assertAlmostEqual(cast(float, return_findings[0][DELTA_B_MINUS_A]), 0.0005)
        self.assertEqual(len(end_mv_findings), 1)
        self.assertAlmostEqual(cast(float, end_mv_findings[0][DELTA_B_MINUS_A]), 500.0)

    def test_changed_value_tolerances_are_resolved_once_per_column(self) -> None:
        """One comparison resolves each shared field tolerance only once."""
        comparison = PerformanceComparison(self._restatement_specification)
        from_date = dt.date(2025, 1, 1)
        thru_date = dt.date(2025, 1, 31)
        snapshot_a = pl.DataFrame(
            {
                pc_cols.PORTFOLIO_ID: ["P1", "P2"],
                pc_cols.FROM_DATE: [from_date, from_date],
                pc_cols.THRU_DATE: [thru_date, thru_date],
                pc_cols.PORTFOLIO_RETURN: [0.01, 0.02],
                pc_cols.END_MARKET_VALUE: [100.0, 200.0],
            }
        )
        snapshot_b = snapshot_a.with_columns(
            (pl.col(pc_cols.PORTFOLIO_RETURN) + 0.01),
            (pl.col(pc_cols.END_MARKET_VALUE) + 10.0),
        )
        compare_columns = {
            pc_cols.PORTFOLIO_RETURN: PC_PORT_RET,
            pc_cols.END_MARKET_VALUE: PC_PORT_MV,
        }

        with mock.patch.object(
            comparison,
            "_tolerance",
            wraps=comparison._tolerance,
        ) as tolerance:
            findings = comparison._changed_value_findings(
                snapshot_a,
                snapshot_b,
                compare_columns=compare_columns,
            )

        self.assertEqual(len(findings), 4)
        self.assertEqual(
            [finding.source_column for finding in findings],
            [
                pc_cols.PORTFOLIO_RETURN,
                pc_cols.END_MARKET_VALUE,
                pc_cols.PORTFOLIO_RETURN,
                pc_cols.END_MARKET_VALUE,
            ],
        )
        self.assertEqual(
            tolerance.call_args_list,
            [
                mock.call(pc_cols.PORTFOLIO_RETURN),
                mock.call(pc_cols.END_MARKET_VALUE),
            ],
        )

    def test_non_transaction_estimates_skip_modified_dietz_eligibility(self) -> None:
        """Unrelated fields do not run external-flow eligibility checks."""
        comparison = PerformanceComparison(self._restatement_specification)

        with mock.patch(
            "ppar.audit.performance_comparison.compare."
            "_modified_dietz_external_flow_eligibility"
        ) as eligibility:
            estimate = comparison._transaction_impact_diagnostic_estimate(
                {},
                pc_cols.HOLDINGS,
                pc_cols.MARKET_VALUE,
                "P1",
                dt.date(2025, 1, 1),
                dt.date(2025, 1, 31),
                100.0,
                10.0,
            )

        self.assertIsNone(estimate)
        eligibility.assert_not_called()

    def test_duplicate_portfolio_comparison_keys_raise_error_112(self) -> None:
        """Duplicate comparison keys are invalid because joins would multiply rows."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _write_duplicate_portfolio_specification(Path(temp_dir))
            specification = AuditSpecification(path)

            with self.assertRaises(PpaError) as context:
                PerformanceComparison(specification).compare_portfolio_performance()

            self.assertTrue(str(context.exception).startswith("Error 112"))
            self.assertIn("portfolio_performance", str(context.exception))
            self.assertIn("snapshot A", str(context.exception))

    def test_blank_portfolio_comparison_key_raises_error_112(self) -> None:
        """Blank portfolio keys fail clearly before comparison output is built."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _write_blank_portfolio_key_specification(Path(temp_dir))
            specification = AuditSpecification(path)

            with self.assertRaises(PpaError) as context:
                PerformanceComparison(specification).compare_portfolio_performance()

            message = str(context.exception)
            self.assertTrue(message.startswith("Error 112"))
            self.assertIn("missing snapshot A comparison key values", message)
            self.assertIn("portfolio_id", message)

    def test_blank_holding_comparison_key_raises_error_112(self) -> None:
        """Blank optional evidence keys fail clearly before unmatched findings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _write_blank_holding_key_specification(Path(temp_dir))
            specification = AuditSpecification(path)

            with self.assertRaises(PpaError) as context:
                PerformanceComparison(specification).compare_holdings()

            message = str(context.exception)
            self.assertTrue(message.startswith("Error 112"))
            self.assertIn("holdings", message)
            self.assertIn("missing snapshot A comparison key values", message)
            self.assertIn("security_id", message)

    def test_identical_baseline_snapshots_have_no_security_findings(self) -> None:
        """The baseline fixture compares identical security performance rows."""
        findings = list(self._baseline_security_findings)

        self.assertEqual(findings, [])

    def test_restatement_fixture_reports_security_changes(self) -> None:
        """The restatement fixture reports controlled security-level changes."""
        findings = list(self._restatement_security_findings)
        finding_dicts = [finding.to_dict() for finding in findings]
        aapl_return_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_SEC_RET
            and finding[SECURITY_ID] == "AAPL"
            and finding[SOURCE_COLUMN] == pc_cols.SECURITY_RETURN
        ]
        aapl_weight_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_SEC_WGT
            and finding[SECURITY_ID] == "AAPL"
            and finding[SOURCE_COLUMN] == pc_cols.WEIGHT
        ]
        aapl_contribution_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_SEC_CONTR
            and finding[SECURITY_ID] == "AAPL"
            and finding[SOURCE_COLUMN] == pc_cols.CONTRIBUTION
        ]
        aapl_market_value_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_SEC_MV
            and finding[SECURITY_ID] == "AAPL"
        ]
        aapl_flow_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_SEC_FLOW
            and finding[SECURITY_ID] == "AAPL"
        ]
        add_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_SEC_ADD
            and finding[SECURITY_ID] == "RESTATED_SEC"
        ]
        drop_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_SEC_DROP
            and finding[SECURITY_ID] == "PFE"
        ]

        self.assertEqual(len(aapl_return_findings), 1)
        self.assertAlmostEqual(
            cast(float, aapl_return_findings[0][DELTA_B_MINUS_A]),
            0.01,
        )
        self.assertEqual(len(aapl_weight_findings), 1)
        self.assertAlmostEqual(
            cast(float, aapl_weight_findings[0][DELTA_B_MINUS_A]),
            0.001,
        )
        self.assertEqual(len(aapl_contribution_findings), 1)
        self.assertAlmostEqual(
            cast(float, aapl_contribution_findings[0][DELTA_B_MINUS_A]),
            0.00058425,
        )
        self.assertGreaterEqual(len(aapl_market_value_findings), 1)
        self.assertGreaterEqual(len(aapl_flow_findings), 1)
        self.assertEqual(len(add_findings), 1)
        self.assertEqual(len(drop_findings), 1)

    def test_security_identifiers_match_case_sensitively(self) -> None:
        """Security identifiers that differ only by case are not the same key."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _write_security_case_sensitive_specification(Path(temp_dir))
            specification = AuditSpecification(path)

            findings = [
                finding.to_dict()
                for finding in PerformanceComparison(
                    specification
                ).compare_security_performance()
            ]

        add_findings = [
            finding
            for finding in findings
            if finding[FINDING_CODE] == PC_SEC_ADD and finding[SECURITY_ID] == "aapl"
        ]
        drop_findings = [
            finding
            for finding in findings
            if finding[FINDING_CODE] == PC_SEC_DROP and finding[SECURITY_ID] == "AAPL"
        ]
        value_change_findings = [
            finding
            for finding in findings
            if finding[FINDING_CODE] in {PC_SEC_RET, PC_SEC_WGT, PC_SEC_CONTR}
        ]

        self.assertEqual(len(add_findings), 1)
        self.assertEqual(len(drop_findings), 1)
        self.assertEqual(value_change_findings, [])

    def test_compare_combines_primary_portfolio_and_source_findings(self) -> None:
        """Portfolio comparison returns portfolio-result and source-data findings."""
        finding_dicts = [
            finding.to_dict() for finding in self._restatement_combined_findings
        ]
        finding_codes = {finding[FINDING_CODE] for finding in finding_dicts}

        self.assertIn(PC_PORT_RET, finding_codes)
        self.assertNotIn(PC_SEC_RET, finding_codes)
        self.assertNotIn(PC_SEC_ADD, finding_codes)
        self.assertNotIn(PC_SEC_DROP, finding_codes)
        self.assertIn(PC_HOLD_QTY, finding_codes)
        self.assertIn(PC_HOLD_COST, finding_codes)
        self.assertIn(PC_HOLD_ACCR, finding_codes)
        self.assertIn(PC_HOLD_MV, finding_codes)
        self.assertIn(PC_FX_RATE, finding_codes)
        self.assertIn(PC_TXN_AMT, finding_codes)
        self.assertIn(PC_TXN_QTY, finding_codes)
        self.assertIn(PC_TXN_PRICE, finding_codes)

    def test_combined_findings_convert_to_polars(self) -> None:
        """Combined findings can be converted to a stable Polars table."""
        findings = list(self._restatement_combined_findings)

        frame = findings_to_polars(findings)

        self.assertFalse(frame.is_empty())
        self.assertIn(FINDING_CODE, frame.columns)
        self.assertIn(EVIDENCE_ROLE, frame.columns)
        self.assertIn(DELTA_B_MINUS_A, frame.columns)
        self.assertIn(SOURCE_FILE, frame.columns)

    def test_restatement_findings_have_explanation_roles(self) -> None:
        """Comparison assigns explicit evidence roles to finding families."""
        findings = list(self._restatement_combined_findings)
        frame = findings_to_polars(findings)

        role_by_code = {
            row[FINDING_CODE]: row[EVIDENCE_ROLE]
            for row in frame.select(FINDING_CODE, EVIDENCE_ROLE).iter_rows(named=True)
        }

        self.assertEqual(role_by_code[PC_PORT_RET], TARGET_OUTPUT)
        self.assertEqual(role_by_code[PC_PORT_MV], DIRECT_INPUT)
        self.assertNotIn(PC_SEC_RET, role_by_code)
        self.assertEqual(role_by_code[PC_HOLD_QTY], DIRECT_INPUT)
        self.assertEqual(role_by_code[PC_HOLD_COST], CONTEXT)
        self.assertEqual(role_by_code[PC_HOLD_MV], DIRECT_INPUT)
        self.assertEqual(role_by_code[PC_FX_RATE], CONTEXT)
        self.assertEqual(role_by_code[PC_TXN_AMT], DIRECT_INPUT)

    def test_baseline_combined_compare_has_empty_polars_output(self) -> None:
        """Identical baseline snapshots produce an empty stable finding table."""
        findings = list(self._baseline_combined_findings)

        frame = findings_to_polars(findings)

        self.assertTrue(frame.is_empty())
        self.assertIn(FINDING_CODE, frame.columns)
        self.assertIn(EVIDENCE_ROLE, frame.columns)
        self.assertIn(SOURCE_FILE, frame.columns)

    def test_identical_baseline_snapshots_have_no_holding_findings(self) -> None:
        """The baseline fixture compares identical holding rows."""
        findings = list(self._baseline_holding_findings)

        self.assertEqual(findings, [])

    def test_restatement_fixture_reports_holding_changes(self) -> None:
        """The restatement fixture reports controlled holding-level changes."""
        findings = list(self._restatement_holding_findings)
        finding_dicts = [finding.to_dict() for finding in findings]
        quantity_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_HOLD_QTY
            and finding[SECURITY_ID] == "AAPL"
            and finding[SOURCE_COLUMN] == pc_cols.QUANTITY
        ]
        market_value_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_HOLD_MV
            and finding[SECURITY_ID] == "AAPL"
            and finding[SOURCE_COLUMN] == pc_cols.MARKET_VALUE
        ]
        cost_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_HOLD_COST
            and finding[SECURITY_ID] == "AAPL"
            and finding[SOURCE_COLUMN] == pc_cols.COST
        ]
        accrued_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_HOLD_ACCR
            and finding[SECURITY_ID] == "AAPL"
            and finding[SOURCE_COLUMN] == pc_cols.ACCRUED
        ]

        self.assertEqual(len(quantity_findings), 1)
        self.assertAlmostEqual(
            cast(float, quantity_findings[0][DELTA_B_MINUS_A]),
            10.0,
        )
        self.assertEqual(len(market_value_findings), 1)
        self.assertAlmostEqual(
            cast(float, market_value_findings[0][DELTA_B_MINUS_A]),
            2648.56,
        )
        self.assertEqual(len(cost_findings), 1)
        self.assertAlmostEqual(
            cast(float, cost_findings[0][DELTA_B_MINUS_A]),
            225.0,
        )
        self.assertEqual(cost_findings[0][EVIDENCE_ROLE], CONTEXT)
        self.assertEqual(len(accrued_findings), 1)
        self.assertAlmostEqual(
            cast(float, accrued_findings[0][DELTA_B_MINUS_A]),
            6.25,
        )

    def test_holding_changes_link_to_containing_portfolio_period(self) -> None:
        """Changed holding rows inherit the containing portfolio period."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_holding_period_specification(Path(temp_dir))
            specification = AuditSpecification(specification_path)

            findings = PerformanceComparison(specification).compare_holdings()
            finding_dicts = [finding.to_dict() for finding in findings]
            market_value_finding = next(
                finding
                for finding in finding_dicts
                if finding[FINDING_CODE] == PC_HOLD_MV
            )

            self.assertEqual(str(market_value_finding[FROM_DATE]), "2025-05-01")
            self.assertEqual(str(market_value_finding[THRU_DATE]), "2025-05-31")

    def test_holding_market_value_policy_is_loaded_from_yaml(self) -> None:
        """Explicit holding market value impact policy is carried into findings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_holding_period_specification(Path(temp_dir))
            configuration = yaml.safe_load(specification_path.read_text(encoding="utf-8"))
            configuration["holding_impact_methods"] = {
                "market_value": {
                    "method": "market_value_delta_over_return_denominator",
                    "denominator_source": "begin_market_value",
                },
            }
            specification_path.write_text(
                yaml.safe_dump(configuration),
                encoding="utf-8",
            )
            specification = AuditSpecification(specification_path)

            findings = PerformanceComparison(specification).compare()
            finding_dicts = [finding.to_dict() for finding in findings]
            market_value_finding = next(
                finding
                for finding in finding_dicts
                if finding[FINDING_CODE] == PC_HOLD_MV
            )
            candidates = portfolio_period_contribution_candidates(
                findings_to_polars(findings)
            )
            holding_candidate = candidates.filter(
                (pl.col(FINDING_CODE) == PC_HOLD_MV)
                & (pl.col(SOURCE_COLUMN) == pc_cols.MARKET_VALUE)
            ).row(0, named=True)

            self.assertEqual(
                market_value_finding[IMPACT_POLICY],
                IMPACT_POLICY_HOLDING_MARKET_VALUE,
            )
            self.assertEqual(market_value_finding[RETURN_DENOMINATOR], 1000.0)
            self.assertAlmostEqual(holding_candidate[ESTIMATED_RETURN_IMPACT], 0.01)
            policies = _holding_impact_policies(specification)
            self.assertEqual(
                policies[pc_cols.MARKET_VALUE],
                IMPACT_POLICY_HOLDING_MARKET_VALUE,
            )

    def test_holding_accrued_policy_is_loaded_from_yaml(self) -> None:
        """Explicit holding accrued impact policy is carried into findings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_holding_period_specification(Path(temp_dir))
            for snapshot_name, accrued in (
                ("snapshot_a", "25.00"),
                ("snapshot_b", "30.00"),
            ):
                holding_path = Path(temp_dir) / snapshot_name / "holdings.csv"
                holding_path.write_text(
                    holding_path.read_text(encoding="utf-8").replace(
                        "25.00\n",
                        f"{accrued}\n",
                    ),
                    encoding="utf-8",
                )
            configuration = yaml.safe_load(specification_path.read_text(encoding="utf-8"))
            configuration["holding_impact_methods"] = {
                "accrued": {
                    "method": "accrued_delta_over_return_denominator",
                    "denominator_source": "begin_market_value",
                },
            }
            specification_path.write_text(
                yaml.safe_dump(configuration),
                encoding="utf-8",
            )
            specification = AuditSpecification(specification_path)

            findings = PerformanceComparison(specification).compare()
            finding_dicts = [finding.to_dict() for finding in findings]
            accrued_finding = next(
                finding
                for finding in finding_dicts
                if finding[FINDING_CODE] == PC_HOLD_ACCR
            )
            candidates = portfolio_period_contribution_candidates(
                findings_to_polars(findings)
            )
            accrued_candidate = candidates.filter(
                (pl.col(FINDING_CODE) == PC_HOLD_ACCR)
                & (pl.col(SOURCE_COLUMN) == pc_cols.ACCRUED)
            ).row(0, named=True)

            self.assertEqual(
                accrued_finding[IMPACT_POLICY],
                IMPACT_POLICY_HOLDING_ACCRUED,
            )
            self.assertEqual(accrued_finding[RETURN_DENOMINATOR], 1000.0)
            self.assertAlmostEqual(accrued_candidate[ESTIMATED_RETURN_IMPACT], 0.005)

    def test_evidence_only_policy_is_loaded_from_yaml(self) -> None:
        """Explicit evidence-only policy is carried into source-data findings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_holding_period_specification(Path(temp_dir))
            holding_path = Path(temp_dir) / "snapshot_b" / "holdings.csv"
            holding_path.write_text(
                holding_path.read_text(encoding="utf-8").replace(
                    "PORT_A,AAPL,2025-05-31,10,1010.00,25.00",
                    "PORT_A,AAPL,2025-05-31,11,1010.00,25.00",
                ),
                encoding="utf-8",
            )
            configuration = yaml.safe_load(specification_path.read_text(encoding="utf-8"))
            configuration["evidence_only_impact_methods"] = {
                "holdings": {
                    "method": "evidence_only",
                    "source_fields": ["quantity"],
                },
            }
            specification_path.write_text(
                yaml.safe_dump(configuration),
                encoding="utf-8",
            )
            specification = AuditSpecification(specification_path)

            findings = PerformanceComparison(specification).compare_holdings()
            quantity_finding = next(
                finding.to_dict()
                for finding in findings
                if finding.to_dict()[FINDING_CODE] == PC_HOLD_QTY
            )

            self.assertEqual(
                quantity_finding[IMPACT_POLICY],
                f"{IMPACT_POLICY_EVIDENCE_ONLY_PREFIX}holdings.quantity",
            )

    def test_holding_quantity_evidence_only_policy_is_loaded_from_yaml(self) -> None:
        """Holding quantity can be marked review-only in holding impact YAML."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_holding_period_specification(Path(temp_dir))
            holding_path = Path(temp_dir) / "snapshot_b" / "holdings.csv"
            holding_path.write_text(
                holding_path.read_text(encoding="utf-8").replace(
                    "PORT_A,AAPL,2025-05-31,10,1010.00,25.00",
                    "PORT_A,AAPL,2025-05-31,11,1010.00,25.00",
                ),
                encoding="utf-8",
            )
            configuration = yaml.safe_load(specification_path.read_text(encoding="utf-8"))
            configuration["holding_impact_methods"] = {
                "quantity": {
                    "method": "evidence_only",
                },
            }
            specification_path.write_text(
                yaml.safe_dump(configuration),
                encoding="utf-8",
            )
            specification = AuditSpecification(specification_path)

            policies = _holding_impact_policies(specification)
            findings = PerformanceComparison(specification).compare_holdings()
            quantity_finding = next(
                finding.to_dict()
                for finding in findings
                if finding.to_dict()[FINDING_CODE] == PC_HOLD_QTY
            )

            self.assertEqual(
                policies[pc_cols.QUANTITY],
                f"{IMPACT_POLICY_EVIDENCE_ONLY_PREFIX}holdings.quantity",
            )
            self.assertEqual(
                quantity_finding[IMPACT_POLICY],
                f"{IMPACT_POLICY_EVIDENCE_ONLY_PREFIX}holdings.quantity",
            )

    def test_holding_quantity_unit_market_value_policy_estimates_impact(self) -> None:
        """Holding quantity can use snapshot A unit market value for estimates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_holding_period_specification(Path(temp_dir))
            holding_path = Path(temp_dir) / "snapshot_b" / "holdings.csv"
            holding_path.write_text(
                holding_path.read_text(encoding="utf-8").replace(
                    "PORT_A,AAPL,2025-05-31,10,1010.00,25.00",
                    "PORT_A,AAPL,2025-05-31,11,1010.00,25.00",
                ),
                encoding="utf-8",
            )
            configuration = yaml.safe_load(specification_path.read_text(encoding="utf-8"))
            configuration["holding_impact_methods"] = {
                "quantity": {
                    "method": (
                        "quantity_delta_times_snapshot_a_unit_market_value_over_return_denominator"
                    ),
                    "denominator_source": "begin_market_value",
                },
            }
            specification_path.write_text(
                yaml.safe_dump(configuration),
                encoding="utf-8",
            )
            specification = AuditSpecification(specification_path)

            findings = PerformanceComparison(specification).compare()
            finding_dicts = [finding.to_dict() for finding in findings]
            quantity_finding = next(
                finding
                for finding in finding_dicts
                if finding[FINDING_CODE] == PC_HOLD_QTY
            )
            candidates = portfolio_period_contribution_candidates(
                findings_to_polars(findings)
            )
            quantity_candidate = candidates.filter(
                (pl.col(FINDING_CODE) == PC_HOLD_QTY)
                & (pl.col(SOURCE_COLUMN) == pc_cols.QUANTITY)
            ).row(0, named=True)

            self.assertEqual(
                quantity_finding[IMPACT_POLICY],
                IMPACT_POLICY_HOLDING_QUANTITY_UNIT_MARKET_VALUE,
            )
            self.assertEqual(quantity_finding[RETURN_DENOMINATOR], 1000.0)
            self.assertEqual(quantity_finding[IMPACT_INPUT_VALUE], 100.0)
            self.assertAlmostEqual(quantity_candidate[ESTIMATED_RETURN_IMPACT], 0.1)
            policies = _holding_impact_policies(specification)
            self.assertEqual(
                policies[pc_cols.QUANTITY],
                IMPACT_POLICY_HOLDING_QUANTITY_UNIT_MARKET_VALUE,
            )

    def test_holding_cost_evidence_only_policy_is_loaded_from_yaml(self) -> None:
        """Holding cost can be marked review-only in holding impact YAML."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for snapshot_name, cost, portfolio_return in (
                ("snapshot_a", "900.00", "0.0100"),
                ("snapshot_b", "925.00", "0.0101"),
            ):
                snapshot_path = root / snapshot_name
                snapshot_path.mkdir()
                (snapshot_path / "portperf.csv").write_text(
                    "PORTFOLIO_CODE,FROM_DATE,THRU_DATE,BEGIN_MV,PORT_RETURN\n"
                    f"PORT_A,2025-05-01,2025-05-31,1000.00,{portfolio_return}\n",
                    encoding="utf-8",
                )
                (snapshot_path / "holdings.csv").write_text(
                    "PORT,SEC,HOLDING_DATE,QTY,MKT_VAL,COST,ACCRUED\n"
                    f"PORT_A,AAPL,2025-05-31,10,1000.00,{cost},25.00\n",
                    encoding="utf-8",
                )
            specification_path = root / "ppar_audit.yaml"
            specification_path.write_text(
                yaml.safe_dump(
                    {
                        "snapshots": {
                            "a": {"path": "snapshot_a"},
                            "b": {"path": "snapshot_b"},
                        },
                        "files": {
                            "portfolio_performance": "portperf.csv",
                            "holdings": "holdings.csv",
                        },
                        "holding_impact_methods": {
                            "cost": {
                                "method": "evidence_only",
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            specification = AuditSpecification(specification_path)

            policies = _holding_impact_policies(specification)
            findings = PerformanceComparison(specification).compare_holdings()
            cost_finding = next(
                finding.to_dict()
                for finding in findings
                if finding.to_dict()[FINDING_CODE] == PC_HOLD_COST
            )

            self.assertEqual(
                policies[pc_cols.COST],
                f"{IMPACT_POLICY_EVIDENCE_ONLY_PREFIX}holdings.cost",
            )
            self.assertEqual(cost_finding[EVIDENCE_ROLE], CONTEXT)
            self.assertEqual(
                cost_finding[IMPACT_POLICY],
                f"{IMPACT_POLICY_EVIDENCE_ONLY_PREFIX}holdings.cost",
            )

    def test_restatement_fixture_models_cash_as_a_holding(self) -> None:
        """The controlled cash change remains a CASHUSD holding restatement."""
        cash_findings = [
            finding.to_dict()
            for finding in self._restatement_holding_findings
            if finding.security_id == "CASHUSD"
        ]
        changes = {
            finding[SOURCE_COLUMN]: finding[DELTA_B_MINUS_A]
            for finding in cash_findings
        }

        self.assertEqual(changes[pc_cols.QUANTITY], 500.0)
        self.assertEqual(changes[pc_cols.MARKET_VALUE], 500.0)
        self.assertEqual(str(cash_findings[0][FROM_DATE]), "2025-05-30")
        self.assertEqual(str(cash_findings[0][THRU_DATE]), "2025-05-30")

    def test_price_policy_is_loaded_from_yaml(self) -> None:
        """Explicit price impact policy is carried into holding price findings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _write_multi_portfolio_holding_price_specification(
                Path(temp_dir),
                include_price_impact_methods=True,
            )
            specification = AuditSpecification(path)

            findings = PerformanceComparison(specification).compare()
            finding_dicts = [finding.to_dict() for finding in findings]
            price_findings = [
                finding
                for finding in finding_dicts
                if finding[FINDING_CODE] == PC_PRICE
            ]
            candidates = portfolio_period_contribution_candidates(
                findings_to_polars(findings)
            )
            price_candidates = candidates.filter(pl.col(FINDING_CODE) == PC_PRICE)
            impact_by_portfolio = {
                row[PORTFOLIO_ID]: row[ESTIMATED_RETURN_IMPACT]
                for row in price_candidates.iter_rows(named=True)
            }

            self.assertEqual(len(price_findings), 2)
            self.assertEqual(
                {finding[IMPACT_POLICY] for finding in price_findings},
                {IMPACT_POLICY_PRICE_WEIGHTED},
            )
            self.assertEqual(
                {finding[RETURN_WEIGHT] for finding in price_findings},
                {0.2, 0.5},
            )
            self.assertAlmostEqual(impact_by_portfolio["PORT_A"], 0.002)
            self.assertAlmostEqual(impact_by_portfolio["PORT_B"], 0.005)
            policies = _price_impact_policies(specification)
            self.assertEqual(policies[pc_cols.PRICE], IMPACT_POLICY_PRICE_WEIGHTED)

    def test_identical_baseline_snapshots_have_no_fx_rate_findings(self) -> None:
        """The baseline fixture compares identical FX rate rows."""
        findings = list(self._baseline_fx_rate_findings)

        self.assertEqual(findings, [])

    def test_restatement_fixture_reports_fx_rate_changes(self) -> None:
        """The restatement fixture reports controlled FX rate changes."""
        findings = list(self._restatement_fx_rate_findings)
        finding_dicts = [finding.to_dict() for finding in findings]
        fx_rate_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_FX_RATE
            and finding[SOURCE_COLUMN] == pc_cols.FX_RATE
        ]

        self.assertEqual(len(fx_rate_findings), 1)
        self.assertAlmostEqual(
            cast(float, fx_rate_findings[0][DELTA_B_MINUS_A]),
            0.005,
        )

    def test_fx_rate_evidence_only_policy_is_loaded_from_yaml(self) -> None:
        """Explicit FX rate review-only policy is carried into findings."""
        specification = AuditSpecification(
            _RESTATEMENT_TRANSACTION_RULES_PATH
        )

        policies = _fx_rate_impact_policies(specification)
        findings = PerformanceComparison(specification).compare_fx_rates()
        fx_rate_finding = next(
            finding.to_dict()
            for finding in findings
            if finding.to_dict()[FINDING_CODE] == PC_FX_RATE
            and finding.to_dict()[SOURCE_COLUMN] == pc_cols.FX_RATE
        )

        self.assertEqual(
            policies[pc_cols.FX_RATE],
            f"{IMPACT_POLICY_EVIDENCE_ONLY_PREFIX}fx_rates.fx_rate",
        )
        self.assertEqual(
            fx_rate_finding[IMPACT_POLICY],
            f"{IMPACT_POLICY_EVIDENCE_ONLY_PREFIX}fx_rates.fx_rate",
        )

    def test_identical_baseline_snapshots_have_no_transaction_findings(self) -> None:
        """The baseline fixture compares identical transaction rows."""
        findings = list(self._baseline_transaction_findings)

        self.assertEqual(findings, [])

    def test_restatement_fixture_reports_transaction_changes(self) -> None:
        """The restatement fixture reports controlled transaction changes."""
        findings = list(self._restatement_transaction_findings)
        finding_dicts = [finding.to_dict() for finding in findings]
        amount_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_TXN_AMT
            and finding[SECURITY_ID] == "AAPL"
            and finding[SOURCE_COLUMN] == pc_cols.AMOUNT
        ]
        quantity_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_TXN_QTY
            and finding[SECURITY_ID] == "AAPL"
            and finding[SOURCE_COLUMN] == pc_cols.QUANTITY
        ]
        price_findings = [
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_TXN_PRICE
            and finding[SECURITY_ID] == "AAPL"
            and finding[SOURCE_COLUMN] == pc_cols.PRICE
        ]

        self.assertEqual(len(amount_findings), 1)
        self.assertEqual(amount_findings[0][TRANSACTION_CATEGORY], "buy")
        self.assertEqual(
            amount_findings[0][TRANSACTION_MATCH_STATUS],
            TRANSACTION_MATCH_STATUS_ID_MATCH,
        )
        self.assertEqual(str(amount_findings[0][FROM_DATE]), "2025-05-30")
        self.assertEqual(str(amount_findings[0][THRU_DATE]), "2025-05-30")
        self.assertEqual(amount_findings[0][RETURN_DENOMINATOR], 999915.0)
        self.assertAlmostEqual(
            cast(float, amount_findings[0][DELTA_B_MINUS_A]),
            -100.0,
        )
        self.assertEqual(len(quantity_findings), 1)
        self.assertEqual(quantity_findings[0][TRANSACTION_CATEGORY], "buy")
        self.assertAlmostEqual(
            cast(float, quantity_findings[0][DELTA_B_MINUS_A]),
            1.0,
        )
        self.assertEqual(len(price_findings), 1)
        self.assertEqual(price_findings[0][TRANSACTION_CATEGORY], "buy")
        self.assertAlmostEqual(
            cast(float, price_findings[0][DELTA_B_MINUS_A]),
            0.5,
        )

    def test_restatement_transaction_rules_fixture_carries_yaml_semantics(
        self,
    ) -> None:
        """YAML transaction rules fill sign/flow semantics in Axys findings."""
        specification = AuditSpecification(
            _RESTATEMENT_TRANSACTION_RULES_PATH
        )

        findings = PerformanceComparison(specification).compare_transactions()
        changed_fields = {
            finding.to_dict()[SOURCE_COLUMN]: finding.to_dict()
            for finding in findings
            if finding.to_dict()[SECURITY_ID] == "AAPL"
        }

        self.assertEqual(
            set(changed_fields),
            {pc_cols.AMOUNT, pc_cols.QUANTITY, pc_cols.PRICE},
        )
        for finding in changed_fields.values():
            self.assertEqual(finding[TRANSACTION_CATEGORY], "buy")
            self.assertEqual(finding[CASH_FLOW_SIGN], "negative")
            self.assertEqual(finding[PERFORMANCE_FLOW_SIGN], "performance")
            self.assertEqual(finding[TRANSACTION_SEMANTICS_SOURCE], "yaml_rule")

    def test_transaction_commission_changes_are_context_by_default(self) -> None:
        """Commission changes remain context without explicit YAML treatment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_transaction_period_specification(Path(temp_dir))
            for snapshot_name, commission in (
                ("snapshot_a", "5.00"),
                ("snapshot_b", "7.50"),
            ):
                transaction_path = Path(temp_dir) / snapshot_name / "transactions.csv"
                transaction_path.write_text(
                    "TRANSACTION_ID,PORT,SEC,TRADE_DATE,SETTLE_DATE,TRAN,QTY,"
                    "PRICE,AMOUNT,COMMISSION,CASH_FLOW_SIGN,PERFORMANCE_FLOW_SIGN\n"
                    "TXN1,PORT_A,AAPL,2025-05-15,2025-05-16,BUY,1,100.00,"
                    f"100.00,{commission},cash out,external\n",
                    encoding="utf-8",
                )
            specification = AuditSpecification(specification_path)

            comparison = PerformanceComparison(specification)
            findings = comparison.compare_transactions()
            commission_finding = next(
                finding.to_dict()
                for finding in findings
                if finding.to_dict()[FINDING_CODE] == PC_TXN_COMM
            )

            self.assertEqual(commission_finding[SOURCE_COLUMN], pc_cols.COMMISSION)
            self.assertEqual(commission_finding[EVIDENCE_ROLE], CONTEXT)
            self.assertAlmostEqual(
                cast(float, commission_finding[DELTA_B_MINUS_A]),
                2.5,
            )

    def test_transaction_commission_evidence_only_policy_loads_from_yaml(self) -> None:
        """Transaction commission can be marked review-only in YAML."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_transaction_period_specification(Path(temp_dir))
            for snapshot_name, commission in (
                ("snapshot_a", "5.00"),
                ("snapshot_b", "7.50"),
            ):
                transaction_path = Path(temp_dir) / snapshot_name / "transactions.csv"
                transaction_path.write_text(
                    "TRANSACTION_ID,PORT,SEC,TRADE_DATE,SETTLE_DATE,TRAN,QTY,"
                    "PRICE,AMOUNT,COMMISSION,CASH_FLOW_SIGN,PERFORMANCE_FLOW_SIGN\n"
                    "TXN1,PORT_A,AAPL,2025-05-15,2025-05-16,BUY,1,100.00,"
                    f"100.00,{commission},cash out,external\n",
                    encoding="utf-8",
                )
            configuration = yaml.safe_load(specification_path.read_text(encoding="utf-8"))
            configuration["transaction_impact_methods"] = {
                "commission": {
                    "method": "evidence_only",
                },
            }
            specification_path.write_text(
                yaml.safe_dump(configuration),
                encoding="utf-8",
            )
            specification = AuditSpecification(specification_path)

            policies = _transaction_impact_policies(specification)
            findings = PerformanceComparison(specification).compare_transactions()
            commission_finding = next(
                finding.to_dict()
                for finding in findings
                if finding.to_dict()[FINDING_CODE] == PC_TXN_COMM
            )

            self.assertEqual(policies[pc_cols.COMMISSION].method, "evidence_only")
            self.assertEqual(
                policies[pc_cols.COMMISSION].finding_label,
                f"{IMPACT_POLICY_EVIDENCE_ONLY_PREFIX}transactions.commission",
            )
            self.assertEqual(commission_finding[EVIDENCE_ROLE], CONTEXT)
            self.assertEqual(
                commission_finding[TRANSACTION_IMPACT_POLICY],
                f"{IMPACT_POLICY_EVIDENCE_ONLY_PREFIX}transactions.commission",
            )

    def test_transaction_changes_link_to_containing_portfolio_period(self) -> None:
        """Changed transaction rows inherit the containing portfolio period."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_transaction_period_specification(Path(temp_dir))
            specification = AuditSpecification(specification_path)

            comparison = PerformanceComparison(specification)
            findings = comparison.compare_transactions()
            finding_dicts = [finding.to_dict() for finding in findings]
            amount_finding = next(
                finding
                for finding in finding_dicts
                if finding[FINDING_CODE] == PC_TXN_AMT
            )

            self.assertEqual(str(amount_finding[FROM_DATE]), "2025-05-01")
            self.assertEqual(str(amount_finding[THRU_DATE]), "2025-05-31")
            self.assertEqual(amount_finding[TRANSACTION_CATEGORY], "buy")
            self.assertEqual(amount_finding[CASH_FLOW_SIGN], "negative")
            self.assertEqual(amount_finding[PERFORMANCE_FLOW_SIGN], "external")
            self.assertEqual(amount_finding[TRANSACTION_SEMANTICS_SOURCE], "source")

    def test_transaction_external_flow_policy_is_loaded_from_yaml(self) -> None:
        """Explicit external-flow impact policy is carried into findings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_transaction_period_specification(Path(temp_dir))
            configuration = yaml.safe_load(specification_path.read_text(encoding="utf-8"))
            configuration["transaction_impact_methods"] = {
                "external_flow": {"method": "evidence_only"}
            }
            specification_path.write_text(
                yaml.safe_dump(configuration),
                encoding="utf-8",
            )
            specification = AuditSpecification(specification_path)

            comparison = PerformanceComparison(specification)
            findings = comparison.compare_transactions()
            amount_finding = next(
                finding.to_dict()
                for finding in findings
                if finding.to_dict()[FINDING_CODE] == PC_TXN_AMT
            )

            self.assertEqual(
                amount_finding[TRANSACTION_IMPACT_POLICY],
                TRANSACTION_IMPACT_POLICY_EXTERNAL_FLOW_EVIDENCE_ONLY,
            )
            self.assertEqual(
                amount_finding[TRANSACTION_IMPACT_DIAGNOSTIC],
                "external-flow evidence-only policy",
            )
            policies = _transaction_impact_policies(specification)
            external_flow_policy = policies["external_flow"]
            self.assertEqual(external_flow_policy.method, "evidence_only")
            self.assertEqual(
                external_flow_policy.finding_label,
                TRANSACTION_IMPACT_POLICY_EXTERNAL_FLOW_EVIDENCE_ONLY,
            )
            self.assertIsNone(external_flow_policy.flow_timing)

    def test_transaction_performance_policy_is_loaded_from_yaml(self) -> None:
        """Explicit performance amount impact policy is carried into findings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_transaction_period_specification(Path(temp_dir))
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                transaction_path = Path(temp_dir) / snapshot_name / "transactions.csv"
                transaction_path.write_text(
                    transaction_path.read_text(encoding="utf-8").replace(
                        "cash out,external",
                        "cash out,performance",
                    ),
                    encoding="utf-8",
                )
            configuration = yaml.safe_load(specification_path.read_text(encoding="utf-8"))
            configuration["transaction_impact_methods"] = {
                "performance": {
                    "method": "transaction_amount_delta_over_return_denominator",
                    "denominator_source": "begin_market_value",
                },
            }
            specification_path.write_text(
                yaml.safe_dump(configuration),
                encoding="utf-8",
            )
            specification = AuditSpecification(specification_path)

            findings = PerformanceComparison(specification).compare_transactions()
            amount_finding = next(
                finding.to_dict()
                for finding in findings
                if finding.to_dict()[FINDING_CODE] == PC_TXN_AMT
            )

            self.assertEqual(
                amount_finding[TRANSACTION_IMPACT_POLICY],
                TRANSACTION_IMPACT_POLICY_PERFORMANCE_AMOUNT_DELTA,
            )
            policies = _transaction_impact_policies(specification)
            performance_policy = policies["performance"]
            self.assertEqual(
                performance_policy.method,
                "transaction_amount_delta_over_return_denominator",
            )
            self.assertEqual(
                performance_policy.finding_label,
                TRANSACTION_IMPACT_POLICY_PERFORMANCE_AMOUNT_DELTA,
            )
            self.assertEqual(performance_policy.denominator_source, "begin_market_value")

    def test_transaction_source_field_evidence_only_policies_load_from_yaml(
        self,
    ) -> None:
        """Transaction source fields can be marked review-only in YAML."""
        specification = AuditSpecification(
            _RESTATEMENT_TRANSACTION_RULES_PATH
        )

        policies = _transaction_impact_policies(specification)
        findings = PerformanceComparison(specification).compare_transactions()
        finding_dicts = [finding.to_dict() for finding in findings]
        quantity_finding = next(
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_TXN_QTY
        )
        price_finding = next(
            finding
            for finding in finding_dicts
            if finding[FINDING_CODE] == PC_TXN_PRICE
        )

        self.assertEqual(policies[pc_cols.QUANTITY].method, "evidence_only")
        self.assertEqual(policies[pc_cols.PRICE].method, "evidence_only")
        self.assertEqual(policies[pc_cols.COMMISSION].method, "evidence_only")
        self.assertEqual(
            policies[pc_cols.QUANTITY].finding_label,
            f"{IMPACT_POLICY_EVIDENCE_ONLY_PREFIX}transactions.quantity",
        )
        self.assertEqual(
            policies[pc_cols.PRICE].finding_label,
            f"{IMPACT_POLICY_EVIDENCE_ONLY_PREFIX}transactions.price",
        )
        self.assertEqual(
            policies[pc_cols.COMMISSION].finding_label,
            f"{IMPACT_POLICY_EVIDENCE_ONLY_PREFIX}transactions.commission",
        )
        self.assertEqual(
            quantity_finding[TRANSACTION_IMPACT_POLICY],
            f"{IMPACT_POLICY_EVIDENCE_ONLY_PREFIX}transactions.quantity",
        )
        self.assertEqual(
            price_finding[TRANSACTION_IMPACT_POLICY],
            f"{IMPACT_POLICY_EVIDENCE_ONLY_PREFIX}transactions.price",
        )

    def test_transaction_modified_dietz_policy_preserves_explicit_yaml_fields(
        self,
    ) -> None:
        """Modified Dietz policy keeps every explicit YAML convention."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_transaction_period_specification(Path(temp_dir))
            specification = AuditSpecification(specification_path)
            external_flow_value = {
                "method": "modified_dietz",
                "flow_timing": "settlement_date",
                "day_count": "actual_days",
                "inclusion_rule": "end_of_day",
                "denominator_source": "begin_market_value",
                "double_count_policy": "cross_check_only",
            }

            policy = _validated_modified_dietz_policy(
                specification,
                external_flow_value,
            )

            self.assertEqual(policy.method, "modified_dietz")
            self.assertEqual(policy.flow_timing, ModifiedDietzFlowTiming.SETTLEMENT_DATE.value)
            self.assertEqual(policy.day_count, ModifiedDietzDayCount.ACTUAL_DAYS.value)
            self.assertEqual(
                policy.inclusion_rule,
                ModifiedDietzInclusionRule.END_OF_DAY.value,
            )
            self.assertEqual(policy.denominator_source, "begin_market_value")
            self.assertEqual(
                policy.double_count_policy,
                ModifiedDietzDoubleCountPolicy.CROSS_CHECK_ONLY.value,
            )

    def test_transaction_modified_dietz_eligibility_accepts_complete_inputs(
        self,
    ) -> None:
        """Modified Dietz eligibility requires explicit row, period, and policy inputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_transaction_period_specification(Path(temp_dir))
            specification = AuditSpecification(specification_path)
            policy = _validated_modified_dietz_policy(
                specification,
                {
                    "method": "modified_dietz",
                    "flow_timing": "trade_date",
                    "day_count": "actual_days",
                    "inclusion_rule": "beginning_of_day",
                    "denominator_source": "begin_market_value",
                    "double_count_policy": "cross_check_only",
                },
            )
            row = {
                pc_cols.PERFORMANCE_FLOW_SIGN: "external",
                pc_cols.TRANSACTION_DATE: dt.date(2025, 5, 15),
                pc_cols.SETTLEMENT_DATE: dt.date(2025, 5, 16),
            }

            eligibility = _modified_dietz_external_flow_eligibility(
                row=row,
                policy=policy,
                portfolio_id="PORT_A",
                from_date=dt.date(2025, 5, 1),
                thru_date=dt.date(2025, 5, 31),
                denominator=10000.0,
            )

            self.assertTrue(eligibility.eligible)
            self.assertEqual(eligibility.missing_inputs, ())
            self.assertEqual(eligibility.flow_date, dt.date(2025, 5, 15))

    def test_transaction_modified_dietz_eligibility_reports_missing_inputs(
        self,
    ) -> None:
        """Modified Dietz eligibility names missing inputs instead of assuming them."""
        row = {
            pc_cols.PERFORMANCE_FLOW_SIGN: "performance",
            pc_cols.TRANSACTION_DATE: dt.date(2025, 6, 1),
        }

        eligibility = _modified_dietz_external_flow_eligibility(
            row=row,
            policy=None,
            portfolio_id=None,
            from_date=None,
            thru_date=dt.date(2025, 5, 31),
            denominator=0.0,
        )

        self.assertFalse(eligibility.eligible)
        self.assertIn(
            "external performance-flow semantics",
            eligibility.missing_inputs,
        )
        self.assertIn("modified_dietz policy", eligibility.missing_inputs)
        self.assertIn("flow date", eligibility.missing_inputs)
        self.assertIn("portfolio", eligibility.missing_inputs)
        self.assertIn("portfolio period", eligibility.missing_inputs)
        self.assertIn(
            "nonzero begin_market_value denominator",
            eligibility.missing_inputs,
        )

    def test_transaction_modified_dietz_eligibility_uses_settlement_date(
        self,
    ) -> None:
        """The flow date comes from the YAML-selected timing convention."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_transaction_period_specification(Path(temp_dir))
            specification = AuditSpecification(specification_path)
            policy = _validated_modified_dietz_policy(
                specification,
                {
                    "method": "modified_dietz",
                    "flow_timing": "settlement_date",
                    "day_count": "actual_days",
                    "inclusion_rule": "end_of_day",
                    "denominator_source": "begin_market_value",
                    "double_count_policy": "cross_check_only",
                },
            )
            row = {
                pc_cols.PERFORMANCE_FLOW_SIGN: "external",
                pc_cols.TRANSACTION_DATE: dt.date(2025, 5, 15),
                pc_cols.SETTLEMENT_DATE: dt.date(2025, 5, 16),
            }

            eligibility = _modified_dietz_external_flow_eligibility(
                row=row,
                policy=policy,
                portfolio_id="PORT_A",
                from_date=dt.date(2025, 5, 1),
                thru_date=dt.date(2025, 5, 31),
                denominator=10000.0,
            )

            self.assertTrue(eligibility.eligible)
            self.assertEqual(eligibility.flow_date, dt.date(2025, 5, 16))

    def test_transaction_modified_dietz_eligibility_rejects_out_of_period_flow(
        self,
    ) -> None:
        """External-flow dates must fall inside the linked portfolio period."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_transaction_period_specification(Path(temp_dir))
            specification = AuditSpecification(specification_path)
            policy = _validated_modified_dietz_policy(
                specification,
                {
                    "method": "modified_dietz",
                    "flow_timing": "trade_date",
                    "day_count": "actual_days",
                    "inclusion_rule": "beginning_of_day",
                    "denominator_source": "begin_market_value",
                    "double_count_policy": "cross_check_only",
                },
            )
            row = {
                pc_cols.PERFORMANCE_FLOW_SIGN: "external",
                pc_cols.TRANSACTION_DATE: dt.date(2025, 6, 1),
            }

            eligibility = _modified_dietz_external_flow_eligibility(
                row=row,
                policy=policy,
                portfolio_id="PORT_A",
                from_date=dt.date(2025, 5, 1),
                thru_date=dt.date(2025, 5, 31),
                denominator=10000.0,
            )

            self.assertFalse(eligibility.eligible)
            self.assertIn("in-period flow date", eligibility.missing_inputs)

    def test_transaction_external_flow_policy_rejects_unsupported_method(self) -> None:
        """Unsupported external-flow methods fail instead of implying a formula."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_transaction_period_specification(Path(temp_dir))
            configuration = yaml.safe_load(specification_path.read_text(encoding="utf-8"))
            configuration["transaction_impact_methods"] = {
                "external_flow": {"method": "not_a_supported_method"}
            }
            specification_path.write_text(
                yaml.safe_dump(configuration),
                encoding="utf-8",
            )

            with self.assertRaises(PpaError) as context:
                PerformanceComparison(
                    AuditSpecification(specification_path)
                )

            self.assertIn("external_flow.method", str(context.exception))

    def test_transaction_modified_dietz_design_contract_validates_fields(self) -> None:
        """Modified Dietz YAML fields have explicit allowed cross-check values."""
        scenarios = [
            ({"flow_timing": "activity_date"}, "external_flow.flow_timing"),
            ({"day_count": "business_days"}, "external_flow.day_count"),
            ({"inclusion_rule": "midday"}, "external_flow.inclusion_rule"),
            (
                {"denominator_source": "average_market_value"},
                "external_flow.denominator_source",
            ),
            ({"double_count_policy": "aggregate"}, "external_flow.double_count_policy"),
            ({"unsupported_key": "value"}, "unsupported modified_dietz keys"),
        ]

        for overrides, expected_message in scenarios:
            with self.subTest(overrides=overrides):
                with tempfile.TemporaryDirectory() as temp_dir:
                    specification_path = _write_transaction_period_specification(
                        Path(temp_dir)
                    )
                    configuration = yaml.safe_load(
                        specification_path.read_text(encoding="utf-8")
                    )
                    modified_dietz = {
                        "method": "modified_dietz",
                        "flow_timing": "trade_date",
                        "day_count": "actual_days",
                        "inclusion_rule": "beginning_of_day",
                        "denominator_source": "begin_market_value",
                        "double_count_policy": "cross_check_only",
                    }
                    modified_dietz.update(overrides)
                    configuration["transaction_impact_methods"] = {
                        "external_flow": modified_dietz
                    }
                    specification_path.write_text(
                        yaml.safe_dump(configuration),
                        encoding="utf-8",
                    )

                    with self.assertRaises(PpaError) as context:
                        PerformanceComparison(
                            AuditSpecification(specification_path)
                        )

                    self.assertIn(expected_message, str(context.exception))

    def test_transaction_external_flow_future_methods_remain_rejected(self) -> None:
        """Future method names are reserved until their formulas are implemented."""
        for method in ("subperiod_linked", "unweighted_flow_delta"):
            with self.subTest(method=method):
                with tempfile.TemporaryDirectory() as temp_dir:
                    specification_path = _write_transaction_period_specification(
                        Path(temp_dir)
                    )
                    configuration = yaml.safe_load(
                        specification_path.read_text(encoding="utf-8")
                    )
                    configuration["transaction_impact_methods"] = {
                        "external_flow": {"method": method}
                    }
                    specification_path.write_text(
                        yaml.safe_dump(configuration),
                        encoding="utf-8",
                    )

                    with self.assertRaises(PpaError) as context:
                        PerformanceComparison(
                            AuditSpecification(specification_path)
                        )

                    self.assertIn("external_flow.method", str(context.exception))
                    self.assertIn(
                        "reserved but not implemented",
                        str(context.exception),
                    )

    def test_transaction_modified_dietz_cross_check_estimate_is_loaded(self) -> None:
        """A fully shaped Modified Dietz policy emits review-only estimates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_transaction_period_specification(Path(temp_dir))
            configuration = yaml.safe_load(specification_path.read_text(encoding="utf-8"))
            configuration["transaction_impact_methods"] = {
                "external_flow": {
                    "method": "modified_dietz",
                    "flow_timing": "trade_date",
                    "day_count": "actual_days",
                    "inclusion_rule": "beginning_of_day",
                    "denominator_source": "begin_market_value",
                    "double_count_policy": "cross_check_only",
                }
            }
            specification_path.write_text(
                yaml.safe_dump(configuration),
                encoding="utf-8",
            )

            specification = AuditSpecification(specification_path)
            findings = PerformanceComparison(specification).compare_transactions()
            amount_finding = next(
                finding.to_dict()
                for finding in findings
                if finding.to_dict()[FINDING_CODE] == PC_TXN_AMT
            )

            self.assertEqual(
                amount_finding[TRANSACTION_IMPACT_POLICY],
                "external_flow:modified_dietz",
            )
            self.assertEqual(
                amount_finding[TRANSACTION_IMPACT_DIAGNOSTIC],
                "modified_dietz cross-check estimate",
            )
            self.assertAlmostEqual(
                cast(float, amount_finding[TRANSACTION_IMPACT_DIAGNOSTIC_ESTIMATE]),
                10.0 * (17.0 / 31.0) / 1000.0,
            )
            policies = _transaction_impact_policies(specification)
            self.assertEqual(
                policies["external_flow"].finding_label,
                "external_flow:modified_dietz",
            )

    def test_transaction_modified_dietz_cross_check_missing_inputs_are_reported(
        self,
    ) -> None:
        """Modified Dietz stays diagnostic-only when row-level inputs are missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_transaction_period_specification(Path(temp_dir))
            configuration = yaml.safe_load(specification_path.read_text(encoding="utf-8"))
            configuration["transaction_impact_methods"] = {
                "external_flow": {
                    "method": "modified_dietz",
                    "flow_timing": "settlement_date",
                    "day_count": "actual_days",
                    "inclusion_rule": "beginning_of_day",
                    "denominator_source": "begin_market_value",
                    "double_count_policy": "cross_check_only",
                }
            }
            specification_path.write_text(
                yaml.safe_dump(configuration),
                encoding="utf-8",
            )

            findings = PerformanceComparison(
                AuditSpecification(specification_path)
            ).compare_transactions()
            amount_finding = next(
                finding.to_dict()
                for finding in findings
                if finding.to_dict()[FINDING_CODE] == PC_TXN_AMT
            )

            self.assertEqual(
                amount_finding[TRANSACTION_IMPACT_DIAGNOSTIC],
                "modified_dietz cross-check estimate",
            )
            self.assertAlmostEqual(
                cast(float, amount_finding[TRANSACTION_IMPACT_DIAGNOSTIC_ESTIMATE]),
                10.0 * (16.0 / 31.0) / 1000.0,
            )

    def test_transaction_modified_dietz_out_of_period_stays_unestimated(self) -> None:
        """Modified Dietz cross-check estimates require in-period flow dates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_transaction_outside_period_specification(
                Path(temp_dir)
            )
            for snapshot_name, amount in (
                ("snapshot_a", "100.00"),
                ("snapshot_b", "110.00"),
            ):
                transaction_path = (
                    Path(temp_dir) / snapshot_name / "transactions.csv"
                )
                transaction_path.write_text(
                    "TRANSACTION_ID,PORT,SEC,TRADE_DATE,SETTLE_DATE,TRAN,QTY,"
                    "PRICE,AMOUNT,CASH_FLOW_SIGN,PERFORMANCE_FLOW_SIGN\n"
                    "TXN1,PORT_A,AAPL,2025-06-15,2025-06-16,BUY,1,100.00,"
                    f"{amount},cash out,external\n",
                    encoding="utf-8",
                )
            configuration = yaml.safe_load(specification_path.read_text(encoding="utf-8"))
            configuration["transaction_impact_methods"] = {
                "external_flow": {
                    "method": "modified_dietz",
                    "flow_timing": "trade_date",
                    "day_count": "actual_days",
                    "inclusion_rule": "beginning_of_day",
                    "denominator_source": "begin_market_value",
                    "double_count_policy": "cross_check_only",
                }
            }
            specification_path.write_text(
                yaml.safe_dump(configuration),
                encoding="utf-8",
            )

            findings = PerformanceComparison(
                AuditSpecification(specification_path)
            ).compare_transactions()
            amount_finding = next(
                finding.to_dict()
                for finding in findings
                if finding.to_dict()[FINDING_CODE] == PC_TXN_AMT
            )

            self.assertIn(
                "modified_dietz missing inputs",
                cast(str, amount_finding[TRANSACTION_IMPACT_DIAGNOSTIC]),
            )
            self.assertIsNone(
                amount_finding[TRANSACTION_IMPACT_DIAGNOSTIC_ESTIMATE]
            )

    def test_transaction_modified_dietz_cross_check_is_not_contribution_estimate(
        self,
    ) -> None:
        """Modified Dietz diagnostics do not populate regular impact totals."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_transaction_period_specification(Path(temp_dir))
            (Path(temp_dir) / "snapshot_b" / "portperf.csv").write_text(
                "PORTFOLIO_CODE,FROM_DATE,THRU_DATE,BEGIN_MV,PORT_RETURN\n"
                "PORT_A,2025-05-01,2025-05-31,1000.00,0.02\n",
                encoding="utf-8",
            )
            configuration = yaml.safe_load(specification_path.read_text(encoding="utf-8"))
            configuration["transaction_impact_methods"] = {
                "external_flow": {
                    "method": "modified_dietz",
                    "flow_timing": "trade_date",
                    "day_count": "actual_days",
                    "inclusion_rule": "beginning_of_day",
                    "denominator_source": "begin_market_value",
                    "double_count_policy": "cross_check_only",
                }
            }
            specification_path.write_text(
                yaml.safe_dump(configuration),
                encoding="utf-8",
            )

            findings = findings_to_polars(
                PerformanceComparison(
                    AuditSpecification(specification_path)
                ).compare()
            )
            candidates = portfolio_period_contribution_candidates(findings)
            transaction_amount = candidates.filter(
                (pl.col(FINDING_CODE) == PC_TXN_AMT)
            ).row(0, named=True)

            self.assertIsNone(transaction_amount[ESTIMATED_RETURN_IMPACT])
            self.assertIsNotNone(
                transaction_amount[TRANSACTION_IMPACT_DIAGNOSTIC_ESTIMATE]
            )

    def test_transaction_modified_dietz_counted_policy_is_rejected(
        self,
    ) -> None:
        """Modified Dietz external-flow policies must remain cross-check only."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_transaction_period_specification(Path(temp_dir))
            configuration = yaml.safe_load(specification_path.read_text(encoding="utf-8"))
            configuration["transaction_impact_methods"] = {
                "external_flow": {
                    "method": "modified_dietz",
                    "flow_timing": "trade_date",
                    "day_count": "actual_days",
                    "inclusion_rule": "beginning_of_day",
                    "denominator_source": "begin_market_value",
                    "double_count_policy": "count_as_explanation",
                }
            }
            specification_path.write_text(
                yaml.safe_dump(configuration),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(PpaError, "external_flow.double_count_policy"):
                PerformanceComparison(
                    AuditSpecification(specification_path)
                ).compare()

    def test_transaction_impact_methods_reject_malformed_yaml(self) -> None:
        """Transaction impact method YAML must use the supported contract."""
        scenarios = [
            ("not-a-mapping", "must be a mapping"),
            ({"unsupported": {"method": "evidence_only"}}, "unsupported"),
            ({"external_flow": "evidence_only"}, "external_flow must be a mapping"),
            ({"external_flow": {}}, "external_flow.method is required"),
            (
                {"external_flow": {"method": "modified_dietz"}},
                "missing required modified_dietz keys",
            ),
            ({"performance": "estimate"}, "performance must be a mapping"),
            ({"performance": {}}, "performance is missing required keys"),
            (
                {
                    "performance": {
                        "method": "unsupported",
                        "denominator_source": "begin_market_value",
                    }
                },
                "performance.method must be",
            ),
            (
                {
                    "performance": {
                        "method": "transaction_amount_delta_over_return_denominator",
                        "denominator_source": "ending_market_value",
                    }
                },
                "performance.denominator_source must be one of",
            ),
            ({"quantity": "review-only"}, "quantity must be a mapping"),
            ({"quantity": {}}, "quantity is missing required keys"),
            (
                {"quantity": {"method": "quantity_delta_over_return_denominator"}},
                "quantity.method must be",
            ),
            (
                {"quantity": {"method": "evidence_only", "denominator_source": "x"}},
                "quantity has unsupported keys",
            ),
            ({"price": "review-only"}, "price must be a mapping"),
            ({"price": {}}, "price is missing required keys"),
            (
                {"price": {"method": "price_delta_over_return_denominator"}},
                "price.method must be",
            ),
            (
                {"price": {"method": "evidence_only", "denominator_source": "x"}},
                "price has unsupported keys",
            ),
            ({"commission": "review-only"}, "commission must be a mapping"),
            ({"commission": {}}, "commission is missing required keys"),
            (
                {"commission": {"method": "commission_delta_over_return_denominator"}},
                "commission.method must be",
            ),
            (
                {"commission": {"method": "evidence_only", "denominator_source": "x"}},
                "commission has unsupported keys",
            ),
        ]

        for transaction_impact_methods, expected_message in scenarios:
            with self.subTest(transaction_impact_methods=transaction_impact_methods):
                with tempfile.TemporaryDirectory() as temp_dir:
                    specification_path = _write_transaction_period_specification(
                        Path(temp_dir)
                    )
                    configuration = yaml.safe_load(
                        specification_path.read_text(encoding="utf-8")
                    )
                    configuration["transaction_impact_methods"] = (
                        transaction_impact_methods
                    )
                    specification_path.write_text(
                        yaml.safe_dump(configuration),
                        encoding="utf-8",
                    )

                    with self.assertRaises(PpaError) as context:
                        PerformanceComparison(
                            AuditSpecification(specification_path)
                        )

                    self.assertIn(expected_message, str(context.exception))

    def test_security_return_impact_methods_are_required_for_security_level(
        self,
    ) -> None:
        """Security comparisons must explicitly configure transaction flows."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_security_transaction_period_specification(
                Path(temp_dir)
            )

            with self.assertRaises(PpaError) as context:
                PerformanceComparison(
                    AuditSpecification(specification_path)
                )

            self.assertIn("security_return_impact_methods is required", str(context.exception))

    def test_security_return_impact_methods_reject_malformed_yaml(self) -> None:
        """Security transaction-flow method YAML must use the supported contract."""
        scenarios = [
            ("not-a-mapping", "security_return_impact_methods must be a mapping"),
            ({"unsupported": {}}, "unsupported security_return_impact_methods keys"),
            ({"transactions": "modified_dietz"}, "transactions must be a mapping"),
            ({"transactions": {}}, "transactions is missing required keys"),
            (
                {
                    "transactions": {
                        "method": "transaction_amount_delta_over_return_denominator",
                        "flow_timing": "transaction_date",
                        "day_count": "actual_days",
                        "inclusion_rule": "beginning_of_day",
                    }
                },
                "transactions.method must be",
            ),
            (
                {
                    "transactions": {
                        "method": "modified_dietz",
                        "flow_timing": "settlement_date",
                        "day_count": "actual_days",
                        "inclusion_rule": "beginning_of_day",
                    }
                },
                "transactions.flow_timing must be one of",
            ),
        ]

        for security_return_impact_methods, expected_message in scenarios:
            with self.subTest(
                security_return_impact_methods=security_return_impact_methods
            ):
                with tempfile.TemporaryDirectory() as temp_dir:
                    specification_path = _write_security_transaction_period_specification(
                        Path(temp_dir)
                    )
                    configuration = yaml.safe_load(
                        specification_path.read_text(encoding="utf-8")
                    )
                    configuration["security_return_impact_methods"] = (
                        security_return_impact_methods
                    )
                    specification_path.write_text(
                        yaml.safe_dump(configuration),
                        encoding="utf-8",
                    )

                    with self.assertRaises(PpaError) as context:
                        PerformanceComparison(
                            AuditSpecification(specification_path)
                        )

                    self.assertIn(expected_message, str(context.exception))

    def test_contribution_impact_methods_reject_malformed_yaml(self) -> None:
        """Contribution impact method YAML must use the supported contract."""
        scenarios = [
            ("not-a-mapping", "contribution_impact_methods must be a mapping"),
            ({"unsupported": {"method": "x"}}, "unsupported"),
            (
                {"portfolio_source_field": "estimate"},
                "portfolio_source_field must be a mapping",
            ),
            (
                {
                    "portfolio_source_field": {
                        "method": "source_field_delta_over_begin_market_value",
                        "denominator_source": "begin_market_value",
                    }
                },
                "portfolio_source_field is missing required keys",
            ),
            (
                {
                    "portfolio_source_field": {
                        "method": "unsupported",
                        "denominator_source": "begin_market_value",
                        "source_fields": ["income"],
                    }
                },
                "portfolio_source_field.method must be",
            ),
            (
                {
                    "portfolio_source_field": {
                        "method": "source_field_delta_over_begin_market_value",
                        "denominator_source": "ending_market_value",
                        "source_fields": ["income"],
                    }
                },
                "portfolio_source_field.denominator_source must be one of",
            ),
            (
                {
                    "portfolio_source_field": {
                        "method": "source_field_delta_over_begin_market_value",
                        "denominator_source": "begin_market_value",
                        "source_fields": ["end_market_value"],
                    }
                },
                "contains unsupported fields",
            ),
            (
                {"security_contribution": {"method": "unsupported"}},
                "security_contribution.method must be",
            ),
            (
                {
                    "security_return": {
                        "method": "security_return_delta_times_weight",
                        "weight_source": "ending_weight",
                    }
                },
                "security_return.weight_source must be one of",
            ),
        ]

        for contribution_impact_methods, expected_message in scenarios:
            with self.subTest(contribution_impact_methods=contribution_impact_methods):
                with tempfile.TemporaryDirectory() as temp_dir:
                    specification_path = _write_transaction_period_specification(
                        Path(temp_dir)
                    )
                    configuration = yaml.safe_load(
                        specification_path.read_text(encoding="utf-8")
                    )
                    configuration["contribution_impact_methods"] = (
                        contribution_impact_methods
                    )
                    specification_path.write_text(
                        yaml.safe_dump(configuration),
                        encoding="utf-8",
                    )

                    with self.assertRaises(PpaError) as context:
                        PerformanceComparison(
                            AuditSpecification(specification_path)
                        )

                    self.assertIn(expected_message, str(context.exception))

    def test_holding_impact_methods_reject_malformed_yaml(self) -> None:
        """Holding impact method YAML must use the supported contract."""
        scenarios = [
            ("not-a-mapping", "holding_impact_methods must be a mapping"),
            ({"unsupported": {"method": "x"}}, "unsupported"),
            ({"market_value": "estimate"}, "market_value must be a mapping"),
            ({"market_value": {}}, "market_value is missing required keys"),
            (
                {
                    "market_value": {
                        "method": "unsupported",
                        "denominator_source": "begin_market_value",
                    }
                },
                "market_value.method must be",
            ),
            (
                {
                    "market_value": {
                        "method": "market_value_delta_over_return_denominator",
                        "denominator_source": "ending_market_value",
                    }
                },
                "market_value.denominator_source must be one of",
            ),
            ({"accrued": "estimate"}, "accrued must be a mapping"),
            ({"accrued": {}}, "accrued is missing required keys"),
            (
                {
                    "accrued": {
                        "method": "unsupported",
                        "denominator_source": "begin_market_value",
                    }
                },
                "accrued.method must be",
            ),
            (
                {
                    "accrued": {
                        "method": "accrued_delta_over_return_denominator",
                        "denominator_source": "ending_market_value",
                    }
                },
                "accrued.denominator_source must be one of",
            ),
            ({"quantity": "review-only"}, "quantity must be a mapping"),
            ({"quantity": {}}, "quantity is missing required keys"),
            (
                {"quantity": {"method": "quantity_delta_over_return_denominator"}},
                "quantity.method must be",
            ),
            (
                {"quantity": {"method": "evidence_only", "denominator_source": "x"}},
                "quantity has unsupported keys",
            ),
            ({"cost": "review-only"}, "cost must be a mapping"),
            ({"cost": {}}, "cost is missing required keys"),
            (
                {"cost": {"method": "cost_delta_over_return_denominator"}},
                "cost.method must be",
            ),
            (
                {"cost": {"method": "evidence_only", "denominator_source": "x"}},
                "cost has unsupported keys",
            ),
        ]

        for holding_impact_methods, expected_message in scenarios:
            with self.subTest(holding_impact_methods=holding_impact_methods):
                with tempfile.TemporaryDirectory() as temp_dir:
                    specification_path = _write_holding_period_specification(
                        Path(temp_dir)
                    )
                    configuration = yaml.safe_load(
                        specification_path.read_text(encoding="utf-8")
                    )
                    configuration["holding_impact_methods"] = holding_impact_methods
                    specification_path.write_text(
                        yaml.safe_dump(configuration),
                        encoding="utf-8",
                    )

                    with self.assertRaises(PpaError) as context:
                        PerformanceComparison(
                            AuditSpecification(specification_path)
                        )

                    self.assertIn(expected_message, str(context.exception))

    def test_price_impact_methods_reject_malformed_yaml(self) -> None:
        """Price impact method YAML must use the supported contract."""
        scenarios = [
            ("not-a-mapping", "price_impact_methods must be a mapping"),
            ({"unsupported": {"method": "x"}}, "unsupported"),
            ({"price": "estimate"}, "price must be a mapping"),
            ({"price": {}}, "price is missing required keys"),
            (
                {
                    "price": {
                        "method": "unsupported",
                        "weight_source": "snapshot_a_weight",
                    }
                },
                "price.method must be",
            ),
            (
                {
                    "price": {
                        "method": "price_delta_over_snapshot_a_price_times_weight",
                        "weight_source": "ending_weight",
                    }
                },
                "price.weight_source must be one of",
            ),
        ]

        for price_impact_methods, expected_message in scenarios:
            with self.subTest(price_impact_methods=price_impact_methods):
                with tempfile.TemporaryDirectory() as temp_dir:
                    specification_path = _write_multi_portfolio_holding_price_specification(
                        Path(temp_dir)
                    )
                    configuration = yaml.safe_load(
                        specification_path.read_text(encoding="utf-8")
                    )
                    configuration["price_impact_methods"] = price_impact_methods
                    specification_path.write_text(
                        yaml.safe_dump(configuration),
                        encoding="utf-8",
                    )

                    with self.assertRaises(PpaError) as context:
                        PerformanceComparison(
                            AuditSpecification(specification_path)
                        )

                    self.assertIn(expected_message, str(context.exception))

    def test_fx_rate_impact_methods_reject_malformed_yaml(self) -> None:
        """FX rate impact method YAML must use the supported contract."""
        scenarios = [
            ("not-a-mapping", "fx_rate_impact_methods must be a mapping"),
            ({"unsupported": {"method": "x"}}, "unsupported"),
            ({"fx_rate": "review-only"}, "fx_rate must be a mapping"),
            ({"fx_rate": {}}, "fx_rate is missing required keys"),
            (
                {"fx_rate": {"method": "fx_delta_over_exposure"}},
                "fx_rate.method must be",
            ),
            (
                {"fx_rate": {"method": "evidence_only", "denominator_source": "x"}},
                "fx_rate has unsupported keys",
            ),
        ]

        for fx_rate_impact_methods, expected_message in scenarios:
            with self.subTest(fx_rate_impact_methods=fx_rate_impact_methods):
                with tempfile.TemporaryDirectory() as temp_dir:
                    specification_path = _write_multi_portfolio_holding_price_specification(
                        Path(temp_dir)
                    )
                    configuration = yaml.safe_load(
                        specification_path.read_text(encoding="utf-8")
                    )
                    configuration["files"]["fx_rates"] = "fx_rates.csv"
                    for snapshot_name, rate in (
                        ("snapshot_a", "1.00000000"),
                        ("snapshot_b", "1.00500000"),
                    ):
                        snapshot_path = Path(temp_dir) / snapshot_name
                        (snapshot_path / "fx_rates.csv").write_text(
                            "FROM_CURRENCY,TO_CURRENCY,RATE_DATE,FX_RATE\n"
                            f"EUR,USD,2025-05-31,{rate}\n",
                            encoding="utf-8",
                        )
                    configuration["fx_rate_impact_methods"] = fx_rate_impact_methods
                    specification_path.write_text(
                        yaml.safe_dump(configuration),
                        encoding="utf-8",
                    )

                    with self.assertRaises(PpaError) as context:
                        PerformanceComparison(
                            AuditSpecification(specification_path)
                        )

                    self.assertIn(expected_message, str(context.exception))

    def test_evidence_only_impact_methods_reject_malformed_yaml(self) -> None:
        """Evidence-only impact method YAML must use the supported contract."""
        scenarios = [
            ("not-a-mapping", "evidence_only_impact_methods must be a mapping"),
            ({"unsupported": {"method": "evidence_only"}}, "unsupported"),
            ({"holdings": "quantity"}, "holdings must be a mapping"),
            ({"holdings": {}}, "holdings is missing required keys"),
            (
                {"holdings": {"method": "unsupported", "source_fields": ["quantity"]}},
                "holdings.method must be",
            ),
            (
                {"holdings": {"method": "evidence_only", "source_fields": []}},
                "source_fields must be a non-empty list",
            ),
            (
                {"holdings": {"method": "evidence_only", "source_fields": [1]}},
                "source_fields values must be strings",
            ),
            (
                {
                    "holdings": {
                        "method": "evidence_only",
                        "source_fields": ["unsupported"],
                    }
                },
                "contains unsupported fields",
            ),
        ]

        for evidence_only_impact_methods, expected_message in scenarios:
            with self.subTest(
                evidence_only_impact_methods=evidence_only_impact_methods
            ):
                with tempfile.TemporaryDirectory() as temp_dir:
                    specification_path = _write_holding_period_specification(
                        Path(temp_dir)
                    )
                    configuration = yaml.safe_load(
                        specification_path.read_text(encoding="utf-8")
                    )
                    configuration["evidence_only_impact_methods"] = (
                        evidence_only_impact_methods
                    )
                    specification_path.write_text(
                        yaml.safe_dump(configuration),
                        encoding="utf-8",
                    )

                    with self.assertRaises(PpaError) as context:
                        PerformanceComparison(
                            AuditSpecification(specification_path)
                        )

                    self.assertIn(expected_message, str(context.exception))

    def test_transaction_outside_period_does_not_get_denominator(self) -> None:
        """Out-of-period transaction rows do not inherit a return denominator."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_transaction_outside_period_specification(
                Path(temp_dir)
            )
            specification = AuditSpecification(specification_path)

            findings = PerformanceComparison(specification).compare_transactions()
            finding_dicts = [finding.to_dict() for finding in findings]
            amount_finding = next(
                finding
                for finding in finding_dicts
                if finding[FINDING_CODE] == PC_TXN_AMT
            )

            self.assertIsNone(amount_finding[FROM_DATE])
            self.assertIsNone(amount_finding[THRU_DATE])
            self.assertIsNone(amount_finding[RETURN_DENOMINATOR])
            self.assertEqual(amount_finding[CASH_FLOW_SIGN], "negative")
            self.assertEqual(amount_finding[PERFORMANCE_FLOW_SIGN], "performance")
            self.assertEqual(amount_finding[TRANSACTION_SEMANTICS_SOURCE], "source")

    def test_transaction_outside_period_uses_single_changed_period_fallback(self) -> None:
        """A single changed portfolio period can supply transaction attribution context."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_transaction_changed_period_fallback_specification(
                Path(temp_dir)
            )
            specification = AuditSpecification(specification_path)

            comparison = PerformanceComparison(specification)
            findings = comparison.compare_transactions()
            amount_finding = next(
                finding.to_dict()
                for finding in findings
                if finding.to_dict()[FINDING_CODE] == PC_TXN_AMT
            )
            candidates = portfolio_period_contribution_candidates(
                findings_to_polars(comparison.compare())
            )
            transaction_candidates = candidates.filter(
                pl.col("impact_basis") == "transaction_performance_amount"
            )

            self.assertEqual(str(amount_finding[FROM_DATE]), "2025-05-30")
            self.assertEqual(str(amount_finding[THRU_DATE]), "2025-05-30")
            self.assertEqual(amount_finding[RETURN_DENOMINATOR], 1000.0)
            self.assertEqual(
                amount_finding[TRANSACTION_IMPACT_POLICY],
                TRANSACTION_IMPACT_POLICY_PERFORMANCE_AMOUNT_DELTA,
            )
            self.assertEqual(transaction_candidates.height, 1)
            self.assertAlmostEqual(
                transaction_candidates[ESTIMATED_RETURN_IMPACT][0],
                0.01,
            )

    def test_transaction_outside_period_does_not_guess_ambiguous_changed_period(
        self,
    ) -> None:
        """Transaction period fallback is withheld when a portfolio has multiple targets."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_transaction_changed_period_fallback_specification(
                Path(temp_dir),
                ambiguous=True,
            )
            specification = AuditSpecification(specification_path)

            comparison = PerformanceComparison(specification)
            findings = comparison.compare_transactions()
            amount_finding = next(
                finding.to_dict()
                for finding in findings
                if finding.to_dict()[FINDING_CODE] == PC_TXN_AMT
            )
            candidates = portfolio_period_contribution_candidates(
                findings_to_polars(comparison.compare())
            )
            transaction_candidates = candidates.filter(
                pl.col("impact_basis") == "transaction_performance_amount"
            )

            self.assertIsNone(amount_finding[FROM_DATE])
            self.assertIsNone(amount_finding[THRU_DATE])
            self.assertIsNone(amount_finding[RETURN_DENOMINATOR])
            self.assertEqual(transaction_candidates.height, 0)

    def test_transaction_singleton_fallback_matches_amount_change(self) -> None:
        """Exact singleton no-ID transaction keys can compare changed fields."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_transaction_fallback_specification(Path(temp_dir))
            specification = AuditSpecification(specification_path)

            findings = PerformanceComparison(specification).compare_transactions()
            finding_dicts = [finding.to_dict() for finding in findings]
            finding_codes = [finding[FINDING_CODE] for finding in finding_dicts]
            amount_finding = next(
                finding
                for finding in finding_dicts
                if finding[FINDING_CODE] == PC_TXN_AMT
            )

            self.assertNotIn(PC_TXN_ADD, finding_codes)
            self.assertNotIn(PC_TXN_DROP, finding_codes)
            self.assertEqual(finding_codes.count(PC_TXN_AMT), 1)
            self.assertEqual(
                amount_finding[TRANSACTION_MATCH_STATUS],
                TRANSACTION_MATCH_STATUS_SINGLETON_FALLBACK_MATCH,
            )
            self.assertAlmostEqual(cast(float, amount_finding[DELTA_B_MINUS_A]), 10.0)

    def test_no_id_transaction_date_move_across_periods_is_timing_add_drop(
        self,
    ) -> None:
        """No-ID date moves remain add/drop evidence with period timing impact."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_no_id_transaction_date_move_specification(
                Path(temp_dir)
            )
            specification = AuditSpecification(specification_path)

            findings = PerformanceComparison(specification).compare_transactions()
            finding_dicts = [finding.to_dict() for finding in findings]
            finding_by_code = {
                finding[FINDING_CODE]: finding
                for finding in finding_dicts
                if finding[FINDING_CODE] in {PC_TXN_ADD, PC_TXN_DROP}
            }

            self.assertEqual(set(finding_by_code), {PC_TXN_ADD, PC_TXN_DROP})
            self.assertNotIn(
                PC_TXN_AMT,
                {finding[FINDING_CODE] for finding in finding_dicts},
            )

            dropped = finding_by_code[PC_TXN_DROP]
            added = finding_by_code[PC_TXN_ADD]
            self.assertEqual(
                dropped[TRANSACTION_MATCH_STATUS],
                TRANSACTION_MATCH_STATUS_MISSING_FROM_SNAPSHOT_B,
            )
            self.assertEqual(
                added[TRANSACTION_MATCH_STATUS],
                TRANSACTION_MATCH_STATUS_ADDED_IN_SNAPSHOT_B,
            )
            self.assertEqual(str(dropped[FROM_DATE]), "2025-05-01")
            self.assertEqual(str(dropped[THRU_DATE]), "2025-05-31")
            self.assertEqual(str(dropped["input_date"]), "2025-05-31")
            self.assertEqual(str(added[FROM_DATE]), "2025-06-01")
            self.assertEqual(str(added[THRU_DATE]), "2025-06-30")
            self.assertEqual(str(added["input_date"]), "2025-06-01")
            self.assertEqual(
                dropped[TRANSACTION_IMPACT_DIAGNOSTIC],
                "modified_dietz cross-check estimate",
            )
            self.assertEqual(
                added[TRANSACTION_IMPACT_DIAGNOSTIC],
                "modified_dietz cross-check estimate",
            )
            self.assertAlmostEqual(
                cast(float, dropped[TRANSACTION_IMPACT_DIAGNOSTIC_ESTIMATE]),
                1000.0 * (1.0 / 31.0) / 10000.0,
            )
            self.assertAlmostEqual(
                cast(float, added[TRANSACTION_IMPACT_DIAGNOSTIC_ESTIMATE]),
                -1000.0 / 10000.0,
            )

    def test_duplicate_singleton_candidates_are_not_fallback_matched(self) -> None:
        """Duplicate singleton candidates stay unpaired instead of guessing linkage."""
        for snapshot_a_count, snapshot_b_count in ((2, 1), (1, 2)):
            with self.subTest(a=snapshot_a_count, b=snapshot_b_count):
                with tempfile.TemporaryDirectory() as temp_dir:
                    specification_path = (
                        _write_transaction_singleton_duplicate_specification(
                            Path(temp_dir),
                            snapshot_a_count=snapshot_a_count,
                            snapshot_b_count=snapshot_b_count,
                        )
                    )
                    specification = AuditSpecification(specification_path)

                    findings = PerformanceComparison(specification).compare_transactions()
                    finding_dicts = [finding.to_dict() for finding in findings]
                    finding_codes = [finding[FINDING_CODE] for finding in finding_dicts]
                    ambiguity = next(
                        finding
                        for finding in finding_dicts
                        if finding[FINDING_CODE] == PC_TXN_AMBIG
                    )

                    self.assertNotIn(PC_TXN_AMT, finding_codes)
                    self.assertEqual(
                        ambiguity[TRANSACTION_MATCH_STATUS],
                        TRANSACTION_MATCH_STATUS_AMBIGUOUS_FALLBACK_MATCH,
                    )
                    self.assertEqual(ambiguity["snapshot_a_value"], snapshot_a_count)
                    self.assertEqual(ambiguity["snapshot_b_value"], snapshot_b_count)

    def test_singleton_fallback_keeps_security_identifiers_case_sensitive(self) -> None:
        """Security identifiers that differ only by case are not fallback matched."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_transaction_case_sensitive_singleton_specification(
                Path(temp_dir),
                snapshot_a_security="AAPL",
                snapshot_b_security="aapl",
            )
            specification = AuditSpecification(specification_path)

            findings = PerformanceComparison(specification).compare_transactions()
            finding_codes = [finding.to_dict()[FINDING_CODE] for finding in findings]

            self.assertEqual(finding_codes.count(PC_TXN_ADD), 1)
            self.assertEqual(finding_codes.count(PC_TXN_DROP), 1)
            self.assertNotIn(PC_TXN_AMT, finding_codes)

    def test_singleton_fallback_keeps_transaction_codes_case_sensitive(self) -> None:
        """Native transaction codes that differ only by case are not fallback matched."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_transaction_case_sensitive_singleton_specification(
                Path(temp_dir),
                snapshot_a_code="BUY",
                snapshot_b_code="buy",
            )
            specification = AuditSpecification(specification_path)

            findings = PerformanceComparison(specification).compare_transactions()
            finding_codes = [finding.to_dict()[FINDING_CODE] for finding in findings]

            self.assertEqual(finding_codes.count(PC_TXN_ADD), 1)
            self.assertEqual(finding_codes.count(PC_TXN_DROP), 1)
            self.assertNotIn(PC_TXN_AMT, finding_codes)

    def test_duplicate_transaction_fallback_keys_are_ambiguity_diagnostics(self) -> None:
        """Duplicate fallback keys are legitimate but unsafe to pair as edits."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_duplicate_transaction_fallback_specification(
                Path(temp_dir)
            )
            specification = AuditSpecification(specification_path)

            findings = PerformanceComparison(specification).compare_transactions()
            finding_dicts = [finding.to_dict() for finding in findings]

            self.assertEqual(len(finding_dicts), 1)
            ambiguity = finding_dicts[0]
            self.assertEqual(ambiguity[FINDING_CODE], PC_TXN_AMBIG)
            self.assertEqual(
                ambiguity[TRANSACTION_MATCH_STATUS],
                TRANSACTION_MATCH_STATUS_AMBIGUOUS_FALLBACK_MATCH,
            )
            self.assertEqual(ambiguity["snapshot_a_value"], 2)
            self.assertEqual(ambiguity["snapshot_b_value"], 2)
            self.assertIn("ambiguous", str(ambiguity["message"]))

    def test_duplicate_transaction_fallback_diagnostics_are_sorted(self) -> None:
        """Ambiguity diagnostics remain deterministic across grouped input order."""
        key_columns = (pc_cols.PORTFOLIO_ID, pc_cols.SECURITY_ID)
        snapshot_a = pl.DataFrame(
            {
                pc_cols.PORTFOLIO_ID: ["PORT_B", "PORT_A", "PORT_B", "PORT_A"],
                pc_cols.SECURITY_ID: ["SEC2", "SEC1", "SEC2", "SEC1"],
            }
        )
        snapshot_b = snapshot_a.reverse()

        duplicate_keys = PerformanceComparison._duplicate_transaction_fallback_keys(
            snapshot_a,
            snapshot_b,
            key_columns,
        )

        self.assertEqual(
            duplicate_keys.select(key_columns).rows(),
            [("PORT_A", "SEC1"), ("PORT_B", "SEC2")],
        )

    def test_duplicate_transaction_ids_still_fail_loudly(self) -> None:
        """Stable transaction IDs remain unique comparison keys."""
        with tempfile.TemporaryDirectory() as temp_dir:
            specification_path = _write_transaction_period_specification(Path(temp_dir))
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                transaction_path = Path(temp_dir) / snapshot_name / "transactions.csv"
                transaction_path.write_text(
                    "TRANSACTION_ID,PORT,SEC,TRADE_DATE,SETTLE_DATE,TRAN,QTY,"
                    "PRICE,AMOUNT,CASH_FLOW_SIGN,PERFORMANCE_FLOW_SIGN\n"
                    "TXN1,PORT_A,AAPL,2025-05-15,2025-05-16,BUY,1,100.00,"
                    "100.00,cash out,external\n"
                    "TXN1,PORT_A,AAPL,2025-05-15,2025-05-16,BUY,1,100.00,"
                    "100.00,cash out,external\n",
                    encoding="utf-8",
                )
            specification = AuditSpecification(specification_path)

            with self.assertRaises(PpaError):
                PerformanceComparison(specification).compare_transactions()


if __name__ == "__main__":
    unittest.main()
