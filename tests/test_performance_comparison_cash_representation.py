"""Tests for the single holdings-based cash representation contract."""

from __future__ import annotations

# Python imports
from pathlib import Path
import tempfile
import unittest

# Third-party imports
import yaml

# Test imports
from tests import test_utilities as test_util

# Project imports
from ppar.errors import PpaError
from ppar.audit import schema as pc_cols
from ppar.audit.source_data_contract import source_data_contract
from ppar.audit.specification import AuditSpecification

_AXYS_APX_STARTER = Path(
    "ppar/setup_templates/axys_apx_audit/"
    "axys_apx_audit.yaml"
)


def _required_yaml_settings() -> dict[str, object]:
    """Return explicit settings unrelated to the cash-contract defect."""
    return {
        "comparison": {"level": "portfolio"},
        "extract_contract": {
            "enforce_ambiguous_axys_flows": True,
            "transaction_semantics_case": "legacy_case_insensitive",
        },
        "tolerances": {
            "return": 0.000001,
            "contribution": 0.000001,
            "weight": 0.000001,
            "market_value": 0.01,
            "quantity": 0.000001,
            "price": 0.000001,
            "split_factor": 0.00000001,
            "fx_rate": 0.00000001,
        },
    }


class TestCashRepresentationContract(unittest.TestCase):
    """Verify cash balances have exactly one normalized dataset representation."""

    def test_source_contract_has_no_standalone_cash_dataset(self) -> None:
        """The normalized source contract represents cash through holdings."""
        dataset_names = {dataset.name for dataset in source_data_contract()}

        self.assertNotIn("cash", dataset_names)
        self.assertFalse(hasattr(pc_cols, "CASH"))
        self.assertFalse(hasattr(pc_cols, "CASH_BALANCE"))

    def test_files_cash_is_rejected(self) -> None:
        """Legacy files.cash configuration fails instead of being ignored."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                snapshot_path = directory / snapshot_name
                snapshot_path.mkdir()
                (snapshot_path / "portperf.csv").write_text(
                    "PORTFOLIO_CODE,FROM_DATE,THRU_DATE,PORT_RETURN\n"
                    "P1,2025-01-01,2025-01-31,0.01\n",
                    encoding="utf-8",
                )
            configuration = {
                **_required_yaml_settings(),
                "snapshots": {
                    "a": {"path": "snapshot_a"},
                    "b": {"path": "snapshot_b"},
                },
                "files": {
                    "portfolio_performance": "portperf.csv",
                    "cash": "cash.csv",
                },
            }
            path = directory / "ppar.yaml"
            test_util.write_audit_test_yaml(path, configuration)

            with self.assertRaisesRegex(PpaError, "files.cash is not supported"):
                AuditSpecification(path)

    def test_axys_apx_starter_uses_holdings_for_cash(self) -> None:
        """The user-facing starter cannot quietly restore files.cash."""
        configuration = yaml.safe_load(_AXYS_APX_STARTER.read_text(encoding="utf-8"))

        self.assertNotIn("cash", configuration["files"])
        self.assertEqual(configuration["files"]["holdings"], "holdings.csv")

    def test_legacy_cash_impact_methods_are_rejected(self) -> None:
        """Legacy cash impact policy cannot survive as ignored configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            for snapshot_name in ("snapshot_a", "snapshot_b"):
                snapshot_path = directory / snapshot_name
                snapshot_path.mkdir()
                (snapshot_path / "portperf.csv").write_text(
                    "PORTFOLIO_CODE,FROM_DATE,THRU_DATE,PORT_RETURN\n"
                    "P1,2025-01-01,2025-01-31,0.01\n",
                    encoding="utf-8",
                )
            configuration = {
                **_required_yaml_settings(),
                "snapshots": {
                    "a": {"path": "snapshot_a"},
                    "b": {"path": "snapshot_b"},
                },
                "files": {"portfolio_performance": "portperf.csv"},
                "cash_impact_methods": {},
            }
            path = directory / "ppar.yaml"
            test_util.write_audit_test_yaml(path, configuration)

            with self.assertRaisesRegex(PpaError, "represent cash as holdings"):
                AuditSpecification(path)


if __name__ == "__main__":
    unittest.main()
