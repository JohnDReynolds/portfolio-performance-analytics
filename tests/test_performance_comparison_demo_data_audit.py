"""Tests for packaged performance-comparison demo data accounting guardrails."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_AUDIT_SCRIPT_PATH = _REPO_ROOT / "scripts" / "audit_performance_comparison_demo_data.py"


def _load_audit_module():
    """Load the demo-data audit script as a test module."""
    spec = importlib.util.spec_from_file_location(
        "audit_performance_comparison_demo_data",
        _AUDIT_SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("Could not load performance-comparison demo-data audit.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestPerformanceComparisonDemoDataAudit(unittest.TestCase):
    """Verify packaged demo data remains internally consistent."""

    def test_packaged_performance_comparison_demo_data_foots(self) -> None:
        """Packaged demo data has no accidental accounting or residual issues."""
        audit_module = _load_audit_module()

        issues = audit_module.audit_demo_data()

        self.assertEqual(issues, [])


if __name__ == "__main__":
    unittest.main()
