"""Check packaged Audit demo data.

This command delegates to
``scripts/operational_demo_data/rebuild_audit_demo_data.py`` so
the generated performance files and their audit guardrails share one
implementation.
"""

from __future__ import annotations

# Python imports
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Project imports  # noqa: E402
from scripts.operational_demo_data.rebuild_audit_demo_data import (
    _DEFAULT_AXYS_APX_DIRECTORY,
    _DEFAULT_COMPARISON_PATH,
    _DEFAULT_HOLDING_SCENARIOS_PATH,
    _DEFAULT_TRANSACTION_SCENARIOS_PATH,
    audit_demo_data,
)


def main() -> int:
    """Run the packaged Audit demo-data check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "audit_path",
        nargs="?",
        type=Path,
        default=_DEFAULT_COMPARISON_PATH,
        help="Audit YAML to check.",
    )
    parser.add_argument(
        "--axys-directory",
        type=Path,
        default=_DEFAULT_AXYS_APX_DIRECTORY,
        help="Directory containing snapshot_a and snapshot_b.",
    )
    parser.add_argument(
        "--holding-scenarios-path",
        type=Path,
        default=_DEFAULT_HOLDING_SCENARIOS_PATH,
        help="CSV file containing scenario adjustments for snapshot B holdings.",
    )
    parser.add_argument(
        "--transaction-scenarios-path",
        type=Path,
        default=_DEFAULT_TRANSACTION_SCENARIOS_PATH,
        help="CSV file containing scenario adjustments for snapshot B transactions.",
    )
    args = parser.parse_args()

    issues = audit_demo_data(
        axys_directory=args.axys_directory,
        comparison_path=args.audit_path,
        holding_scenarios_path=args.holding_scenarios_path,
        transaction_scenarios_path=args.transaction_scenarios_path,
    )
    print(json.dumps([asdict(issue) for issue in issues], indent=2))
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
