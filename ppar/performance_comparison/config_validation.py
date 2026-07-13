"""Validate performance comparison YAML configurations."""

from __future__ import annotations

# Python imports
import argparse
from pathlib import Path
import sys

# Project imports
from ppar.errors import PpaError
from ppar.performance_comparison import schema as _pc_cols
from ppar.performance_comparison.compare import PerformanceComparison
from ppar.performance_comparison.extract_contract import extract_contract_summary
from ppar.performance_comparison.findings import findings_to_polars
from ppar.performance_comparison.runner import validate_yaml_setup_complete
from ppar.performance_comparison.specification import PerformanceComparisonSpecification
from ppar.performance_comparison.source_data_contract import (
    comparison_required_dataset_names,
    source_data_contract_summary,
)
from ppar.performance_comparison.transaction_summary import (
    format_codes,
    format_semantics_source_counts,
    transaction_rule_codes,
    transaction_semantics_summary,
)
from ppar.performance_comparison.transactions import TransactionsLoader

__all__ = [
    "main",
    "validate_config",
]


def main(argv: list[str] | None = None) -> int:
    """Validate a performance comparison YAML file from the command line.

    Args:
        argv: Optional command-line arguments excluding the executable name.

    Returns:
        Process exit code. ``0`` means validation passed; ``1`` means the
        configuration has validation issues.
    """
    args = _argument_parser().parse_args(argv)
    try:
        summary = validate_config(
            args.comparison_path,
            require_complete_yaml_setup=not args.allow_incomplete_yaml,
        )
    except PpaError as error:
        print(f"Config validation failed: {args.comparison_path}", file=sys.stderr)
        print(f"- {error}", file=sys.stderr)
        return 1

    print(f"Config validation passed: {args.comparison_path}")
    print(f"Snapshot A: {summary['snapshot_a']}")
    print(f"Snapshot B: {summary['snapshot_b']}")
    print(f"Configured datasets: {summary['dataset_names']}")
    print(f"Minimum required datasets: {summary['minimum_required_datasets']}")
    print(f"Required source-data columns: {summary['required_source_data_columns']}")
    print(f"Missing optional files: {summary['missing_optional_files']}")
    print(f"Contribution impact methods: {summary['contribution_impact_methods']}")
    print(f"Holding impact methods: {summary['holding_impact_methods']}")
    print(f"Price impact methods: {summary['price_impact_methods']}")
    print(f"FX rate impact methods: {summary['fx_rate_impact_methods']}")
    print(f"Evidence-only impact methods: {summary['evidence_only_impact_methods']}")
    print(f"Transaction rules configured: {summary['transaction_rule_count']}")
    print(f"Transaction impact methods: {summary['transaction_impact_methods']}")
    print(f"Transaction files checked: {summary['transaction_files_checked']}")
    print(f"Extract contract: {summary['extract_contract']}")
    print(
        "Enforce ambiguous Axys/APX flows: "
        f"{summary['enforce_ambiguous_axys_flows']}"
    )
    print(
        "Required transaction context columns: "
        f"{summary['required_transaction_context_columns']}"
    )
    print(
        "Report-bundle source context: manifest.json records the comparison path, "
        "extract contract, and transaction semantics summary."
    )
    print(f"Transaction codes observed: {summary['transaction_codes_observed']}")
    print(
        "Transaction codes without YAML rules: "
        f"{summary['transaction_codes_without_yaml_rules']}"
    )
    print(f"Transaction semantics sources: {summary['transaction_semantics_sources']}")
    return 0


def validate_config(
    comparison_path: Path,
    *,
    require_complete_yaml_setup: bool = True,
) -> dict[str, object]:
    """Validate one comparison YAML file and return a compact summary.

    Args:
        comparison_path: Path to a performance comparison YAML file.
        require_complete_yaml_setup: Whether to reject changed source-data
            fields that lack additive, evidence-only, or suppression YAML.

    Returns:
        Summary fields for the resolved snapshots, configured datasets, and
        transaction files checked.

    Raises:
        PpaError: If the comparison specification, configured files,
            transaction rules, or transaction impact methods are invalid.
    """
    return _validate_config(
        comparison_path,
        require_complete_yaml_setup=require_complete_yaml_setup,
    )


def _validate_config(
    comparison_path: Path,
    *,
    require_complete_yaml_setup: bool,
) -> dict[str, object]:
    """Validate one comparison YAML file with explicit YAML setup strictness."""
    specification = PerformanceComparisonSpecification(comparison_path)
    comparison = PerformanceComparison(specification)
    findings = findings_to_polars(comparison.compare())
    if require_complete_yaml_setup:
        validate_yaml_setup_complete(findings)
    transaction_preview = _validate_transactions(specification)
    contract_summary = extract_contract_summary(
        specification.values,
        specification_path=specification.path,
    )
    dataset_names = ", ".join(sorted(specification.files))
    minimum_contract = source_data_contract_summary(
        comparison_level=specification.comparison_level,
        include_reconstruction_sources=(
            specification.portfolio_return_reconstruction is not None
            or specification.security_return_reconstruction is not None
        ),
        include_security_performance=(
            specification.security_return_reconstruction is not None
        ),
    )
    return {
        "snapshot_a": specification.snapshot_a.path,
        "snapshot_b": specification.snapshot_b.path,
        "dataset_names": dataset_names,
        "minimum_required_datasets": ", ".join(
            comparison_required_dataset_names(specification)
        ),
        "required_source_data_columns": minimum_contract["required_columns"],
        "missing_optional_files": _missing_optional_files(specification),
        "contribution_impact_methods": _contribution_impact_methods(specification),
        "holding_impact_methods": _holding_impact_methods(specification),
        "price_impact_methods": _price_impact_methods(specification),
        "fx_rate_impact_methods": _fx_rate_impact_methods(specification),
        "evidence_only_impact_methods": _evidence_only_impact_methods(specification),
        "transaction_rule_count": _transaction_rule_count(specification),
        "transaction_impact_methods": _transaction_impact_methods(specification),
        "extract_contract": contract_summary["path"],
        "enforce_ambiguous_axys_flows": (
            contract_summary["enforce_ambiguous_axys_flows"]
        ),
        "required_transaction_context_columns": format_codes(
            contract_summary["required_transaction_context_columns"]
        ),
        "transaction_files_checked": transaction_preview["files_checked"],
        "transaction_codes_observed": transaction_preview["codes_observed"],
        "transaction_codes_without_yaml_rules": (
            transaction_preview["codes_without_yaml_rules"]
        ),
        "transaction_semantics_sources": transaction_preview["semantics_sources"],
        "transaction_semantics": transaction_preview["summary"],
        "extract_contract_summary": contract_summary,
    }


def _validate_transactions(
    specification: PerformanceComparisonSpecification,
) -> dict[str, object]:
    """Validate configured transaction files and return preview fields."""
    if _pc_cols.TRANSACTIONS not in specification.files:
        return {
            "files_checked": 0,
            "codes_observed": "none",
            "codes_without_yaml_rules": "none",
            "semantics_sources": "none",
            "summary": transaction_semantics_summary([]),
        }
    loader = TransactionsLoader(specification)
    checked = 0
    frames = []
    for snapshot_key in ("a", "b"):
        frame = loader.load(snapshot_key)
        if frame is None:
            continue
        checked += 1
        frames.append(frame)
    summary = transaction_semantics_summary(
        frames,
        rule_codes=transaction_rule_codes(specification.values),
    )
    return {
        "files_checked": checked,
        "codes_observed": format_codes(summary["observed_codes"]),
        "codes_without_yaml_rules": format_codes(summary["codes_without_yaml_rules"]),
        "semantics_sources": format_semantics_source_counts(
            summary["semantics_source_counts"]
        ),
        "summary": summary,
    }


def _missing_optional_files(
    specification: PerformanceComparisonSpecification,
) -> str:
    """Return a readable list of configured optional files that are absent."""
    missing_files: list[str] = []
    for comparison_file in specification.files.values():
        if comparison_file.required:
            continue
        for snapshot_key, file_path in (
            ("a", comparison_file.snapshot_a_path),
            ("b", comparison_file.snapshot_b_path),
        ):
            if not file_path.exists():
                missing_files.append(f"{comparison_file.name}:{snapshot_key}")
    return ", ".join(sorted(missing_files)) if missing_files else "none"


def _transaction_rule_count(
    specification: PerformanceComparisonSpecification,
) -> int:
    """Return the number of configured transaction code rules."""
    rules_value = specification.values.get("transaction_rules", {})
    return len(rules_value) if isinstance(rules_value, dict) else 0


def _transaction_impact_methods(
    specification: PerformanceComparisonSpecification,
) -> str:
    """Return configured transaction impact method keys."""
    methods_value = specification.values.get("transaction_impact_methods", {})
    if not isinstance(methods_value, dict) or not methods_value:
        return "none"
    return ", ".join(sorted(str(key) for key in methods_value))


def _contribution_impact_methods(
    specification: PerformanceComparisonSpecification,
) -> str:
    """Return configured contribution impact method keys."""
    methods_value = specification.values.get("contribution_impact_methods", {})
    if not isinstance(methods_value, dict) or not methods_value:
        return "none"
    return ", ".join(sorted(str(key) for key in methods_value))


def _holding_impact_methods(
    specification: PerformanceComparisonSpecification,
) -> str:
    """Return configured holding impact method keys."""
    methods_value = specification.values.get("holding_impact_methods", {})
    if not isinstance(methods_value, dict) or not methods_value:
        return "none"
    return ", ".join(sorted(str(key) for key in methods_value))


def _price_impact_methods(
    specification: PerformanceComparisonSpecification,
) -> str:
    """Return configured price impact method keys."""
    methods_value = specification.values.get("price_impact_methods", {})
    if not isinstance(methods_value, dict) or not methods_value:
        return "none"
    return ", ".join(sorted(str(key) for key in methods_value))


def _fx_rate_impact_methods(
    specification: PerformanceComparisonSpecification,
) -> str:
    """Return configured FX rate impact method keys."""
    methods_value = specification.values.get("fx_rate_impact_methods", {})
    if not isinstance(methods_value, dict) or not methods_value:
        return "none"
    return ", ".join(sorted(str(key) for key in methods_value))


def _evidence_only_impact_methods(
    specification: PerformanceComparisonSpecification,
) -> str:
    """Return configured evidence-only impact method keys."""
    methods_value = specification.values.get("evidence_only_impact_methods", {})
    if not isinstance(methods_value, dict) or not methods_value:
        return "none"
    return ", ".join(sorted(str(key) for key in methods_value))


def _argument_parser() -> argparse.ArgumentParser:
    """Return the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Validate a performance comparison YAML configuration.",
    )
    parser.add_argument(
        "comparison_path",
        type=Path,
        help="Path to a performance comparison YAML file.",
    )
    parser.add_argument(
        "--allow-incomplete-yaml",
        action="store_true",
        help=(
            "Validate file/schema/contract shape even when changed source-data "
            "fields are not explicitly classified by additive, evidence-only, "
            "or suppression YAML."
        ),
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
