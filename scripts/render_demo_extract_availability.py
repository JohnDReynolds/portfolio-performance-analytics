"""Render the packaged Axys/APX demo extract-availability contract from YAML."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import yaml


_REPO_ROOT: Final = Path(__file__).resolve().parents[1]
_DEFAULT_CONTRACT_PATH: Final = (
    _REPO_ROOT
    / "ppar"
    / "demos"
    / "data"
    / "axysapx_performance_comparison"
    / "demo_extract_availability.yaml"
)
_DEFAULT_OUTPUT_PATH: Final = (
    _REPO_ROOT
    / "docs"
    / "axys-apx-reference"
    / "contracts"
    / "demo_extract_availability.md"
)
_DATASET_LABELS: Final[dict[str, str]] = {
    "holdings.csv": "holdings",
    "portperf.csv": "portfolio performance",
    "sec_ref.csv": "security master",
    "secperf.csv": "security performance",
    "transactions.csv": "transactions",
}
_CONFIDENCE_DISPLAY: Final[dict[str, str]] = {
    "high": "High",
    "medium_high": "Medium / High",
    "medium": "Medium",
    "medium_unknown": "Medium / Unknown",
    "low_medium": "Low / Medium",
    "low": "Low",
    "unknown": "Unknown",
    "imex_preferred": "IMEX preferred",
    "rep_preferred": "REP preferred",
    "imex_or_rep": "IMEX or REP",
    "imex_then_rep_cross_check": "IMEX then REP cross-check",
    "local_discovery_required": "Local discovery required",
}
_PRACTICAL_GUIDANCE_ROWS: Final[tuple[tuple[str, str, str], ...]] = (
    (
        "Holdings values for return reconstruction",
        "IMEX or REP",
        "Holdings/positions are relatively well-supported by both export and "
        "appraisal/report paths.",
    ),
    (
        "Transaction amount, quantity, price, and commission",
        "IMEX first, REP as cross-check",
        "Transaction import/export evidence is strong for core transaction data; "
        "REP can validate user-visible report output.",
    ),
    (
        "Ambiguous external-flow classification for `li`, `lo`, `dp`, `wd`-style rows",
        "IMEX only if source/destination and special-security context is exposed; "
        "otherwise REP/custom report or another source is required",
        "Code alone is not enough for all cases. The required context fields must "
        "be proven locally.",
    ),
    (
        "Portfolio and security reported returns",
        "REP/report extract preferred",
        "The local reference corpus treats performance IMEX objects and fields as "
        "Unknown/validate-locally.",
    ),
    (
        "Security classifications",
        "IMEX or security-master report extract",
        "Asset class, sector, country, currency, and industry are plausible, but "
        "history/timing must be validated.",
    ),
)
_LOCAL_VALIDATION_STEPS: Final[tuple[str, ...]] = (
    "IMEX profile or REP report name.",
    "Axys/APX version and client/reporting tool version.",
    "Exact field names and aliases used in the export.",
    "Report parameters, date basis, portfolio list, currency basis, and gross/net "
    "return basis.",
    "A paired report/export sample for at least one portfolio and period.",
    "Transaction examples for `li`, `lo`, `dp`, and `wd` with source/destination "
    "and special-security context.",
    "Evidence whether performance fields are stored values, report-calculated "
    "values, or export-calculated values.",
)
_RELATED_REFERENCES: Final[tuple[str, ...]] = (
    "[Chapter_05_Transactions.md](../reference/Chapter_05_Transactions.md)",
    "[Chapter_06_Holdings.md](../reference/Chapter_06_Holdings.md)",
    "[Chapter_10_Performance.md](../reference/Chapter_10_Performance.md)",
    "[Chapter_12_Imex.md](../reference/Chapter_12_Imex.md)",
    "[Chapter_13_Rep.md](../reference/Chapter_13_Rep.md)",
    "[Chapter_15_Data_Dictionary.md](../reference/Chapter_15_Data_Dictionary.md)",
    "[axysapx_common_core_export.md](../../axysapx_common_core_export.md)",
    "[performance_comparison_demo_source_contract.md]"
    "(../../performance_comparison_demo_source_contract.md)",
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the contract renderer.

    Args:
        argv: Optional command-line arguments. Defaults to ``sys.argv`` when
            omitted by ``argparse``.

    Returns:
        Process-style exit code. ``0`` means the contract was written or already
        current. ``1`` means ``--check`` found stale output.
    """
    args = _parse_args(argv)
    contract = _load_contract(args.contract)
    rendered = render_markdown(contract)

    if args.check:
        current = args.output.read_text(encoding="utf-8")
        if current != rendered:
            print(f"{args.output} is stale; rerun {Path(__file__).as_posix()}.")
            return 1
        return 0

    args.output.write_text(rendered, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


def render_markdown(contract: Mapping[str, Any]) -> str:
    """Return rendered contract markdown from the YAML contract."""
    datasets = _mapping(contract, "datasets")
    lines = [
        "# Demo Extract Availability Contract",
        "",
        "Repository: AXYS / APX Reference Repository",
        "Scope: `ppar/setup_templates/axysapx_performance_comparison/snapshot_a` and",
        "`ppar/setup_templates/axysapx_performance_comparison/snapshot_b`",
        "Status: Draft confidence matrix generated from the packaged YAML contract.",
        "",
        "<!-- GENERATED FROM ppar/setup_templates/axysapx_performance_comparison/demo_extract_availability.yaml. -->",
        "<!-- Run scripts/render_demo_extract_availability.py after editing the YAML. -->",
        "",
        "---",
        "",
        "## Purpose",
        "",
        "This contract estimates how likely each packaged Axys/APX demo dataset and "
        "column is to be obtainable from an Axys/APX installation through IMEX "
        "and/or REP-style report extracts.",
        "",
        "The machine-readable source of truth is "
        "`ppar/setup_templates/axysapx_performance_comparison/demo_extract_availability.yaml`. Tests verify "
        "that the YAML covers every packaged comparison demo CSV header and "
        "that this contract is current.",
        "",
        "The packaged demo files are normalized demo extracts. They are not "
        "official Axys/APX schemas, not universal IMEX profiles, and not claims "
        "that every Axys/APX site can export every field with these exact names.",
        "",
        "The two packaged snapshots currently use the same file layouts:",
        "",
        *[f"- `{dataset_name}`" for dataset_name in datasets],
        "",
        "## Confidence Labels",
        "",
        "| Label | Meaning |",
        "|---|---|",
    ]
    for label, meaning in _mapping(contract, "confidence_labels").items():
        lines.append(f"| {_confidence_label(label)} | {_escape_table_text(meaning)} |")

    lines.extend(
        [
            "",
            "## Evidence Boundaries",
            "",
            "The confidence ratings below rely on these local reference conclusions:",
            "",
            "- `Chapter_12_Imex.md` establishes IMEX / Import-Export as an Axys/APX "
            "import/export mechanism, but says the complete native object and "
            "field dictionaries are not available in the supplied source "
            "material.",
            "- `Chapter_12_Imex.md` documents Axys/APX CI import workflows for "
            "transactions, positions, prices, security information, Trade "
            "Blotter context, and selected transaction translation fields.",
            "- `Chapter_13_Rep.md` establishes REP / RepLang / REP32 as a "
            "report-driven extraction path and distinguishes it from IMEX.",
            "- `Chapter_10_Performance.md` says `portperf` and `secperf` "
            "should be treated as normalized/local names unless a live IMEX "
            "object, report output, or vendor manual confirms native names.",
            "- `docs/axysapx_common_core_export.md` is a starter reference only. "
            "It proposes common field aliases but does not override the more "
            "conservative chapter confidence boundaries.",
            "",
            "## Interpretation Rules",
            "",
            "Use this matrix as an implementation planning aid:",
            "",
            "- **IMEX confidence** asks whether the value is likely obtainable "
            "from a structured Axys/APX IMEX-style export or adjacent import/export "
            "workflow.",
            "- **REP confidence** asks whether the value is likely obtainable "
            "from a standard or custom REP/Replang report extract.",
            "- A high confidence rating does not mean the exact demo column name "
            "exists in Axys/APX. It means the underlying value is likely available.",
            "- Performance fields are rated more conservatively for IMEX because "
            "the local reference corpus does not contain an official performance "
            "IMEX object dictionary.",
            "- Transaction source/destination and special-security fields are "
            "rated as integration-context fields. The corpus supports them in "
            "transaction translation workflows, but not as guaranteed columns in "
            "every posted-transaction export.",
            "",
            "## Availability Matrix",
            "",
        ]
    )

    for dataset_name, dataset in datasets.items():
        lines.extend(_dataset_table(dataset_name, _mapping(dataset, "columns")))

    lines.extend(_name_mapping_section(contract, datasets))
    lines.extend(_source_strategy_section(contract, datasets))

    lines.extend(
        [
            "## Practical Extraction Guidance",
            "",
            "| Need | Preferred path | Reason |",
            "|---|---|---|",
            *[_table_row(row) for row in _PRACTICAL_GUIDANCE_ROWS],
            "",
            "## Local Validation Checklist",
            "",
            "Before treating a site extract as equivalent to these demo files, collect:",
            "",
            *[
                f"{index}. {step}"
                for index, step in enumerate(_LOCAL_VALIDATION_STEPS, start=1)
            ],
            "",
            "## Related References",
            "",
            *[f"- {reference}" for reference in _RELATED_REFERENCES],
            "",
        ]
    )
    return "\n".join(lines)


def _dataset_table(dataset_name: str, columns: Mapping[str, Any]) -> list[str]:
    """Return one rendered dataset availability table."""
    dataset_label = _DATASET_LABELS.get(dataset_name, dataset_name.removesuffix(".csv"))
    lines = [
        f"### `{dataset_name}`",
        "",
        "| Dataset | Demo column | Normalized meaning | "
        "IMEX confidence | REP confidence | Evidence basis | "
        "Open questions | Comments |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for column_name, metadata in columns.items():
        metadata_map = _as_mapping(metadata)
        open_questions = "; ".join(_required_text_list(metadata_map, "open_questions"))
        lines.append(
            "| "
            f"{_escape_table_text(dataset_label)} | "
            f"`{column_name}` | "
            f"{_escape_table_text(_required_text(metadata_map, 'normalized_meaning'))} | "
            f"{_confidence_label(_required_text(metadata_map, 'imex_confidence'))} | "
            f"{_confidence_label(_required_text(metadata_map, 'rep_confidence'))} | "
            f"{_escape_table_text(_required_text(metadata_map, 'basis'))} | "
            f"{_escape_table_text(open_questions)} | "
            f"{_escape_table_text(_required_text(metadata_map, 'comments'))} |"
        )
    lines.extend(["", ""])
    return lines


def _name_mapping_section(
    contract: Mapping[str, Any],
    datasets: Mapping[str, Any],
) -> list[str]:
    """Return rendered candidate native/report-name mapping tables."""
    lines = [
        "## Candidate Name Mapping",
        "",
        "These names are candidate aliases for local discovery. They are not "
        "assertions that the packaged demo headers are official Axys/APX IMEX or "
        "REP names.",
        "",
        "| Label | Meaning |",
        "|---|---|",
    ]
    for label, meaning in _mapping(contract, "name_confidence_labels").items():
        lines.append(f"| {_confidence_label(label)} | {_escape_table_text(meaning)} |")

    lines.extend([""])

    for dataset_name, dataset in datasets.items():
        lines.extend(
            _name_mapping_table(
                dataset_name,
                _mapping(dataset, "columns"),
            )
        )

    return lines


def _name_mapping_table(dataset_name: str, columns: Mapping[str, Any]) -> list[str]:
    """Return one candidate name-mapping table."""
    dataset_label = _DATASET_LABELS.get(dataset_name, dataset_name.removesuffix(".csv"))
    lines = [
        f"### `{dataset_name}` Name Candidates",
        "",
        "| Dataset | Demo column | Candidate Axys/APX export names | "
        "Candidate report labels | Name confidence | Notes |",
        "|---|---|---|---|---|---|",
    ]
    for column_name, metadata in columns.items():
        metadata_map = _as_mapping(metadata)
        axys_names = ", ".join(_required_text_list(metadata_map, "candidate_axys_names"))
        report_labels = ", ".join(
            _required_text_list(metadata_map, "candidate_report_labels")
        )
        lines.append(
            "| "
            f"{_escape_table_text(dataset_label)} | "
            f"`{column_name}` | "
            f"{_escape_table_text(axys_names)} | "
            f"{_escape_table_text(report_labels)} | "
            f"{_confidence_label(_required_text(metadata_map, 'name_confidence'))} | "
            f"{_escape_table_text(_required_text(metadata_map, 'name_notes'))} |"
        )
    lines.extend(["", ""])
    return lines


def _source_strategy_section(
    contract: Mapping[str, Any],
    datasets: Mapping[str, Any],
) -> list[str]:
    """Return rendered preferred-source strategy tables."""
    lines = [
        "## Source Strategy Matrix",
        "",
        "This matrix translates availability confidence into implementation "
        "guidance. `Blocking if missing` means ppar should not silently proceed "
        "for that field in a workflow that depends on the corresponding dataset.",
        "",
        "| Label | Meaning |",
        "|---|---|",
    ]
    for label, meaning in _mapping(contract, "source_strategy_labels").items():
        lines.append(f"| {_confidence_label(label)} | {_escape_table_text(meaning)} |")

    lines.extend([""])

    for dataset_name, dataset in datasets.items():
        lines.extend(
            _source_strategy_table(
                dataset_name,
                _mapping(dataset, "columns"),
            )
        )

    return lines


def _source_strategy_table(dataset_name: str, columns: Mapping[str, Any]) -> list[str]:
    """Return one preferred-source strategy table."""
    dataset_label = _DATASET_LABELS.get(dataset_name, dataset_name.removesuffix(".csv"))
    lines = [
        f"### `{dataset_name}` Source Strategy",
        "",
        "| Dataset | Demo column | Preferred source | Fallback source | "
        "Context required | Blocking if missing | Notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for column_name, metadata in columns.items():
        metadata_map = _as_mapping(metadata)
        lines.append(
            "| "
            f"{_escape_table_text(dataset_label)} | "
            f"`{column_name}` | "
            f"{_confidence_label(_required_text(metadata_map, 'preferred_source'))} | "
            f"{_confidence_label(_required_text(metadata_map, 'fallback_source'))} | "
            f"{_bool_label(_required_bool(metadata_map, 'requires_context_for_semantics'))} | "
            f"{_bool_label(_required_bool(metadata_map, 'blocking_if_missing'))} | "
            f"{_escape_table_text(_required_text(metadata_map, 'source_strategy_notes'))} |"
        )
    lines.extend(["", ""])
    return lines


def _table_row(values: Sequence[object]) -> str:
    """Return one markdown table row."""
    return "| " + " | ".join(_escape_table_text(value) for value in values) + " |"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Return parsed command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=_DEFAULT_CONTRACT_PATH,
        help="Path to demo_extract_availability.yaml.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT_PATH,
        help="Path to the rendered markdown contract.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the rendered contract differs from the current file.",
    )
    return parser.parse_args(argv)


def _load_contract(path: Path) -> Mapping[str, Any]:
    """Load the YAML availability contract."""
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return _as_mapping(loaded)


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Return a required nested mapping."""
    return _as_mapping(parent.get(key))


def _as_mapping(value: object) -> Mapping[str, Any]:
    """Return ``value`` as a mapping or raise a useful error."""
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping, got {type(value).__name__}.")
    return value


def _required_text(metadata: Mapping[str, Any], key: str) -> str:
    """Return a required nonblank scalar value as text."""
    value = metadata.get(key)
    if value is None or not str(value).strip():
        raise ValueError(f"Missing required metadata field {key!r}.")
    return str(value).strip()


def _required_text_list(metadata: Mapping[str, Any], key: str) -> list[str]:
    """Return a required list of nonblank text values."""
    value = metadata.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Missing required metadata list {key!r}.")

    text_values = [str(item).strip() for item in value]
    if any(not item for item in text_values):
        raise ValueError(f"Metadata list {key!r} contains a blank value.")
    return text_values


def _required_bool(metadata: Mapping[str, Any], key: str) -> bool:
    """Return a required boolean metadata value."""
    value = metadata.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"Missing required boolean metadata field {key!r}.")
    return value


def _confidence_label(value: object) -> str:
    """Return display text for a confidence label."""
    key = str(value)
    return _CONFIDENCE_DISPLAY.get(key, key.replace("_", " ").title())


def _bool_label(value: bool) -> str:
    """Return display text for a boolean metadata value."""
    return "Yes" if value else "No"


def _escape_table_text(value: object) -> str:
    """Escape a scalar for safe markdown table output."""
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


if __name__ == "__main__":
    raise SystemExit(main())
