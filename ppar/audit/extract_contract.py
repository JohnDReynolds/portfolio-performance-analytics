"""Validate source extracts against the packaged Axys/APX extract contract."""

from __future__ import annotations

# Python imports
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Any, Final

# Third-party imports
import polars as pl
import yaml

# Project imports
from ppar.errors import PpaError
from ppar.audit import schema as pc_cols
from ppar.audit.transaction_policy import transaction_boundary_codes
import ppar.utilities as util

_CONTRACT_RESOURCE: Final[str] = "ppar.setup_templates"
_CONTRACT_RESOURCE_DIRECTORY: Final[str] = "axys_apx_audit"
_CONTRACT_FILE_NAME: Final[str] = "demo_extract_availability.yaml"
_AXYS_AMBIGUOUS_FLOW_CODES: Final[frozenset[str]] = transaction_boundary_codes(
    "ambiguous_context_required"
)
_EXTRACT_CONTRACT_KEY: Final[str] = "extract_contract"
_PATH_KEY: Final[str] = "path"
_ENFORCE_AMBIGUOUS_AXYS_FLOWS_KEY: Final[str] = "enforce_ambiguous_axys_flows"
_TRANSACTION_SEMANTICS_CASE_KEY: Final[str] = "transaction_semantics_case"
_TRANSACTION_SEMANTICS_CASE_EXACT: Final[str] = "exact"
_TRANSACTION_SEMANTICS_CASE_LEGACY: Final[str] = "legacy_case_insensitive"
_TRANSACTION_SEMANTICS_CASE_VALUES: Final[frozenset[str]] = frozenset(
    {
        _TRANSACTION_SEMANTICS_CASE_EXACT,
        _TRANSACTION_SEMANTICS_CASE_LEGACY,
    }
)


@dataclass(frozen=True)
class ExtractContractSettings:
    """Resolved extract-contract settings.

    Attributes:
        path: Filesystem path for a local contract, or packaged resource label.
        enforce_ambiguous_axys_flows: Whether ambiguous Axys/APX transaction codes
            require source/destination and special-security context fields.
        transaction_semantics_case: Case-matching mode for transaction-rule
            codes and native context-condition values.
        contract: Parsed extract-contract YAML.
    """

    path: str
    enforce_ambiguous_axys_flows: bool
    transaction_semantics_case: str
    contract: dict[str, Any]


def validate_extract_contract(
    contract: Mapping[str, Any],
    *,
    contract_label: str,
    require_ambiguous_flow_context: bool = True,
) -> None:
    """Validate the extract-contract shape required by runtime guards.

    Args:
        contract: Parsed extract-contract YAML.
        contract_label: User-facing contract path or resource label for errors.
        require_ambiguous_flow_context: Whether the transaction contract must
            define at least one blocking context field for ambiguous Axys/APX flow
            semantics.

    Raises:
        PpaError: If the contract is structurally invalid, omits transaction
            columns needed by runtime guards, contains non-boolean guard flags,
            or names an unsupported transaction column.
    """
    datasets = _required_mapping(contract, "datasets", contract_label)
    transactions = _required_mapping(datasets, "transactions.csv", contract_label)
    columns = _required_mapping(transactions, "columns", contract_label)

    required_context_columns: set[str] = set()
    for column_name in columns:
        if not isinstance(column_name, str) or not column_name.strip():
            raise PpaError(
                f"{contract_label}: transactions.csv column names must be strings.",
                504,
            )
        metadata = _required_mapping(
            columns,
            column_name,
            contract_label,
        )
        requires_context = _required_bool(
            metadata,
            "requires_context_for_semantics",
            contract_label,
            column_name,
        )
        blocking = _required_bool(
            metadata,
            "blocking_if_missing",
            contract_label,
            column_name,
        )
        _normalized_transaction_column(column_name)
        if requires_context and blocking:
            required_context_columns.add(column_name)

    if require_ambiguous_flow_context and not required_context_columns:
        raise PpaError(
            (
                f"{contract_label}: transactions.csv must define at least one "
                "column with requires_context_for_semantics: true and "
                "blocking_if_missing: true when ambiguous Axys/APX flow enforcement "
                "is enabled."
            ),
            504,
        )


def extract_contract_settings(
    values: Mapping[str, Any],
    *,
    specification_path: util.PathLike,
) -> ExtractContractSettings:
    """Return resolved extract-contract settings from comparison YAML values.

    Args:
        values: Parsed comparison YAML settings.
        specification_path: Comparison YAML path used to resolve local contract
            paths and report validation errors.

    Returns:
        Resolved extract-contract settings. The safety and case-matching choices
        must be explicit; omitting ``path`` selects the packaged contract.

    Raises:
        PpaError: If ``extract_contract`` has an invalid shape, omits a required
            safety choice, or references a missing/unreadable contract file.
    """
    raw_settings = values.get(_EXTRACT_CONTRACT_KEY)
    if raw_settings is None:
        raise PpaError(
            f"{specification_path}: extract_contract must be a mapping.",
            504,
        )
    if not isinstance(raw_settings, dict):
        raise PpaError(
            f"{specification_path}: extract_contract must be a mapping.",
            504,
        )

    enforce_value = raw_settings.get(_ENFORCE_AMBIGUOUS_AXYS_FLOWS_KEY)
    if not isinstance(enforce_value, bool):
        raise PpaError(
            (
                f"{specification_path}: "
                "extract_contract.enforce_ambiguous_axys_flows must be a boolean."
            ),
            504,
        )

    case_value = raw_settings.get(
        _TRANSACTION_SEMANTICS_CASE_KEY,
    )
    if (
        not isinstance(case_value, str)
        or case_value not in _TRANSACTION_SEMANTICS_CASE_VALUES
    ):
        allowed = ", ".join(sorted(_TRANSACTION_SEMANTICS_CASE_VALUES))
        raise PpaError(
            (
                f"{specification_path}: extract_contract."
                f"{_TRANSACTION_SEMANTICS_CASE_KEY} must be one of {allowed}; "
                f"received {case_value!r}."
            ),
            504,
        )

    raw_path = raw_settings.get(_PATH_KEY)
    if raw_path is None:
        contract = _load_packaged_extract_contract()
        validate_extract_contract(
            contract,
            contract_label=_packaged_contract_label(),
            require_ambiguous_flow_context=enforce_value,
        )
        _validate_exact_case_contract_version(
            contract,
            case_value=case_value,
            contract_label=_packaged_contract_label(),
        )
        return ExtractContractSettings(
            path=_packaged_contract_label(),
            enforce_ambiguous_axys_flows=enforce_value,
            transaction_semantics_case=case_value,
            contract=contract,
        )
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise PpaError(
            f"{specification_path}: extract_contract.path must be a nonblank string.",
            504,
        )

    contract_path = _resolve_contract_path(raw_path, specification_path)
    contract = _load_local_extract_contract(str(contract_path))
    validate_extract_contract(
        contract,
        contract_label=str(contract_path),
        require_ambiguous_flow_context=enforce_value,
    )
    _validate_exact_case_contract_version(
        contract,
        case_value=case_value,
        contract_label=str(contract_path),
    )
    return ExtractContractSettings(
        path=str(contract_path),
        enforce_ambiguous_axys_flows=enforce_value,
        transaction_semantics_case=case_value,
        contract=contract,
    )


def transaction_semantics_exact_case(
    values: Mapping[str, Any],
    *,
    specification_path: util.PathLike,
) -> bool:
    """Return whether transaction semantics use exact native-case matching.

    Args:
        values: Parsed comparison YAML settings.
        specification_path: Comparison YAML path used to resolve and validate
            the selected extract contract.

    Returns:
        ``True`` only when the versioned extract-contract configuration selects
        exact-case transaction semantics. The setting is required.
    """
    settings = extract_contract_settings(
        values,
        specification_path=specification_path,
    )
    return settings.transaction_semantics_case == _TRANSACTION_SEMANTICS_CASE_EXACT


def validate_transaction_extract_contract(
    frame: pl.DataFrame,
    *,
    path: util.PathLike,
    specification_path: util.PathLike,
    specification_values: Mapping[str, Any],
) -> None:
    """Raise if a transaction extract lacks required ambiguous-flow context.

    Args:
        frame: Normalized transaction source rows.
        path: Source CSV path.
        specification_path: Comparison YAML path for error context.
        specification_values: Parsed comparison YAML settings.

    Raises:
        PpaError: If ambiguous Axys/APX transaction codes are present and the
            transaction extract does not include the context fields marked as
            required by the packaged Axys/APX availability contract.

    Notes:
        Axys/APX ``dp``, ``li``, ``lo``, ``ti``, and ``wd`` rows cannot always be
        classified from transaction code alone. Source/destination and
        special-security context must be available before YAML transaction rules
        are allowed to classify those rows.
    """
    settings = extract_contract_settings(
        specification_values,
        specification_path=specification_path,
    )
    if not settings.enforce_ambiguous_axys_flows:
        return

    if pc_cols.TRANSACTION_CODE not in frame.columns:
        return

    ambiguous_codes = _observed_ambiguous_codes(
        frame,
        exact_case=(
            settings.transaction_semantics_case
            == _TRANSACTION_SEMANTICS_CASE_EXACT
        ),
    )
    if not ambiguous_codes:
        return

    required_columns = _transaction_semantics_context_columns(settings.contract)
    missing_columns = sorted(required_columns - set(frame.columns))
    if not missing_columns:
        return

    raise PpaError(
        (
            f"{specification_path}: transactions file {path} contains ambiguous "
            f"Axys/APX transaction codes {', '.join(ambiguous_codes)} but is missing "
            f"required transaction semantics/context fields {missing_columns}. "
            "IMEX transaction code alone is not enough to classify external "
            "flows for dp/li/lo/ti/wd rows. Use an IMEX profile that exposes "
            "source/destination and special-security context, or use a "
            "REP/report extract, custom report, or local discovery that "
            "supplies reviewed category/sign semantics before running "
            "performance comparison."
        ),
        504,
    )


def extract_contract_summary(
    values: Mapping[str, Any],
    *,
    specification_path: util.PathLike,
) -> dict[str, object]:
    """Return compact extract-contract metadata for review artifacts.

    Args:
        values: Parsed comparison YAML settings.
        specification_path: Comparison YAML path used to resolve local contract
            paths and report validation errors.

    Returns:
        JSON-serializable extract-contract metadata for manifest/report
        handoff artifacts.
    """
    settings = extract_contract_settings(
        values,
        specification_path=specification_path,
    )
    required_context_columns = sorted(
        _transaction_semantics_context_columns(settings.contract)
    )
    return {
        "path": settings.path,
        "enforce_ambiguous_axys_flows": settings.enforce_ambiguous_axys_flows,
        "required_transaction_context_columns": required_context_columns,
    }


def _observed_ambiguous_codes(
    frame: pl.DataFrame,
    *,
    exact_case: bool,
) -> list[str]:
    """Return ambiguous Axys/APX transaction codes observed in a frame."""
    observed: set[str] = set()
    for value in frame.get_column(pc_cols.TRANSACTION_CODE):
        native_code = _native_transaction_code(value)
        matching_code = native_code if exact_case else native_code.lower()
        if matching_code in _AXYS_AMBIGUOUS_FLOW_CODES:
            observed.add(native_code if exact_case else native_code.upper())
    return sorted(observed)


def _transaction_semantics_context_columns(contract: Mapping[str, Any]) -> frozenset[str]:
    """Return normalized transaction columns required for ambiguous semantics."""
    transaction_columns = contract["datasets"]["transactions.csv"]["columns"]
    required_demo_columns = {
        column_name
        for column_name, metadata in transaction_columns.items()
        if (
            metadata.get("requires_context_for_semantics") is True
            and metadata.get("blocking_if_missing") is True
        )
    }
    return frozenset(
        _normalized_transaction_column(column_name)
        for column_name in sorted(required_demo_columns)
    )


@cache
def _load_packaged_extract_contract() -> dict[str, Any]:
    """Load the packaged Axys/APX demo extract availability contract."""
    contract_path = files(_CONTRACT_RESOURCE).joinpath(
        _CONTRACT_RESOURCE_DIRECTORY,
        _CONTRACT_FILE_NAME,
    )
    with contract_path.open(encoding=util.ENCODING) as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise TypeError(f"{_CONTRACT_FILE_NAME} must contain a YAML mapping.")
    return loaded


def _packaged_contract_label() -> str:
    """Return the user-facing packaged contract resource label."""
    return (
        f"packaged:{_CONTRACT_RESOURCE}/{_CONTRACT_RESOURCE_DIRECTORY}/"
        f"{_CONTRACT_FILE_NAME}"
    )


@cache
def _load_local_extract_contract(path: str) -> dict[str, Any]:
    """Load a site-specific extract availability contract."""
    try:
        with open(path, encoding=util.ENCODING) as handle:
            loaded = yaml.safe_load(handle)
    except OSError as error:
        raise PpaError(
            f"Could not read extract contract {path!r}: {error}",
            504,
        ) from error
    if not isinstance(loaded, dict):
        raise PpaError(f"Extract contract {path!r} must contain a YAML mapping.", 504)
    return loaded


def _required_mapping(
    parent: Mapping[str, Any],
    key: str,
    contract_label: str,
) -> Mapping[str, Any]:
    """Return a required nested mapping from an extract contract."""
    value = parent.get(key)
    if not isinstance(value, dict):
        raise PpaError(f"{contract_label}: {key} must be a mapping.", 504)
    return value


def _required_bool(
    metadata: Mapping[str, Any],
    key: str,
    contract_label: str,
    column_name: str,
) -> bool:
    """Return a required boolean column metadata value."""
    value = metadata.get(key)
    if not isinstance(value, bool):
        raise PpaError(
            (
                f"{contract_label}: transactions.csv column {column_name!r} "
                f"must define boolean {key}."
            ),
            504,
        )
    return value


def _validate_exact_case_contract_version(
    contract: Mapping[str, Any],
    *,
    case_value: object,
    contract_label: str,
) -> None:
    """Require a versioned source contract before exact-case matching."""
    if case_value != _TRANSACTION_SEMANTICS_CASE_EXACT:
        return
    version = contract.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise PpaError(
            (
                f"{contract_label}: exact transaction semantics require a "
                "positive integer contract version."
            ),
            504,
        )


def _resolve_contract_path(
    raw_path: str,
    specification_path: util.PathLike,
) -> Path:
    """Return an absolute extract-contract path."""
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return Path(specification_path).parent / path


def _normalized_transaction_column(demo_column: str) -> str:
    """Return the normalized transaction field named by an extract contract.

    This vocabulary validates fields explicitly declared by an extract
    contract; it is not used to infer CSV source mappings.
    """
    common_axys_apx_columns = {
        "PORT": pc_cols.PORTFOLIO_ID,
        "Portfolio Code": pc_cols.PORTFOLIO_ID,
        "Transaction Date": pc_cols.TRANSACTION_DATE,
        "Settlement Date": pc_cols.SETTLEMENT_DATE,
        "SETTLE_DATE": pc_cols.SETTLEMENT_DATE,
        "SEC": pc_cols.SECURITY_ID,
        "Security Symbol": pc_cols.SECURITY_ID,
        # Security Type is the second component of security_id; it is distinct
        # from the transaction row's auditable Transaction Security Type.
        "Security Type": pc_cols.SECURITY_ID,
        "Transaction Code": pc_cols.TRANSACTION_CODE,
        "TRAN": pc_cols.TRANSACTION_CODE,
        "Transaction Security Type": pc_cols.SECURITY_TYPE,
        "SEC_TYPE": pc_cols.SECURITY_TYPE,
        "Source/Destination Type": pc_cols.SOURCE_DESTINATION_TYPE,
        "SRC_DEST_TYPE": pc_cols.SOURCE_DESTINATION_TYPE,
        "Source/Destination Symbol": pc_cols.SOURCE_DESTINATION_SYMBOL,
        "SRC_DEST_SYMBOL": pc_cols.SOURCE_DESTINATION_SYMBOL,
        "Special Security Type": pc_cols.SPECIAL_SECURITY_TYPE,
        "SPECIAL_SEC_TYPE": pc_cols.SPECIAL_SECURITY_TYPE,
        "Special Security Symbol": pc_cols.SPECIAL_SECURITY_SYMBOL,
        "SPECIAL_SEC_SYMBOL": pc_cols.SPECIAL_SECURITY_SYMBOL,
        "Currency Code": pc_cols.CURRENCY,
        "Base Currency": pc_cols.BASE_CURRENCY,
        "Quantity": pc_cols.QUANTITY,
        "Price": pc_cols.PRICE,
        "Amount": pc_cols.AMOUNT,
        "Base Amount": pc_cols.BASE_AMOUNT,
        "Commission": pc_cols.COMMISSION,
    }
    if demo_column in common_axys_apx_columns:
        return common_axys_apx_columns[demo_column]
    normalized_demo_column = demo_column.strip().upper()
    for internal_column in (
        pc_cols.TRANSACTIONS_REQUIRED_COLUMNS
        + pc_cols.TRANSACTIONS_OPTIONAL_COLUMNS
    ):
        if normalized_demo_column == internal_column.upper():
            return internal_column
    raise PpaError(f"Unsupported contract transaction column {demo_column!r}.", 504)


def _native_transaction_code(value: object) -> str:
    """Return a stripped native transaction code or blank for missing values."""
    if value is None:
        return ""
    return str(value).strip()
