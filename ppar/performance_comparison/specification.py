"""Read and validate performance comparison YAML specifications."""

from __future__ import annotations

# Python imports
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

# Third-party imports
import yaml

# Project imports
from ppar.errors import PpaError
from ppar.performance_comparison.methods import (
    ModifiedDietzDayCount,
    ModifiedDietzFlowTiming,
    ModifiedDietzInclusionRule,
    ReturnBasis,
    ReturnReconstructionFlowSource,
    ReturnReconstructionMethod,
    ReturnReconstructionSignConvention,
    ReturnReconstructionValueSource,
)
import ppar.utilities as util

_SNAPSHOT_A_KEY: Final[str] = "a"
_SNAPSHOT_B_KEY: Final[str] = "b"
_SNAPSHOTS_KEY: Final[str] = "snapshots"
_FILES_KEY: Final[str] = "files"
_PATH_KEY: Final[str] = "path"
_LABEL_KEY: Final[str] = "label"
_VENDOR_KEY: Final[str] = "vendor"
_SCHEMA_KEY: Final[str] = "schema"
_REQUIRED_KEY: Final[str] = "required"
_COMPARISON_KEY: Final[str] = "comparison"
_LEVEL_KEY: Final[str] = "level"
_PORTFOLIO_RETURN_RECONSTRUCTION_KEY: Final[str] = "portfolio_return_reconstruction"
_SECURITY_RETURN_RECONSTRUCTION_KEY: Final[str] = "security_return_reconstruction"
_PORTFOLIO_PERFORMANCE_KEY: Final[str] = "portfolio_performance"
_SECURITY_PERFORMANCE_KEY: Final[str] = "security_performance"
_SUPPORTED_FILE_KEYS: Final[frozenset[str]] = frozenset(
    {
        _PORTFOLIO_PERFORMANCE_KEY,
        _SECURITY_PERFORMANCE_KEY,
        "splits",
        "holdings",
        "transactions",
        "fx_rates",
    }
)
_REMOVED_CASH_IMPACT_METHODS_KEY: Final[str] = "cash_impact_methods"
PORTFOLIO_COMPARISON_LEVEL: Final[str] = "portfolio"
SECURITY_COMPARISON_LEVEL: Final[str] = "security"
COMPARISON_LEVELS: Final[frozenset[str]] = frozenset(
    {PORTFOLIO_COMPARISON_LEVEL, SECURITY_COMPARISON_LEVEL}
)
_RECONSTRUCTION_REQUIRED_KEYS: Final[tuple[str, ...]] = (
    "method",
    "beginning_value_source",
    "ending_value_source",
    "flow_source",
    "flow_categories",
    "income_categories",
    "return_basis",
    "sign_convention",
)
_RECONSTRUCTION_TIMED_FLOW_KEYS: Final[tuple[str, ...]] = (
    "flow_timing",
    "day_count",
    "inclusion_rule",
)
_RETURN_RECONSTRUCTION_METHODS: Final[frozenset[str]] = frozenset(
    method.value for method in ReturnReconstructionMethod
)


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """YAML safe loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    """Construct a mapping while rejecting duplicate keys.

    YAML's default behavior keeps the last duplicate key, which is dangerous for
    audit configuration because a repeated ``method`` or policy block can hide
    the setting a reviewer thought they were using.
    """
    loader.flatten_mapping(node)
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate YAML key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class ComparisonSnapshot:
    """Resolved comparison snapshot configuration.

    Attributes:
        key: Neutral snapshot key, currently ``"a"`` or ``"b"``.
        label: User-facing snapshot label.
        path: Resolved snapshot directory path.
        vendor: Optional source-system adapter name.
        schema_path: Optional resolved vendor schema YAML path.
    """

    key: str
    label: str
    path: Path
    vendor: str | None
    schema_path: Path | None


@dataclass(frozen=True)
class ComparisonFile:
    """Resolved comparison source-file configuration for both snapshots.

    Attributes:
        name: Normalized dataset name such as ``portfolio_performance``.
        relative_path: File path configured relative to each snapshot.
        snapshot_a_path: Resolved file path in snapshot A.
        snapshot_b_path: Resolved file path in snapshot B.
        required: Whether optional-file existence must be validated up front.
    """

    name: str
    relative_path: Path
    snapshot_a_path: Path
    snapshot_b_path: Path
    required: bool


@dataclass(frozen=True)
class PortfolioReturnReconstruction:
    """Resolved portfolio return-reconstruction settings.

    Attributes:
        method: Return reconstruction method.
        beginning_value_source: Dataset used for beginning value.
        ending_value_source: Dataset used for ending value.
        flow_source: Dataset used for dated external flows.
        flow_timing: Transaction date field used for dated flow weighting, or
            ``None`` for methods that do not use dated flow weights.
        day_count: Day-count convention, or ``None`` for methods that do not
            use dated flow weights.
        inclusion_rule: Beginning/end-of-day flow inclusion rule, or ``None``
            for methods that do not use dated flow weights.
        flow_categories: Transaction categories treated as external flows.
        income_categories: Transaction categories treated as income inputs.
        return_basis: Reported-return basis for fee/expense interpretation.
        sign_convention: Transaction amount sign convention.
    """

    method: str
    beginning_value_source: str
    ending_value_source: str
    flow_source: str
    flow_timing: str | None
    day_count: str | None
    inclusion_rule: str | None
    flow_categories: tuple[str, ...]
    income_categories: tuple[str, ...]
    return_basis: str
    sign_convention: str


SecurityReturnReconstruction = PortfolioReturnReconstruction


class PerformanceComparisonSpecification:
    """Read performance comparison YAML settings and resolve fixture paths.

    Attributes:
        path: Filesystem path to the comparison YAML specification.
        values: Parsed YAML settings dictionary.
        snapshot_a: Resolved snapshot A settings.
        snapshot_b: Resolved snapshot B settings.
        files: Resolved file settings keyed by normalized dataset name.
        comparison_level: Primary performance-result level to compare. The
            default is ``"portfolio"``; ``"security"`` uses
            ``security_performance`` as the target performance-result dataset.

    Notes:
        The primary performance-result file is always required. Other files are
        optional unless configured with ``required: true``, which only controls
        preflight file-existence validation.
    """

    def __init__(
        self,
        path: util.PathLike,
        *,
        comparison_level: str | None = None,
    ) -> None:
        """Read and validate a performance comparison specification.

        Args:
            path: Path to the comparison YAML specification.
            comparison_level: Optional primary performance-result level override.
                When omitted, ``comparison.level`` from the YAML is used.

        Raises:
            PpaError: If the YAML cannot be parsed, its shape is invalid, the
                required snapshots are missing, or required files do not exist.
        """
        self.path = Path(path)
        with open(self.path, "r", encoding=util.ENCODING) as file:
            try:
                loaded_yaml: Any = yaml.load(file, Loader=_UniqueKeySafeLoader)
            except Exception as error:
                raise PpaError(self._error_message(f"Invalid YAML: {error}"), 504) from error
        if not isinstance(loaded_yaml, dict):
            raise PpaError(self._error_message("YAML must be a dictionary."), 504)

        self.values: dict[str, Any] = loaded_yaml
        self._validate_removed_cash_configuration()
        self.comparison_level = self._comparison_level(comparison_level)
        self.portfolio_return_reconstruction = (
            self._portfolio_return_reconstruction()
        )
        self.security_return_reconstruction = self._security_return_reconstruction()
        self.snapshot_a = self._snapshot(_SNAPSHOT_A_KEY)
        self.snapshot_b = self._snapshot(_SNAPSHOT_B_KEY)
        self.files = self._files()
        self._validate_reconstruction_files()
        self._validate_required_files()

    def _validate_removed_cash_configuration(self) -> None:
        """Reject the retired standalone cash-dataset policy section.

        Raises:
            PpaError: If legacy cash impact configuration remains in YAML.
        """
        if _REMOVED_CASH_IMPACT_METHODS_KEY not in self.values:
            return
        raise PpaError(
            self._error_message(
                "cash_impact_methods is not supported; represent cash as holdings "
                "and use holding_impact_methods."
            ),
            504,
        )

    def resolve_path(self, file_path: util.PathLike) -> Path:
        """Return an absolute or comparison-YAML-relative path.

        Args:
            file_path: Path from the comparison specification.

        Returns:
            Absolute paths unchanged; relative paths resolved beside the
            comparison YAML file.
        """
        path = Path(file_path)
        return path if path.is_absolute() else self.path.parent / path

    def _snapshot(self, key: str) -> ComparisonSnapshot:
        """Return one resolved snapshot definition."""
        snapshots = self.values.get(_SNAPSHOTS_KEY)
        if not isinstance(snapshots, dict):
            raise PpaError(self._error_message("snapshots must be a mapping."), 504)

        snapshot = snapshots.get(key)
        if not isinstance(snapshot, dict):
            raise PpaError(
                self._error_message(f"snapshots.{key} must be a mapping."),
                504,
            )

        snapshot_path_value = snapshot.get(_PATH_KEY)
        if not isinstance(snapshot_path_value, str) or not snapshot_path_value:
            raise PpaError(
                self._error_message(f"snapshots.{key}.path must be a string."),
                504,
            )
        snapshot_path = self.resolve_path(snapshot_path_value)

        label_value = snapshot.get(_LABEL_KEY, key)
        if not isinstance(label_value, str) or not label_value:
            raise PpaError(
                self._error_message(f"snapshots.{key}.label must be a string."),
                504,
            )

        vendor_value = snapshot.get(_VENDOR_KEY)
        if vendor_value is not None and not isinstance(vendor_value, str):
            raise PpaError(
                self._error_message(f"snapshots.{key}.vendor must be a string."),
                504,
            )

        schema_value = snapshot.get(_SCHEMA_KEY)
        schema_path = self._schema_path(key, schema_value)

        return ComparisonSnapshot(
            key=key,
            label=label_value,
            path=snapshot_path,
            vendor=vendor_value,
            schema_path=schema_path,
        )

    def _schema_path(self, key: str, schema_value: object) -> Path | None:
        """Return a resolved schema path or ``None`` for inline schema mappings."""
        if schema_value is None or isinstance(schema_value, dict):
            return None
        if not isinstance(schema_value, str) or not schema_value:
            raise PpaError(
                self._error_message(
                    f"snapshots.{key}.schema must be a string or mapping."
                ),
                504,
            )
        return self.resolve_path(schema_value)

    def _files(self) -> dict[str, ComparisonFile]:
        """Return resolved comparison files keyed by normalized dataset name."""
        files_value = self.values.get(_FILES_KEY)
        if not isinstance(files_value, dict):
            raise PpaError(self._error_message("files must be a mapping."), 504)
        required_performance_file = self._required_performance_file_name()
        if required_performance_file not in files_value:
            raise PpaError(
                self._error_message(f"files.{required_performance_file} is required."),
                504,
            )

        files: dict[str, ComparisonFile] = {}
        for file_name, file_value in files_value.items():
            if not isinstance(file_name, str) or not file_name:
                raise PpaError(self._error_message("File names must be strings."), 504)
            if file_name not in _SUPPORTED_FILE_KEYS:
                supported = ", ".join(sorted(_SUPPORTED_FILE_KEYS))
                raise PpaError(
                    self._error_message(
                        f"files.{file_name} is not supported. Supported files: "
                        f"{supported}."
                    ),
                    504,
                )
            files[file_name] = self._file(file_name, file_value)
        return files

    def _portfolio_return_reconstruction(
        self,
    ) -> PortfolioReturnReconstruction | None:
        """Return validated portfolio return-reconstruction settings."""
        return self._return_reconstruction(_PORTFOLIO_RETURN_RECONSTRUCTION_KEY)

    def _security_return_reconstruction(
        self,
    ) -> SecurityReturnReconstruction | None:
        """Return validated security return-reconstruction settings."""
        return self._return_reconstruction(_SECURITY_RETURN_RECONSTRUCTION_KEY)

    def _return_reconstruction(
        self,
        section: str,
    ) -> PortfolioReturnReconstruction | None:
        """Return validated return-reconstruction settings for a YAML section."""
        reconstruction = self.values.get(section)
        if reconstruction is None:
            return None
        if not isinstance(reconstruction, dict):
            raise PpaError(
                self._error_message(
                    f"{section} must be a mapping."
                ),
                504,
            )

        supported_keys = set(_RECONSTRUCTION_REQUIRED_KEYS) | set(
            _RECONSTRUCTION_TIMED_FLOW_KEYS
        )
        unsupported_keys = sorted(str(key) for key in reconstruction if key not in supported_keys)
        if unsupported_keys:
            raise PpaError(
                self._error_message(
                    f"{section} has unsupported keys: {', '.join(unsupported_keys)}."
                ),
                504,
            )

        missing_keys = [
            key for key in _RECONSTRUCTION_REQUIRED_KEYS if key not in reconstruction
        ]
        if missing_keys:
            raise PpaError(
                self._error_message(
                    f"{section} missing required keys: {', '.join(missing_keys)}."
                ),
                504,
            )
        method = self._required_choice(
            section,
            "method",
            reconstruction["method"],
            set(_RETURN_RECONSTRUCTION_METHODS),
        )
        timing_values = self._return_reconstruction_timing_values(
            section,
            reconstruction,
            method,
        )

        return PortfolioReturnReconstruction(
            method=method,
            beginning_value_source=self._required_choice(
                section,
                "beginning_value_source",
                reconstruction["beginning_value_source"],
                {ReturnReconstructionValueSource.HOLDINGS.value},
            ),
            ending_value_source=self._required_choice(
                section,
                "ending_value_source",
                reconstruction["ending_value_source"],
                {ReturnReconstructionValueSource.HOLDINGS.value},
            ),
            flow_source=self._required_choice(
                section,
                "flow_source",
                reconstruction["flow_source"],
                {ReturnReconstructionFlowSource.TRANSACTIONS.value},
            ),
            flow_timing=timing_values[0],
            day_count=timing_values[1],
            inclusion_rule=timing_values[2],
            flow_categories=self._required_string_list(
                section,
                "flow_categories",
                reconstruction["flow_categories"],
            ),
            income_categories=self._required_string_list(
                section,
                "income_categories",
                reconstruction["income_categories"],
            ),
            return_basis=self._required_choice(
                section,
                "return_basis",
                reconstruction["return_basis"],
                {ReturnBasis.GROSS.value, ReturnBasis.NET.value},
            ),
            sign_convention=self._required_choice(
                section,
                "sign_convention",
                reconstruction["sign_convention"],
                {ReturnReconstructionSignConvention.SIGNED_AMOUNT.value},
            ),
        )

    def _return_reconstruction_timing_values(
        self,
        section: str,
        reconstruction: dict[object, object],
        method: str,
    ) -> tuple[str | None, str | None, str | None]:
        """Return validated timing fields for one reconstruction method."""
        if method == ReturnReconstructionMethod.MODIFIED_DIETZ.value:
            missing_keys = [
                key for key in _RECONSTRUCTION_TIMED_FLOW_KEYS if key not in reconstruction
            ]
            if missing_keys:
                raise PpaError(
                    self._error_message(
                        f"{section} missing required keys for method "
                        f"{method}: {', '.join(missing_keys)}."
                    ),
                    504,
                )
            return (
                self._required_choice(
                    section,
                    "flow_timing",
                    reconstruction["flow_timing"],
                    {
                        "transaction_date",
                        ModifiedDietzFlowTiming.TRADE_DATE.value,
                        ModifiedDietzFlowTiming.SETTLEMENT_DATE.value,
                    },
                ),
                self._required_choice(
                    section,
                    "day_count",
                    reconstruction["day_count"],
                    {ModifiedDietzDayCount.ACTUAL_DAYS.value},
                ),
                self._required_choice(
                    section,
                    "inclusion_rule",
                    reconstruction["inclusion_rule"],
                    {
                        ModifiedDietzInclusionRule.BEGINNING_OF_DAY.value,
                        ModifiedDietzInclusionRule.END_OF_DAY.value,
                    },
                ),
            )

        unsupported_keys = [
            key for key in _RECONSTRUCTION_TIMED_FLOW_KEYS if key in reconstruction
        ]
        if unsupported_keys:
            raise PpaError(
                self._error_message(
                    f"{section} keys are not valid for method {method}: "
                    f"{', '.join(unsupported_keys)}."
                ),
                504,
            )
        return (None, None, None)

    def _required_choice(
        self,
        section: str,
        key: str,
        value: object,
        allowed_values: set[str],
    ) -> str:
        """Return a required string enum value or raise."""
        if not isinstance(value, str) or not value:
            raise PpaError(
                self._error_message(f"{section}.{key} must be a string."),
                504,
            )
        if value not in allowed_values:
            allowed = ", ".join(sorted(allowed_values))
            raise PpaError(
                self._error_message(
                    f"{section}.{key} must be one of: {allowed}."
                ),
                504,
            )
        return value

    def _required_string_list(
        self,
        section: str,
        key: str,
        value: object,
    ) -> tuple[str, ...]:
        """Return a required list of strings or raise."""
        if not isinstance(value, list):
            raise PpaError(
                self._error_message(f"{section}.{key} must be a list."),
                504,
            )
        if not all(isinstance(item, str) and item for item in value):
            raise PpaError(
                self._error_message(
                    f"{section}.{key} must contain only non-empty strings."
                ),
                504,
            )
        return tuple(value)

    def _file(self, file_name: str, file_value: object) -> ComparisonFile:
        """Return one resolved comparison file definition."""
        required_file_names = self._required_file_names()
        required = file_name in required_file_names
        if isinstance(file_value, str):
            relative_path = Path(file_value)
        elif isinstance(file_value, dict):
            path_value = file_value.get(_PATH_KEY)
            if not isinstance(path_value, str) or not path_value:
                raise PpaError(
                    self._error_message(f"files.{file_name}.path must be a string."),
                    504,
                )
            if _REQUIRED_KEY in file_value:
                if file_name in required_file_names:
                    raise PpaError(
                        self._error_message(
                            f"files.{file_name} is required by the comparison "
                            "contract and must not specify required."
                        ),
                        504,
                    )
                required_value = file_value[_REQUIRED_KEY]
                if not isinstance(required_value, bool):
                    raise PpaError(
                        self._error_message(
                            f"files.{file_name}.required must be a boolean."
                        ),
                        504,
                    )
                required = required_value
            relative_path = Path(path_value)
        else:
            raise PpaError(
                self._error_message(f"files.{file_name} must be a string or mapping."),
                504,
            )

        return ComparisonFile(
            name=file_name,
            relative_path=relative_path,
            snapshot_a_path=self._snapshot_file_path(self.snapshot_a, relative_path),
            snapshot_b_path=self._snapshot_file_path(self.snapshot_b, relative_path),
            required=required,
        )

    def _comparison_level(self, override: str | None = None) -> str:
        """Return the primary comparison level from YAML settings."""
        comparison_value = self.values.get(_COMPARISON_KEY, {})
        if comparison_value is None:
            comparison_value = {}
        if not isinstance(comparison_value, dict):
            raise PpaError(self._error_message("comparison must be a mapping."), 504)
        level_value = (
            override
            if override is not None
            else comparison_value.get(_LEVEL_KEY, PORTFOLIO_COMPARISON_LEVEL)
        )
        if not isinstance(level_value, str) or level_value not in COMPARISON_LEVELS:
            allowed_values = ", ".join(sorted(COMPARISON_LEVELS))
            raise PpaError(
                self._error_message(
                    f"comparison.level must be one of: {allowed_values}."
                ),
                504,
            )
        return level_value

    def _required_performance_file_name(self) -> str:
        """Return the required performance-result file for the comparison level."""
        if self.comparison_level == SECURITY_COMPARISON_LEVEL:
            return _SECURITY_PERFORMANCE_KEY
        return _PORTFOLIO_PERFORMANCE_KEY

    def _required_file_names(self) -> frozenset[str]:
        """Return source file names required by comparison level and formulas."""
        required_names = {self._required_performance_file_name()}
        if self._active_return_reconstruction_configured():
            required_names.update({"holdings", "transactions"})
        return frozenset(required_names)

    def _active_return_reconstruction_configured(self) -> bool:
        """Return whether the active comparison level has reconstruction enabled."""
        if self.comparison_level == SECURITY_COMPARISON_LEVEL:
            return self.security_return_reconstruction is not None
        return self.portfolio_return_reconstruction is not None

    @staticmethod
    def _snapshot_file_path(
        snapshot: ComparisonSnapshot,
        relative_path: Path,
    ) -> Path:
        """Return a snapshot file path resolved relative to the snapshot path."""
        return relative_path if relative_path.is_absolute() else snapshot.path / relative_path

    def _validate_required_files(self) -> None:
        """Validate existence for portfolio performance and required files."""
        for comparison_file in self.files.values():
            if not comparison_file.required:
                continue
            for snapshot_key, file_path in (
                ("a", comparison_file.snapshot_a_path),
                ("b", comparison_file.snapshot_b_path),
            ):
                if not util.file_path_exists(file_path):
                    raise PpaError(
                        self._error_message(
                            f"files.{comparison_file.name} is required but "
                            f"snapshot {snapshot_key} is missing {file_path}."
                        ),
                        802,
                    )

    def _validate_reconstruction_files(self) -> None:
        """Raise if opted-in return reconstruction lacks required source files."""
        if not self._active_return_reconstruction_configured():
            return
        for file_name in ("holdings", "transactions"):
            if file_name not in self.files:
                raise PpaError(
                    self._error_message(
                        f"return reconstruction requires files.{file_name}."
                    ),
                    504,
                )

    def _error_message(self, message: str) -> str:
        """Return an error message with comparison specification context."""
        return f"{message}  |  comparison_specification_path={self.path}"
