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
_PORTFOLIO_PERFORMANCE_KEY: Final[str] = "portfolio_performance"
_SECURITY_PERFORMANCE_KEY: Final[str] = "security_performance"
_SUPPORTED_FILE_KEYS: Final[frozenset[str]] = frozenset(
    {
        _PORTFOLIO_PERFORMANCE_KEY,
        _SECURITY_PERFORMANCE_KEY,
        "security_master",
        "holdings",
        "transactions",
        "cash",
        "fx_rates",
    }
)
PORTFOLIO_COMPARISON_LEVEL: Final[str] = "portfolio"
SECURITY_COMPARISON_LEVEL: Final[str] = "security"
COMPARISON_LEVELS: Final[frozenset[str]] = frozenset(
    {PORTFOLIO_COMPARISON_LEVEL, SECURITY_COMPARISON_LEVEL}
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

    def __init__(self, path: util.PathLike) -> None:
        """Read and validate a performance comparison specification.

        Args:
            path: Path to the comparison YAML specification.

        Raises:
            PpaError: If the YAML cannot be parsed, its shape is invalid, the
                required snapshots are missing, or required files do not exist.
        """
        self.path = Path(path)
        with open(self.path, "r", encoding=util.ENCODING) as file:
            try:
                loaded_yaml: Any = yaml.safe_load(file)
            except Exception as error:
                raise PpaError(self._error_message(f"Invalid YAML: {error}"), 504) from error
        if not isinstance(loaded_yaml, dict):
            raise PpaError(self._error_message("YAML must be a dictionary."), 504)

        self.values: dict[str, Any] = loaded_yaml
        self.comparison_level = self._comparison_level()
        self.snapshot_a = self._snapshot(_SNAPSHOT_A_KEY)
        self.snapshot_b = self._snapshot(_SNAPSHOT_B_KEY)
        self.files = self._files()
        self._validate_required_files()

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

    def _file(self, file_name: str, file_value: object) -> ComparisonFile:
        """Return one resolved comparison file definition."""
        required = file_name == self._required_performance_file_name()
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
                if file_name == self._required_performance_file_name():
                    raise PpaError(
                        self._error_message(
                            f"files.{file_name} must not specify required."
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

    def _comparison_level(self) -> str:
        """Return the primary comparison level from YAML settings."""
        comparison_value = self.values.get(_COMPARISON_KEY, {})
        if comparison_value is None:
            comparison_value = {}
        if not isinstance(comparison_value, dict):
            raise PpaError(self._error_message("comparison must be a mapping."), 504)
        level_value = comparison_value.get(_LEVEL_KEY, PORTFOLIO_COMPARISON_LEVEL)
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
            for file_path in (
                comparison_file.snapshot_a_path,
                comparison_file.snapshot_b_path,
            ):
                if not util.file_path_exists(file_path):
                    raise PpaError(self._error_message(util.file_path_error(file_path)), 802)

    def _error_message(self, message: str) -> str:
        """Return an error message with comparison specification context."""
        return f"{message}  |  comparison_specification_path={self.path}"
