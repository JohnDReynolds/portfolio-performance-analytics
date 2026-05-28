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
_PORTFOLIO_PERFORMANCE_KEY: Final[str] = "portfolio_performance"


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

    Notes:
        ``portfolio_performance`` is always required. Other files are optional
        unless they are configured with ``required: true``, which only controls
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
        if _PORTFOLIO_PERFORMANCE_KEY not in files_value:
            raise PpaError(
                self._error_message("files.portfolio_performance is required."),
                504,
            )

        files: dict[str, ComparisonFile] = {}
        for file_name, file_value in files_value.items():
            if not isinstance(file_name, str) or not file_name:
                raise PpaError(self._error_message("File names must be strings."), 504)
            files[file_name] = self._file(file_name, file_value)
        return files

    def _file(self, file_name: str, file_value: object) -> ComparisonFile:
        """Return one resolved comparison file definition."""
        required = file_name == _PORTFOLIO_PERFORMANCE_KEY
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
                if file_name == _PORTFOLIO_PERFORMANCE_KEY:
                    raise PpaError(
                        self._error_message(
                            "files.portfolio_performance must not specify required."
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
