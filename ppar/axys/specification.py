"""Read and validate Axys YAML specifications."""

from __future__ import annotations

# Python imports
import datetime as dt
from pathlib import Path
from typing import Any, Callable, Literal, cast

# Third-party imports
import yaml

# Project imports
from ppar.errors import PpaError
import ppar.utilities as util

ErrorMessage = Callable[[str], str]
SourceType = Literal["classification", "mapping"]
_DEFAULTS_KEY = "defaults"
_DEFAULT_FROM_DATE_KEY = "from_date"
_DEFAULT_THRU_DATE_KEY = "thru_date"
_DEFAULT_CLASSIFICATION_KEY = "classification"
_AXYS_PORTFOLIO_NAME_SEPARATOR = " - "


class AxysSpecification:
    """Read Axys YAML settings and resolve referenced source paths.

    Attributes:
        path: Filesystem path to the YAML specification.
        values: Parsed YAML settings dictionary.
    """

    def __init__(self, path: util.PathLike, error_message: ErrorMessage) -> None:
        """Read and validate an Axys YAML specification.

        Args:
            path: Path to the YAML specification file.
            error_message: Callback that adds facade-level source context to
                validation messages.

        Raises:
            PpaError: If the YAML cannot be parsed or its root object is not a
                dictionary.
        """
        self.path = Path(path)
        self._error_message = error_message
        with open(self.path, "r", encoding=util.ENCODING) as file:
            try:
                loaded_yaml: Any = yaml.safe_load(file)
            except Exception as error:
                raise PpaError(error_message(f"Invalid YAML: {error}"), 504) from error
        if not isinstance(loaded_yaml, dict):
            raise PpaError(error_message("YAML must be a dictionary"), 504)
        self.values: dict[str, Any] = loaded_yaml

    def resolve_path(self, file_path: util.PathLike) -> Path:
        """Return an absolute or specifications-relative source path.

        Args:
            file_path: Source path from an argument or specification setting.

        Returns:
            ``file_path`` when it includes a directory component; otherwise,
            the source name resolved beside the specification file.
        """
        path = Path(file_path)
        return path if util.has_directory(path) else self.path.parent / path

    def performance_path(
        self,
        argument_path: util.PathLike | None,
        specification_key: Literal[
            "portfolio_performance_path",
            "security_performance_path",
        ],
        error_message: ErrorMessage,
    ) -> Path:
        """Resolve an explicit or configured performance source path.

        Args:
            argument_path: Explicit source path provided to ``AxysData``, if
                any.
            specification_key: Specification setting holding the fallback path.
            error_message: Callback that adds facade-level source context to
                validation messages.

        Returns:
            Resolved portfolio- or security-performance source path.

        Raises:
            PpaError: If neither an argument path nor a configured path is
                available.
        """
        file_path = argument_path or self.values.get(specification_key)
        if not file_path:
            raise PpaError(
                error_message(
                    f"{specification_key} not in specifications file and not provided "
                    "as an argument."
                ),
                504,
            )
        return self.resolve_path(file_path)

    @property
    def prefix_portfolio_code(self) -> str | None:
        """Return Axys account-code separator for report display names.

        Returns:
            Separator text inserted between the Axys account code and account
            name.
        """
        return _AXYS_PORTFOLIO_NAME_SEPARATOR

    @property
    def default_from_date(self) -> dt.date | None:
        """Return the optional default earliest Axys reporting date.

        Returns:
            Configured default ``from_date``, or ``None`` when omitted.

        Raises:
            PpaError: If the configured value is not a date or ISO date string.
        """
        return self._default_date(_DEFAULT_FROM_DATE_KEY)

    @property
    def default_thru_date(self) -> dt.date | None:
        """Return the optional default latest Axys thru date.

        Returns:
            Configured default ``thru_date``, or ``None`` when omitted.

        Raises:
            PpaError: If the configured value is not a date or ISO date string.
        """
        return self._default_date(_DEFAULT_THRU_DATE_KEY)

    @property
    def default_classification_name(self) -> str | None:
        """Return the optional default Axys classification name.

        Returns:
            Configured default classification, or ``None`` when omitted.

        Raises:
            PpaError: If the configured value is not a string.
        """
        value = self._defaults.get(_DEFAULT_CLASSIFICATION_KEY)
        if value is None:
            return None
        if not isinstance(value, str):
            raise PpaError(
                self._error_message(
                    f"{_DEFAULTS_KEY}.{_DEFAULT_CLASSIFICATION_KEY} must be a string."
                ),
                504,
            )
        return value or None

    @property
    def classifications(self) -> dict[str, dict[str, Any]]:
        """Return configured Axys classification definitions.

        Returns:
            Classification definitions keyed by user-facing configuration
            name. Missing sections are treated as empty.
        """
        return cast(dict[str, dict[str, Any]], self.values.get("classifications", {}))

    @property
    def mappings(self) -> dict[str, dict[str, Any]]:
        """Return configured Axys security-to-grouping mappings.

        Returns:
            Mapping definitions keyed by user-facing configuration name.
            Missing sections are treated as empty.
        """
        return cast(dict[str, dict[str, Any]], self.values.get("mappings", {}))

    def is_security_master(self, classification_name: str) -> bool:
        """Return whether a configured classification is the security master.

        Args:
            classification_name: Configured classification to inspect.

        Returns:
            ``True`` if the classification is marked as the security master;
            otherwise, ``False``.

        Notes:
            ``Security`` is a built-in security-master classification when it
            is not explicitly configured.
        """
        if classification_name == "Security":
            return True
        return bool(
            self.classifications.get(classification_name, {}).get(
                "is_security_master",
                False,
            )
        )

    @property
    def _defaults(self) -> dict[str, Any]:
        """Return optional user defaults from the Axys specification."""
        defaults = self.values.get(_DEFAULTS_KEY, {})
        if not isinstance(defaults, dict):
            raise PpaError(
                self._error_message(f"{_DEFAULTS_KEY} must be a mapping."),
                504,
            )
        return cast(dict[str, Any], defaults)

    def _default_date(self, key: Literal["from_date", "thru_date"]) -> dt.date | None:
        """Return an optional default date from a YAML date or ISO string."""
        value = self._defaults.get(key)
        if value is None:
            return None
        if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
            return value
        if isinstance(value, str):
            try:
                return dt.date.fromisoformat(value)
            except ValueError as error:
                raise PpaError(
                    self._error_message(
                        f"{_DEFAULTS_KEY}.{key} must be an ISO date like 2024-01-01."
                    ),
                    504,
                ) from error
        raise PpaError(
            self._error_message(
                f"{_DEFAULTS_KEY}.{key} must be a date or ISO date string."
            ),
            504,
        )
