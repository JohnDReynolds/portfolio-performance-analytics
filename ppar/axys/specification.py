"""Read and validate Axys YAML specifications."""

from __future__ import annotations

# Python imports
from pathlib import Path
from typing import Any, Callable, Literal

# Third-party imports
import yaml

# Project imports
from ppar.errors import PpaError
import ppar.utilities as util

ErrorMessage = Callable[[str], str]
SourceType = Literal["classification", "mapping"]


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
        specification_key: Literal["portperf_path", "secperf_path"],
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
        """Return optional text inserted between portfolio code and name.

        Returns:
            Configured separator text, or ``None`` if portfolio codes should
            not be prefixed to names.
        """
        return self.values.get("settings", {}).get("prefix_portfolio_code")

    def default_source_names(self, source_type: SourceType) -> tuple[str, ...]:
        """Return configured classification or mapping source names.

        Args:
            source_type: Kind of supporting source to enumerate.

        Returns:
            Configured source names in specification insertion order.
        """
        return tuple(self.values.get(f"{source_type}s", {}).keys())

    def is_security_master(self, classification_name: str) -> bool:
        """Return whether a configured classification is the security master.

        Args:
            classification_name: Configured classification to inspect.

        Returns:
            ``True`` if the classification is marked as the security master;
            otherwise, ``False``.

        Raises:
            KeyError: If ``classification_name`` is not configured.
        """
        return bool(
            self.values["classifications"][classification_name].get(
                "is_security_master", False
            )
        )
