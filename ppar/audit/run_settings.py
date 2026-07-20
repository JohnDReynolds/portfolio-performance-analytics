"""Validate and resolve normal Audit run settings."""

from __future__ import annotations

# Python imports
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

# Project imports
from ppar.errors import PpaError

AUDIT_SECTION: Final[str] = "audit"
_AUDIT_SETTING_DEFAULTS: Final[dict[str, str | bool | None]] = {
    "output_directory": "output",
    "title": None,
    "xlsx_output": True,
    "html_output": True,
    "exclude_suppressed": False,
    "reconstruction_diagnostics": False,
    "expand_all_supporting_files": False,
    "require_causal_attribution": False,
}
_AUDIT_SETTING_NAMES: Final[frozenset[str]] = frozenset(_AUDIT_SETTING_DEFAULTS)


@dataclass(frozen=True)
class AuditRunSettings:
    """Resolved presentation and validation settings for one audit run.

    Attributes:
        output_directory: Root directory receiving both report levels.
        title: Shared visible report title, or ``None`` for level-specific titles.
        exclude_suppressed: Whether suppressed findings are omitted from output.
        include_reconstruction_diagnostics: Whether detailed reconstruction
            diagnostics are included.
        require_causal_attribution: Whether supported changed periods require
            complete causal-method setup.
        allow_incomplete_yaml: Whether the CLI-only diagnostic bypass is active.
        include_workbook: Whether each report level writes XLSX.
        include_html_output: Whether each report level writes HTML.
        expand_all_supporting_files: Whether supporting artifacts are expanded
            instead of archived.
    """

    output_directory: Path
    title: str | None
    exclude_suppressed: bool
    include_reconstruction_diagnostics: bool
    require_causal_attribution: bool
    allow_incomplete_yaml: bool
    include_workbook: bool
    include_html_output: bool
    expand_all_supporting_files: bool


def audit_settings(
    config_values: dict[str, Any],
    *,
    required: bool,
) -> dict[str, Any]:
    """Return and validate the Audit run-settings mapping.

    Args:
        config_values: Parsed root YAML mapping.
        required: Whether absence of the section is an error. Normal site runs
            require it; lower-level comparison specifications do not.

    Returns:
        The validated settings mapping, or an empty mapping when the optional
        section is absent.

    Raises:
        PpaError: If the section is missing when required, malformed, contains
            an unsupported setting, or has an invalid configured value.
    """
    if AUDIT_SECTION not in config_values and not required:
        return {}
    settings = config_values.get(AUDIT_SECTION)
    if not isinstance(settings, dict):
        raise PpaError(f"{AUDIT_SECTION} must be a mapping.", 504)
    unsupported = sorted(
        str(key) for key in settings if key not in _AUDIT_SETTING_NAMES
    )
    if unsupported:
        raise PpaError(
            "audit has unsupported keys: " + ", ".join(unsupported) + ".",
            504,
        )
    _validate_configured_values(settings)
    return settings


def resolve_settings(
    site_path: Path,
    values: dict[str, Any],
    *,
    output_directory: Path | None,
    title: str | None,
    exclude_suppressed: bool | None,
    include_reconstruction_diagnostics: bool | None,
    require_causal_attribution: bool | None,
    allow_incomplete_yaml: bool,
    include_workbook: bool | None,
    include_html_output: bool | None,
    expand_all_supporting_files: bool | None,
) -> AuditRunSettings:
    """Resolve command-line overrides, YAML settings, and defaults.

    Args:
        site_path: Audit site directory used to anchor configured output.
        values: Validated ``audit`` YAML mapping.
        output_directory: One-run output-directory override.
        title: One-run report-title override.
        exclude_suppressed: One-run suppression-display override.
        include_reconstruction_diagnostics: One-run diagnostics override.
        require_causal_attribution: One-run causal-attribution guard override.
        allow_incomplete_yaml: CLI-only diagnostic safety bypass.
        include_workbook: One-run XLSX-output override.
        include_html_output: One-run HTML-output override.
        expand_all_supporting_files: One-run supporting-file layout override.

    Returns:
        Fully resolved settings for the Audit run.
    """
    configured_output = cast(
        str,
        values.get("output_directory", _AUDIT_SETTING_DEFAULTS["output_directory"]),
    )
    configured_title = cast(
        str | None,
        values.get("title", _AUDIT_SETTING_DEFAULTS["title"]),
    )
    return AuditRunSettings(
        output_directory=output_directory or site_path / configured_output,
        title=title if title is not None else configured_title,
        exclude_suppressed=_boolean_setting(
            exclude_suppressed,
            values,
            "exclude_suppressed",
        ),
        include_reconstruction_diagnostics=_boolean_setting(
            include_reconstruction_diagnostics,
            values,
            "reconstruction_diagnostics",
        ),
        require_causal_attribution=_boolean_setting(
            require_causal_attribution,
            values,
            "require_causal_attribution",
        ),
        allow_incomplete_yaml=allow_incomplete_yaml,
        include_workbook=_boolean_setting(
            include_workbook,
            values,
            "xlsx_output",
        ),
        include_html_output=_boolean_setting(
            include_html_output,
            values,
            "html_output",
        ),
        expand_all_supporting_files=_boolean_setting(
            expand_all_supporting_files,
            values,
            "expand_all_supporting_files",
        ),
    )


def _validate_configured_values(values: dict[str, Any]) -> None:
    """Validate value types before CLI overrides are considered."""
    if "output_directory" in values:
        configured_output = values["output_directory"]
        if not isinstance(configured_output, str) or not configured_output:
            raise PpaError("audit.output_directory must be a non-empty string.", 504)
    if "title" in values:
        configured_title = values["title"]
        if configured_title is not None and (
            not isinstance(configured_title, str) or not configured_title
        ):
            raise PpaError("audit.title must be null or a non-empty string.", 504)
    for name in (_AUDIT_SETTING_NAMES - {"output_directory", "title"}) & values.keys():
        if not isinstance(values[name], bool):
            raise PpaError(f"audit.{name} must be true or false.", 504)


def _boolean_setting(
    override: bool | None,
    values: dict[str, Any],
    name: str,
) -> bool:
    """Return one boolean using CLI, YAML, then default precedence."""
    value = (
        override
        if override is not None
        else values.get(name, _AUDIT_SETTING_DEFAULTS[name])
    )
    return cast(bool, value)
