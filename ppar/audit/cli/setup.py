"""Create a local PPAR Audit or PPAR Analytics workspace."""

from __future__ import annotations

# Python imports
import argparse
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
import sys
from typing import Any, Final, Literal

import yaml

# Project imports
from ppar.errors import PpaError
from ppar.audit.config_validation import validate_config
import ppar.common as util

_CONFIG_FILE_NAME: Final[str] = "ppar.yaml"
_SNAPSHOT_A_DIR: Final[str] = "snapshot_a"
_SNAPSHOT_B_DIR: Final[str] = "snapshot_b"
_PACKAGED_DEMO_RESOURCE: Final[str] = "ppar.setup_templates"
_PACKAGED_ANALYTICS_DIRECTORY: Final[str] = "axys_apx_analytics"
_PACKAGED_AUDIT_DIRECTORY: Final[str] = "axys_apx_audit"
_PACKAGED_GENERIC_ANALYTICS_DIRECTORY: Final[str] = "generic_analytics"
_PACKAGED_ANALYTICS_YAML: Final[str] = "axys_apx_analytics.yaml"
_PACKAGED_AUDIT_YAML: Final[str] = "axys_apx_audit.yaml"
_WorkspaceKind = Literal["audit", "analytics", "generic_analytics"]
_ANALYTICS_SETUP_FILES: Final[tuple[str, ...]] = (
    "portperf.csv",
    "secperf.csv",
    "secmast.csv",
    "run_analytics.py",
)
_PORTFOLIO_SETUP_FILES: Final[tuple[str, ...]] = (
    "portperf.csv",
    "holdings.csv",
    "transactions.csv",
    "secmast.csv",
    "splits.csv",
)
_AUDIT_SETUP_FILES: Final[tuple[str, ...]] = (
    "run_audit.py",
)


def main(argv: list[str] | None = None) -> int:
    """Run the PPAR workspace setup command.

    Args:
        argv: Optional command-line arguments excluding the executable name.

    Returns:
        Process exit code. ``0`` indicates that the requested workspace exists
        and, for Audit, its configuration validates.
    """
    args = _argument_parser().parse_args(argv)
    try:
        result = run_setup(
            args.workspace_directory,
            analytics=args.analytics,
            generic_analytics=args.generic_analytics,
            overwrite=args.overwrite,
        )
    except PpaError as error:
        print(f"Setup failed: {error}", file=sys.stderr)
        return 1

    _print_success(result)
    return 0


def run_setup(
    workspace_directory: Path,
    *,
    analytics: bool = False,
    generic_analytics: bool = False,
    overwrite: bool = False,
) -> dict[str, Path | str]:
    """Create one self-contained PPAR workspace.

    Args:
        workspace_directory: Folder that will receive the selected workflow.
        analytics: Whether to create an Analytics workspace instead of the
            default Audit workspace.
        generic_analytics: Whether to create a Generic Analytics Python
            workspace instead of the default Audit workspace.
        overwrite: Whether to replace existing packaged workspace files.

    Returns:
        Paths and status labels for the selected workspace.

    Raises:
        PpaError: If a destination path is an existing non-directory, if the
            requested workflow conflicts with an existing workspace, if files
            cannot be written, or if Audit validation fails.
    """
    if analytics and generic_analytics:
        raise PpaError(
            "--analytics and --generic-analytics are mutually exclusive.",
            802,
        )

    workspace_path = Path(workspace_directory).expanduser()
    workflow: _WorkspaceKind = (
        "generic_analytics"
        if generic_analytics
        else "analytics" if analytics else "audit"
    )
    setup_status = _ensure_directory(workspace_path)
    _validate_workspace_kind(workspace_path, workflow)
    if generic_analytics:
        workflow_status, readme_status = _ensure_generic_analytics_workspace(
            workspace_path,
            overwrite=overwrite,
        )
        return {
            "workspace_directory": workspace_path,
            "workflow": workflow,
            "setup_status": setup_status,
            "readme_path": workspace_path / "README.md",
            "readme_status": readme_status,
            "workflow_status": workflow_status,
        }

    readme_status = (
        _write_text_file(
            workspace_path / "README.md",
            _analytics_workspace_readme_text(workspace_path),
            overwrite=overwrite,
        )
        if analytics
        else _copy_resource_file(
            files(_PACKAGED_DEMO_RESOURCE).joinpath(
                _PACKAGED_AUDIT_DIRECTORY,
                "README.md",
            ),
            workspace_path / "README.md",
            overwrite=overwrite,
        )
    )
    if analytics:
        workflow_status = _ensure_analytics_workspace(
            workspace_path,
            overwrite=overwrite,
        )
        result: dict[str, Path | str] = {
            "workspace_directory": workspace_path,
            "workflow": workflow,
            "setup_status": setup_status,
            "readme_path": workspace_path / "README.md",
            "readme_status": readme_status,
            "workflow_status": workflow_status,
            "config_path": workspace_path / _CONFIG_FILE_NAME,
        }
        return result

    workflow_status = _ensure_audit_workspace(
        workspace_path,
        overwrite=overwrite,
    )
    audit_config_path = workspace_path / _CONFIG_FILE_NAME
    result = {
        "workspace_directory": workspace_path,
        "workflow": workflow,
        "setup_status": setup_status,
        "readme_path": workspace_path / "README.md",
        "readme_status": readme_status,
        "workflow_status": workflow_status,
        "config_path": audit_config_path,
    }
    missing_files = _missing_snapshot_files(workspace_path)
    if missing_files:
        result["missing_files"] = "\n".join(missing_files)
        return result

    validate_config(audit_config_path)
    result["validation_status"] = "audit-ready"
    return result


def _argument_parser() -> argparse.ArgumentParser:
    """Return the setup argument parser."""
    parser = argparse.ArgumentParser(
        prog="ppar setup",
        description="Create a PPAR Audit workspace.",
        epilog=(
            "Examples:\n"
            "  ppar setup ./my_ppar_audit\n"
            "  ppar setup ./my_ppar_analytics --analytics\n"
            "  ppar setup ./my_ppar_generic_analytics --generic-analytics"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "workspace_directory",
        type=Path,
        help="Folder that will contain the selected PPAR workspace.",
    )
    workflow_options = parser.add_mutually_exclusive_group()
    workflow_options.add_argument(
        "--analytics",
        action="store_true",
        help="Create a PPAR Analytics workspace instead of a PPAR Audit workspace.",
    )
    workflow_options.add_argument(
        "--generic-analytics",
        action="store_true",
        help=(
            "Create a Generic Analytics Python workspace instead of a PPAR "
            "Audit workspace."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing workspace files with packaged files.",
    )
    return parser


def _ensure_directory(directory: Path) -> str:
    """Create a directory when needed and return a status label."""
    if directory.exists() and not directory.is_dir():
        raise PpaError(
            f"{directory} exists but is not a directory.",
            802,
        )
    if directory.exists():
        return "existing"
    directory.mkdir(parents=True)
    return "created"


def _validate_workspace_kind(
    workspace_path: Path,
    requested_kind: _WorkspaceKind,
) -> None:
    """Reject setup requests that would mix workflow files.

    Args:
        workspace_path: Existing or newly created workspace directory.
        requested_kind: Workflow requested by the current setup invocation.

    Raises:
        PpaError: If the destination is a legacy combined setup root or its
            configuration belongs to the other workflow.
    """
    legacy_audit_config = workspace_path / "audit" / _CONFIG_FILE_NAME
    legacy_analytics_config = workspace_path / "analytics" / _CONFIG_FILE_NAME
    if legacy_audit_config.exists() or legacy_analytics_config.exists():
        raise PpaError(
            f"{workspace_path} is a legacy combined PPAR workspace. Continue "
            "using its audit/ and analytics/ folders, or choose a new workspace "
            "directory.",
            802,
        )

    existing_kinds: set[_WorkspaceKind] = set()
    if (workspace_path / "run_generic_analytics.py").is_file():
        existing_kinds.add("generic_analytics")

    config_path = workspace_path / _CONFIG_FILE_NAME
    if config_path.exists():
        try:
            values: Any = yaml.safe_load(
                config_path.read_text(encoding=util.ENCODING)
            )
        except yaml.YAMLError:
            values = None
        if isinstance(values, dict):
            existing_kinds.update(
                kind for kind in ("audit", "analytics") if kind in values
            )

    if not existing_kinds or existing_kinds == {requested_kind}:
        return
    existing_label = " and ".join(sorted(existing_kinds))
    raise PpaError(
        f"{workspace_path} contains an existing {existing_label} workspace and cannot "
        f"be initialized as {requested_kind}. Choose a different workspace "
        "directory.",
        802,
    )


def _missing_snapshot_files(site_path: Path) -> list[str]:
    """Return user-facing labels for expected portfolio files not present yet."""
    missing_files: list[str] = []
    for snapshot_dir in (_SNAPSHOT_A_DIR, _SNAPSHOT_B_DIR):
        snapshot_path = site_path / snapshot_dir
        for file_name in _PORTFOLIO_SETUP_FILES:
            if not (snapshot_path / file_name).exists():
                missing_files.append(f"{snapshot_dir}/{file_name}")
    return missing_files


def _ensure_analytics_workspace(directory: Path, *, overwrite: bool) -> str:
    """Copy the Axys/APX Analytics workspace files into ``directory``."""
    status = _ensure_directory(directory)
    source_directory = files(_PACKAGED_DEMO_RESOURCE).joinpath(
        _PACKAGED_ANALYTICS_DIRECTORY
    )
    config_status = _copy_resource_file(
        source_directory.joinpath(_PACKAGED_ANALYTICS_YAML),
        directory / _CONFIG_FILE_NAME,
        overwrite=overwrite,
    )
    for file_name in _ANALYTICS_SETUP_FILES:
        _copy_resource_file(
            source_directory.joinpath(file_name),
            directory / file_name,
            overwrite=overwrite,
        )
    return _combined_status(status, config_status)


def _ensure_audit_workspace(directory: Path, *, overwrite: bool) -> str:
    """Copy the Axys/APX Audit workspace files into ``directory``."""
    status = _ensure_directory(directory)
    source_directory = files(_PACKAGED_DEMO_RESOURCE).joinpath(
        _PACKAGED_AUDIT_DIRECTORY
    )
    config_status = _write_text_file(
        directory / _CONFIG_FILE_NAME,
        _audit_workspace_config_text(
            source_directory.joinpath(_PACKAGED_AUDIT_YAML)
        ),
        overwrite=overwrite,
    )
    for snapshot_dir in (_SNAPSHOT_A_DIR, _SNAPSHOT_B_DIR):
        _copy_resource_tree(
            source_directory.joinpath(snapshot_dir),
            directory / snapshot_dir,
            overwrite=overwrite,
        )
    for file_name in _AUDIT_SETUP_FILES:
        _copy_resource_file(
            source_directory.joinpath(file_name),
            directory / file_name,
            overwrite=overwrite,
        )
    return _combined_status(status, config_status)


def _ensure_generic_analytics_workspace(
    directory: Path,
    *,
    overwrite: bool,
) -> tuple[str, str]:
    """Copy the Generic Analytics workspace files into ``directory``."""
    status = _ensure_directory(directory)
    source_directory = files(_PACKAGED_DEMO_RESOURCE).joinpath(
        _PACKAGED_GENERIC_ANALYTICS_DIRECTORY
    )
    readme_status = _copy_resource_file(
        source_directory.joinpath("README.md"),
        directory / "README.md",
        overwrite=overwrite,
    )
    for resource in source_directory.iterdir():
        if resource.name == "README.md":
            continue
        destination = directory / resource.name
        if resource.is_dir():
            _copy_resource_tree(resource, destination, overwrite=overwrite)
        elif resource.is_file():
            _copy_resource_file(resource, destination, overwrite=overwrite)
    return _combined_status(status, readme_status), readme_status


def _audit_workspace_config_text(resource: Traversable) -> str:
    """Return the documented one-file Audit workspace YAML."""
    return resource.read_text(encoding=util.ENCODING)


def _analytics_workspace_readme_text(workspace_path: Path) -> str:
    """Return the README for a generated PPAR Analytics workspace."""
    return f"""# PPAR Analytics Workspace

This folder was created by:

```bash
ppar setup {workspace_path} --analytics
```

## What This Folder Is For

This workspace contains demonstration portfolio, benchmark, and classification
exports for Performance Analytics. Run the demonstration first, then replace
the CSV files with approved exports from your own environment.

Performance Analytics explains portfolio results relative to a benchmark:

- **Performance Attribution:** Brinson-Fachler attribution, Carino-smoothed
  multi-period effects, and contribution views.
- **Ex-Post Risk:** risk statistics calculated from realized returns.

## First Run

Install the optional Analytics dependencies and run the workspace:

```bash
pip install "ppar[analytics]"
ppar analytics {workspace_path}
```

## Customizing With Your Own Data

1. Replace `portperf.csv` with your portfolio-performance export.
2. Replace `secperf.csv` with your security-performance export.
3. Replace `secmast.csv` with your security-master export.
4. Edit `ppar.yaml` if your filenames, headers, or report choices differ.
5. Run `ppar analytics {workspace_path}`.

## Optional Python Script

`run_analytics.py` is the visible Python equivalent of the normal command and
is the starting point for local output customization.

```bash
ppar analytics -h
python run_analytics.py -h
```

## Folder Map

```text
{workspace_path.name}/
  README.md
  ppar.yaml
  portperf.csv
  secperf.csv
  secmast.csv
  run_analytics.py
```
"""


def _copy_resource_tree(
    source_directory: Traversable,
    destination_directory: Path,
    *,
    overwrite: bool,
) -> None:
    """Copy resource files from one packaged directory to a local directory."""
    _ensure_directory(destination_directory)
    for resource in source_directory.iterdir():
        destination = destination_directory / resource.name
        if resource.is_dir():
            _copy_resource_tree(resource, destination, overwrite=overwrite)
        elif resource.is_file():
            _copy_resource_file(resource, destination, overwrite=overwrite)


def _copy_resource_file(
    resource: Traversable,
    destination: Path,
    *,
    overwrite: bool,
) -> str:
    """Copy one packaged resource file and return its status label."""
    if destination.exists() and not overwrite:
        return "existing"
    existed_before = destination.exists()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(resource.read_bytes())
    return "updated" if existed_before else "written"


def _write_text_file(destination: Path, text: str, *, overwrite: bool) -> str:
    """Write a workspace text file and return its status label."""
    if destination.exists() and not overwrite:
        return "existing"
    existed_before = destination.exists()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding=util.ENCODING)
    return "updated" if existed_before else "written"


def _combined_status(directory_status: str, config_status: str) -> str:
    """Return a compact status label for a workspace folder."""
    if directory_status == "created":
        return "created"
    return config_status


def _print_success(result: dict[str, Path | str]) -> None:
    """Print a concise user handoff."""
    workspace_path = result["workspace_directory"]
    workflow = str(result["workflow"])
    if workflow == "generic_analytics":
        print(f"PPAR Generic Analytics workspace ready: {workspace_path}")
        print()
        print("To run Generic Analytics:")
        print(f"  python {Path(workspace_path) / 'run_generic_analytics.py'}")
        print()
        print("To customize with your own data:")
        print(
            "  Refer to the \"Customizing With Your Own Data\" section in "
            f"{Path(workspace_path) / 'README.md'}"
        )
        return

    label = "Analytics" if workflow == "analytics" else "Audit"
    print(f"PPAR {label} workspace ready: {workspace_path}")
    print()
    print(f"To run {label}:")
    print(f"  ppar {workflow} {workspace_path}")
    print()
    if "missing_files" in result:
        print("Next step: add these portfolio source files, then run setup again:")
        print(result["missing_files"])
        return

    print("To customize with your own data:")
    readme_path = Path(str(workspace_path)) / "README.md"
    print(
        f"  Refer to the \"Customizing With Your Own Data\" section in "
        f"{readme_path}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
