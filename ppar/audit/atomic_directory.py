"""Stage and atomically promote complete Audit output directories."""

from __future__ import annotations

# Python imports
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import shutil
import tempfile

# Project imports
from ppar.errors import PpaError
import ppar.common as util


@contextmanager
def staged_directory(destination: util.PathLike) -> Iterator[Path]:
    """Yield a sibling staging directory and promote it after successful use.

    Args:
        destination: Final directory that should receive the staged contents.

    Yields:
        Empty sibling directory for building and validating complete output.

    Raises:
        PpaError: If the destination exists but is not a directory, or if a
            failed promotion cannot restore the previous directory.
        OSError: If staging or promotion fails.

    Notes:
        The previous destination remains unchanged if work inside the context
        fails. Promotion briefly renames an existing destination to a sibling
        backup, installs the staged directory, and then removes the backup.
    """
    destination_path = Path(destination)
    if destination_path.exists() and not destination_path.is_dir():
        raise PpaError(
            f"{destination_path} exists but is not a directory.",
            802,
        )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_path.name}.staging-",
            dir=destination_path.parent,
        )
    )
    try:
        yield staging_path
    except BaseException:
        shutil.rmtree(staging_path, ignore_errors=True)
        raise

    try:
        _promote_staged_directory(staging_path, destination_path)
    except BaseException:
        shutil.rmtree(staging_path, ignore_errors=True)
        raise


@contextmanager
def staged_children(
    destination_root: util.PathLike,
    child_names: tuple[str, ...],
) -> Iterator[Path]:
    """Yield a staging root and atomically replace selected child directories.

    Args:
        destination_root: Parent containing the managed output directories.
        child_names: Exact child directory names owned by the current run.

    Yields:
        Empty sibling staging root. A missing managed child at successful
        promotion removes any older destination child with the same name.

    Raises:
        PpaError: If a managed destination is not a directory or recovery fails.
        OSError: If staging or promotion fails.

    Notes:
        Files and directories below ``destination_root`` whose names are not in
        ``child_names`` remain untouched.
    """
    destination_path = Path(destination_root)
    if destination_path.exists() and not destination_path.is_dir():
        raise PpaError(
            f"{destination_path} exists but is not a directory.",
            802,
        )
    if not child_names or len(set(child_names)) != len(child_names):
        raise ValueError("child_names must contain unique managed directory names.")
    if any(Path(name).name != name or name in {"", ".", ".."} for name in child_names):
        raise ValueError("child_names must contain plain directory names.")

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_path.name}.run-staging-",
            dir=destination_path.parent,
        )
    )
    try:
        yield staging_root
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise

    try:
        _promote_staged_children(
            staging_root,
            destination_path,
            child_names,
        )
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise


def remap_staged_path(
    path: Path,
    *,
    staging_root: Path,
    destination_root: Path,
) -> Path:
    """Return the final path corresponding to a path inside a staging root.

    Args:
        path: Artifact path returned while writing the staging directory.
        staging_root: Root directory used for the staged build.
        destination_root: Final promoted directory.

    Returns:
        Artifact path below ``destination_root`` with the same relative path.
    """
    return destination_root / path.relative_to(staging_root)


def _promote_staged_directory(staging_path: Path, destination_path: Path) -> None:
    """Replace one destination directory while preserving recovery on failure."""
    if not destination_path.exists():
        staging_path.replace(destination_path)
        return

    backup_path = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_path.name}.backup-",
            dir=destination_path.parent,
        )
    )
    backup_path.rmdir()
    destination_path.replace(backup_path)
    try:
        staging_path.replace(destination_path)
    except BaseException as promotion_error:
        try:
            backup_path.replace(destination_path)
        except OSError as restoration_error:
            raise PpaError(
                "Audit output promotion failed and the previous output could "
                f"not be restored from {backup_path}: {restoration_error}",
                999,
            ) from promotion_error
        raise
    shutil.rmtree(backup_path, ignore_errors=True)


def _promote_staged_children(
    staging_root: Path,
    destination_root: Path,
    child_names: tuple[str, ...],
) -> None:
    """Promote a complete set of managed children with rollback."""
    for child_name in child_names:
        staged_child = staging_root / child_name
        destination_child = destination_root / child_name
        if staged_child.exists() and not staged_child.is_dir():
            raise PpaError(
                f"Staged Audit output {staged_child} is not a directory.",
                999,
            )
        if destination_child.exists() and not destination_child.is_dir():
            raise PpaError(
                f"{destination_child} exists but is not a directory.",
                802,
            )

    destination_existed = destination_root.exists()
    destination_root.mkdir(parents=True, exist_ok=True)
    backup_root = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_root.name}.run-backup-",
            dir=destination_root.parent,
        )
    )
    promoted_names: list[str] = []
    try:
        for child_name in child_names:
            destination_child = destination_root / child_name
            if destination_child.exists():
                destination_child.replace(backup_root / child_name)
        for child_name in child_names:
            staged_child = staging_root / child_name
            if staged_child.exists():
                staged_child.replace(destination_root / child_name)
                promoted_names.append(child_name)
    except BaseException as promotion_error:
        for child_name in promoted_names:
            shutil.rmtree(destination_root / child_name, ignore_errors=True)
        try:
            for child_name in child_names:
                backup_child = backup_root / child_name
                if backup_child.exists():
                    backup_child.replace(destination_root / child_name)
            if not destination_existed and not any(destination_root.iterdir()):
                destination_root.rmdir()
        except OSError as restoration_error:
            raise PpaError(
                "Audit run promotion failed and previous report directories "
                f"could not be restored from {backup_root}: {restoration_error}",
                999,
            ) from promotion_error
        raise

    shutil.rmtree(backup_root, ignore_errors=True)
    shutil.rmtree(staging_root, ignore_errors=True)
