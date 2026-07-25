"""Lightweight constants, path types, and file validation shared by PPAR."""

from pathlib import Path
from typing import TypeAlias

import ppar.errors as errs

__all__ = [
    "ENCODING",
    "PathLike",
    "file_basename_without_extension",
    "file_path_error",
    "file_path_exists",
]

PathLike: TypeAlias = str | Path
ENCODING = "utf-8"


def file_basename_without_extension(file_path: PathLike) -> str:
    """Return a file name without its directory or extension.

    Args:
        file_path: File path to evaluate.

    Returns:
        Base file name before the first period in the file name.
    """
    return Path(file_path).name.split(".")[0]


def file_path_error(file_path: PathLike) -> str:
    """Return the appropriate file-path validation message.

    Args:
        file_path: File path that failed validation.

    Returns:
        Empty-path error text for a blank string, otherwise missing-file error
        text followed by the path.
    """
    is_blank_path = isinstance(file_path, str) and not file_path.strip()
    return errs.ERRORS[804] if is_blank_path else f"{errs.ERRORS[802]}{file_path}"


def file_path_exists(file_path: PathLike) -> bool:
    """Return whether a nonblank path exists and points to a file.

    Args:
        file_path: File path to test.

    Returns:
        ``True`` only for an existing file.
    """
    if isinstance(file_path, str) and not file_path.strip():
        return False
    return Path(file_path).is_file()
