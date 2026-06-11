"""Tests for the local project health-check script."""

# Python Imports
from pathlib import Path
import unittest
from unittest.mock import patch

# Project Imports
import scripts.check_project as check_project


def _command_text(command: tuple[str, ...]) -> str:
    """Return a readable command line from a captured command tuple."""
    return " ".join(command)


class TestCheckProjectScript(unittest.TestCase):
    """Verify check-project command selection without running nested checks."""

    def test_quick_mode_skips_mypy_but_keeps_fast_error_gates(self) -> None:
        """Quick mode runs tests, pyright errors, and pylint errors without mypy."""
        commands: list[tuple[str, ...]] = []

        with (
            patch.object(check_project, "_require_venv_python"),
            patch.object(
                check_project,
                "_run",
                side_effect=lambda command: commands.append(
                    tuple(str(part) for part in command)
                ),
            ),
            patch.object(check_project, "_run_build_check") as build_check,
        ):
            exit_code = check_project.main(["--quick"])

        command_texts = [_command_text(command) for command in commands]
        self.assertEqual(exit_code, 0)
        self.assertTrue(any("-m unittest discover tests" in text for text in command_texts))
        self.assertTrue(any("-m pyright --level error" in text for text in command_texts))
        self.assertTrue(any("-m pylint --errors-only" in text for text in command_texts))
        self.assertFalse(any("-m mypy" in text for text in command_texts))
        build_check.assert_not_called()

    def test_full_mode_runs_mypy_before_pyright(self) -> None:
        """The default mode retains the full static-check behavior."""
        commands: list[tuple[str, ...]] = []

        with (
            patch.object(check_project, "_require_venv_python"),
            patch.object(
                check_project,
                "_run",
                side_effect=lambda command: commands.append(
                    tuple(str(part) for part in command)
                ),
            ),
            patch.object(check_project, "_run_build_check"),
        ):
            exit_code = check_project.main([])

        command_texts = [_command_text(command) for command in commands]
        mypy_index = next(
            index for index, text in enumerate(command_texts) if "-m mypy" in text
        )
        pyright_index = next(
            index for index, text in enumerate(command_texts) if "-m pyright" in text
        )
        self.assertEqual(exit_code, 0)
        self.assertLess(mypy_index, pyright_index)

    def test_build_option_still_runs_with_quick_mode(self) -> None:
        """Quick mode can still request the temporary build check explicitly."""
        with (
            patch.object(check_project, "_require_venv_python"),
            patch.object(check_project, "_run"),
            patch.object(check_project, "_run_build_check") as build_check,
        ):
            exit_code = check_project.main(["--quick", "--build"])

        self.assertEqual(exit_code, 0)
        build_check.assert_called_once_with()

    def test_script_requires_project_venv_python(self) -> None:
        """The script documents and enforces the project virtual environment."""
        self.assertEqual(check_project._VENV_PYTHON, Path.cwd() / ".venv" / "bin" / "python")

    def test_run_sets_ignored_matplotlib_cache_dir(self) -> None:
        """Subprocess checks use a writable Matplotlib cache location."""
        with patch("scripts.check_project.subprocess.run") as run:
            check_project._run([check_project._VENV_PYTHON, "-V"])

        self.assertEqual(
            run.call_args.kwargs["env"]["MPLCONFIGDIR"],
            str(check_project._MPLCONFIGDIR),
        )
        self.assertEqual(
            run.call_args.kwargs["env"]["XDG_CACHE_HOME"],
            str(check_project._CHECK_CACHE_DIR),
        )


if __name__ == "__main__":
    unittest.main()
