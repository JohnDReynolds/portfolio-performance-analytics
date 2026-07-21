"""Tests for the local project health-check script."""

# Python Imports
import os
from pathlib import Path
import tempfile
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
        self.assertEqual(
            run.call_args.kwargs["env"]["PIP_CACHE_DIR"],
            str(check_project._CHECK_CACHE_DIR / "pip"),
        )

    def test_run_can_isolate_installed_package_checks(self) -> None:
        """Installed-wheel commands cannot inherit a checkout import path."""
        with (
            patch.dict(os.environ, {"PYTHONPATH": "/unexpected/source"}),
            patch("scripts.check_project.subprocess.run") as run,
        ):
            check_project._run(
                [check_project._VENV_PYTHON, "-V"],
                cwd=Path("/tmp"),
                isolate_python_path=True,
            )

        self.assertEqual(run.call_args.kwargs["cwd"], Path("/tmp"))
        self.assertNotIn("PYTHONPATH", run.call_args.kwargs["env"])

    def test_installed_wheel_smoke_covers_setup_analytics_and_audit(self) -> None:
        """The candidate wheel smoke exercises both products and validators."""
        calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

        def record_run(
            command: list[str | Path],
            **kwargs: object,
        ) -> None:
            calls.append((tuple(str(part) for part in command), kwargs))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel_path = root / "ppar-0.0.0-py3-none-any.whl"
            with patch.object(check_project, "_run", side_effect=record_run):
                check_project._run_installed_wheel_smoke(
                    wheel_path,
                    root / "smoke",
                )

        command_texts = [_command_text(command) for command, _ in calls]
        self.assertTrue(any("-m venv" in text for text in command_texts))
        self.assertTrue(any("-m pip install" in text for text in command_texts))
        self.assertTrue(any("-m pip check" in text for text in command_texts))
        self.assertTrue(any("ppar setup" in text for text in command_texts))
        self.assertTrue(any("ppar analytics" in text for text in command_texts))
        self.assertTrue(any("ppar audit" in text for text in command_texts))
        validator_calls = [
            text for text in command_texts if "ppar.audit.cli.validate_bundle" in text
        ]
        self.assertEqual(len(validator_calls), 2)
        self.assertTrue(any(text.endswith("output/portfolio") for text in validator_calls))
        self.assertTrue(any(text.endswith("output/security") for text in validator_calls))
        self.assertTrue(
            all(kwargs["isolate_python_path"] is True for _, kwargs in calls)
        )


if __name__ == "__main__":
    unittest.main()
