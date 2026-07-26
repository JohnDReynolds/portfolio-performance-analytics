"""Tests for the packaged Audit demo health-check script."""

from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import scripts.check_audit_demo_health as demo_health


def _command_text(command: tuple[str, ...]) -> str:
    """Return a readable command line from a captured command tuple."""
    return " ".join(command)


class TestAuditDemoHealthScript(unittest.TestCase):
    """Verify demo-health command selection without running nested checks."""

    def test_default_mode_runs_all_demo_health_guards(self) -> None:
        """Default mode runs rebuild, docs, bundle, and matrix checks."""
        commands: list[tuple[str, ...]] = []

        with (
            patch.object(demo_health, "_require_venv_python"),
            patch.object(
                demo_health,
                "_run",
                side_effect=lambda command: commands.append(
                    tuple(str(part) for part in command)
                ),
            ),
        ):
            exit_code = demo_health.main([])

        command_texts = [_command_text(command) for command in commands]
        self.assertEqual(exit_code, 0)
        self.assertTrue(
            any(
                "rebuild_audit_demo_data.py" in text
                for text in command_texts
            )
        )
        self.assertTrue(
            any(
                "render_demo_extract_availability.py --check" in text
                for text in command_texts
            )
        )
        self.assertTrue(
            any(
                "ppar.cli setup" in text
                and "my_ppar_audit" in text
                for text in command_texts
            )
        )
        self.assertTrue(
            any(
                "my_ppar_audit/run_audit.py" in text
                for text in command_texts
            )
        )
        self.assertFalse(any("run_analytics.py" in text for text in command_texts))
        self.assertTrue(
            any(
                "validate_bundle" in text
                and "output/portfolio" in text
                for text in command_texts
            )
        )
        self.assertTrue(
            any(
                "validate_bundle" in text
                and "output/security" in text
                for text in command_texts
            )
        )
        self.assertTrue(any("validate_demo_matrix" in text for text in command_texts))

    def test_skip_options_can_run_matrix_only(self) -> None:
        """Skip options make it possible to run only scenario-matrix validation."""
        commands: list[tuple[str, ...]] = []

        with (
            patch.object(demo_health, "_require_venv_python"),
            patch.object(
                demo_health,
                "_run",
                side_effect=lambda command: commands.append(
                    tuple(str(part) for part in command)
                ),
            ),
        ):
            exit_code = demo_health.main(
                [
                    "--skip-rebuild-audit",
                    "--skip-extract-availability",
                    "--skip-bundles",
                ]
            )

        command_texts = [_command_text(command) for command in commands]
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(command_texts), 1)
        self.assertIn("validate_demo_matrix", command_texts[0])

    def test_write_packaged_assets_is_forwarded_to_rebuild(self) -> None:
        """The release wrapper can request an intentional demo-data rewrite."""
        commands: list[tuple[str, ...]] = []

        with (
            patch.object(demo_health, "_require_venv_python"),
            patch.object(
                demo_health,
                "_run",
                side_effect=lambda command: commands.append(
                    tuple(str(part) for part in command)
                ),
            ),
        ):
            exit_code = demo_health.main(
                [
                    "--write-packaged-assets",
                    "--skip-extract-availability",
                    "--skip-bundles",
                    "--skip-demo-matrix",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(commands), 1)
        self.assertIn("rebuild_audit_demo_data.py", _command_text(commands[0]))
        self.assertEqual(commands[0][-1], "--write")

    def test_script_requires_project_venv_python(self) -> None:
        """The script documents and enforces the project virtual environment."""
        self.assertEqual(
            demo_health._VENV_PYTHON,
            Path.cwd() / ".venv" / "bin" / "python",
        )


if __name__ == "__main__":
    unittest.main()
