"""Tests for deterministic README image rendering helpers."""

from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from scripts import render_readme_images


class TestRenderReadmeImages(unittest.TestCase):
    """Verify headless-browser failures remain bounded and recoverable."""

    def test_image_families_run_in_isolated_processes(self) -> None:
        """Large chart and browser renderers do not retain each other's memory."""
        with mock.patch.object(
            render_readme_images.subprocess,
            "run",
        ) as run:
            render_readme_images._render_isolated_image_families(
                ("analytics-charts", "security-attribution")
            )

        self.assertEqual(run.call_count, 2)
        self.assertEqual(
            [
                call.args[0][-2:]
                for call in run.call_args_list
            ],
            [
                ["--only", "analytics-charts"],
                ["--only", "security-attribution"],
            ],
        )
        self.assertTrue(
            all(call.kwargs["check"] is True for call in run.call_args_list)
        )
        self.assertEqual(
            [
                call.kwargs["env"]["PPAR_README_IMAGE_CACHE_SCOPE"]
                for call in run.call_args_list
            ],
            ["analytics-charts", "security-attribution"],
        )

    def test_render_png_retries_one_transient_browser_crash(self) -> None:
        """A first Chrome abort receives one fresh-profile retry."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            html_path = root / "report.html"
            png_path = root / "report.png"
            html_path.write_text("<html></html>", encoding="utf-8")
            with mock.patch.object(
                render_readme_images.subprocess,
                "run",
                side_effect=(
                    subprocess.CalledProcessError(1, ["chrome"]),
                    mock.DEFAULT,
                ),
            ) as run:
                render_readme_images._render_png(
                    "chrome",
                    html_path,
                    png_path,
                    (100, 100),
                    root / "chrome_profile",
                )

        self.assertEqual(run.call_count, 2)
        first_command = run.call_args_list[0].args[0]
        retry_command = run.call_args_list[1].args[0]
        self.assertIn(f"--user-data-dir={root / 'chrome_profile'}", first_command)
        self.assertIn(
            f"--user-data-dir={root / 'chrome_profile_retry'}",
            retry_command,
        )

    def test_render_png_raises_after_two_browser_crashes(self) -> None:
        """Persistent Chrome failure remains a release-stopping error."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            html_path = root / "report.html"
            png_path = root / "report.png"
            html_path.write_text("<html></html>", encoding="utf-8")
            with mock.patch.object(
                render_readme_images.subprocess,
                "run",
                side_effect=subprocess.CalledProcessError(1, ["chrome"]),
            ) as run:
                with self.assertRaises(subprocess.CalledProcessError):
                    render_readme_images._render_png(
                        "chrome",
                        html_path,
                        png_path,
                        (100, 100),
                        root / "chrome_profile",
                    )

        self.assertEqual(run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
