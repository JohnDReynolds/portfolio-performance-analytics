"""Tests for package metadata maintained alongside runtime dependencies."""

# Python Imports
import subprocess
import sys
import tomllib
import unittest

# Project Imports
import ppar.utilities as util
from ppar import axys
from ppar.axys import AxysData, AxysPortfolio, AxysSpecification, AxysSupportingSources


class TestPackageMetadata(unittest.TestCase):
    """Verify package dependency metadata agrees with development requirements."""

    def test_dependency_metadata(self) -> None:
        """Runtime dependencies are represented by the requirements file."""
        with open("pyproject.toml", "rb") as file:
            pyproject = tomllib.load(file)
        pyproject_dependencies = {
            dependency.split(">=", maxsplit=1)[0].lower()
            for dependency in pyproject["project"]["dependencies"]
        }
        with open("requirements.txt", "r", encoding=util.ENCODING) as file:
            requirements_dependencies = {
                line.split(">=", maxsplit=1)[0].strip().lower()
                for line in file
                if line.strip()
            }

        self.assertNotIn("great_tables", pyproject_dependencies)
        self.assertNotIn("great_tables", requirements_dependencies)
        self.assertIn("pyyaml", pyproject_dependencies)
        self.assertTrue(pyproject_dependencies.issubset(requirements_dependencies))

    def test_axys_package_is_included(self) -> None:
        """The Axys subpackage is included in distribution metadata."""
        with open("pyproject.toml", "rb") as file:
            pyproject = tomllib.load(file)

        self.assertIn("ppar.axys", pyproject["tool"]["setuptools"]["packages"])

    def test_public_axys_import_contract(self) -> None:
        """The documented Axys package exports remain importable."""
        expected_exports = {
            "AxysData",
            "AxysPortfolio",
            "AxysSpecification",
            "AxysSupportingSources",
        }

        self.assertEqual(set(axys.__all__), expected_exports)
        self.assertIs(AxysData, axys.AxysData)
        self.assertIs(AxysPortfolio, axys.AxysPortfolio)
        self.assertIs(AxysSpecification, axys.AxysSpecification)
        self.assertIs(AxysSupportingSources, axys.AxysSupportingSources)

    def test_chart_dependencies_are_optional(self) -> None:
        """Normal package imports do not load optional chart rendering code."""
        with open("pyproject.toml", "rb") as file:
            pyproject = tomllib.load(file)

        chart_dependencies = {
            dependency.split(">=", maxsplit=1)[0].lower()
            for dependency in pyproject["project"]["optional-dependencies"]["charts"]
        }
        core_dependencies = {
            dependency.split(">=", maxsplit=1)[0].lower()
            for dependency in pyproject["project"]["dependencies"]
        }
        command = (
            "import sys; import ppar; "
            "raise SystemExit(1 if 'ppar.format_chart' in sys.modules else 0)"
        )

        self.assertEqual(chart_dependencies, {"matplotlib", "seaborn"})
        self.assertTrue(chart_dependencies.isdisjoint(core_dependencies))
        subprocess.run([sys.executable, "-c", command], check=True)


if __name__ == "__main__":
    unittest.main()
