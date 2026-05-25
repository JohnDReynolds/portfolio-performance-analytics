"""Tests for package metadata maintained alongside runtime dependencies."""

# Python Imports
import tomllib
import unittest

# Project Imports
import ppar.utilities as util


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


if __name__ == "__main__":
    unittest.main()
