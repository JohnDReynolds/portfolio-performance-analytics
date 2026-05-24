"""Focused tests for AxysData source and specification validation failures."""

# Python Imports
from pathlib import Path
import tempfile
from typing import cast
import unittest

# Third-Party Imports
import yaml

# Test Imports
from tests import test_utilities as test_util

# Project Imports
from ppar.axysdata import AxysData
import ppar.errors as errs
from ppar.errors import PpaError


def _assert_axys_error(
    test: unittest.TestCase,
    error_code: int,
    specifications_path: Path | None = None,
    portperf_file_name: str | None = "imex_portperf.csv",
    secperf_file_name: str | None = "imex_secperf.csv",
    portfolio_code: str = "PORT_SMALL",
    classification_name: str | None = None,
    mapping_name: str | None = None,
) -> None:
    """Assert that constructing AxysData fails with a numbered PpaError."""
    if specifications_path is None:
        specifications_path = test_util.axys_data_path("axysdata.yaml", ".yaml")
    portperf_path = (
        test_util.axys_data_path(portperf_file_name) if portperf_file_name is not None else None
    )
    secperf_path = (
        test_util.axys_data_path(secperf_file_name) if secperf_file_name is not None else None
    )

    with test.assertRaises(PpaError) as context:
        AxysData(
            specifications_path,
            portperf_path,
            secperf_path,
            portfolio_codes=(portfolio_code,),
            classification_names=classification_name,
            mapping_names=mapping_name,
        )

    test.assertTrue(
        str(context.exception).startswith(errs.ERRORS[error_code]),
        str(context.exception),
    )


def _write_yaml(directory: Path, contents: object) -> Path:
    """Write temporary YAML contents and return its path."""
    path = directory / "axysdata.yaml"
    path.write_text(yaml.safe_dump(contents), encoding="utf-8")
    return path


def _fixture_specification() -> dict[str, object]:
    """Load the committed valid/bad-case specification as mutable data."""
    path = test_util.axys_data_path("axysdata.yaml", ".yaml")
    specification: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(specification, dict)
    return cast(dict[str, object], specification)


class TestAxysValidation(unittest.TestCase):
    """Verify Axys input validation and numbered error behavior."""

    def test_missing_portperf_columns_raise_error_502(self) -> None:
        """Required portperf columns are validated before processing."""
        _assert_axys_error(self, 502, portperf_file_name="error_502_portperf.csv")

    def test_missing_secperf_columns_raise_error_502(self) -> None:
        """Required secperf columns are validated before processing."""
        _assert_axys_error(self, 502, secperf_file_name="error_502_secperf.csv")

    def test_material_reconciliation_difference_raises_error_503(self) -> None:
        """An unreconciled return difference outside tolerance is rejected."""
        _assert_axys_error(
            self,
            503,
            portperf_file_name="error_503_a_portperf.csv",
            secperf_file_name="error_503_a_secperf.csv",
            portfolio_code="PORT_FAIL_HIGH",
        )

    def test_equal_return_reconciliation_failure_raises_error_503(self) -> None:
        """Unachievable equal-security target returns are rejected."""
        _assert_axys_error(
            self,
            503,
            portperf_file_name="error_503_b_portperf.csv",
            secperf_file_name="error_503_b_secperf.csv",
            portfolio_code="PORT_FAIL_EQUAL",
        )

    def test_invalid_yaml_raises_error_504(self) -> None:
        """A syntactically invalid YAML specification is rejected."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "axysdata.yaml"
            path.write_text("classifications: [", encoding="utf-8")
            _assert_axys_error(self, 504, specifications_path=path)

    def test_non_mapping_yaml_root_raises_error_504(self) -> None:
        """A YAML list cannot be used as the specification object."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _write_yaml(Path(temp_dir), ["not", "a", "mapping"])
            _assert_axys_error(self, 504, specifications_path=path)

    def test_missing_portperf_path_raises_error_504(self) -> None:
        """Portperf must be provided either by argument or specification."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _write_yaml(Path(temp_dir), {})
            _assert_axys_error(self, 504, specifications_path=path, portperf_file_name=None)

    def test_missing_secperf_path_raises_error_504(self) -> None:
        """Secperf must be provided either by argument or specification."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _write_yaml(Path(temp_dir), {})
            _assert_axys_error(self, 504, specifications_path=path, secperf_file_name=None)

    def test_missing_security_master_classification_raises_error_504(self) -> None:
        """A requested classification set must include a security master."""
        _assert_axys_error(self, 504, classification_name="Sector1")

    def test_unknown_classification_raises_error_504(self) -> None:
        """Requested classification names must be defined in the specification."""
        _assert_axys_error(self, 504, classification_name="unknown")

    def test_missing_required_source_field_raises_error_504(self) -> None:
        """Classification and mapping definitions require their source fields."""
        _assert_axys_error(self, 504, classification_name="MissingFilePath")

    def test_nonexistent_source_column_raises_error_504(self) -> None:
        """Specified source columns must exist in their CSV source."""
        _assert_axys_error(self, 504, classification_name="BadFilterColumnName")

    def test_unknown_source_field_raises_error_504(self) -> None:
        """Unrecognized source-definition fields are rejected."""
        _assert_axys_error(self, 504, mapping_name="BadUnknownField")

    def test_non_boolean_security_master_setting_raises_error_504(self) -> None:
        """The security-master setting accepts booleans only."""
        specification = _fixture_specification()
        classifications = cast(dict[str, object], specification["classifications"])
        mappings = cast(dict[str, object], specification["mappings"])
        security = cast(dict[str, object], classifications["Security"])
        security_mapping = cast(dict[str, object], mappings["SecurityToSector"])
        security["file_path"] = str(test_util.axys_data_path("imex_security_master.csv"))
        security["is_security_master"] = "true"
        security_mapping["file_path"] = str(test_util.axys_data_path("imex_security_master.csv"))

        with tempfile.TemporaryDirectory() as temp_dir:
            path = _write_yaml(Path(temp_dir), specification)
            _assert_axys_error(self, 504, specifications_path=path, classification_name="Security")

    def test_no_common_periods_raise_error_505(self) -> None:
        """Portperf and secperf must retain at least one common period."""
        _assert_axys_error(
            self,
            505,
            portperf_file_name="error_505_portperf.csv",
            secperf_file_name="error_505_secperf.csv",
        )


if __name__ == "__main__":
    unittest.main()
