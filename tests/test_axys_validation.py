"""Focused tests for AxysData source and specification validation failures."""

# Python Imports
from collections.abc import Mapping
from dataclasses import dataclass, field
import datetime as dt
from pathlib import Path
import tempfile
from typing import cast
import unittest

# Third-Party Imports
import polars as pl
import yaml

# Test Imports
from tests import test_utilities as test_util

# Project Imports
from ppar.axys import AxysData
import ppar.errors as errs
from ppar.errors import PpaError


@dataclass(frozen=True)
class _AxysArguments:
    """Constructor inputs that a validation test needs to override."""

    specifications_path: Path = field(
        default_factory=lambda: test_util.axys_data_path("axysdata.yaml", ".yaml")
    )
    portperf_path: Path | None = field(
        default_factory=lambda: test_util.axys_data_path("portperf.csv")
    )
    secperf_path: Path | None = field(
        default_factory=lambda: test_util.axys_data_path("secperf.csv")
    )
    source_path_overrides: Mapping[str, Path] | None = None
    portfolio_code: str = "PORT_SMALL"
    classification_name: str | None = None


def _assert_axys_error(
    test: unittest.TestCase,
    error_code: int,
    arguments: _AxysArguments | None = None,
    message_contains: str | None = None,
) -> None:
    """Assert that constructing AxysData fails with a numbered PpaError."""
    arguments = arguments or _AxysArguments()

    with test.assertRaises(PpaError) as context:
        data = AxysData(
            arguments.specifications_path,
            arguments.portperf_path,
            arguments.secperf_path,
            arguments.source_path_overrides,
        )
        portfolio = data.get_portfolio(arguments.portfolio_code)
        if arguments.classification_name is not None:
            data.get_classification_sources(arguments.classification_name, portfolio)

    test.assertTrue(
        str(context.exception).startswith(errs.ERRORS[error_code]),
        str(context.exception),
    )
    if message_contains is not None:
        test.assertIn(message_contains, str(context.exception))


def _write_yaml(directory: Path, contents: object) -> Path:
    """Write temporary YAML contents and return its path."""
    path = directory / "axysdata.yaml"
    path.write_text(yaml.safe_dump(contents), encoding="utf-8")
    return path


def _write_text_csv(directory: Path, file_name: str, contents: str) -> Path:
    """Write a CSV fixture whose raw header is material to validation."""
    path = directory / file_name
    path.write_text(contents, encoding="utf-8")
    return path


def _write_frame_csv(directory: Path, file_name: str, data: dict[str, list[object]]) -> Path:
    """Write compact valid-schema input rows for a validation failure."""
    path = directory / file_name
    pl.DataFrame(data).write_csv(path)
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
        with tempfile.TemporaryDirectory() as temp_dir:
            portperf_path = _write_text_csv(
                Path(temp_dir),
                "portperf.csv",
                (
                    "PORTFOLIO_CODEX,PORTFOLIO_NAME,FROM_DATE,THRU_DATE,PORT_RETURN\n"
                    "PORT_SMALL,Small Portfolio,2024-01-01,2024-01-31,0.01\n"
                ),
            )
            _assert_axys_error(self, 502, _AxysArguments(portperf_path=portperf_path))

    def test_missing_secperf_columns_raise_error_502(self) -> None:
        """Required secperf columns are validated before processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            secperf_path = _write_text_csv(
                Path(temp_dir),
                "secperf.csv",
                (
                    "PORTFOLIO_CODE,FROM_DATEX,THRU_DATE,SECURITY_ID,BEGIN_WEIGHT,"
                    "SEC_RETURN,CONTRIBUTION_W_X_R\n"
                    "PORT_SMALL,2024-01-01,2024-01-31,S001,1.0,0.01,0.01\n"
                ),
            )
            _assert_axys_error(self, 502, _AxysArguments(secperf_path=secperf_path))

    def test_material_reconciliation_difference_raises_error_503(self) -> None:
        """An unreconciled return difference outside tolerance is rejected."""
        with tempfile.TemporaryDirectory() as temp_dir:
            portperf_path = _write_frame_csv(
                Path(temp_dir),
                "portperf.csv",
                {
                    "PORTFOLIO_CODE": ["PORT_FAIL_HIGH"],
                    "PORTFOLIO_NAME": ["Failure Demo High Target"],
                    "FROM_DATE": ["2024-03-01"],
                    "THRU_DATE": ["2024-03-31"],
                    "PORT_RETURN": [0.50],
                },
            )
            _assert_axys_error(
                self,
                503,
                _AxysArguments(
                    portperf_path=portperf_path,
                    secperf_path=test_util.axys_data_path("unreachable_target_secperf.csv"),
                    portfolio_code="PORT_FAIL_HIGH",
                ),
            )

    def test_equal_return_reconciliation_failure_raises_error_503(self) -> None:
        """Unachievable equal-security target returns are rejected."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            portperf_path = _write_frame_csv(
                directory,
                "portperf.csv",
                {
                    "PORTFOLIO_CODE": ["PORT_FAIL_EQUAL"],
                    "PORTFOLIO_NAME": ["Failure Demo Equal Weight"],
                    "FROM_DATE": ["2024-04-11"],
                    "THRU_DATE": ["2024-04-20"],
                    "PORT_RETURN": [0.20],
                },
            )
            secperf_path = _write_frame_csv(
                directory,
                "secperf.csv",
                {
                    "PORTFOLIO_CODE": ["PORT_FAIL_EQUAL"],
                    "FROM_DATE": ["2024-04-11"],
                    "THRU_DATE": ["2024-04-20"],
                    "SECURITY_ID": ["S001"],
                    "BEGIN_WEIGHT": [1.0],
                    "SEC_RETURN": [0.0],
                    "CONTRIBUTION_W_X_R": [0.0],
                },
            )
            _assert_axys_error(
                self,
                503,
                _AxysArguments(
                    portperf_path=portperf_path,
                    secperf_path=secperf_path,
                    portfolio_code="PORT_FAIL_EQUAL",
                ),
            )

    def test_invalid_yaml_raises_error_504(self) -> None:
        """A syntactically invalid YAML specification is rejected."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "axysdata.yaml"
            path.write_text("classifications: [", encoding="utf-8")
            _assert_axys_error(self, 504, _AxysArguments(specifications_path=path))

    def test_non_mapping_yaml_root_raises_error_504(self) -> None:
        """A YAML list cannot be used as the specification object."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _write_yaml(Path(temp_dir), ["not", "a", "mapping"])
            _assert_axys_error(self, 504, _AxysArguments(specifications_path=path))

    def test_missing_portperf_path_raises_error_504(self) -> None:
        """Portperf must be provided either by argument or specification."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _write_yaml(Path(temp_dir), {})
            _assert_axys_error(
                self,
                504,
                _AxysArguments(specifications_path=path, portperf_path=None),
            )

    def test_missing_secperf_path_raises_error_504(self) -> None:
        """Secperf must be provided either by argument or specification."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _write_yaml(Path(temp_dir), {})
            _assert_axys_error(
                self,
                504,
                _AxysArguments(specifications_path=path, secperf_path=None),
            )

    def test_unknown_classification_raises_error_504(self) -> None:
        """Requested classification names must be defined in the specification."""
        _assert_axys_error(self, 504, _AxysArguments(classification_name="unknown"))

    def test_missing_portfolio_error_includes_requested_dates(self) -> None:
        """Portfolio-loading errors report the requested date window."""
        data = AxysData(
            test_util.axys_data_path("axysdata.yaml", ".yaml"),
            test_util.axys_data_path("portperf.csv"),
            test_util.axys_data_path("secperf.csv"),
        )

        with self.assertRaises(PpaError) as context:
            data.get_portfolio(
                "UNKNOWN_PORTFOLIO",
                from_date=dt.date(2024, 1, 1),
                thru_date=dt.date(2024, 12, 31),
            )

        self.assertIn("from_date=2024-01-01", str(context.exception))
        self.assertIn("thru_date=2024-12-31", str(context.exception))

    def test_missing_required_source_field_raises_error_504(self) -> None:
        """Classification and mapping definitions require their default path fields."""
        _assert_axys_error(self, 504, _AxysArguments(classification_name="MissingFilePath"))

    def test_unknown_source_path_override_raises_error_504(self) -> None:
        """Source path overrides must reference configured source names."""
        _assert_axys_error(
            self,
            504,
            _AxysArguments(
                source_path_overrides={"UnknownSource": Path("x.csv")},
            ),
            "Unknown source path override names",
        )

    def test_nonexistent_source_column_raises_error_504(self) -> None:
        """Specified source columns must exist in their CSV source."""
        _assert_axys_error(self, 504, _AxysArguments(classification_name="BadFilterColumnName"))

    def test_unknown_source_field_raises_error_504(self) -> None:
        """Unrecognized source-definition fields are rejected."""
        specification = _fixture_specification()
        classifications = cast(dict[str, object], specification["classifications"])
        sector = cast(dict[str, object], classifications["Sector1"])
        sector["default_file_path"] = str(test_util.axys_data_path("classification_lookup.csv"))
        sector["mapping"] = "BadUnknownField"

        with tempfile.TemporaryDirectory() as temp_dir:
            path = _write_yaml(Path(temp_dir), specification)
            _assert_axys_error(
                self,
                504,
                _AxysArguments(specifications_path=path, classification_name="Sector1"),
            )

    def test_non_security_master_classification_requires_mapping(self) -> None:
        """Classifications below security grain must identify their mapping."""
        specification = _fixture_specification()
        classifications = cast(dict[str, object], specification["classifications"])
        sector = cast(dict[str, object], classifications["Sector1"])
        del sector["mapping"]

        with tempfile.TemporaryDirectory() as temp_dir:
            path = _write_yaml(Path(temp_dir), specification)
            _assert_axys_error(
                self,
                504,
                _AxysArguments(specifications_path=path, classification_name="Sector1"),
                "Missing mapping for classification 'Sector1'",
            )

    def test_classification_mapping_must_be_configured(self) -> None:
        """Classification mapping references must point to configured mappings."""
        specification = _fixture_specification()
        classifications = cast(dict[str, object], specification["classifications"])
        sector = cast(dict[str, object], classifications["Sector1"])
        sector["mapping"] = "UnknownMapping"

        with tempfile.TemporaryDirectory() as temp_dir:
            path = _write_yaml(Path(temp_dir), specification)
            _assert_axys_error(
                self,
                504,
                _AxysArguments(specifications_path=path, classification_name="Sector1"),
                "Unknown mapping 'UnknownMapping' for classification 'Sector1'",
            )

    def test_mapping_definition_rejects_classification_mapping_field(self) -> None:
        """The mapping field belongs to classifications, not mapping sources."""
        specification = _fixture_specification()
        classifications = cast(dict[str, object], specification["classifications"])
        mappings = cast(dict[str, object], specification["mappings"])
        security = cast(dict[str, object], classifications["Security"])
        sector = cast(dict[str, object], classifications["Sector1"])
        security_to_sector = cast(dict[str, object], mappings["SecurityToSector"])
        security["default_file_path"] = str(test_util.axys_data_path("security_master.csv"))
        sector["default_file_path"] = str(test_util.axys_data_path("classification_lookup.csv"))
        security_to_sector["default_file_path"] = str(
            test_util.axys_data_path("security_master.csv")
        )
        security_to_sector["mapping"] = "SecurityToSector"

        with tempfile.TemporaryDirectory() as temp_dir:
            path = _write_yaml(Path(temp_dir), specification)
            _assert_axys_error(
                self,
                504,
                _AxysArguments(
                    specifications_path=path,
                    classification_name="Sector1",
                ),
                "Unknown fields for mapping 'SecurityToSector'",
            )

    def test_non_boolean_security_master_setting_raises_error_504(self) -> None:
        """The security-master setting accepts booleans only."""
        specification = _fixture_specification()
        classifications = cast(dict[str, object], specification["classifications"])
        mappings = cast(dict[str, object], specification["mappings"])
        security = cast(dict[str, object], classifications["Security"])
        security_mapping = cast(dict[str, object], mappings["SecurityToSector"])
        security["default_file_path"] = str(test_util.axys_data_path("security_master.csv"))
        security["is_security_master"] = "true"
        security_mapping["default_file_path"] = str(
            test_util.axys_data_path("security_master.csv")
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = _write_yaml(Path(temp_dir), specification)
            _assert_axys_error(
                self,
                504,
                _AxysArguments(specifications_path=path, classification_name="Security"),
            )

    def test_no_common_periods_raise_error_505(self) -> None:
        """Portperf and secperf must retain at least one common period."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            portperf_path = _write_frame_csv(
                directory,
                "portperf.csv",
                {
                    "PORTFOLIO_CODE": ["PORT_SMALL"],
                    "PORTFOLIO_NAME": ["Small Demo Portfolio"],
                    "FROM_DATE": ["2024-01-01"],
                    "THRU_DATE": ["2024-01-30"],
                    "PORT_RETURN": [0.01],
                },
            )
            secperf_path = _write_frame_csv(
                directory,
                "secperf.csv",
                {
                    "PORTFOLIO_CODE": ["PORT_SMALL"],
                    "FROM_DATE": ["2024-01-01"],
                    "THRU_DATE": ["2024-01-31"],
                    "SECURITY_ID": ["S001"],
                    "BEGIN_WEIGHT": [1.0],
                    "SEC_RETURN": [0.01],
                    "CONTRIBUTION_W_X_R": [0.01],
                },
            )
            _assert_axys_error(
                self,
                505,
                _AxysArguments(
                    portperf_path=portperf_path,
                    secperf_path=secperf_path,
                ),
            )


if __name__ == "__main__":
    unittest.main()
