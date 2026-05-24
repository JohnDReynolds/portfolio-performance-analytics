"""Focused in-memory tests for performance input normalization and validation."""

# Python Imports
import datetime as dt
import re
import unittest

# Third-Party Imports
import pandas as pd
import polars as pl

# Project Imports
import ppar.columns as cols
import ppar.errors as errs
from ppar.errors import PpaError
from ppar.performance import Performance

_PERIODS = (
    (dt.date(2023, 12, 31), dt.date(2024, 1, 31)),
    (dt.date(2024, 1, 31), dt.date(2024, 2, 29)),
)


def _wide_performance_df() -> pl.DataFrame:
    """Return a valid two-period, two-asset wide performance input."""
    return pl.DataFrame(
        {
            cols.BEGINNING_DATE: [period[0] for period in _PERIODS],
            cols.ENDING_DATE: [period[1] for period in _PERIODS],
            "A.ret": [0.10, 0.02],
            "B.ret": [-0.05, 0.03],
            "A.wgt": [0.60, 0.40],
            "B.wgt": [0.40, 0.60],
        }
    )


def _narrow_performance_df(include_names: bool = False) -> pl.DataFrame:
    """Return the wide fixture represented as four narrow input rows."""
    data: dict[str, list[dt.date] | list[str] | list[float]] = {
        cols.BEGINNING_DATE: [_PERIODS[0][0], _PERIODS[0][0], _PERIODS[1][0], _PERIODS[1][0]],
        cols.ENDING_DATE: [_PERIODS[0][1], _PERIODS[0][1], _PERIODS[1][1], _PERIODS[1][1]],
        cols.IDENTIFIER: ["A", "B", "A", "B"],
        cols.RETURN: [0.10, -0.05, 0.02, 0.03],
        cols.WEIGHT: [0.60, 0.40, 0.40, 0.60],
    }
    if include_names:
        data[cols.NAME] = ["Alpha", "Beta", "Alpha", "Beta"]
    return pl.DataFrame(data)


class TestPerformanceNormalization(unittest.TestCase):
    """Test supported input forms and validation boundaries without CSV fixtures."""

    def test_narrow_and_wide_inputs_produce_same_normalized_dataframe(self) -> None:
        """Equivalent input layouts produce equivalent calculated performance."""
        wide = Performance(_wide_performance_df())
        narrow = Performance(_narrow_performance_df())

        self.assertTrue(wide.df.equals(narrow.df))

    def test_pandas_and_polars_inputs_produce_same_normalized_dataframe(self) -> None:
        """Equivalent supported DataFrame implementations normalize identically."""
        polars_input = _wide_performance_df()
        pandas_input = pd.DataFrame(polars_input.to_dict(as_series=False))

        from_polars = Performance(polars_input)
        from_pandas = Performance(pandas_input)

        self.assertTrue(from_polars.df.equals(from_pandas.df))

    def test_input_rows_are_sorted_by_ending_date(self) -> None:
        """Chronologically reversed input rows are normalized to period order."""
        expected = Performance(_wide_performance_df())
        reversed_rows = Performance(_wide_performance_df().reverse())

        self.assertTrue(expected.df.equals(reversed_rows.df))
        self.assertEqual(reversed_rows.df[cols.ENDING_DATE].item(0), _PERIODS[0][1])

    def test_input_column_order_does_not_affect_calculations(self) -> None:
        """Return and weight columns may arrive in any order."""
        expected = Performance(_wide_performance_df())
        shuffled_columns = _wide_performance_df().select(
            "B.wgt",
            cols.ENDING_DATE,
            "A.wgt",
            "B.ret",
            cols.BEGINNING_DATE,
            "A.ret",
        )
        normalized = Performance(shuffled_columns)

        self.assertEqual(normalized.identifiers, ["A", "B"])
        self.assertTrue(expected.df.equals(normalized.df))

    def test_inclusive_beginning_dates_are_normalized(self) -> None:
        """Inclusive month beginnings are converted to prior period endings."""
        inclusive_input = pl.DataFrame(
            {
                cols.BEGINNING_DATE: [dt.date(2024, 1, 1), dt.date(2024, 2, 1)],
                cols.ENDING_DATE: [dt.date(2024, 1, 31), dt.date(2024, 2, 29)],
                "A.ret": [0.01, 0.02],
                "A.wgt": [1.0, 1.0],
            }
        )

        performance = Performance(inclusive_input)

        self.assertEqual(
            performance.df[cols.BEGINNING_DATE].to_list(),
            [_PERIODS[0][0], _PERIODS[1][0]],
        )

    def test_narrow_names_populate_classification_items(self) -> None:
        """Narrow security names remain available for inferred classifications."""
        performance = Performance(_narrow_performance_df(include_names=True))
        items = performance.classification_items.sort(cols.CLASSIFICATION_IDENTIFIER)

        self.assertEqual(
            items.to_dict(as_series=False),
            {
                cols.CLASSIFICATION_IDENTIFIER: ["A", "B"],
                cols.CLASSIFICATION_NAME: ["Alpha", "Beta"],
            },
        )

    def test_narrow_input_without_names_has_no_classification_items(self) -> None:
        """Classification metadata is absent when names were not supplied."""
        performance = Performance(_narrow_performance_df())

        self.assertTrue(performance.classification_items.is_empty())

    def test_duplicate_narrow_date_identifier_rows_raise_error_112(self) -> None:
        """A duplicate narrow asset row is rejected before pivoting."""
        duplicate = pl.concat([_narrow_performance_df(), _narrow_performance_df().head(1)])

        with self.assertRaisesRegex(PpaError, errs.ERRORS[112]):
            Performance(duplicate)

    def test_unmatched_return_and_weight_columns_raise_error_107(self) -> None:
        """Every return identifier must have a matching weight identifier."""
        unmatched = _wide_performance_df().drop("B.wgt")

        with self.assertRaisesRegex(PpaError, re.escape(errs.ERRORS[107])):
            Performance(unmatched)

    def test_weights_that_do_not_net_to_one_raise_error_108(self) -> None:
        """Input rows whose weights do not sum to one are rejected."""
        invalid_weights = _wide_performance_df().with_columns(pl.lit(0.20).alias("B.wgt"))

        with self.assertRaisesRegex(PpaError, errs.ERRORS[108]):
            Performance(invalid_weights)

    def test_null_returns_raise_error_104(self) -> None:
        """Missing numeric observations are rejected from in-memory inputs."""
        null_returns = _wide_performance_df().with_columns(pl.lit(None).alias("A.ret"))

        with self.assertRaisesRegex(PpaError, errs.ERRORS[104]):
            Performance(null_returns)

    def test_discontinuous_dates_raise_error_106(self) -> None:
        """A gap between adjacent performance periods is rejected."""
        discontinuous = _wide_performance_df().with_columns(
            pl.Series(
                cols.BEGINNING_DATE,
                [dt.date(2023, 12, 31), dt.date(2024, 2, 2)],
            )
        )

        with self.assertRaisesRegex(PpaError, errs.ERRORS[106]):
            Performance(discontinuous)


if __name__ == "__main__":
    unittest.main()
