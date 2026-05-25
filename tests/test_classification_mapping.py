"""Focused in-memory tests for classification inference and mapping behavior."""

import datetime as dt
import unittest

import polars as pl

from ppar.analytics import Analytics
from ppar.attribution import View
from ppar.classification import Classification
import ppar.columns as cols
import ppar.errors as errs
from ppar.errors import PpaError
from ppar.mapping import Mapping
from ppar.performance import Performance
import ppar.utilities as util


def _named_performance(
    a_name: str = "Alpha",
    b_name: str = "Beta",
    classification_name: str = "Security",
) -> Performance:
    """Return a minimal named performance data set for classification tests."""
    return Performance(
        pl.DataFrame(
            {
                cols.BEGINNING_DATE: [dt.date(2023, 12, 31)] * 2,
                cols.ENDING_DATE: [dt.date(2024, 1, 31)] * 2,
                cols.IDENTIFIER: ["A", "B"],
                cols.RETURN: [0.10, -0.05],
                cols.WEIGHT: [0.60, 0.40],
                cols.NAME: [a_name, b_name],
            }
        ),
        classification_name=classification_name,
    )


def _wide_performance() -> pl.DataFrame:
    """Return a minimal wide performance data set for attribution tests."""
    return pl.DataFrame(
        {
            cols.BEGINNING_DATE: [dt.date(2023, 12, 31)],
            cols.ENDING_DATE: [dt.date(2024, 1, 31)],
            "A.ret": [0.10],
            "B.ret": [-0.05],
            "A.wgt": [0.60],
            "B.wgt": [0.40],
        }
    )


class ClassificationTests(unittest.TestCase):
    """Verify classification inference and explicit classification sources."""

    def test_classification_is_inferred_from_named_performances(self) -> None:
        """Matching named inputs provide an inferred classification."""
        portfolio = _named_performance()
        benchmark = _named_performance()

        classification = Classification("", None, (portfolio, benchmark))

        self.assertEqual(classification.name, "Security")
        self.assertEqual(
            classification.df.sort(cols.CLASSIFICATION_IDENTIFIER).to_dict(as_series=False),
            {
                cols.CLASSIFICATION_IDENTIFIER: ["A", "B"],
                cols.CLASSIFICATION_NAME: ["Alpha", "Beta"],
            },
        )

    def test_inferred_classification_prefers_portfolio_name_on_overlap(self) -> None:
        """Portfolio names take precedence for identifiers present in both inputs."""
        portfolio = _named_performance(a_name="Portfolio Alpha")
        benchmark = _named_performance(a_name="Benchmark Alpha")

        classification = Classification("", None, (portfolio, benchmark))

        names = dict(
            zip(
                classification.df[cols.CLASSIFICATION_IDENTIFIER].to_list(),
                classification.df[cols.CLASSIFICATION_NAME].to_list(),
            )
        )
        self.assertEqual(names["A"], "Portfolio Alpha")

    def test_inferred_classification_is_empty_when_names_differ(self) -> None:
        """Different input classification names prevent implicit classification."""
        portfolio = _named_performance(classification_name="Security")
        benchmark = _named_performance(classification_name="Holding")

        classification = Classification("", None, (portfolio, benchmark))

        self.assertEqual(classification.name, util.EMPTY)
        self.assertEqual(
            classification.df[cols.CLASSIFICATION_IDENTIFIER].item(),
            util.EMPTY,
        )

    def test_explicit_classification_filters_and_keeps_last_duplicate(self) -> None:
        """Explicit sources filter unused items and use the last duplicate name."""
        source = pl.DataFrame(
            {
                "identifier": ["A", "A", "B", "UNUSED"],
                "name": ["Old Alpha", "Alpha", "Beta", "Unused"],
            }
        )

        classification = Classification(
            "Security",
            source,
            (_named_performance(), _named_performance()),
        )

        self.assertEqual(
            classification.df.sort(cols.CLASSIFICATION_IDENTIFIER).to_dict(as_series=False),
            {
                cols.CLASSIFICATION_IDENTIFIER: ["A", "B"],
                cols.CLASSIFICATION_NAME: ["Alpha", "Beta"],
            },
        )

    def test_one_column_classification_source_raises_error_302(self) -> None:
        """Explicit classification sources must supply identifier and name columns."""
        source = pl.DataFrame({"identifier": ["A", "B"]})

        with self.assertRaisesRegex(PpaError, errs.ERRORS[302]):
            Classification("Security", source, (_named_performance(), _named_performance()))


class MappingTests(unittest.TestCase):
    """Verify the direct mapping contract and mapped attribution result."""

    def test_mapping_rolls_multiple_items_to_same_target(self) -> None:
        """Several source identifiers may roll up to one target identifier."""
        mapping = Mapping(("A", "B"), {"A": "TECH", "B": "TECH"})

        self.assertEqual(dict(mapping.to_froms), {"TECH": ["A", "B"]})

    def test_mapping_keeps_unmapped_item_at_its_own_identifier(self) -> None:
        """An unmapped identifier remains a standalone mapped group."""
        mapping = Mapping(("A", "B"), {"A": "TECH"})

        self.assertEqual(
            dict(mapping.to_froms),
            {"TECH": ["A"], "B": ["B"]},
        )

    def test_mapping_filters_unused_source_items(self) -> None:
        """Mappings for identifiers outside the source performance are discarded."""
        mapping = Mapping(("A", "B"), {"A": "TECH", "B": "FIN", "C": "OTHER"})

        self.assertEqual(
            dict(mapping.to_froms),
            {"TECH": ["A"], "FIN": ["B"]},
        )

    def test_mapping_duplicate_source_item_uses_last_value(self) -> None:
        """Duplicate mapping rows resolve to the final target value."""
        mapping = Mapping(
            ("A",),
            pl.DataFrame({"from": ["A", "A"], "to": ["TECH", "HEALTH"]}),
        )

        self.assertEqual(dict(mapping.to_froms), {"HEALTH": ["A"]})

    def test_one_column_mapping_source_raises_error_353(self) -> None:
        """Mapping sources must supply both from and to identifier columns."""
        with self.assertRaisesRegex(PpaError, errs.ERRORS[353]):
            Mapping(("A", "B"), pl.DataFrame({"from": ["A", "B"]}))

    def test_mapped_attribution_rollup_preserves_portfolio_contribution(self) -> None:
        """Mapped attribution totals retain underlying portfolio contribution."""
        analytics = Analytics(
            _wide_performance(),
            _wide_performance(),
            portfolio_classification_name="Security",
            benchmark_classification_name="Security",
        )

        attribution = analytics.get_attribution(
            "Sector",
            {"TECH": "Technology"},
            ({"A": "TECH", "B": "TECH"}, {"A": "TECH", "B": "TECH"}),
        )
        details = attribution.to_polars(View.SUBPERIOD_ATTRIBUTION)

        self.assertEqual(details.height, 1)
        self.assertEqual(details[cols.CLASSIFICATION_IDENTIFIER].item(), "TECH")
        self.assertEqual(details[cols.CLASSIFICATION_NAME].item(), "Technology")
        self.assertAlmostEqual(details[cols.PORTFOLIO_WEIGHT].item(), 1.0)
        self.assertAlmostEqual(details[cols.PORTFOLIO_CONTRIB_SIMPLE].item(), 0.04)


if __name__ == "__main__":
    unittest.main()
