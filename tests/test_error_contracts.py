"""Public contracts for structured PPAR errors and exact pair inputs."""

import unittest
from typing import cast, Sequence

import numpy as np

from ppar.analytics.attribution import Attribution
from ppar.analytics.frequency import Frequency
from ppar.analytics.performance import Performance
from ppar.analytics.riskstatistics import RiskStatistics
import ppar.errors as errs
from ppar.errors import PpaError


class TestPpaErrorContracts(unittest.TestCase):
    """Verify package failures expose stable machine-readable metadata."""

    def test_error_exposes_code_detail_context_and_compatible_text(self) -> None:
        """Structured metadata supplements rather than replaces readable text."""
        error = PpaError(
            "calculation detail",
            203,
            context={"portfolio_id": "BALANCED", "period": "2024-01"},
        )

        self.assertEqual(error.code, 203)
        self.assertEqual(error.detail, "calculation detail")
        self.assertEqual(
            error.context,
            {"portfolio_id": "BALANCED", "period": "2024-01"},
        )
        self.assertEqual(str(error), f"{errs.ERRORS[203]}calculation detail")

    def test_uncoded_error_retains_detail_and_empty_context(self) -> None:
        """Compatibility failures without a registry code remain structured."""
        error = PpaError("plain detail", None)

        self.assertIsNone(error.code)
        self.assertEqual(error.detail, "plain detail")
        self.assertEqual(error.context, {})
        self.assertEqual(str(error), "plain detail")


class TestExactPairContracts(unittest.TestCase):
    """Verify public portfolio/benchmark boundaries require exactly two items."""

    def test_risk_statistics_rejects_wrong_sequence_lengths(self) -> None:
        """Risk inputs cannot raise IndexError or silently ignore extra arrays."""
        returns = np.array([0.01, 0.02], dtype=np.float64)
        for sequence in ((), (returns,), (returns, returns, returns)):
            with self.subTest(length=len(sequence)):
                with self.assertRaisesRegex(PpaError, errs.ERRORS[805]):
                    RiskStatistics(sequence, Frequency.MONTHLY)

    def test_attribution_rejects_wrong_performance_sequence_lengths(self) -> None:
        """Direct Attribution construction validates its pair before indexing."""
        for sequence in ((), (None,), (None, None, None)):
            with self.subTest(length=len(sequence)):
                with self.assertRaisesRegex(PpaError, errs.ERRORS[805]):
                    Attribution(
                        cast(Sequence[Performance], sequence),
                        classification_name=None,
                        classification_data_source=None,
                        frequency=Frequency.AS_OFTEN_AS_POSSIBLE,
                    )


if __name__ == "__main__":
    unittest.main()
