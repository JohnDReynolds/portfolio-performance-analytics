"""Tests for the executable Audit safety-invariant catalog."""

from __future__ import annotations

# Python imports
from pathlib import Path
import unittest

# Project imports
from ppar.audit.safety_invariants import (
    DIFFERENCE_DISPOSITION_RULE,
    MATERIAL_DIFFERENCE_DEFINITION,
    SAFETY_INVARIANTS,
    DifferenceDisposition,
    InvariantCoverage,
    InvariantFailureClass,
    safety_invariant,
)

_SAFETY_DOCUMENT = Path(
    "docs/audit/safety_invariants.md"
)


class TestAuditSafetyInvariants(unittest.TestCase):
    """Protect the stable safety-net definitions and enforcement catalog."""

    def test_active_program_has_no_partial_invariants(self) -> None:
        """Every active safety net remains enforced after the SN-04 retirement."""
        self.assertEqual(
            {invariant.coverage for invariant in SAFETY_INVARIANTS},
            {InvariantCoverage.ENFORCED, InvariantCoverage.RETIRED},
        )
        self.assertEqual(
            [
                invariant.identifier
                for invariant in SAFETY_INVARIANTS
                if invariant.coverage == InvariantCoverage.RETIRED
            ],
            ["SN-04"],
        )

    def test_catalog_has_twelve_stable_unique_identifiers(self) -> None:
        """The twelve agreed safety nets retain stable ordered identifiers."""
        identifiers = [invariant.identifier for invariant in SAFETY_INVARIANTS]

        self.assertEqual(identifiers, [f"SN-{number:02d}" for number in range(1, 13)])
        self.assertEqual(len(identifiers), len(set(identifiers)))

    def test_every_invariant_has_a_complete_phase_one_audit(self) -> None:
        """Every catalog row states a guarantee, baseline, gap, and failure mode."""
        for invariant in SAFETY_INVARIANTS:
            with self.subTest(identifier=invariant.identifier):
                self.assertTrue(invariant.name)
                self.assertTrue(invariant.guarantee)
                self.assertIsInstance(invariant.failure_class, InvariantFailureClass)
                self.assertIsInstance(invariant.coverage, InvariantCoverage)
                self.assertTrue(invariant.existing_controls)
                self.assertTrue(invariant.control_tests)
                if invariant.coverage == InvariantCoverage.ENFORCED:
                    self.assertFalse(invariant.known_gaps)
                elif invariant.coverage != InvariantCoverage.RETIRED:
                    self.assertTrue(invariant.known_gaps)
                else:
                    self.assertFalse(invariant.known_gaps)
                self.assertIn(invariant.implementation_phase, range(2, 7))
                self.assertIs(safety_invariant(invariant.identifier), invariant)

    def test_every_catalog_entry_names_an_existing_executable_test(self) -> None:
        """Catalog claims remain linked to concrete executable regression tests."""
        for invariant in SAFETY_INVARIANTS:
            for node_id in invariant.control_tests:
                with self.subTest(identifier=invariant.identifier, node_id=node_id):
                    path_text, separator, selector = node_id.partition("::")
                    self.assertEqual(separator, "::")
                    self.assertTrue(selector)
                    test_path = Path(path_text)
                    self.assertTrue(test_path.is_file())
                    test_source = test_path.read_text(encoding="utf-8")
                    for selector_part in selector.split("::"):
                        declaration = (
                            f"def {selector_part}("
                            if selector_part.startswith("test_")
                            else f"class {selector_part}("
                        )
                        self.assertIn(declaration, test_source)

    def test_planned_phases_preserve_the_agreed_grouping(self) -> None:
        """Implementation phases retain the dependencies established in Phase 1."""
        expected_phases = {
            "SN-01": 2,
            "SN-02": 2,
            "SN-03": 2,
            "SN-04": 3,
            "SN-05": 4,
            "SN-06": 3,
            "SN-07": 3,
            "SN-08": 5,
            "SN-09": 5,
            "SN-10": 6,
            "SN-11": 6,
            "SN-12": 4,
        }

        self.assertEqual(
            {
                invariant.identifier: invariant.implementation_phase
                for invariant in SAFETY_INVARIANTS
            },
            expected_phases,
        )

    def test_difference_dispositions_do_not_allow_hidden_state(self) -> None:
        """A reportable difference can be counted or reviewable, never hidden."""
        self.assertEqual(
            {disposition.value for disposition in DifferenceDisposition},
            {"counted_cause", "review_evidence"},
        )
        self.assertIn("Suppression", DIFFERENCE_DISPOSITION_RULE)
        self.assertIn("must not erase", DIFFERENCE_DISPOSITION_RULE)
        self.assertIn("independent of finding severity", MATERIAL_DIFFERENCE_DEFINITION)

    def test_maintainer_document_covers_every_catalog_entry(self) -> None:
        """The design document remains synchronized with the executable catalog."""
        contents = _SAFETY_DOCUMENT.read_text(encoding="utf-8")

        for invariant in SAFETY_INVARIANTS:
            with self.subTest(identifier=invariant.identifier):
                self.assertIn(
                    f"| `{invariant.identifier}` | {invariant.name} |",
                    contents,
                )
        self.assertIn("## Current Guarantee Summary", contents)
        self.assertIn("## Existing Enforcement Map", contents)
        self.assertIn("## Change-Control Rule", contents)

    def test_unknown_invariant_identifier_raises_key_error(self) -> None:
        """Catalog lookup does not silently accept an unknown safety net."""
        with self.assertRaises(KeyError):
            safety_invariant("SN-99")


if __name__ == "__main__":
    unittest.main()
