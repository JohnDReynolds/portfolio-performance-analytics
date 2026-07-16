# Performance Comparison Evidence-Pack Review

This review pack inventories the accumulated evidence-pack and transaction
boundary work before commit. It is a reviewer aid, not a replacement for the
roadmap or the machine-readable transaction semantics matrix.

## Change Inventory

Evidence-pack manifests:
Report bundles now validate source context, extract-contract metadata,
transaction-semantics summaries, row counts, and reviewer entrypoints.

Site extract readiness:
IMEX context, REP/report semantics, code-only failure, and reviewed local opt-out
paths are documented and covered by site-variant fixtures.

Transaction boundaries:
Fixed-income, capital-return, short-side, review-only, context-only, and
ambiguous-flow boundaries are named in helper modules and matrix tests.

Test-only fixtures:
`imex_context`, `imex_code_only`, `rep_semantics`, `local_opt_out`, and
`review_only_actions` cover context-driven and blocked semantics.

Validator coverage:
`validate_demo_matrix` now acts as a compact coverage contract for baseline,
attribution, site variants, review-only quarantine, and backlog gates.

Reviewer docs:
Roadmap phases, the source contract, readiness checklist, and boundary snapshot
explain what is covered, what is test-only, and what remains backlog.

## Public Surface

The runtime package keeps its top-level public API focused on comparison
workflows. Boundary helpers are importable as direct submodules:

- `ppar.performance_comparison.fixed_income`
- `ppar.performance_comparison.backlog_gates`
- `ppar.performance_comparison.transaction_boundary_registry`
- `ppar.performance_comparison.transaction_summary`

Those helpers support tests, validators, and reviewer documentation. They are
not exported from `ppar.performance_comparison.__all__`.

## Review Notes

- The work stays scoped to Modified Dietz formula inputs and evidence review.
- Packaged demo behavior remains narrower than the test-only fixture surface.
- Backlog gates intentionally block code-only treatment for high-risk families.
- The full suite should pass before committing this pack.

## Suggested Commit Message

```text
Add performance comparison evidence-pack boundaries
```
