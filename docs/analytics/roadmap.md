# PPAR Analytics Roadmap

This document owns forward-looking direction for the PPAR Analytics product.
It is intentionally compact. Operational instructions for regenerating demo
data and README assets remain in the
[Analytics Demo Refresh Guide](analytics_demo_refresh.md).

## Product Boundary

PPAR Analytics creates portfolio-versus-benchmark attribution, contribution,
cumulative-effect, and ex-post risk reporting from local portfolio accounting
data.

It is a separate product from PPAR Audit. It may share setup, normalized data,
Axys/APX integration knowledge, packaging, and reporting infrastructure, but it
does not inherit Audit's changed-performance investigation roadmap.

## Current Position

- The installed-user path starts with `ppar setup` and the packaged Axys/APX
  Analytics starter workspace.
- The `generic_analytics` dataset remains maintainer/demo infrastructure for
  README images, analytics regression tests, optional-value tests, and selected
  operational demo-data derivation.
- The generic dataset is not the primary new-user onboarding path.
- Demo refresh and README image generation are maintained through the
  [refresh guide](analytics_demo_refresh.md).

## Current Priorities

### Keep the public story reproducible

Keep the packaged Analytics inputs, generated reports, README narrative,
screenshots, and regression expectations synchronized. A demo change is
complete only when its source-data, generated artifacts, documentation, and
tests tell the same defensible story.

### Preserve the installed workflow boundary

Keep user-facing Analytics runners independent of source-checkout-only demo
helpers. New users should be able to work entirely from the setup-created
Analytics folder and its documented YAML.

### Improve the demo only with a clear analytical benefit

The current Mega-Cap Alpha versus Mega-Cap Benchmark story is adequate. Revisit
its allocation effect or overall outperformance only if the change makes the
analytical lesson clearer without making the data look contrived.

## Evidence-Gated Candidates

| Candidate | Gate |
| --- | --- |
| Effective-dated names and classifications | Define an explicit source contract, deterministic gap/overlap behavior, period-correct joins, validation, reporting, and tests before implementation. |
| Broader currency and source coverage | Validate quote conventions, portfolio currencies, mappings, and report interpretation before expanding claims. |
| Larger-scale optimization | Start from measured timings after meaningful data growth or report-shape changes; preserve the maintained scale gates. |
| Additional vendor starters | Require a documented, tested, supportable source contract rather than assuming configurable normalization proves compatibility. |

## Deferred Cleanup

Remove the `generic_analytics` packaged dataset and optional setup script only
after every remaining maintainer dependency has an accepted replacement:

- README image rendering uses another approved input set;
- regression and optional-value tests no longer depend on it;
- operational derivation scripts use another accepted source universe;
- package-data rules can be simplified safely; and
- documentation no longer treats it as active infrastructure.

Until those conditions hold, retention is deliberate and removal is not active
work.

## Maintenance Rule

Record Analytics product direction and backlog here. Keep repeatable refresh
steps in `analytics_demo_refresh.md`, shared architecture in
`../architecture.md`, Axys/APX evidence in `../axys_apx/`, and completed
cross-product implementation history in
`../archive/roadmap_through_v0.1.5.md`.
