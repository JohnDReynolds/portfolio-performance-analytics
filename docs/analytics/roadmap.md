# PPAR Analytics Roadmap

This document owns forward-looking direction for the PPAR Analytics product.
It is intentionally compact. Operational instructions for regenerating demo
data and README assets remain in the
[Analytics Demo Refresh Guide](analytics_demo_refresh.md).

## Product Boundary

PPAR Analytics creates portfolio-versus-benchmark attribution, contribution,
cumulative-effect, and ex-post risk reporting from local portfolio accounting
data.

It is a maintained additional module with a separate product identity from
PPAR Audit. It may share setup, normalized data, Axys/APX integration knowledge,
packaging, and reporting infrastructure, but it does not inherit Audit's
changed-performance investigation roadmap. It can be evaluated as a separately
positioned product when market evidence supports that step.

## Current Position

- The installed-user path starts with
  `ppar setup ./my_ppar_analytics --analytics` and the packaged Axys/APX
  Analytics workspace.
- Normal run choices live together in the workspace's strict `analytics:` section;
  matching CLI options are one-run overrides rather than a second source of
  defaults.
- Conventional Analytics source filenames are `portperf.csv`, `secperf.csv`,
  and `secmast.csv`. Vendor-specific source headings require explicit YAML
  mappings; an omitted mapping accepts only the exact normalized field name.
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

## Eventual Security Display Identity

This is a long-term, cross-product usability candidate, not a current priority
and not authorized for implementation. Axys/APX users often recognize a
security by its symbol, while PPAR must use a canonical identifier such as
`csusAAPL` to distinguish the rarer cases in which the same symbol belongs to
more than one security type. An eventual display policy could suppress the
usually redundant type prefix without weakening internal identity.

The canonical `security_id` must remain the only key used for joins, grouping,
filtering, lineage, fingerprints, calculations, and YAML rules. The feature
should not add a general-purpose display column to financial DataFrames because
that would make accidental use of a noncanonical value more likely. Instead,
the shared Axys/APX security-identity layer should build an immutable mapping
from canonical security ID to presentation label, and Analytics and Audit should
apply that mapping only at explicit presentation boundaries.

The eventual YAML could extend the existing identity definition rather than add
an unrelated top-level setting:

```yaml
security_id:
  components:
    - security_type
    - security_symbol
  display_component: security_symbol
```

Omitting `display_component` must preserve current behavior exactly by showing
the canonical `security_id`. The configured display component should be
restricted to a validated security-ID component so that it has a defined
relationship to canonical identity.

The display mapping must be exact-case, deterministic, and independent of input
row order. It should be derived from the complete configured security-bearing
source universe rather than only the portfolios, dates, or rows selected for a
particular report. For Audit, that universe includes both snapshots. For
Analytics, it includes security performance and the complete security master.
Using the same complete security-master universe wherever possible reduces the
risk that Analytics and Audit choose different labels for the same security.

For each canonical security ID, the configured component may be displayed only
when:

- every observed value for that ID is present and identical;
- the candidate value maps to exactly one canonical security ID across the
  complete source universe; and
- the candidate would not collide with another security's canonical-ID
  fallback value.

When any condition fails, the presentation label must fall back to the canonical
`security_id`. All IDs participating in a shared-symbol collision must fall back,
not just one arbitrarily selected row. The implementation must also handle the
less obvious case in which one security's symbol equals another security's full
canonical ID. A mapping constructed from sorted distinct pairs, with explicit
collision resolution, can make the result reproducible without depending on
file or row order.

Reviewer-facing outputs are the intended scope. That can eventually include
Analytics HTML and charts, Audit XLSX and HTML, root review CSVs, root
`source_detail.csv`, and user-facing explanations or review guidance. Merely
replacing the visible Security column would be incomplete because current
narrative text can also contain canonical IDs.

Machine-integrity artifacts should retain canonical identifiers. In particular,
Audit `findings.csv`, `cause_lineage.csv`, matching and reconstruction
diagnostics, source locators, and fingerprint inputs depend on stable technical
identity. Replacing those values while also forbidding an additional output
column would make evidence harder or impossible to reconcile to source-data.
The durable boundary is therefore display labels in reviewer artifacts and
canonical IDs in machine evidence, even when expanded support files are visible
to a user.

YAML filters must continue to use canonical IDs such as `csusAAPL`, regardless
of what a report displays. Documentation and validation must make that
distinction explicit. The bundle manifest should record the selected display
policy and deterministic counts of symbol displays and canonical-ID fallbacks;
an output-contract change may require a deliberate manifest-version increment.

Before implementation, require focused decisions and tests for:

- the precise complete-source universe for each product;
- behavior when a configured or optional dataset lacks the display component;
- exact-case symbols, blanks, inconsistent values, and cross-type collisions;
- candidate-label collisions with canonical fallback IDs;
- stability when portfolios or report dates change;
- complete projection of user-facing narrative as well as tabular values;
- unchanged calculations, joins, grouping, lineage, conservation, and YAML
  matching;
- byte-for-byte preservation for configurations that omit the setting; and
- report parity, determinism, manifest validation, and maintained 500x scale
  gates.

The shared policy belongs in the Axys/APX security-identity module, but there is
not one universal report call site: Analytics and Audit have separate output
stacks and would each need a narrow presentation adapter. Treat this as a
separately scoped feature slice if it is eventually promoted from the roadmap.

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
