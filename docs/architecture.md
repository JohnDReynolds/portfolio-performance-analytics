# PPAR Architecture

This document is the compact system map for PPAR. It explains how the main
parts fit together without replacing the roadmap, setup README, YAML comments,
or deeper performance-comparison design notes.

## Product Surface

PPAR presents two market-facing products—PPAR Audit and PPAR Analytics—that
share one codebase, common command infrastructure, and integration support.
Audit contains sibling Performance Comparison and Data Issues sub-features.

The public installed command is:

```bash
ppar
```

The main user path is:

```text
ppar setup <site_directory>
  -> creates analytics/ and audit/ starter folders
  -> installs heavily commented ppar.yaml files
  -> copies Axys/APX starter CSV files
```

After setup, users run:

```bash
ppar analytics <site_directory>/analytics
ppar audit <site_directory>/audit
```

## Package Map

| Package | Role |
| --- | --- |
| `ppar.analytics` | Performance analytics, attribution, contribution, risk statistics, and analytics report generation. |
| `ppar.axys_apx` | Axys/APX ingestion and normalization support. |
| `ppar.audit` | Audit orchestration, shared source loading, reports, bundles, safety controls, and validation. |
| `ppar.audit.performance_comparison` | Performance-difference attribution, Modified Dietz evidence, and explanation logic. |
| `ppar.audit.data_issues` | Source-data relationship checks, configuration, and issue vocabulary. |
| `ppar.setup_templates` | Packaged setup workspaces and tutorial scripts copied by `ppar setup`. |

## Data Flow

PPAR generally treats source files as configured site inputs, not as a hidden
database.

For Performance Analytics:

```text
Axys/APX portfolio and security performance CSVs
  -> analytics ppar.yaml
  -> ppar analytics
  -> HTML tables and PNG charts
```

For Audit:

```text
snapshot_a CSVs + snapshot_b CSVs
  -> audit ppar.yaml
  -> normalized datasets
  -> Performance Comparison differences and explanations
  -> independent Data Issues checks
  -> portfolio_audit.* or security_audit.*
```

When one `ppar audit` run requests both report levels, holdings, FX rates,
splits, and transactions are compared once as canonical shared-source
findings. Portfolio performance and security performance remain separate
calculations, and each report view applies its own transaction-impact policy
and suppression rules to the shared findings. Return-reconstruction inputs are
also cached across both report builds. A single-level run uses the same path
without calculating the unused primary performance level.

Within each report view, the workbook, HTML, and supporting CSV layers reuse
one set of portfolio-period summaries, contribution candidates, Modified Dietz
formula rows, and context-evidence tables. The explanation, lineage,
conservation, and displayed-value reconciliation invariant checks run against those
same cached tables; caching avoids reconstruction work but does not bypass a
production safety check.

The Performance Comparison sub-feature does not try to rebuild a full accounting
ledger. It counts configured Modified Dietz formula inputs, shows supporting
source-data evidence, and leaves unsupported or ambiguous rows as review
context unless explicit YAML policy says otherwise.

## Setup Data Versus Maintainer Data

| Area | Intended audience | Notes |
| --- | --- | --- |
| `ppar/setup_templates/axys_apx_analytics` | Installed users and demos | Copied by `ppar setup` into `analytics/`. |
| `ppar/setup_templates/axys_apx_audit` | Installed users and demos | Copied by `ppar setup` into `audit/`. |
| `ppar/setup_templates/generic_analytics` | Maintainers | Feeds README images, analytics regression tests, and demo-data derivation. It is not the primary onboarding path. |
| `tests/data/axys` | Test authors | Synthetic fixtures for narrow validation and edge-case behavior. |
| `_demo_output` | Maintainers | Generated local report/image output; not source-data and not shipped as user setup input. |

## Configuration Boundary

The YAML files are the main configuration and onboarding surface.

Performance Analytics YAML answers:

- Where are the portfolio and security performance files?
- Which columns contain identifiers, dates, returns, weights, and attribution fields?
- Which classifications and mappings should reports use?
- Which reports should be written?

Audit YAML answers:

- Which two snapshots are being compared?
- Which CSV files belong to each snapshot?
- Which columns map into normalized datasets?
- Which transaction codes are external flows, income, fees, trades, or review-only rows?
- Which changed fields are allowed to explain Modified Dietz differences?
- Which changed fields are context evidence or intentionally suppressed?

The package should fail early when required YAML treatment is missing. Silent
guessing is worse than a blocked setup because unexplained performance
differences need an auditable reason.

## Calculation And Validation Boundaries

Performance Analytics and Audit validate different contracts:

- Analytics validates normalized periods, identifiers, weights, classifications,
  mappings, reporting-frequency coverage, and risk-statistic inputs before using
  them in calculations.
- Audit validates source mappings and accounting roles, then checks that every
  source difference keeps a traceable disposition and that counted Modified
  Dietz causes remain conserved and reconcile to the reviewer-facing totals.
- A `Fully Explained` audit period is allowed only when its counted cause amounts
  reconcile to its explained difference and its displayed cause amounts reconcile
  after report rounding.

Fixed-frequency Analytics validation is based on calendar-month, calendar-quarter,
or calendar-year coverage. It does not use a market-holiday calendar and does not
require adjacent dates. Weekends, holidays, and other within-period date gaps are
therefore accepted; a wholly missing requested reporting bucket is rejected.

These production checks protect outcomes that can vary with real user data. Broader
structural checks, metamorphic financial tests, artifact parity checks, demo-matrix
coverage, and scale regressions remain in the test and release-candidate layers so
ordinary report runs do not repeatedly pay their full cost.

Package validation failures use `PpaError`. Callers may inspect its stable `code`,
human-readable `detail`, and copied machine-readable `context`; command-line entry
points continue to present the complete formatted message.

## Report Boundary

Performance Analytics outputs are presentation artifacts for
portfolio-vs-benchmark review.
They emphasize attribution, contribution, cumulative effects, and risk
statistics.

Audit outputs are reviewer artifacts. The normal review order is:

1. `Performance Differences`
2. `Performance Difference Causes`
3. `Data Issues`

`source_detail.csv`, stored under `supporting_files/` when expanded, retains the
active row-level evidence used for audit and troubleshooting; it is supporting
detail rather than an ordinary review sheet.

The complete supporting evidence is stored compactly in `audit_support.zip` by
default. `--expand-all-supporting-files` writes the same validated artifacts as
individual files under `supporting_files/`.

Optional reconstruction diagnostics are secondary. They help debug reported
returns, but they should not become the first review surface for ordinary users.

## Where To Extend

Use this rough guide before adding code or docs:

| Goal | Start here |
| --- | --- |
| Add user setup behavior | `ppar/audit/cli/setup.py` and packaged `ppar.yaml` files. |
| Add analytics behavior | `ppar/analytics/` and the Axys/APX analytics starter YAML. |
| Add performance-comparison logic | `ppar/audit/performance_comparison/` plus focused tests under `tests/`. |
| Add Data Issues behavior | `ppar/audit/data_issues/` plus focused tests under `tests/`. |
| Add transaction coverage | Update evidence/docs first, then test-only fixtures, then packaged demo rows only when realistic. |
| Add portfolio-level or shared-platform direction | `docs/roadmap.md`. |
| Add Audit product direction or active MVP scope | `docs/audit/product_constitution.md` or `docs/audit/mvp_plan.md`. |
| Add Analytics product direction | `docs/analytics/roadmap.md`. |
| Add Axys/APX facts or evidence gaps | `docs/axys_apx/` using its documented chapter, evidence, and contract roles. |

Keep new docs rare. Prefer updating the owning product roadmap, portfolio
roadmap, architecture map, maintainer guide, setup README, or YAML comments
before adding another document.
