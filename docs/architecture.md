# PPAR Architecture

This document is the compact system map for PPAR. It explains how the main
parts fit together without replacing the roadmap, setup README, YAML comments,
or deeper performance-comparison design notes.

## Product Surface

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
| `ppar.axys` | Axys/APX ingestion and normalization support. This name may eventually become `ppar.axysapx`, but it remains the current import boundary. |
| `ppar.performance_comparison` | Snapshot loading, source-data comparison, Modified Dietz evidence assembly, report/workbook generation, and validation. |
| `ppar.setup_templates` | Packaged setup workspaces and tutorial scripts copied by `ppar setup`. |

## Data Flow

PPAR generally treats source files as configured site inputs, not as a hidden
database.

For Performance Analytics:

```text
Axys/APX portfolio and security performance CSVs
  -> analytics ppar.yaml
  -> ppar analytics
  -> HTML/PNG/XLSX-style analytics outputs
```

For Performance Comparison:

```text
snapshot_a CSVs + snapshot_b CSVs
  -> performance_comparison ppar.yaml
  -> normalized datasets
  -> source-data differences
  -> Modified Dietz evidence and explanation tables
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
conservation, and displayed-value reconciliation assertions run against those
same cached tables; caching avoids reconstruction work but does not bypass a
production safety check.

The performance-comparison engine does not try to rebuild a full accounting
ledger. It counts configured Modified Dietz formula inputs, shows supporting
source-data evidence, and leaves unsupported or ambiguous rows as review
context unless explicit YAML policy says otherwise.

## Setup Data Versus Maintainer Data

| Area | Intended audience | Notes |
| --- | --- | --- |
| `ppar/setup_templates/axysapx_analytics` | Installed users and demos | Copied by `ppar setup` into `analytics/`. |
| `ppar/setup_templates/axysapx_performance_comparison` | Installed users and demos | Copied by `ppar setup` into `audit/`. |
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

Performance Comparison YAML answers:

- Which two snapshots are being compared?
- Which CSV files belong to each snapshot?
- Which columns map into normalized datasets?
- Which transaction codes are external flows, income, fees, trades, or review-only rows?
- Which changed fields are allowed to explain Modified Dietz differences?
- Which changed fields are context evidence or intentionally suppressed?

The package should fail early when required YAML treatment is missing. Silent
guessing is worse than a blocked setup because unexplained performance
differences need an auditable reason.

## Report Boundary

Performance Analytics outputs are presentation artifacts for
portfolio-vs-benchmark review.
They emphasize attribution, contribution, cumulative effects, and risk
statistics.

Performance-comparison outputs are reviewer artifacts. The normal review order
is:

1. `Performance Differences`
2. `Performance Difference Causes`
3. `source_detail.csv`

The complete supporting evidence is stored compactly in `audit_support.zip` by
default. `--expand-all-supporting-files` writes the same validated artifacts as
individual files under `supporting_files/`.

Optional reconstruction diagnostics are secondary. They help debug reported
returns, but they should not become the first review surface for ordinary users.

## Where To Extend

Use this rough guide before adding code or docs:

| Goal | Start here |
| --- | --- |
| Add user setup behavior | `ppar/performance_comparison/cli/setup.py` and packaged `ppar.yaml` files. |
| Add analytics behavior | `ppar/analytics/` and the Axys/APX analytics starter YAML. |
| Add performance-comparison logic | `ppar/performance_comparison/` plus focused tests under `tests/`. |
| Add transaction coverage | Update evidence/docs first, then test-only fixtures, then packaged demo rows only when realistic. |
| Add broad product direction | `docs/roadmap.md`. |

Keep new docs rare. Prefer updating the existing roadmap, architecture map,
repository guide, setup README, or YAML comments before adding another document.
