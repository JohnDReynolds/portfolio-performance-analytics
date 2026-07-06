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
  -> creates analytics/ and performance_comparison/ starter folders
  -> installs heavily commented ppar.yaml files
  -> copies Axys/APX starter CSV files
```

After setup, users run:

```bash
ppar analytics <site_directory>/analytics
ppar performance_comparison <site_directory>/performance_comparison
```

`ppar perfcomp` is the short alias for `ppar performance_comparison`.

## Package Map

| Package | Role |
| --- | --- |
| `ppar.analytics` | Performance analytics, attribution, contribution, risk statistics, and analytics report generation. |
| `ppar.axys` | Axys/APX ingestion and normalization support. This name may eventually become `ppar.axysapx`, but it remains the current import boundary. |
| `ppar.performance_comparison` | Snapshot loading, source-data comparison, Modified Dietz evidence assembly, report/workbook generation, and validation. |
| `ppar.demos` | Source-checkout smoke modules and packaged starter-data helpers. Public users should start with `ppar setup`, not demo modules. |

## Data Flow

PPAR generally treats source files as configured site inputs, not as a hidden
database.

For Analytics:

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
  -> report.xlsx and report.html
```

The performance-comparison engine does not try to rebuild a full accounting
ledger. It counts configured Modified Dietz formula inputs, shows supporting
source-data evidence, and leaves unsupported or ambiguous rows as review
context unless explicit YAML policy says otherwise.

## Setup Data Versus Maintainer Data

| Area | Intended audience | Notes |
| --- | --- | --- |
| `ppar/setup_templates/axysapx_analytics` | Installed users and demos | Copied by `ppar setup` into `analytics/`. |
| `ppar/setup_templates/axysapx_performance_comparison` | Installed users and demos | Copied by `ppar setup` into `performance_comparison/`. |
| `ppar/setup_templates/generic_analytics` | Maintainers | Feeds README images, analytics regression tests, and demo-data derivation. It is not the primary onboarding path. |
| `tests/data/axys` | Test authors | Synthetic fixtures for narrow validation and edge-case behavior. |
| `_demo_output` | Maintainers | Generated local report/image output; not source-data and not shipped as user setup input. |

## Configuration Boundary

The YAML files are the main configuration and onboarding surface.

Analytics YAML answers:

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

Analytics outputs are presentation artifacts for portfolio-vs-benchmark review.
They emphasize attribution, contribution, cumulative effects, and risk
statistics.

Performance-comparison outputs are reviewer artifacts. The normal review order
is:

1. `Performance Differences`
2. `Performance Difference Causes`
3. `Raw Audit Trail`

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
