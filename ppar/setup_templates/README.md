# Packaged Demo Data

This directory contains CSV inputs, YAML files, and Python runner scripts used
by packaged setup templates, maintainer checks, and README image generation.
The public onboarding path starts with the PPAR Audit workspace created by
`ppar setup ./my_ppar_audit`.

## Axys/APX Workspace Data

- `axys_apx_audit/`: Audit snapshots and YAML copied directly into
  `my_ppar_audit`.
- `axys_apx_analytics/`: Analytics CSVs and YAML copied directly into
  `my_ppar_analytics` by `ppar setup ./my_ppar_analytics --analytics`.

## Generic Analytics Data

The Mega-Cap analytics files are generated from historical holdings of the
iShares S&P 100 ETF as a public, reproducible proxy for a U.S. mega-cap
benchmark. `ppar setup ./my_ppar_generic_analytics --generic-analytics` copies
these files into a standalone, vendor-neutral Python workspace. The same data
also supports regression tests, README marketing images, and maintainer
data-derivation scripts. Audit remains the default onboarding path.

Files:

- `generic_analytics/performance/Mega-Cap Alpha Portfolio.csv`
- `generic_analytics/performance/Mega-Cap Benchmark.csv`
- `generic_analytics/classifications/Security.csv`
- `generic_analytics/classifications/Economic Sector.csv`
- `generic_analytics/mappings/Security--to--Economic Sector.csv`

The data covers 60 consecutive monthly periods from June 2021 through May 2026.
The March 2024 holdings snapshot uses March 28, 2024 because U.S. equity markets
were closed on March 29, 2024 for Good Friday.

Cash and derivative rows from the source holdings are aggregated into
`CASHUSD`, mapped to the `Cash` sector. `CASHUSD` uses BIL adjusted monthly
returns as its cash-return proxy.

## Refresh Notes

The packaged CSVs are the source of truth for the optional Generic Analytics
workspace, `docs/analytics/README.md`, images under `docs/images/readme/`, and
selected operational demo derivation scripts. Refresh helpers live under
`scripts/generic_analytics_demo_data/`; generated files under
`_demo_output/generic_analytics_data_generation/` are cache/provenance output,
not packaged demo inputs.

When regenerating this dataset, follow
[`docs/analytics/analytics_demo_refresh.md`](../../docs/analytics/analytics_demo_refresh.md).
That guide covers candidate data generation, validation, promotion into this
directory, README story updates, image regeneration with
`scripts/render_readme_images.py`, and final test verification.
