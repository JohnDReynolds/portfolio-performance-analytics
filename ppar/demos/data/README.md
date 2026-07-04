# Packaged Demo Data

This directory contains CSV inputs used by the user-facing demo commands.

## Mega-Cap Analytics Data

The Mega-Cap analytics files are generated from historical holdings of the
iShares S&P 100 ETF as a public, reproducible proxy for a U.S. mega-cap
benchmark. The user-facing name is "Mega-Cap"; OEF is only the source
provenance.

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
`CASH_USD`, mapped to the `Cash` sector. `CASH_USD` uses BIL adjusted monthly
returns as its cash-return proxy.

## Refresh Notes

The packaged CSVs are the source of truth for `ppar-generic-analytics-demo`,
`README.md`, and the README images under `docs/images/readme/`. Refresh helpers
live under `scripts/generic_analytics_demo_data/`; generated files under
`_demo_output/generic_analytics_data_generation/` are cache/provenance output, not
packaged demo inputs.

When regenerating this dataset, follow
[`docs/analytics_demo_refresh.md`](../../../docs/analytics_demo_refresh.md).
That guide covers candidate data generation, validation, promotion into this
directory, README story updates, image regeneration with
`scripts/render_readme_images.py`, and final test verification.
