# Packaged Demo Data

This directory contains CSV inputs used by the user-facing demo commands.

## Mega-Cap Analytics Data

The Mega-Cap analytics files are generated from historical holdings of the
iShares S&P 100 ETF as a public, reproducible proxy for a U.S. mega-cap
benchmark. The user-facing name is "Mega-Cap"; OEF is only the source
provenance.

Files:

- `performance/Mega-Cap Alpha Portfolio.csv`
- `performance/Mega-Cap Benchmark.csv`
- `classifications/Mega-Cap Security.csv`
- `classifications/Mega-Cap Economic Sector.csv`
- `mappings/Mega-Cap Security--to--Mega-Cap Economic Sector.csv`

The data covers 60 consecutive monthly periods from June 2021 through May 2026.
The March 2024 holdings snapshot uses March 28, 2024 because U.S. equity markets
were closed on March 29, 2024 for Good Friday.

Cash and derivative rows from the source holdings are aggregated into
`CASHBAL`, mapped to the `Cash` sector. `CASHBAL` uses BIL adjusted monthly
returns as its cash-return proxy.

## Refresh Notes

The packaged CSVs are the source of truth for `ppar-analytics-demo`,
`README.md`, and the README images under `images/`. Temporary generation files
under `_demo_output/analytics_data_generation/` are provenance and refresh aids,
not packaged demo inputs.

When regenerating this dataset, follow
[`docs/analytics_demo_refresh.md`](../../../docs/analytics_demo_refresh.md).
That guide covers candidate data generation, validation, promotion into this
directory, README story updates, image regeneration with
`scripts/render_readme_images.py`, and final test verification.
