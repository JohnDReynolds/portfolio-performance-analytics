# Analytics Demo Data Refresh

This directory contains the maintained helpers for refreshing the Mega-Cap
Analytics demo data. Analytics product direction belongs in
[`../../docs/analytics/roadmap.md`](../../docs/analytics/roadmap.md); this file
documents only the repeatable data-generation workflow and its current source
assumptions.

Generated outputs and caches live under
`_demo_output/generic_analytics_data_generation/` and are ignored by Git.
Promote reviewed CSVs into `ppar/setup_templates/` only deliberately.

Analytics and Audit share normalized yFinance observations at
`_demo_output/demo_market_data/yfinance_market_history.csv`. The cache retains
Yahoo's source `Close`, a reconstructed contemporaneous close for holdings,
adjusted close for returns, dividends, stock splits, and repair provenance.
Once coverage is present, ordinary demo construction is offline and does not
download the same observations again.

## Maintained Helpers

- `generate_mega_cap_analytics_demo_data.py` downloads or loads historical
  BlackRock OEF holdings and prices, applies the cash-return proxy, and writes
  candidate Analytics CSVs.
- `audit_mega_cap_history.py` audits holdings-date continuity and price
  coverage.
- `validate_generated_analytics_demo_data.py` loads candidate CSVs through
  `Analytics` and verifies the intended portfolio/benchmark story.

The retired SPY and calibration prototypes are not part of the maintained
workflow. Market-data download libraries are generation-time conveniences,
not PPAR runtime dependencies.

## Refresh Workflow

Generate candidate files:

```bash
./.venv/bin/python scripts/generic_analytics_demo_data/generate_mega_cap_analytics_demo_data.py
```

Validate the generated files:

```bash
./.venv/bin/python scripts/generic_analytics_demo_data/validate_generated_analytics_demo_data.py
```

Audit longer source history when investigating coverage:

```bash
./.venv/bin/python scripts/generic_analytics_demo_data/audit_mega_cap_history.py
```

The helpers write to:

```text
_demo_output/generic_analytics_data_generation/generated_oef_files/
_demo_output/generic_analytics_data_generation/cache/oef/
```

Before promotion, confirm that periods align, weights sum to 1.0, identifiers
and sector mappings are complete, Analytics can load the data, economic-sector
attribution can be produced, and the intended cumulative-return and Sharpe-ratio
story remains true.

## Source and Modeling Assumptions

- The user-facing dataset is called Mega-Cap; OEF identifies the source
  provenance.
- Historical holdings come from BlackRock's product-data JSON endpoint for
  portfolio `239723`.
- The maintained history is five years because the ten-year OEF archive has a
  non-repairable early gap.
- BlackRock cash and derivative rows are aggregated into `CASHUSD` in the
  `Cash` sector.
- BIL adjusted monthly returns provide the cash-return proxy. Audit also uses
  BIL, SHY, IEI, and MBB as disclosed public-market proxies for its synthetic
  cash and fixed-income identifiers.
- The benchmark is a capitalization-weighted proxy; the portfolio applies a
  modest synthetic tilt intended to tell a clear demonstration story.
- Current survivors and current GICS classifications are acceptable for this
  demo when the limitation remains documented.

One requested holdings date, 2024-03-29, has no source rows. The generator uses
2024-03-28, which remains in the same calendar month, while retaining March
2024 as the performance period.

## Current Accepted Dataset

The current five-year candidate covers 61 usable monthly holdings snapshots
from 2021-06-01 through 2026-05-29 and produces 60 performance periods with no
missing months. It contains 117 securities including `CASHUSD`.

At the last accepted refresh:

- benchmark cumulative return: 1.165913;
- portfolio cumulative return: 1.251763;
- active return: 0.085850;
- portfolio annualized Sharpe ratio: 1.082316;
- benchmark annualized Sharpe ratio: 1.041324;
- average `CASHUSD` weight: approximately 0.268%; and
- maximum `CASHUSD` weight: approximately 0.466%.

Treat these values as refresh-review anchors, not permanent product gates.
After accepting a new source snapshot, update this section, refresh the
packaged generic Analytics data, derive the related Axys/APX starter data as
needed, regenerate README images, and run the repository validation workflow.
