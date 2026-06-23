# Analytics Demo Data Run Notes

## Current Prototype Status

The legacy SPY prototype generator is:

```text
scripts/analytics_demo_data/generate_analytics_demo_data.py
```

It writes only under:

```text
_demo_output/analytics_data_generation/generated_files/
_demo_output/analytics_data_generation/cache/
```

No packaged demo files have been changed.

## Current Data Source Behavior

- The script supports ETF/seed holdings sources, top-N holdings selection, and
  configurable price history length.
- The State Street SPY holdings workbook is reachable and provides ticker, name,
  current weight, and shares held.
- The SPY workbook reports sector as `-`, so the script merges current GICS
  sector from the S&P 500 constituents table.
- The current generated files use SPY holdings, top 200 requested, and a
  calibrated static SPY proxy weight vector. The calibration keeps weights
  nonnegative and sums to 1.0, while fitting public SPY adjusted returns over
  the same monthly window.
- The generated alpha portfolio uses a simple hindsight tilt toward stronger
  realized risk-adjusted return names while preserving benchmark sector weights.
- The current prototype uses `--alpha-tilt-multiplier 2.0`, roughly double the
  original tilt strength.

## Current Output Files

```text
generated_files/performance/Generated Large-Cap Benchmark.csv
generated_files/performance/Generated Large-Cap Alpha Portfolio.csv
generated_files/classifications/Generated Security.csv
generated_files/classifications/Generated Economic Sector.csv
generated_files/mappings/Generated Security--to--Generated Economic Sector.csv
generated_files/summary.json
generated_files/benchmark_calibration_probe.json
```

## Current Run Summary

- Source command:

```bash
./.venv/bin/python scripts/analytics_demo_data/generate_analytics_demo_data.py \
  --holdings-source spy \
  --top-holdings 200 \
  --years 10 \
  --alpha-tilt-multiplier 2.0 \
  --benchmark-weight-model calibrated_static_spy
```

- Usable securities: 186
- Periods: 119 monthly periods
- Date range: 2016-07-31 to 2026-05-31
- Portfolio cumulative return: 3.590719
- Benchmark cumulative return: 3.228568
- Active return: 0.362152
- Portfolio annualized Sharpe from generator: 1.102442
- Benchmark annualized Sharpe from generator: 1.037269

The benchmark cumulative return now matches public SPY adjusted-close returns
for the same generated monthly period window. This avoids the inflated result
from applying today's SPY mega-cap weights across the full 10-year history.

## Package Validation Snapshot

The generated files load successfully through `Analytics` with monthly
frequency.

- Period count: 119
- Security count: 186 usable securities
- Security attribution rows: 187
- Sector attribution rows: 12
- Portfolio cumulative return from generated performance rows: 3.590719
- Benchmark cumulative return from generated performance rows: 3.228568
- Annualized Sharpe from ppar risk statistics:
  - Portfolio: 0.905901
  - Benchmark: 0.842816
- Sector allocation effect is effectively zero because the alpha tilt preserves
  benchmark sector weights.
- Sector selection effect carries the attribution story.

Validation command:

```bash
./.venv/bin/python scripts/analytics_demo_data/validate_generated_analytics_demo_data.py
```

Calibration probe command:

```bash
./.venv/bin/python scripts/analytics_demo_data/probe_benchmark_calibration.py
```

## Current Caveats

- The current output is cap-weighted-like, but still simplified. It uses
  current SPY constituents and inferred static weights, not actual historical
  SPY constituent/share history.
- The inferred weights are intentionally a proxy. They match the public SPY
  return experience better than current-weight or current-share methods, but
  they should not be described as real historical SPY holdings.
- It uses current survivors historically.
- GICS sectors are current-sector assignments from the S&P 500 constituents
  table.
- The portfolio is intentionally synthetic and hindsight-tilted to tell a clear
  demo story.
- The current run is good enough for prototype review, but not yet ready to
  replace the packaged analytics demo files.

## OEF Historical Holdings Prototype

BlackRock's current product-data JSON endpoint works for historical OEF holdings:

```text
https://www.blackrock.com/varnish-api/blk-one01-product-data/product-data/api/v2/get-product-data
```

Required parameters:

```text
appSubType=ISHARES
appType=PRODUCT_PAGE
locale=en_US
targetSite=us-ishares
userType=individual
portfolioId=239723
component=holdings
asOfDate=YYYYMMDD
```

The older `1467271812596.ajax` CSV route returned the product page HTML in this
environment, despite a `text/csv` response header. The JSON endpoint returned
real historical rows with `asOfDate`, `ticker`, `issueName`, `sectorName`,
`holdingPercent`, `marketValue`, and `unitsHeld`.

OEF generator:

```bash
./.venv/bin/python scripts/analytics_demo_data/generate_oef_analytics_demo_data.py
```

The default is now 5 years because monthly continuity is non-negotiable and the
10-year archive has a non-repairable gap from January-May 2017.

Current 5-year OEF prototype summary:

- Holdings snapshots: 61 usable monthly snapshots
- Date range: 2021-06-01 to 2026-05-29
- Benchmark source: OEF month-end holdings weights and adjusted constituent
  returns
- Cash source: BlackRock cash and derivative rows are aggregated into
  `CASHBAL` under the `Cash` sector.
- Cash return proxy: `BIL` adjusted monthly returns from yfinance.
- Usable securities across history: 117, including `CASHBAL`
- Period count: 60
- Missing performance months: none
- Benchmark cumulative return: 1.165913
- Portfolio cumulative return: 1.251763
- Active return: 0.085850
- Portfolio annualized Sharpe from generator: 1.082316
- Benchmark annualized Sharpe from generator: 1.041324
- Average `CASHBAL` weight in generated performance rows: approximately 0.268%
- Maximum `CASHBAL` weight in generated performance rows: approximately 0.466%

One requested holdings date in the 5-year window, 2024-03-29, returns no rows
from BlackRock. The generator substitutes 2024-03-28, which is one day earlier
and still in the same calendar month. The performance period remains March 2024.

Package validation snapshot:

- Security attribution rows: 112
- Sector attribution rows: 13, including `Cash`
- Risk-statistics rows: 26
- Annualized Sharpe from ppar risk statistics:
  - Portfolio: 1.068076
  - Benchmark: 1.018017
- Max period weight-sum error:
  - Portfolio: approximately 5.8e-15
  - Benchmark: approximately 5.6e-15

History audit:

```bash
./.venv/bin/python scripts/analytics_demo_data/audit_oef_history.py
```

Audit findings:

- 10-year requested month-ends: 121
- 10-year available exact month-ends: 113
- Non-repairable missing 10-year month-ends:
  - 2017-01-31
  - 2017-02-28
  - 2017-03-31
  - 2017-04-28
  - 2017-05-31
- Repairable 10-year gaps:
  - 2017-06-30 -> 2017-07-06
  - 2018-03-30 -> 2018-03-29
  - 2024-03-29 -> 2024-03-28
- 5-year price coverage before adding cash: 116 holdings securities, 116 with
  adjusted-price coverage.
