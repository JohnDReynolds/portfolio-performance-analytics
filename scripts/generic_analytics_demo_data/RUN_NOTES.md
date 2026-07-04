# Analytics Demo Data Run Notes

These notes summarize the current Mega-Cap analytics demo refresh workflow and
the source-data choices behind the packaged CSVs.

## Current Helper Scripts

- `generate_mega_cap_analytics_demo_data.py`: downloads/loads historical
  BlackRock OEF holdings, prices, cash proxy returns, and writes generated
  candidate CSVs.
- `audit_mega_cap_history.py`: audits holdings-date continuity and price
  coverage.
- `validate_generated_analytics_demo_data.py`: loads generated candidate CSVs
  through `Analytics` and verifies the intended portfolio/benchmark story.

The retired SPY/calibration prototype helpers have been removed. Their old
generated outputs and caches are not part of the maintained refresh workflow.

## Output Workspace

The helpers write generated artifacts under:

```text
_demo_output/generic_analytics_data_generation/generated_oef_files/
_demo_output/generic_analytics_data_generation/cache/oef/
```

These files are ignored by Git. Promote generated CSVs into `ppar/demos/data/`
only after review.

## Source Data

BlackRock's product-data JSON endpoint works for historical OEF holdings:

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

The user-facing dataset name is `Mega-Cap`; OEF is source provenance. The older
`1467271812596.ajax` CSV route returned product-page HTML in this environment,
despite a `text/csv` response header. The JSON endpoint returned real
historical rows with `asOfDate`, `ticker`, `issueName`, `sectorName`,
`holdingPercent`, `marketValue`, and `unitsHeld`.

## Current Candidate Summary

The default generated candidate uses 5 years because monthly continuity is
non-negotiable and the 10-year archive has a non-repairable gap from
January-May 2017.

- Holdings snapshots: 61 usable monthly snapshots
- Date range: 2021-06-01 to 2026-05-29
- Benchmark source: OEF month-end holdings weights and adjusted constituent
  returns
- Cash source: BlackRock cash and derivative rows are aggregated into
  `CASH_USD` under the `Cash` sector.
- Cash return proxy: `BIL` adjusted monthly returns from yfinance.
- Usable securities across history: 117, including `CASH_USD`
- Period count: 60
- Missing performance months: none
- Benchmark cumulative return: 1.165913
- Portfolio cumulative return: 1.251763
- Active return: 0.085850
- Portfolio annualized Sharpe from generator: 1.082316
- Benchmark annualized Sharpe from generator: 1.041324
- Average `CASH_USD` weight in generated performance rows: approximately 0.268%
- Maximum `CASH_USD` weight in generated performance rows: approximately 0.466%

One requested holdings date in the 5-year window, 2024-03-29, returns no rows
from BlackRock. The generator substitutes 2024-03-28, which is one day earlier
and still in the same calendar month. The performance period remains March 2024.

## Useful Commands

Generate candidate files:

```bash
./.venv/bin/python scripts/generic_analytics_demo_data/generate_mega_cap_analytics_demo_data.py
```

Validate generated candidate files:

```bash
./.venv/bin/python scripts/generic_analytics_demo_data/validate_generated_analytics_demo_data.py
```

Audit longer source history:

```bash
./.venv/bin/python scripts/generic_analytics_demo_data/audit_mega_cap_history.py
```
