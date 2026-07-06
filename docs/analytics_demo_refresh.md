# Analytics Demo Refresh Guide

This maintainer guide documents the refresh workflow for the generic Mega-Cap
analytics dataset, README story, and README images. Use it when replacing the
packaged Mega-Cap inputs with a newer or better historical dataset.

The public first-user onboarding path is the Axys/APX setup workspace created by
`ppar setup`. The generic Mega-Cap dataset is retained as maintainer/demo
infrastructure because it feeds README marketing images, analytics regression
tests, and Axys/APX demo-data derivation.

## Current Demo Shape

The generic analytics refresh path uses packaged CSV files under
`ppar/setup_templates/generic_analytics/`:

- `performance/Mega-Cap Alpha Portfolio.csv`
- `performance/Mega-Cap Benchmark.csv`
- `classifications/Security.csv`
- `classifications/Economic Sector.csv`
- `mappings/Security--to--Economic Sector.csv`

The current files are generated from historical iShares S&P 100 ETF holdings as
a public proxy for a U.S. mega-cap benchmark. The user-facing name is
`Mega-Cap`; `OEF` is only source provenance. Cash and derivative rows are
aggregated into `CASH_USD`, mapped to the `Cash` sector, and use BIL adjusted
monthly returns as a cash-return proxy.

The README story and images are generated from these packaged files, not from
the temporary data-generation workspace. The packaged data remains monthly, but
the README images are rendered quarterly to keep date-heavy charts readable.
Installed setup users do not need this refresh path.

## Refresh Workflow

### 1. Generate Candidate Data

Use the maintained data-refresh helper for candidate generation:

```bash
./.venv/bin/python scripts/generic_analytics_demo_data/generate_mega_cap_analytics_demo_data.py
```

Useful options:

```bash
./.venv/bin/python scripts/generic_analytics_demo_data/generate_mega_cap_analytics_demo_data.py \
  --years 5 \
  --alpha-tilt 0.8
```

Notes:

- The generator lives under `scripts/generic_analytics_demo_data/` because it is a
  maintained refresh/provenance tool, not a package runtime API.
- It may require network access, `requests`, and local `yfinance`.
- It writes generated files under
  `_demo_output/generic_analytics_data_generation/generated_oef_files/`.
- Keep monthly continuity non-negotiable. If a market holiday falls on month
  end, use a nearby available business-day holdings snapshot and document it.
- Do not promote equal-weight benchmark data. The benchmark should remain
  holdings-weighted/cap-weight-like.

### 2. Review Candidate Data

Before promotion, inspect at least:

- Date coverage and monthly continuity.
- Holdings count and any dropped/renamed securities.
- Cash weight average and maximum.
- Sector weights for believability.
- Benchmark cumulative return versus a broad published proxy for the same
  period.
- Alpha portfolio cumulative return, active return, and risk statistics.
- Whether the story is believable without looking engineered.

Run or update the candidate validation helper if it still exists:

```bash
./.venv/bin/python scripts/generic_analytics_demo_data/validate_generated_analytics_demo_data.py
```

### 3. Promote Packaged CSVs

Once accepted, copy the generated files into `ppar/setup_templates/` using the
stable user-facing `Mega-Cap` names:

```text
generated_oef_files/performance/Generated OEF Alpha Portfolio.csv
  -> ppar/setup_templates/generic_analytics/performance/Mega-Cap Alpha Portfolio.csv

generated_oef_files/performance/Generated OEF Benchmark.csv
  -> ppar/setup_templates/generic_analytics/performance/Mega-Cap Benchmark.csv

generated_oef_files/classifications/Generated OEF Security.csv
  -> ppar/setup_templates/generic_analytics/classifications/Security.csv

generated_oef_files/classifications/Generated OEF Economic Sector.csv
  -> ppar/setup_templates/generic_analytics/classifications/Economic Sector.csv

generated_oef_files/mappings/Generated OEF Security--to--Generated OEF Economic Sector.csv
  -> ppar/setup_templates/generic_analytics/mappings/Security--to--Economic Sector.csv
```

After promotion, update:

- `ppar/setup_templates/README.md`
- `tests/test_mega_cap_demo_data_contract.py`
- any expected coverage dates, cash notes, and story metrics.

### 4. Verify Packaged Data

Run the demo-data contract and package metadata tests:

```bash
./.venv/bin/python -m pytest \
  tests/test_mega_cap_demo_data_contract.py \
  tests/test_package_metadata.py
```

Then run the optional generic analytics setup script from a temporary setup
workspace:

```bash
./.venv/bin/python -m ppar.cli setup /tmp/ppar_generic_smoke --include-generic-analytics
./.venv/bin/python /tmp/ppar_generic_smoke/generic_analytics/run_generic_analytics.py
```

Review output under `/tmp/ppar_generic_smoke/generic_analytics/output/`. This
is a maintainer smoke output, not the primary user setup workspace.

### 5. Update README Story

Use the packaged data and the same frequency used by
`scripts/render_readme_images.py`, not the generator summary alone, to compute
README numbers. The README currently renders quarterly images from monthly
packaged inputs. The useful values are:

- Portfolio cumulative return.
- Benchmark cumulative return.
- Active return.
- Total allocation effect.
- Total selection effect.
- Largest sector total effect.
- Annualized Sharpe ratio for portfolio and benchmark.
- Annualized Sortino ratio for portfolio and benchmark.

The README story should be concise and believable. Prefer phrasing like
"public proxy" or "realistic, inspectable example" instead of implying exact
index replication.

### 6. Regenerate README Images

The README images are generated from packaged demo files:

```bash
./.venv/bin/python scripts/render_readme_images.py
```

The script regenerates:

- chart PNGs in `docs/images/readme/`
- table screenshots as JPGs in `docs/images/readme/`

Headless Chrome may require normal local-machine permissions. If the sandboxed
runner cannot launch Chrome, rerun the same command with the tool approval used
for GUI/headless-browser execution.

Visually spot-check at least:

- `docs/images/readme/OverallAttributionByEconomicSector.png`
- `docs/images/readme/RiskStatistics.jpg`
- one large table screenshot such as
  `docs/images/readme/OverallAttributionBySecurity.jpg`

### 7. Final Verification

Run the full test suite:

```bash
./.venv/bin/python -m pytest
```

Before committing, check:

- README story matches regenerated image titles and values.
- The optional generic analytics setup script uses the same packaged files.
- `scripts/render_readme_images.py` uses the same packaged files.
- `ppar/setup_templates/README.md` documents source provenance and any date
  substitutions.
- No generated `_demo_output/` files are accidentally staged.

## Refresh Helper Location

Reusable refresh helpers live in `scripts/generic_analytics_demo_data/`. Generated
candidate CSVs, downloaded holdings, price caches, and audit output stay under
`_demo_output/generic_analytics_data_generation/` and should not be packaged as demo
inputs until explicitly promoted.

Avoid adding new package runtime dependencies solely for data refresh unless
that is an explicit product decision.
