# Analytics Demo Data Generation Roadmap

This roadmap records how the current Mega-Cap analytics demo data was created
and what still remains if the same source universe is later extended to Axys
analytics or performance-comparison demos. Reusable refresh helpers live in
`scripts/analytics_demo_data/`; generated outputs and caches live in
`_demo_output/analytics_data_generation/`.

## Objective

Create a believable, simple, reproducible large-cap demo data set that can
eventually replace the current `ppar-analytics-demo` files and become the
source universe for related Axys analytics and performance comparison demos.

The demo should tell a story similar to the current README Features section:

- the portfolio outperforms the benchmark;
- security selection is a major source of active return;
- the portfolio has a better Sharpe ratio than the benchmark;
- GICS sector reporting remains understandable and realistic.

The data does not need to be research-grade. Current survivors, current GICS,
and modest synthetic portfolio tilts are acceptable if documented.

## Guiding Principles

- Do not touch the existing packaged demo CSV files in phase 1.
- Do not add `yfinance` or other market-data downloaders as runtime
  dependencies.
- It is acceptable to use packages already installed in the local `.venv` for
  temporary generation work.
- Prefer simple, inspectable logic over perfect index reconstruction.
- Use all usable holdings from the selected universe unless presentation quality
  becomes a problem.
- Use a cap-weighted benchmark proxy. Equal weighting is not acceptable for the
  main candidate data set.
- Keep generated output names distinct until we explicitly switch demos.

## Phase 1: Generate New Analytics Files

Create a script that downloads or assembles a current large-cap universe,
downloads recent monthly adjusted prices, constructs benchmark and alpha
portfolio performance rows, and writes new CSV files under distinct names.

Candidate script:

```text
scripts/analytics_demo_data/generate_analytics_demo_data.py
```

Candidate generated files:

```text
ppar/demos/data/performance/Generated Large-Cap Alpha Portfolio.csv
ppar/demos/data/performance/Generated Large-Cap Benchmark.csv
ppar/demos/data/classifications/Generated Security.csv
ppar/demos/data/classifications/Generated Economic Sector.csv
ppar/demos/data/mappings/Generated Security--to--Generated Economic Sector.csv
```

The script should also print summary metrics:

- usable security count;
- period count;
- cumulative portfolio return;
- cumulative benchmark return;
- active return;
- portfolio Sharpe ratio;
- benchmark Sharpe ratio;
- top security and sector contributors if easy to calculate.

## Phase 2: Validate The Generated Story

Add lightweight validation to the generation script before any demo code uses
the new files.

Minimum checks:

- portfolio and benchmark periods align;
- each period's weights sum to 1.0;
- every performance identifier has a security name;
- every security maps to a GICS sector;
- every GICS sector has a display name;
- portfolio cumulative return is greater than benchmark cumulative return;
- portfolio Sharpe ratio is greater than benchmark Sharpe ratio;
- generated CSVs can be loaded by `Analytics`;
- economic-sector attribution can be produced.

## Phase 3: Switch The Analytics Demo

After manual review, update `ppar/demos/analytics_demo.py` to use the generated
files as the main data source.

Possible follow-up choices:

- keep generated file names distinct;
- or rename them to the existing canonical demo names once we are confident.

Refresh README images and Features text only after the new output is accepted.

## Phase 4: Derive Axys-Style Analytics Data

Use the same generated source universe to create Axys-style exports for the
Axys analytics demo.

Likely files:

- `portperf.csv`;
- `secperf.csv`;
- `sec_ref.csv`;
- classification or lookup data if useful.

The goal is for `ppar-axys-analytics-demo` to tell the same broad portfolio vs.
benchmark story as `ppar-analytics-demo`, but through Axys-style source files.

## Phase 5: Derive Performance Comparison Snapshots

Create controlled snapshot A and snapshot B data from the generated Axys-style
universe.

Snapshot B should include small, understandable restatements such as:

- one price change;
- one position market value or quantity change;
- one transaction amount change;
- one cash or accrued income change;
- one classification/reference change as context.

The portfolio and security performance comparison workbooks should be able to
answer:

```text
This performance difference happened, and these underlying source-data
differences explain it.
```

## Phase 6: Promote Or Retire Old Demo Data

Once all three demo families use the generated source universe:

- decide which old packaged demo files remain useful;
- move test-only scenarios out of user-facing demo paths if needed;
- remove obsolete generated or legacy files;
- update documentation and contract tests.

## Current Choices And Open Questions

- Current universe choice: current SPY holdings, top 200 by source weight, with
  usable securities retained after 10 years of monthly adjusted price history.
- Should the holdings source be downloaded live or provided as a checked-in
  seed CSV?
- Current benchmark-weight choice: calibrated static nonnegative weights fitted
  to public SPY adjusted returns over the generated monthly window. This is a
  proxy for historical beginning weights, not real historical holdings.
- Open question: should the promoted version use a checked-in seed holdings CSV
  to avoid relying on issuer download availability?
- Open question: should the calibrated benchmark proxy be retained, or should we
  accept a broader top-N universe with less calibration once reviewed?
- Current active-weight tilt: `--alpha-tilt-multiplier 2.0`, which produces a
  believable positive active-return and Sharpe story without sector allocation
  becoming the headline.
- Where should raw download caches live during generation?

## Current Local Notes

- `.cache/` and `_demo_output/` are ignored by Git.
- `yfinance` is available in the current `.venv`, but should remain a
  generation-time convenience rather than a runtime dependency.
