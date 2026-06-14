# portfolio-performance-analytics
portfolio-performance-analytics (ppar) is a python package that produces holdings-based multi-period performance attribution, contribution, and benchmark-relative ex-post risk statistics.

[License](LICENSE)

## Table of Contents

- [Description](#description)
- [Features](#features)
- [Inputs](#inputs)
- [Outputs](#outputs)
- [Installation](#installation)
- [Usage](#usage)
- [Performance Comparison](#performance-comparison)
- [Technical](#technical)
- [Enhancements](#enhancements)
- [Support](#support)

---

## Description

portfolio-performance-analytics is a python package (https://pypi.org/project/ppar/) that produces holdings-based multi-period performance attribution, contribution, and benchmark-relative ex-post risk statistics. It uses the Brinson-Fachler methodology for calculating attribution effects, and uses the Carino method for logarithmically-smoothing cumulative effects over multi-period time frames.

---

## Features

The below sample outputs portray a large-cap alpha strategy that has achieved a high active return of 1737 bps over the benchmark.  In the total lines of the Economic Sector Attribution reports, you can see that this active return can be broken down into 359 bps in sector allocation and 1378 bps in selecting securities.  From the Risk Statistics report, you can see that this has been accomplished with a lower downside probabilty than the benchmark (29% vs 36%), and a higher annualized sharpe ratio than the benchmark (2.02 vs 1.27).  The largest contributor to active performance was in the Information Technology Sector.  Although the portfolio was slightly under-allocated in the Information Technology sector (by -0.05%), it did an excellent job of selecting securities for a total active contribution of 431 bps in the sector.

<!--
Image sources are repository-relative so they render for authorized GitHub
viewers when this repository is private. PyPI cannot render these private
assets; public image URLs are required there if image display is needed.
-->

- **Attribution & Contribution**:
<img src="images/OverallAttributionByEconomicSector.png" alt="Overall Attribution by Economic Sector Chart" width="100%" />
<br><br><br>
<img src="images/OverallContributionByEconomicSector.png" alt="Overall Contribution by Economic Sector Chart" width="100%" />
<br><br><br>
<img src="images/SubPeriodAttributionEffectsByEconomicSector.png" alt="Sub-Period Attribution Effects by Economic Sector Chart" width="100%" />
<br><br><br>
<img src="images/SubPeriodReturns.png" alt="Sub-Period Returns Chart" width="100%" />
<br><br><br>
<img src="images/ActiveContributionsByEconomicSector.png" alt="Active Contributions by Economic Sector Chart" width="100%" />
<br><br><br>
<img src="images/TotalAttributionEffectsByEconomicSector.png" alt="Total Attribution Effects by Economic Sector Chart" width="100%" />
<br><br><br>
<img src="images/CumulativeAttributionEffectsByEconomicSector.png" alt="Cumulative Attribution Effect by Economic Sector Chart" width="100%" />
<br><br><br>
<img src="images/CumulativeReturns.png" alt="Cumulative Returns" width="100%" />
<br><br><br>
<img src="images/CumulativeAttributionByEconomicSector.jpg" alt="Cumulative Attribution by Economic Sector Table" width="100%" />
<br><br><br>
<img src="images/OverallAttributionByEconomicSector.jpg" alt="Overall Attribution by Economic Sector Table" width="100%" />
<br><br><br>
<img src="images/OverallAttributionBySecurity.jpg" alt="Overall Attribution by Security Table" width="100%" />
<br><br><br>

- **Ex-Post Risk Statistics**:
<img src="images/RiskStatistics.jpg" alt="Risk Statistics" width="100%" />
<br>

---

## Inputs

The inputs required to produce the analytics fall into three categories:
1. Periodic "classification-level" weights and returns for a portfolio and its benchmark.  A "classification" can be any category such as region, country, economic sector, industry, security, etc.  The weights and returns must satisfy the formula: *SumOf(weights * returns) = Total Return*. They will typically be from-of-period weights and period returns. (Required)
2. Classification items and descriptions. (Optional)
3. Mappings from the classification scheme of the weights and returns to a reporting classification. (Optional)

The input data may be provided directly as either:
1. Pandas DataFrames.
2. Polars DataFrames.
3. Python dictionaries (for Classifications and Mappings).
4. csv files.

Portfolio and benchmark performance sources use narrow rows. The `name` column
is optional; `contribution` and total return are calculated by the package.

```csv
from_date,thru_date,identifier,weight,return,name
2023-12-31,2024-01-31,AAPL,0.4,-0.0422272121,Apple Inc.
2023-12-31,2024-01-31,MSFT,0.6,0.0572811503,Microsoft
```

Wide performance files with per-identifier columns such as `AAPL.ret` and
`AAPL.wgt` are not supported.

For sample input data sources, please refer to the ``ppar-analytics-demo``
command and the ppar/demo_data directory. Once the input data has been
provided, then the analytics may be requested using different calculation
parameters, time-periods, and frequencies:
1. Daily (or for whatever data frequency is provided).
2. Monthly
3. Quarterly
4. Yearly

Typically, a user will develop their own "data source" functions that provide
the data in one of the above formats. The ``ppar-analytics-demo`` command uses
sample data source functions.

---

## Outputs

The outputs are represented by different views and charts.  See [Features](#features) above.  They may be delivered in different formats:
1. csv files
2. html strings
3. json strings
4. Pandas DataFrames
5. png files
6. Polars DataFrames
7. Lightweight Python HTML table objects
8. xml strings

The ``to_html()`` methods return complete HTML document strings. The ``to_table()`` methods return lightweight table objects whose ``as_raw_html()`` method can emit either a complete HTML document or a table fragment. Users can also develop their own "presentation layer" using the various output formats as the inputs to their presentation layer.

---

## Installation
pip install ppar

For chart output, including the chart option in ``ppar-analytics-demo``:
```
pip install "ppar[charts]"
```

---

## Usage
Run the bundled demos from an installed environment:

```bash
ppar-analytics-demo
ppar-axys-analytics-demo
ppar-performance-comparison-demo
ppar-performance-comparison-validate-config
```

---

## Performance Comparison

The performance comparison feature compares two source-data snapshots and helps
explain why reported portfolio performance changed between extraction dates. It
loads normalized portfolio performance, security performance, transactions,
positions, prices, FX rates, cash, and security-reference data; emits stable
findings; and writes reviewer-oriented Markdown, HTML, CSV, and bundle outputs.

Start with these references:

- [Performance Comparison Design Notes](docs/performance_comparison_design.md)
  for the feature model, current implementation checkpoint, YAML semantics, and
  report bundle shape.
- [Axys Common-Core Export Reference](docs/axys_common_core_export.md) for a
  starter Axys export template and field-reference tables.
- [Axys test data notes](tests/data/axys/README.md) for the synthetic fixture
  layouts used by demos and tests.

The performance comparison demo writes a review bundle under `_demo_output`.
Start with `_demo_output/performance_comparison_bundle/report.html`; its Review
Problems grid gives one row per actionable issue. The default demo uses a
multi-portfolio restatement fixture so the grid has several issue shapes to
review. Rows surface severity, portfolio, period, return delta, the problem,
the required action, why it matters, and an optional evidence link. The grid
includes lightweight browser filters and sortable headers. The HTML report
keeps the first view short: Problems first, with backing tables under Evidence
Appendix. Then inspect
`needs_review_summary.csv` for the periods and issues that need review first.
The `review_key` column links period-level bundle tables, and
`review_detail_artifacts` names the CSVs most relevant to each triage row. The
bundle `README.md` and `manifest.json` describe the generated reports and CSV
tables.

A practical review order is:

1. `report.html`: browser-readable Problems grid and optional Evidence
   Appendix.
2. `needs_review_summary.csv`: changed periods, suggested next steps, and
   drilldown artifacts.
3. `impact_coverage.csv`: estimated versus evidence-only cause areas, missing
   inputs, and reviewer-facing coverage status.
4. `context_evidence.csv`: context-only items such as cost basis, commission,
   and security-master changes that are excluded from return-impact estimates.
5. `findings.csv`: complete finding-level audit output.

Other useful bundle tables include `impact_estimates.csv` for currently
quantified impacts, `transaction_activity.csv` for changed transaction rows,
`transaction_cross_checks.csv` for review-only external-flow diagnostics, and
`top_evidence.csv` for ranked evidence rows shown in the report.
`context_evidence_summary.csv` includes reviewer priority labels so linked
portfolio-period context appears ahead of broader reference-data context.
Linked high-priority context is also surfaced in `needs_review_summary.csv`
review cues for the affected portfolio period. `context_evidence.csv` carries
the same priority labels on row-level detail.

Transaction impact estimates are never inferred from transaction codes alone.
When a source does not provide category/sign/flow semantics, supply explicit
rules in the comparison YAML:

```yaml
transaction_rules:
  BUY:
    transaction_category: buy
    cash_flow_sign: negative
    performance_flow_sign: performance
  DEP:
    transaction_category: external_flow
    cash_flow_sign: positive
    performance_flow_sign: external
```

Impact methods are also explicit. Omitted methods leave transaction activity as
review evidence and produce missing-input diagnostics rather than estimates.

```yaml
contribution_impact_methods:
  portfolio_source_field:
    method: source_field_delta_over_begin_market_value
    denominator_source: begin_market_value
    source_fields:
      - income
      - gain_loss
  security_contribution:
    method: vendor_contribution_delta
  security_return:
    method: security_return_delta_times_weight
    weight_source: snapshot_a_weight
```

```yaml
transaction_impact_methods:
  external_flow:
    method: evidence_only
  performance:
    method: transaction_amount_delta_over_return_denominator
    denominator_source: begin_market_value
```

For external-flow review cross-checks, the supported Modified Dietz diagnostic
requires every convention to be named:

```yaml
transaction_impact_methods:
  external_flow:
    method: modified_dietz
    flow_timing: trade_date
    day_count: actual_days
    inclusion_rule: beginning_of_day
    denominator_source: begin_market_value
    double_count_policy: cross_check_only
```

To smoke-test the performance comparison demo from a source checkout:

```bash
./.venv/bin/python -m ppar.demos.performance_comparison_demo
./.venv/bin/python scripts/performance_comparison_validate_bundle.py _demo_output/performance_comparison_bundle
```

Then open `_demo_output/performance_comparison_bundle/report.html` and confirm
the `Reviewer Triage` and `Context Evidence Summary` sections are present. The
`_demo_output/` directory is generated output and is intentionally ignored by
Git.

To validate an existing performance comparison bundle from a source checkout:

```bash
./.venv/bin/python scripts/performance_comparison_validate_bundle.py _demo_output/performance_comparison_bundle
```

To validate a comparison YAML file before running a report:

```bash
ppar-performance-comparison-validate-config tests/data/axys/ppar_performance_comparison_restatement.yaml
```

From a source checkout, the same validator is also available as:

```bash
./.venv/bin/python scripts/performance_comparison_validate_config.py tests/data/axys/ppar_performance_comparison_restatement.yaml
```

---

## Technical
Being built on top of Polars dataframes, ppar is able to efficiently process large datasets through parallel processing, vectorization, lazy evaluation, and using Apache Arrow as its underlying data format.

Run local project checks from a source checkout with the repository virtual
environment:

```bash
./.venv/bin/python scripts/check_project.py
```

For faster routine feedback during small changes:

```bash
./.venv/bin/python scripts/check_project.py --quick
```

To include a temporary wheel and source-distribution build check:

```bash
./.venv/bin/python scripts/check_project.py --build
```

---

## Enhancements
Future enhancements may include:
1. Break out the interaction (cross-product) effect.  It is currently included in the selection effect.
2. Break out the currency effect.
3. Break out the long and short sides.
4. Add additional multi-period smoothing algorithms (e.g. Menchero).
5. Support time-series of risk-free rates (as opposed to a single annual rate).
6. Calculate additional risk statistics.

---

## Support
If you find this project helpful, consider sponsoring it at https://github.com/sponsors/JohnDReynolds to help keep it going!
