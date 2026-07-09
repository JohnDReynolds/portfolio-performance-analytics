# PPAR

PPAR uses local Axys/APX data to perform two main functions:

1. Produce Brinson-Fachler attribution, Carino-smoothed multi-period effects,
   contribution views, and ex-post risk statistics.
2. Audit reported performance, source-data changes, and source-data quality.

PPAR is built for local execution. Your Axys/APX exports stay on your machine or
inside your environment.

[License](LICENSE)

---

## Performance Analytics

Use Analytics when you want a clean explanation of performance versus a
benchmark. PPAR produces Brinson-Fachler attribution, Carino-smoothed
multi-period effects, contribution views, and ex-post risk statistics.

Typical questions from the Mega-Cap Alpha vs Mega-Cap Benchmark demo:

- Did Mega-Cap Alpha outperform? Yes. The portfolio returned about 89.3%
  versus 82.4% for the benchmark, or roughly 684 bps of active return.
- Was outperformance mostly allocation or selection? Mostly selection. The
  overall sector attribution view shows about 19 bps from allocation and about
  665 bps from selection.
- Which area drove the result? Information Technology was the largest positive
  contributor, with roughly 351 bps of total attribution effect.
- Did the portfolio take more risk? Slightly, but risk-adjusted results still
  improved: Sharpe was about 0.70 versus 0.67, and Sortino was about 1.82
  versus 1.74.

<img
  src="docs/images/readme/OverallAttributionByEconomicSector.png"
  alt="Overall attribution by economic sector"
  width="100%"
/>

<img
  src="docs/images/readme/OverallContributionByEconomicSector.png"
  alt="Overall contribution by economic sector"
  width="100%"
/>

<img
  src="docs/images/readme/SubPeriodAttributionEffectsByEconomicSector.png"
  alt="Sub-period attribution effects by economic sector"
  width="100%"
/>

<img
  src="docs/images/readme/SubPeriodReturns.png"
  alt="Sub-period returns"
  width="100%"
/>

<img
  src="docs/images/readme/ActiveContributionsByEconomicSector.png"
  alt="Active contributions by economic sector"
  width="100%"
/>

<img
  src="docs/images/readme/TotalAttributionEffectsByEconomicSector.png"
  alt="Total attribution effects by economic sector"
  width="100%"
/>

<img
  src="docs/images/readme/CumulativeAttributionEffectsByEconomicSector.png"
  alt="Cumulative attribution effect by economic sector"
  width="100%"
/>

<img
  src="docs/images/readme/CumulativeReturns.png"
  alt="Cumulative returns"
  width="100%"
/>

<img
  src="docs/images/readme/CumulativeAttributionByEconomicSector.jpg"
  alt="Cumulative attribution by economic sector table"
  width="100%"
/>

<img
  src="docs/images/readme/OverallAttributionByEconomicSector.jpg"
  alt="Overall attribution by economic sector table"
  width="100%"
/>

<img
  src="docs/images/readme/OverallAttributionBySecurity.jpg"
  alt="Overall attribution by security table"
  width="100%"
/>

<img src="docs/images/readme/RiskStatistics.jpg" alt="Risk statistics report" width="100%" />

---

## Performance Auditing

Use Performance Auditing when reported returns have changed and you need to know
why. It compares an original source-data snapshot with a newer or restated
snapshot, then separates three things:

- changed reported performance;
- source-data rows that explain the change;
- suspicious source-data relationships that deserve review.

The workbook includes Performance Comparison sheets plus a **Data Audit Issues**
sheet for price ranges, dividend rates, accrued-interest rates, missing
dividends, and holding value math.

Typical questions:

- Which portfolio or security returns changed?
- Which holdings or transactions explain the change?
- Which differences still need human review?
- Are there data-quality issues that might explain suspicious results?

<img
  src="docs/images/readme/PerformanceComparisonPortfolio.jpg"
  alt="Portfolio performance comparison report"
  width="100%"
/>

<img
  src="docs/images/readme/PerformanceComparisonSecurity.jpg"
  alt="Security performance comparison report"
  width="100%"
/>

The workbook output includes performance differences, explanation rows, data
audit issues, residual review status, and source detail.

---

## Setup

Start by creating a local starter workspace. The starter data lets you run the
full workflow before replacing anything with your own exports.

```bash
pip install ppar
ppar setup ./my_ppar_data
```

Then run the two workflows:

```bash
ppar analytics ./my_ppar_data/analytics
ppar performance_audit ./my_ppar_data/performance_audit
```

Open the files printed by each command.

The setup folder stays intentionally small:

```text
my_ppar_data/
  README.md
  analytics/
    ppar.yaml
    portperf.csv
    secperf.csv
    run_analytics.py
  performance_audit/
    ppar.yaml
    run_portfolio_comparison.py
    run_security_comparison.py
    snapshot_a/
      portperf.csv
      holdings.csv
      transactions.csv
      secperf.csv
    snapshot_b/
      portperf.csv
      holdings.csv
      transactions.csv
      secperf.csv
```

To customize with your own data, replace the starter CSVs and follow the
`Customizing` section in `./my_ppar_data/README.md`.

---

## Inputs

The Axys/APX starter path focuses on IMEX-style CSV exports:

- portfolio performance
- security performance
- holdings
- transactions

Analytics can run from portfolio/security performance exports. Performance
Auditing uses two source-data snapshots, usually an older/original snapshot
and a newer/restated snapshot.

PPAR normalizes those files through YAML so each site can document its own local
field names, transaction-code treatment, and report assumptions.

---

## Outputs

Analytics writes browser-ready attribution/risk files and chart images:

```text
analytics/output/
  security_overall_attribution.html
  sector_overall_attribution.html
  sector_cumulative_attribution.html
  risk_statistics.html
  *.png
```

Performance Auditing writes review packages:

```text
performance_audit/output/
  portfolio/report.xlsx
  portfolio/report.html
  security/report.xlsx
  security/report.html
```
