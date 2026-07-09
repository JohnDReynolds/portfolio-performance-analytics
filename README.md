# PPAR

PPAR uses local Axys/APX data for two main workflows:

1. **Performance Auditing:** review changed reported returns and suspicious
   source-data relationships.
2. **Performance Analytics:** produce attribution, contribution, and Ex-Post
   Risk reports.

PPAR is built for local execution. Your Axys/APX exports stay on your machine or
inside your environment.

[License](LICENSE)

---

## Performance Auditing

Use Performance Auditing when reported returns have changed and you need to know
why. It compares an original source-data snapshot with a newer or restated
snapshot, then separates three things:

- changed reported performance;
- source-data rows that explain the change;
- suspicious source-data relationships that deserve review.

Performance Auditing includes two related features:

- **Performance Comparison:** explains changed reported performance.
- **Data Auditing:** flags suspicious source-data relationships such as price
  ranges, dividend rates, accrued-interest rates, missing dividends, and holding
  value math.

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

## Performance Analytics

Use Performance Analytics when you want a clean explanation of performance
versus a benchmark. It includes:

- **Performance Attribution:** Brinson-Fachler attribution, Carino-smoothed
  multi-period effects, and contribution views.
- **Ex-Post Risk:** risk statistics calculated from realized returns.

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

<img src="docs/images/readme/RiskStatistics.jpg" alt="Ex-post risk report" width="100%" />

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
ppar performance_audit ./my_ppar_data/performance_audit
ppar analytics ./my_ppar_data/analytics
```

Open the files printed by each command.

The setup folder stays intentionally small:

```text
my_ppar_data/
  README.md
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
  analytics/
    ppar.yaml
    portperf.csv
    secperf.csv
    run_analytics.py
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

Performance Auditing uses two source-data snapshots, usually an older/original
snapshot and a newer/restated snapshot. Performance Analytics can run from
portfolio/security performance exports.

PPAR normalizes those files through YAML so each site can document its own local
field names, transaction-code treatment, and report assumptions.

---

## Outputs

Performance Auditing writes review packages:

```text
performance_audit/output/
  portfolio/report.xlsx
  portfolio/report.html
  security/report.xlsx
  security/report.html
```

Performance Analytics writes browser-ready attribution and Ex-Post Risk files
and chart images:

```text
analytics/output/
  security_overall_attribution.html
  sector_overall_attribution.html
  sector_cumulative_attribution.html
  risk_statistics.html
  *.png
```
