# PPAR - Portfolio Performance Auditing & Analytics Reporting

PPAR is a Python package that creates Audit and Performance
Analytics reports from local portfolio accounting data.

1. You can generate standard charts, xlsx, and html output.
2. Being built in Python, it is highly customizable.  You can automate batch
   runs and produce custom output in CSV, Pandas, Polars, JSON, and XML formats.
3. Everything runs locally, so your data stays inside your environment.

[Download the product overview (PDF)](PPAR.pdf) | [License](LICENSE)

---

## Audit

Use Audit to answer: "Why did my reported performance change?"

- **Performance Comparison:** identifies changed portfolio and security
  performance for each time period, quantitatively attributes the differences
  to changes in holdings and transactions, and highlights anything that needs human
  review.
- **Data Issues:** flags suspicious source-data relationships — including price
  ranges, dividend rates, accrued-interest rates, and missing dividends — that
  may indicate data-quality issues.

<img
  src="docs/images/readme/PerformanceAuditPortfolio.jpg"
  alt="Portfolio Audit report"
  width="100%"
/>

---

## Performance Analytics

Use Performance Analytics when you want a clean explanation of portfolio
performance versus a benchmark. It includes:

- **Performance Attribution:** Brinson-Fachler attribution, Carino-smoothed
  multi-period effects, and contribution views.
- **Ex-Post Risk:** ex-post risk statistics calculated from realized returns.

Typical questions from the below Mega-Cap Alpha vs Mega-Cap Benchmark demo:

- Did Mega-Cap Alpha outperform the benchmark? Yes. The portfolio returned
  about 89.3% versus 82.4% for the benchmark, or roughly 684 bps of active
  return.
- Was outperformance mostly allocation or selection? Mostly selection. The
  overall sector attribution view shows about 19 bps from allocation and about
  665 bps from selection.
- Which area drove the result? Information Technology was the largest positive
  contributor, with roughly 351 bps of total attribution effect.
- Did the portfolio take more risk than the benchmark? Slightly, but
  risk-adjusted results still improved: Sharpe was about 0.70 versus 0.67, and
  Sortino was about 1.96 versus 1.85.

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

Install the ppar package.

```bash
pip install ppar
```

Create a local starter workspace. The workspace is seeded with demo data that
lets you run full demos before replacing the demo data with your own data.

```bash
ppar setup ./my_ppar_data
```

Run the demos.

```bash
ppar audit ./my_ppar_data/audit
ppar analytics ./my_ppar_data/analytics
```

To customize the workspace with your own data, follow the `Customizing` section in
`./my_ppar_data/README.md`.

```text
my_ppar_data/
  README.md
  audit/
    ppar.yaml
    run_audit.py
    snapshot_a/
      portperf.csv
      holdings.csv
      transactions.csv
      secref.csv
      secperf.csv
      fx_rates.csv
      splits.csv
    snapshot_b/
      portperf.csv
      holdings.csv
      transactions.csv
      secref.csv
      secperf.csv
      fx_rates.csv
      splits.csv
  analytics/
    ppar.yaml
    portperf.csv
    secperf.csv
    secref.csv
    run_analytics.py
```

---

## Inputs

The source-data files are typically IMEX-style CSV exports:

- portfolio performance
- security performance
- holdings
- transactions
- FX rates
- split factors

Audit uses two source-data snapshots, usually an older/original
snapshot and a newer/restated snapshot.

Performance Analytics uses portfolio performance, security performance, and security
reference exports.

PPAR normalizes those files through YAML, so that each site can configure its own
local field names, transaction-code treatment, and report assumptions.

---

## Outputs

Audit writes review packages:

To prevent unusably large report artifacts, Audit stops with a nonzero exit code
before writing a report when any primary review table would exceed 100,000 rows.
The error identifies the oversized table and its largest contributors so the user
can narrow the portfolio/date scope or correct the upstream differences.

```text
audit/output/
  portfolio/
    portfolio_audit.xlsx
    portfolio_audit.html
    source_detail.csv
    audit_support.zip
  security/
    security_audit.xlsx
    security_audit.html
    source_detail.csv
    audit_support.zip
```

Use `--no-xlsx-output` for HTML-only output or `--no-html-output` for XLSX-only
output. Supplying both options writes a CSV-only audit and promotes
`performance_differences.csv`, `performance_difference_causes.csv`,
`data_issues.csv`, and `source_detail.csv` to each report directory.

Use `--expand-all-supporting-files` to replace each `audit_support.zip` with an
expanded `supporting_files/` directory.

Performance Analytics writes attribution and ex-post risk reports and chart images:

```text
analytics/output/
  security_overall_attribution.html
  sector_overall_attribution.html
  sector_cumulative_attribution.html
  risk_statistics.html
  *.png
```

The Python API can also return CSV, Pandas, Polars, JSON, and XML formats for
Performance Analytics result tables.
