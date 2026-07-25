# PPAR Analytics

**Turn portfolio and benchmark returns into a clear, decision-ready story.**

PPAR Analytics shows whether a portfolio added value, identifies the allocation
and selection decisions that drove the result, and places those returns in the
context of realized risk. It creates attribution, contribution, cumulative-return,
and ex-post risk reports from local portfolio-accounting exports.

- Everything runs locally, so portfolio data stays inside the user's environment.
- Standard output includes HTML tables and PNG charts.
- The Python API supports CSV, Pandas, Polars, JSON, and XML result formats.

PPAR Analytics is maintained in the PPAR codebase but is not part of the current
PPAR Audit validation program or default onboarding workflow.

---

## What PPAR Analytics Answers

PPAR Analytics is built around a portfolio-versus-benchmark review:

- Did the portfolio outperform its benchmark?
- Was active performance driven primarily by allocation or selection?
- Which sectors and securities contributed most to the result?
- How did active effects develop over time?
- Did the return justify the realized risk?

It includes:

- **Performance Attribution:** Brinson-Fachler attribution, Carino-smoothed
  multi-period effects, and contribution views.
- **Ex-Post Risk:** risk statistics calculated from realized portfolio and
  benchmark returns.

---

## Mega-Cap Alpha Demonstration

The maintained demonstration compares Mega-Cap Alpha with the Mega-Cap Benchmark:

- **Did Mega-Cap Alpha outperform?** Yes. The portfolio returned about 89.3%
  versus 82.4% for the benchmark, or roughly 684 bps of active return.
- **Was outperformance mostly allocation or selection?** Mostly selection. The
  overall sector attribution view shows about 19 bps from allocation and about
  665 bps from selection.
- **Which area drove the result?** Information Technology was the largest
  positive contributor, with roughly 351 bps of total attribution effect.
- **Did the portfolio take more risk?** Slightly, but risk-adjusted results still
  improved: Sharpe was about 0.70 versus 0.67, and Sortino was about 1.96 versus
  1.85.

<img
  src="../images/readme/OverallAttributionByEconomicSector.png"
  alt="Overall attribution by economic sector"
  width="100%"
/>

<img
  src="../images/readme/OverallContributionByEconomicSector.png"
  alt="Overall contribution by economic sector"
  width="100%"
/>

<img
  src="../images/readme/SubPeriodAttributionEffectsByEconomicSector.png"
  alt="Sub-period attribution effects by economic sector"
  width="100%"
/>

<img
  src="../images/readme/SubPeriodReturns.png"
  alt="Sub-period returns"
  width="100%"
/>

<img
  src="../images/readme/ActiveContributionsByEconomicSector.png"
  alt="Active contributions by economic sector"
  width="100%"
/>

<img
  src="../images/readme/TotalAttributionEffectsByEconomicSector.png"
  alt="Total attribution effects by economic sector"
  width="100%"
/>

<img
  src="../images/readme/CumulativeAttributionEffectsByEconomicSector.png"
  alt="Cumulative attribution effect by economic sector"
  width="100%"
/>

<img
  src="../images/readme/CumulativeReturns.png"
  alt="Cumulative returns"
  width="100%"
/>

<img
  src="../images/readme/CumulativeAttributionByEconomicSector.jpg"
  alt="Cumulative attribution by economic sector table"
  width="100%"
/>

<img
  src="../images/readme/OverallAttributionByEconomicSector.jpg"
  alt="Overall attribution by economic sector table"
  width="100%"
/>

<img
  src="../images/readme/OverallAttributionBySecurity.jpg"
  alt="Overall attribution by security table"
  width="100%"
/>

<img
  src="../images/readme/RiskStatistics.jpg"
  alt="Ex-post risk report"
  width="100%"
/>

---

## Setup

Install PPAR with its optional Analytics chart dependencies:

```bash
pip install "ppar[analytics]"
```

Create a dedicated Analytics workspace:

```bash
ppar setup ./my_ppar_analytics --analytics
```

Run Analytics:

```bash
ppar analytics ./my_ppar_analytics
```

The workspace includes demonstration data so the complete workflow can run
before the CSV files are replaced with approved local exports.

```text
my_ppar_analytics/
  README.md
  ppar.yaml
  run_analytics.py
  portperf.csv
  secperf.csv
  secmast.csv
```

---

## Inputs

PPAR Analytics uses:

- portfolio performance;
- security performance;
- security master and classification data; and
- portfolio-to-benchmark mappings.

The setup-created `ppar.yaml` maps local filenames and columns, selects the
portfolio and benchmark, defines the reporting frequency and classification,
and records risk assumptions and output choices.

---

## Outputs

The standard workspace writes HTML tables and PNG charts:

```text
output/
  security_overall_attribution.html
  sector_overall_attribution.html
  sector_cumulative_attribution.html
  risk_statistics.html
  *.png
```

The Python API can also return CSV, Pandas, Polars, JSON, and XML formats for
Analytics result tables.

---

## Product and Maintenance Status

PPAR Analytics remains a maintained additional module with its own use case,
demonstration, and roadmap. It is intentionally outside the current Audit-focused
validation-client message and can be evaluated as a separately positioned product
when market evidence supports that step.

See:

- [Analytics roadmap](roadmap.md)
- [Demonstration and image refresh guide](analytics_demo_refresh.md)
- [PPAR architecture](../architecture.md)
