# PPAR

PPAR helps Axys/APX teams explain investment performance and review source-data
changes without sending portfolio data outside the firm.

It has two workflows:

- **Analytics:** explain portfolio performance versus a benchmark with HTML and
  PNG attribution/risk reports.
- **Performance Comparison:** explain why reported performance changed between
  two source-data snapshots with Excel workbooks and HTML review reports.

PPAR is built for local execution. Your Axys/APX exports stay on your machine or
inside your environment.

[License](LICENSE)

---

## Performance Analytics

Use Analytics when you want a clean explanation of performance versus a
benchmark. PPAR produces Brinson-Fachler attribution, contribution views,
Carino-smoothed multi-period effects, and ex-post risk statistics.

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

## Performance Comparison

Use Performance Comparison when reported returns changed and you need to know
why. PPAR compares two source-data snapshots, finds changed performance inputs,
and writes reviewer-friendly workbooks that separate explained differences from
items that need human review.

Typical questions:

- Did a revised holding market value change Modified Dietz return?
- Did a transaction correction change weighted external flows?
- Which rows explain the performance difference?
- Which differences are evidence-only or still unexplained?

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

The workbook output is designed for review, not just export. It includes
performance differences, explanation rows, residual review status, and raw audit
trail detail.

---

## Quick Setup

Install PPAR and create a local starter workspace:

```bash
pip install ppar
ppar setup ./my_ppar_data
```

Run Analytics:

```bash
ppar analytics ./my_ppar_data/analytics
```

Run Performance Comparison:

```bash
ppar performance_comparison ./my_ppar_data/performance_comparison
```

The setup folder is intentionally small:

```text
my_ppar_data/
  README.md
  analytics/
    ppar.yaml
    portperf.csv
    secperf.csv
  performance_comparison/
    ppar.yaml
    snapshot_a/
    snapshot_b/
```

Replace the starter CSV files with your own Axys/APX IMEX exports, then edit the
nearby `ppar.yaml`. The YAML files are heavily commented and are intended to be
the primary onboarding guide. The generated `README.md` in `my_ppar_data/`
keeps the day-to-day run commands in one place.

After installation, this command prints the setup guide:

```bash
ppar setup --guide
```

---

## Inputs

The Axys/APX starter path focuses on IMEX-style CSV exports:

- portfolio performance
- security performance
- holdings
- transactions

Analytics can run from portfolio/security performance exports. Performance
Comparison uses two source-data snapshots, usually an older/original snapshot
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

Performance Comparison writes review packages:

```text
performance_comparison/output/
  portfolio/report.xlsx
  portfolio/report.html
  security/report.xlsx
  security/report.html
```

Open `report.xlsx` when present; use `report.html` for browser review.
CSV artifacts support audit traceability in detailed report bundles.

---

## For Maintainers

Repository orientation:

- [Repository Guide](docs/repository_guide.md)
- [Roadmap](docs/roadmap.md)
- [Performance Comparison Design Notes](docs/performance_comparison_design.md)
- [Axys/APX Reference](docs/axys-apx-reference/README.md)

Useful source-checkout checks:

```bash
./.venv/bin/python scripts/check_project.py --quick
./.venv/bin/python scripts/check_performance_comparison_demo_health.py
./.venv/bin/python scripts/render_readme_images.py
```

To refresh only the Performance Comparison README screenshots:

```bash
./.venv/bin/python scripts/render_readme_images.py --only performance-comparison
```
