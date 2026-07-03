# PPAR Performance Comparison Quick Start

Use this when the first goal is simple: compare two source-data snapshots and
open the first Modified Dietz review reports.

## 1. Create One Site Folder

Put the two extracts in `snapshot_a` and `snapshot_b`:

```text
my_site_extracts/
  snapshot_a/
    portperf.csv
    secperf.csv
    holdings.csv
    transactions.csv
  snapshot_b/
    portperf.csv
    secperf.csv
    holdings.csv
    transactions.csv
```

Keep native transaction codes and security identifiers case-sensitive.

## 2. Run Quickstart

From a terminal:

```bash
ppar quickstart ./my_site_extracts
```

If `my_site_extracts`, `snapshot_a`, or `snapshot_b` do not exist yet, they are
created for you. Add the CSV files. Run the same command again.

The direct command is also available:

```bash
ppar-performance-comparison-quickstart ./my_site_extracts
```

The command creates `ppar.yaml`, validates the source files, and writes both
portfolio and security reports.

## 3. Open The Reports

```text
my_site_extracts/
  ppar.yaml
  output/
    portfolio/report.xlsx
    portfolio/report.html
    security/report.xlsx
    security/report.html
```

Review In This Order:

1. `Performance Differences`
2. `Performance Difference Causes`
3. `Raw Audit Trail`
4. `manifest.json`

Stay focused on Modified Dietz inputs: beginning value, ending value, and dated
flows.

## 4. Iterate Carefully

Edit `ppar.yaml` only after the first report runs.

Start with the packaged rules for `by`, `sl`, `dv`, `in`, `dp`, `li`, `lo`,
`wd`, and fixed-income `pa`/`sa`. Add local overrides only when the site extract
proves the treatment through source/destination, special-security, REP, or
reviewed local evidence.

Run quickstart again after each change. Existing `ppar.yaml` is reused unless
you pass `--overwrite`.

Cost-basis handling is best-efforts for demo construction; Modified Dietz
explanations do not depend on cost.

## Packaged Demo Commands

Use these when you want the shipped sample exactly as-is:

```bash
ppar-performance-comparison-portfolio-demo
ppar-performance-comparison-security-demo
```
