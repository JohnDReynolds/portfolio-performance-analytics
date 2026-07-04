# PPAR Performance Comparison Setup

Use setup once to create the local site folder and starter `ppar.yaml`.
After setup is working, use `ppar report` whenever you need a report package.

## 1. Run Setup

From a terminal:

```bash
ppar setup ./my_site_extracts
```

If `my_site_extracts`, `snapshot_a`, or `snapshot_b` do not exist yet, they are
created for you.

## 2. Add Portfolio Source Files

Put the portfolio source files in `snapshot_a` and `snapshot_b`:

```text
my_site_extracts/
  ppar.yaml
  snapshot_a/
    portperf.csv
    holdings.csv
    transactions.csv
  snapshot_b/
    portperf.csv
    holdings.csv
    transactions.csv
```

Keep native transaction codes and security identifiers case-sensitive.

## 3. Validate Setup

Run setup again:

```bash
ppar setup ./my_site_extracts
```

When setup says `Portfolio setup validated`, the site is ready for portfolio
reporting.

## 4. Run Reports

Create the standard portfolio report package:

```bash
ppar report ./my_site_extracts
```

Output:

```text
my_site_extracts/
  output/
    portfolio/report.xlsx
    portfolio/report.html
```

Review In This Order:

1. `Performance Differences`
2. `Performance Difference Causes`
3. `Raw Audit Trail`
4. `manifest.json`

Stay focused on Modified Dietz inputs: beginning value, ending value, and dated
flows.

## 5. Iterate Carefully

Edit `ppar.yaml` only after the first portfolio report runs.

Start with the packaged rules for `by`, `sl`, `dv`, `in`, `dp`, `li`, `lo`,
`wd`, and fixed-income `pa`/`sa`. Add local overrides only when the site extract
proves the treatment through source/destination, special-security, REP, or
reviewed local evidence.

Run `ppar report ./my_site_extracts` after each change. Existing `ppar.yaml` is
reused unless you pass `--overwrite` to `ppar setup`.

Cost-basis handling is best-efforts for demo construction; Modified Dietz
explanations do not depend on cost.

## Optional: Security-Level Review

After portfolio reporting is working, add `secperf.csv` to both snapshots:

```text
my_site_extracts/
  snapshot_a/
    secperf.csv
  snapshot_b/
    secperf.csv
```

Then run:

```bash
ppar report ./my_site_extracts --report security
```

Or regenerate both report packages:

```bash
ppar report ./my_site_extracts --report both
```

## Packaged Demo Commands

Use these when you want the shipped sample exactly as-is:

```bash
ppar-performance-comparison-portfolio-demo
ppar-performance-comparison-security-demo
```
