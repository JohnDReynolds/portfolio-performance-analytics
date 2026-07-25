# PPAR Audit Workspace

This self-contained workspace was created with `ppar setup`. It includes
PPAR-normalized demonstration data modeled on Axys/APX source and report data,
plus the documented Audit configuration in `ppar.yaml`.

Run the demonstration first, then replace the CSV files with reviewed exports
from your own environment.

## What This Folder Is For

PPAR Audit answers the question: "Why did my reported performance change?"

- **Performance Comparison:** identifies changed portfolio and security
  performance for each time period, quantitatively attributes the differences
  to supported source-data changes, and highlights anything that still needs
  human review.

- **Data Issues:** flags suspicious source-data relationships—including price
  ranges, dividend rates, accrued-interest rates, and missing dividends—that
  may indicate data-quality issues.

## First Run

Run Audit from this directory:

```bash
ppar audit .
```

Open the output files printed by the command. Normal output is written under
`output/portfolio` and, when security-performance files are available,
`output/security`.

## Customizing With Your Own Data

Audit compares two snapshots:

- `snapshot_a`: the original or older source-data snapshot.
- `snapshot_b`: the newer, corrected, or restated source-data snapshot.

Steps:

1. Replace the CSV data in `snapshot_a` with reviewed exports from your own
   environment.
2. Replace the CSV data in `snapshot_b` with reviewed exports from your own
   environment.
3. Edit `ppar.yaml`.
4. Run `ppar audit .`.

### Getting Data from Axys/APX

Start by reviewing the comments under `files:` in `ppar.yaml`. They classify
every workspace field as **Required**, **Required only when applicable**, or
**Optional**.

Required data is intentionally narrow: it includes only what PPAR needs to
account for a reported return change with supported evidence, within the
configured tolerance. The report labels that outcome as **Fully Explained**.

The most defensible source plan from the currently available Axys/APX evidence
is:

- Portfolio and security reported returns: use a REP performance or attribution
  report. PPAR does not assume that a native performance IMEX object exists.
- Holdings: use an IMEX positions/holdings export or a REP appraisal report.
- Transactions: try IMEX first. If `dp`, `li`, `lo`, or `wd` rows can occur, the
  extract must include the source/destination and special-security context named
  in `ppar.yaml`; otherwise use REP, a custom report, or another reviewed
  source.
- Security master: needed only when Data Issues filters use
  `security_master.*` qualifiers. Use a reviewed security-information IMEX
  export, security-master report, or equivalent extract and preserve exact case.
- FX rates: needed only when a changed FX rate itself must be explained. Use a
  locally validated REP, FX/price, or other controlled rate source.
- Split factors: optional review information, usually from `split.inf` or an
  equivalent local export.

The demonstration CSV names and headers are PPAR-normalized examples, not
guaranteed native Axys/APX schemas. Confirm the exact local object, report,
field names, date basis, currency basis, and return basis before relying on an
extract.

## Optional Python Script

`run_audit.py` shows the standard Python workflow and is the starting point for
local customization.

View the available command-line options:

```bash
ppar audit -h
python run_audit.py -h
```

## Folder Map

```text
./
  README.md
  ppar.yaml
  run_audit.py
  snapshot_a/
    portperf.csv
    holdings.csv
    transactions.csv
    secmast.csv
    secperf.csv
    fx_rates.csv
    splits.csv
  snapshot_b/
    portperf.csv
    holdings.csv
    transactions.csv
    secmast.csv
    secperf.csv
    fx_rates.csv
    splits.csv
```
