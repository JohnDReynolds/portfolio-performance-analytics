# PPAR Generic Analytics Workspace

This workspace is a vendor-neutral Python example for PPAR Analytics. It turns
portfolio and benchmark performance into attribution, contribution,
cumulative-return, and ex-post risk reports.

Unlike the Axys/APX Analytics workspace, this example constructs `Analytics`
directly in Python. There is no `ppar.yaml`: the visible paths and settings in
`run_generic_analytics.py` are the customization surface.

## First Run

Install PPAR with its optional Analytics chart dependencies:

```bash
pip install "ppar[analytics]"
```

Run the example:

```bash
python run_generic_analytics.py
```

The script writes HTML tables and PNG charts under `output/`, then lists the
files to open.

## Customizing With Your Own Data

1. Replace the portfolio and benchmark CSVs under `performance/`.
2. Replace the classification and mapping CSVs if your hierarchy differs.
3. Replace `holidays.csv` with the relevant reporting holidays, or remove the
   `holidays` argument from the script when weekends are the only nonbusiness
   days. The file is headerless, with one `YYYY-MM-DD` date per line.
4. Edit the clearly marked paths, classification name, and frequency in
   `run_generic_analytics.py`.
5. Run the script again.

The example keeps the complete Analytics workflow visible so it can serve as a
starting point for a client-owned runner, scheduled process, or application
integration.

## Workspace Layout

```text
my_ppar_generic_analytics/
  README.md
  run_generic_analytics.py
  holidays.csv
  performance/
    Mega-Cap Alpha Portfolio.csv
    Mega-Cap Benchmark.csv
  classifications/
    Security.csv
    Economic Sector.csv
  mappings/
    Security--to--Economic Sector.csv
```

## What the Example Produces

- security and sector attribution tables;
- overall contribution and attribution charts;
- sub-period and cumulative analysis; and
- ex-post portfolio and benchmark risk statistics.

For the full product demonstration and methodology overview, see the [PPAR
Analytics documentation][analytics-docs].

[analytics-docs]: https://github.com/JohnDReynolds/portfolio-performance-analytics/blob/main/docs/analytics/README.md
