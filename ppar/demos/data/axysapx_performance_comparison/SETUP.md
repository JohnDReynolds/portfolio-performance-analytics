# PPAR Axys/APX Setup

Run setup once to create a local starter workspace:

```bash
ppar setup ./my_ppar_data
```

Setup copies starter files into:

```text
my_ppar_data/
  README.md
  analytics/
    run_analytics.py
  performance_comparison/
    run_portfolio_comparison.py
    run_security_comparison.py
```

The setup command prints the run commands for the copied folders:

```bash
ppar analytics ./my_ppar_data/analytics
ppar performance_comparison ./my_ppar_data/performance_comparison
```

`ppar perfcomp` is a shorter alias for `ppar performance_comparison`.

Open `my_ppar_data/README.md`, section `Customizing`, when you are ready to
replace the starter CSV files with your own Axys/APX IMEX or export data.
The copied Python scripts are optional examples for users who want to call PPAR
from Python instead of using the `ppar` command.

Existing files are kept unless you pass `--overwrite`.
