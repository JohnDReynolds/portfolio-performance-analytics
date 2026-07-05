# PPAR Axys/APX Setup

Run setup once to create a local starter workspace:

```bash
ppar setup ./my_ppar_data
```

After installation, this guide is also available in the terminal:

```bash
ppar setup --guide
```

Setup copies starter files into:

```text
my_ppar_data/
  README.md
  PYTHON_TUTORIAL.md
  analytics/
  performance_comparison/
```

The setup command prints the run commands for the copied folders:

```bash
ppar analytics ./my_ppar_data/analytics
ppar performance_comparison ./my_ppar_data/performance_comparison
```

`ppar perfcomp` is a shorter alias for `ppar performance_comparison`.

Open `my_ppar_data/README.md`, section `Customizing`, when you are ready to
replace the starter CSV files with your own Axys/APX IMEX or export data.
Open `my_ppar_data/PYTHON_TUTORIAL.md` when you want to run the same workflows
from Python instead of the `ppar` command.

Existing files are kept unless you pass `--overwrite`.
