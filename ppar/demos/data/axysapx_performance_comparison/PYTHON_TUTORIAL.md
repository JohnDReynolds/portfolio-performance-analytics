# Run PPAR From Python

Most users should start with the `ppar` commands in `README.md`. Use this
document when you want to call PPAR from a Python script, scheduler, notebook,
or internal automation job.

The examples assume you already ran:

```bash
ppar setup ./my_ppar_data
```

## Analytics

This is the Python equivalent of:

```bash
ppar analytics ./my_ppar_data/analytics
```

Create a file such as `run_analytics.py`:

```python
from pathlib import Path

from ppar.analytics.cli import run_analytics


site_directory = Path("my_ppar_data") / "analytics"

run_analytics(site_directory)
```

Run it:

```bash
python run_analytics.py
```

The output files are written to `my_ppar_data/analytics/output` unless you
change `analytics.output_directory` in `my_ppar_data/analytics/ppar.yaml`.

Optional overrides:

```python
from pathlib import Path

from ppar.analytics.cli import run_analytics


site_directory = Path("my_ppar_data") / "analytics"

run_analytics(
    site_directory,
    portfolio_code="MEGA_ALPHA",
    benchmark_code="MEGA_BENCH",
    frequency_value="quarterly",
)
```

## Performance Comparison

This is the Python equivalent of:

```bash
ppar performance_comparison ./my_ppar_data/performance_comparison
```

Create a file such as `run_performance_comparison.py`:

```python
from pathlib import Path

from ppar.performance_comparison.cli.site_report import run_report


site_directory = Path("my_ppar_data") / "performance_comparison"

run_report(site_directory)
```

Run it:

```bash
python run_performance_comparison.py
```

The default writes both portfolio and security report workbooks when both are
available:

```text
my_ppar_data/performance_comparison/output/portfolio/report.xlsx
my_ppar_data/performance_comparison/output/security/report.xlsx
```

To write only one report family:

```python
from pathlib import Path

from ppar.performance_comparison.cli.site_report import run_report


site_directory = Path("my_ppar_data") / "performance_comparison"

run_report(site_directory, report="portfolio")
run_report(site_directory, report="security")
```

Valid `report` values are `portfolio`, `security`, and `both`.

## Customizing

Keep the Python scripts small. Put site-specific decisions in the nearby
`ppar.yaml` files:

- `my_ppar_data/analytics/ppar.yaml`
- `my_ppar_data/performance_comparison/ppar.yaml`

That keeps your scheduled jobs stable while your YAML documents the Axys/APX
extract columns, transaction treatment, and report assumptions.
