# Axys/APX Performance Comparison Quick Start

Use this when the first goal is simple: get an Axys/APX-style Modified Dietz
comparison running, then change only what the local site proves is different.

## Starter Kit

Copy the starter files as a group. Do not copy only the YAML unless you also
edit its relative paths.

```text
axys_site_start/
  axys_site_start.yaml
  axys_column_mappings.yaml
  demo_extract_availability.yaml
  axys_full_spec_a/
  axys_full_spec_b/
  _site_output/
```

## 1. Copy The Starter Kit

```bash
mkdir -p axys_site_start
cp ppar/demos/data/axys/axys_performance_comparison.yaml \
  axys_site_start/axys_site_start.yaml
cp ppar/demos/data/axys/axys_column_mappings.yaml \
  axys_site_start/axys_column_mappings.yaml
cp ppar/demos/data/axys/demo_extract_availability.yaml \
  axys_site_start/demo_extract_availability.yaml
cp -R ppar/demos/data/axys/axys_full_spec_a axys_site_start/axys_full_spec_a
cp -R ppar/demos/data/axys/axys_full_spec_b axys_site_start/axys_full_spec_b
```

Edit the copied files in `axys_site_start/`. Leave the packaged files unchanged.

## 2. Replace The Demo Extracts

For a real site, replace `axys_full_spec_a/` and `axys_full_spec_b/` with the
site's Snapshot A/B extract folders, or edit `snapshots.a.path` and
`snapshots.b.path` in `axys_site_start.yaml`.

Start with these edits only:

| File | First edits |
| --- | --- |
| `axys_site_start.yaml` | Snapshot paths, `comparison.level`, and filenames under `files:`. |
| `axys_column_mappings.yaml` | Local column headers. |
| Snapshot folders | Local portfolio performance, security performance, holdings, and transactions CSVs. |

Keep native transaction codes and security identifiers case-sensitive.

## 3. Validate

```bash
./.venv/bin/python -m ppar.performance_comparison.cli.validate_config \
  ./axys_site_start/axys_site_start.yaml
```

Validation should pass before you generate a report. Fix missing files, missing
columns, and unmapped transaction codes here.

## 4. Generate The Portfolio Report

```bash
./.venv/bin/python -m ppar.performance_comparison.cli.report_bundle \
  ./axys_site_start/axys_site_start.yaml \
  ./axys_site_start/_site_output/performance_comparison_portfolio \
  --include-workbook
```

Open:

- `./axys_site_start/_site_output/performance_comparison_portfolio/report.xlsx`
- `./axys_site_start/_site_output/performance_comparison_portfolio/report.html`

## 5. Generate The Security Report

Copy the YAML and change only `comparison.level` to `security`:

```bash
cp ./axys_site_start/axys_site_start.yaml ./axys_site_start/axys_site_security.yaml
```

```yaml
comparison:
  name: Axys performance comparison demo
  level: security
```

Then run:

```bash
./.venv/bin/python -m ppar.performance_comparison.cli.report_bundle \
  ./axys_site_start/axys_site_security.yaml \
  ./axys_site_start/_site_output/performance_comparison_security \
  --include-workbook
```

## 6. Review In This Order

1. `Performance Differences`
2. `Performance Difference Causes`
3. `Raw Audit Trail`
4. `manifest.json`

Stay focused on Modified Dietz inputs: beginning value, ending value, and dated
flows.

## 7. Iterate Carefully

Keep the first run vanilla. Start from the packaged rules for `by`, `sl`, `dv`,
`in`, `dp`, `li`, `lo`, `wd`, and fixed-income `pa`/`sa`.

Only add local overrides when the site extract proves the treatment through
source/destination, special-security, REP, or reviewed local evidence. Re-run
`validate_config` after every YAML change.

Cost-basis handling is best-efforts for demo construction; Modified Dietz
explanations do not depend on cost.

## Packaged Demo Commands

Use these when you want the shipped marketing/onboarding sample exactly as-is:

```bash
./.venv/bin/python -m ppar.demos.performance_comparison_portfolio_demo
./.venv/bin/python -m ppar.demos.performance_comparison_security_demo
```
