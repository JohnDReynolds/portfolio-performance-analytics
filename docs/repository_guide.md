# Repository Guide

This guide is the orientation layer for the repository. Use it when the root
README, demo fixtures, commands, validators, and tests start to feel like too
many disconnected entry points.

## Start Here

| Need | Start With | Why |
| --- | --- | --- |
| Package overview | [`README.md`](../README.md) | Top-level project description, installation, public demo commands, and common smoke tests. |
| Analytics demo refresh | [`docs/analytics_demo_refresh.md`](analytics_demo_refresh.md) | How to regenerate Mega-Cap demo data, update the README story, and refresh README images. |
| Performance comparison roadmap | [`docs/performance_comparison_roadmap.md`](performance_comparison_roadmap.md) | Central forward-looking plan for return reconstruction, explanations, report evolution, and demo-data guardrails. |
| Performance comparison concepts | [`docs/performance_comparison_design.md`](performance_comparison_design.md) | Deep feature notes, YAML vocabulary, report bundle structure, and implementation status. |
| Axys export shape | [`docs/axys_common_core_export.md`](axys_common_core_export.md) | Starter Axys export template and field-reference notes. |
| Packaged Axys demos | [`ppar/demos/data/axys/README.md`](../ppar/demos/data/axys/README.md) | User-facing Axys analytics and performance comparison demo inputs. |
| Test Axys configs | [`tests/data/axys/README.md`](../tests/data/axys/README.md) | Synthetic Axys snapshots, test-focused comparison YAML files, and validation matrix fixtures. |
| Historical checkpoint notes | [`docs/performance_comparison_checkpoint_notes.md`](performance_comparison_checkpoint_notes.md) | Working notes from earlier implementation sessions; useful context, not durable product documentation. |

## Directory Map

| Path | Purpose |
| --- | --- |
| `ppar/` | Installable package code. |
| `ppar/analytics/` | Core analytics engine, analytics column schema, attribution, contribution, and risk calculations. |
| `ppar/axys/` | Axys-specific ingestion and normalization support. |
| `ppar/performance_comparison/` | Performance comparison model, loaders, comparison logic, explanation tables, report writers, and workbook export. |
| `ppar/demos/` | Installed demo entry points, including the performance comparison demo commands. |
| `ppar/demos/data/` | Packaged demo data shipped with source distributions and wheels. |
| `ppar/demos/data/axys/` | Packaged user-facing Axys demo snapshots and comparison YAML files. Use these for demos and review workflows. |
| `tests/` | Unit, integration, metadata, packaging, and report tests. |
| `tests/data/` | Test fixtures. These are allowed to be narrower and more surgical than packaged demos. |
| `scripts/` | Repository-maintenance helpers; Performance Comparison commands live under ppar.performance_comparison.cli. |
| `docs/` | Durable project documentation and design notes. |
| `docs/images/` | Durable documentation images. README-rendered analytics assets live under `docs/images/readme/`. |
| `_demo_output/` | Generated demo/report output. This is intentionally ignored by Git. |

## README Files

| File | Audience | Contents |
| --- | --- | --- |
| [`README.md`](../README.md) | New users and maintainers | Package overview, installation, bundled demo commands, performance-comparison smoke tests, and project checks. |
| [`docs/analytics_demo_refresh.md`](analytics_demo_refresh.md) | Maintainers | Analytics demo data-generation, story, and README image refresh workflow. |
| [`docs/performance_comparison_roadmap.md`](performance_comparison_roadmap.md) | Maintainers | Central future-work roadmap for performance comparison, return reconstruction, explanations, reports, and demo-data guardrails. |
| [`docs/performance_comparison_design.md`](performance_comparison_design.md) | Maintainers | Deep design reference, YAML vocabulary, implementation status, and open design issues. |
| [`ppar/demos/data/axys/README.md`](../ppar/demos/data/axys/README.md) | Demo reviewers | Packaged Axys demo commands, data/YAML descriptions, and expected workbook outputs. |
| [`tests/data/axys/README.md`](../tests/data/axys/README.md) | Test authors | Synthetic Axys snapshots, test-only comparison YAML files, and validation matrix fixtures. |
| Snapshot README files under `tests/data/axys/snapshots/*/README.md` | Test authors | Small notes about individual synthetic snapshot directories. |

## Scripts

The `scripts/` directory contains repository-maintenance helpers. Performance
Comparison command implementations live in `ppar.performance_comparison.cli` and
can be run from a source checkout with `./.venv/bin/python -m <module>`.

| Command Module Or Script | Purpose | Common Use |
| --- | --- | --- |
| `scripts/check_project.py` | Runs project checks. | `./.venv/bin/python scripts/check_project.py --quick` |
| `scripts/check_performance_comparison_demo_health.py` | Runs packaged performance-comparison demo guardrails. | `./.venv/bin/python scripts/check_performance_comparison_demo_health.py` |
| `scripts/render_readme_images.py` | Regenerates README images from packaged Mega-Cap analytics demo files. | Documentation image maintenance after analytics demo refresh. |
| `scripts/operational_demo_data/rebuild_performance_comparison_demo_data.py` | Rebuilds and audits packaged performance-comparison demo accounting. | Refresh scenario-derived `holdings.csv` plus derived `secperf.csv`/`portperf.csv` and verify fixture consistency after demo-data edits. |
| `scripts/audit_performance_comparison_demo_data.py` | Wrapper around the packaged demo-data audit. | Backward-compatible audit command. |
| `ppar.performance_comparison.cli.report_bundle` | Writes HTML, CSV, manifest, and optional XLSX workbook artifacts for a comparison YAML. | Generate review bundles and workbooks. |
| `ppar.performance_comparison.cli.validate_bundle` | Validates a generated report bundle. | Check that expected artifacts and manifest references exist. |
| `ppar.performance_comparison.cli.validate_demo_matrix` | Validates performance comparison scenario fixtures. | Prove validation fixtures still cover documented scenarios. |
| `ppar.performance_comparison.cli.validate_config` | Validates a comparison YAML file. | Catch YAML setup issues before generating reports. |

## Installed Commands

The installed command names are declared in `pyproject.toml`.

| Command | Purpose |
| --- | --- |
| `ppar-analytics-demo` | Runs the core analytics demo. |
| `ppar-axys-analytics-demo` | Runs the Axys analytics demo. |
| `ppar-performance-comparison-portfolio-demo` | Runs the portfolio-level performance comparison demo. |
| `ppar-performance-comparison-security-demo` | Runs the security-level performance comparison demo. |

## Demo Data

Packaged demos and test configuration fixtures intentionally live in different
places:

- `ppar/demos/data/axys/` is for packaged examples that users and reviewers can
  run. These fixtures should be understandable, documented, and stable enough
  to explain from workbook/report output.
- `tests/data/axys/` is for synthetic Axys snapshots, test-only comparison YAML
  files, and validation matrix fixtures.

The packaged Axys YAML files are limited to user-facing workflows. Broader
scenario coverage lives under `tests/data/axys`.

| Role | YAML | Snapshot A | Snapshot B | Use |
| --- | --- | --- | --- | --- |
| Workbook demos | `ppar_performance_comparison.yaml` | `axys_full_spec_a` | `axys_full_spec_b` | Shared portfolio/security demo spec. The demo command selects the primary review level. |

For the recommended XLSX workbook command and expected output, use
[`ppar/demos/data/axys/README.md`](../ppar/demos/data/axys/README.md).
For the validation fixture matrix, use
[`tests/data/axys/README.md`](../tests/data/axys/README.md).

## Common Workflows

### Run Demo Commands

The installed demo commands write durable artifacts under `_demo_output/`:

| Command | Output Directory | Notes |
| --- | --- | --- |
| `ppar.demos.analytics_demo` | `_demo_output/analytics` | Writes table and chart artifacts; default frequency is quarterly. |
| `ppar.demos.axys_analytics_demo` | `_demo_output/axys_analytics` | Axys-backed analytics demo. |
| `ppar.demos.performance_comparison_portfolio_demo` | `_demo_output/performance_comparison_portfolio` | Portfolio review bundle with `report.xlsx`, `report.html`, CSVs, and manifest. |
| `ppar.demos.performance_comparison_security_demo` | `_demo_output/performance_comparison_security` | Security review bundle with `report.xlsx`, `report.html`, CSVs, and manifest. |

All demo commands print the generated artifact paths and leave browser opening
to the reviewer.

The performance comparison demo can be smoke-tested noninteractively:

```bash
./.venv/bin/python -m ppar.demos.performance_comparison_portfolio_demo
./.venv/bin/python -m ppar.demos.performance_comparison_security_demo
./.venv/bin/python -m ppar.performance_comparison.cli.validate_bundle \
  _demo_output/performance_comparison_portfolio
./.venv/bin/python -m ppar.performance_comparison.cli.validate_bundle \
  _demo_output/performance_comparison_security
```

Open the generated `report.xlsx` first when it is present. Use `report.html`
for browser review, and keep the CSV artifacts for supplementary diagnostics
and audit traceability.

For the full packaged-demo guardrail pass, run:

```bash
./.venv/bin/python scripts/check_performance_comparison_demo_health.py
```

This consolidates the operational rebuild drift audit, extract-availability
appendix check, portfolio/security bundle generation, bundle validation, and
packaged scenario-matrix validation.

### Generate A Custom Review Bundle

```bash
./.venv/bin/python -m ppar.performance_comparison.cli.report_bundle \
  ppar/demos/data/axys/ppar_performance_comparison.yaml \
  _demo_output/custom_portfolio \
  --include-workbook \
  --require-causal-attribution
```

Use this lower-level command when you want to choose a YAML file or output
directory yourself. The packaged demo command above is the recommended
user-facing demo.

Code that needs to inspect the generated report-bundle handoff surface should
use `ppar.performance_comparison.report_bundle_contract()`. That helper returns
the required artifact keys, manifest keys, review entrypoints, review-summary
keys, Modified Dietz review basis, and review vocabulary keys.

### Validate The Packaged Demo Matrix

```bash
./.venv/bin/python -m ppar.performance_comparison.cli.validate_demo_matrix
```

This is a scenario-coverage check. It is not just a bundle validator; it proves
that packaged fixtures still demonstrate the documented review situations.

### Validate A Generated Bundle

```bash
./.venv/bin/python -m ppar.performance_comparison.cli.validate_bundle \
  _demo_output/performance_comparison_portfolio
```

Use this after generating report/workbook output.

### Validate A Comparison YAML

```bash
./.venv/bin/python -m ppar.performance_comparison.cli.validate_config \
  ppar/demos/data/axys/ppar_performance_comparison.yaml
```

Use this before report generation when you are editing YAML.

### Run Project Checks

```bash
./.venv/bin/python scripts/check_project.py --quick
```

Use the full check before a larger handoff:

```bash
./.venv/bin/python scripts/check_project.py
```

## Tests And Validators

Think of tests and validators as different tools:

- Unit and integration tests under `tests/` protect code behavior.
- Metadata tests protect packaging, public exports, and documentation promises.
- Bundle validation checks generated artifact structure.
- Demo matrix validation checks that test-only scenario fixtures still explain the
  intended review cases.
- YAML validation checks configuration completeness and vocabulary before a
  comparison is run.

When changing report/workbook behavior, the most relevant focused tests are:

```bash
./.venv/bin/python -m unittest \
  tests.test_package_metadata \
  tests.test_performance_comparison_report \
  tests.test_performance_comparison_cli \
  tests.test_performance_comparison_workbook_contract
```

When changing demo data or YAML, also run:

```bash
./.venv/bin/python scripts/audit_performance_comparison_demo_data.py
./.venv/bin/python scripts/operational_demo_data/rebuild_performance_comparison_demo_data.py
./.venv/bin/python -m ppar.performance_comparison.cli.validate_demo_matrix
```

## Generated Output

Generated output normally belongs under `_demo_output/`:

- `_demo_output/analytics`: Core analytics demo HTML/PNG artifacts.
- `_demo_output/axys_analytics`: Axys-backed analytics demo HTML artifacts.
- `_demo_output/performance_comparison_portfolio`: Portfolio comparison review bundle.
- `_demo_output/performance_comparison_security`: Security comparison review bundle.

Performance comparison report bundles include:

- `report.html`: browser-friendly review report.
- `report.xlsx`: primary Excel reviewer artifact when generated.
- `needs_review_summary.csv`: changed periods and suggested review notes.
- `findings.csv`: complete finding-level audit trail.
- `manifest.json`: machine-readable artifact inventory.
- `README.md`: generated bundle handoff notes.

The generated bundle README is not source documentation. It describes one
specific output directory after a bundle is written.

Both portfolio and security workbooks start with the `Performance Differences`
sheet. They then use the `Performance Difference Causes` sheet, `Other Data
Differences` sheet, and `Raw Audit Trail` sheet. Explained amounts appear on
`Performance Difference Causes` sheet rows when ppar has a defensible
input-level explanation. The action sheets use an `Explanation` column written
for reviewers; the raw audit sheet and bundle CSV artifacts keep lower-level
fields for troubleshooting.

## Suggested Consolidation Rules

Before moving files, prefer clarifying the map:

1. Keep the root README short enough to answer "what is this project and how do
   I run the main workflows?"
2. Keep detailed packaged demo instructions in `ppar/demos/data/axys/README.md`.
3. Keep future performance-comparison plans in
   `docs/performance_comparison_roadmap.md`.
4. Keep deep design rationale in `docs/performance_comparison_design.md`.
5. Keep checkpoint/session notes in
   `docs/performance_comparison_checkpoint_notes.md`, but treat them as
   temporary working context.
6. Add new scripts only when they serve automation or a distinct workflow.
   Otherwise, prefer improving this guide or the existing script help text.
