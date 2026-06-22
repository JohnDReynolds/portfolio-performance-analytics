# Repository Guide

This guide is the orientation layer for the repository. Use it when the root
README, demo fixtures, commands, validators, and tests start to feel like too
many disconnected entry points.

## Start Here

| Need | Start With | Why |
| --- | --- | --- |
| Package overview | [`README.md`](../README.md) | Top-level project description, installation, public demo commands, and common smoke tests. |
| Performance comparison concepts | [`docs/performance_comparison_design.md`](performance_comparison_design.md) | Deep feature notes, YAML vocabulary, report bundle structure, and implementation status. |
| Axys export shape | [`docs/axys_common_core_export.md`](axys_common_core_export.md) | Starter Axys export template and field-reference notes. |
| Packaged Axys demos | [`ppar/demos/data/axys/README.md`](../ppar/demos/data/axys/README.md) | Four XLSX workbook demos, four validation fixtures, exact commands, and expected outputs. |
| Test Axys fixtures | [`tests/data/axys/README.md`](../tests/data/axys/README.md) | Synthetic test fixture layout and test-focused comparison YAML files. |
| Session continuity | [`docs/work_session_notes.md`](work_session_notes.md) | Working notes from recent implementation sessions; useful context, not durable product documentation. |

## Directory Map

| Path | Purpose |
| --- | --- |
| `ppar/` | Installable package code. |
| `ppar/analytics/` | Core analytics engine, analytics column schema, attribution, contribution, and risk calculations. |
| `ppar/axys/` | Axys-specific ingestion and normalization support. |
| `ppar/performance_comparison/` | Performance comparison model, loaders, comparison logic, explanation tables, report writers, and workbook export. |
| `ppar/demos/` | Installed demo entry points, including `ppar-performance-comparison-demo`. |
| `ppar/demos/data/` | Packaged demo data shipped with source distributions and wheels. |
| `ppar/demos/data/axys/` | Packaged Axys demo snapshots and comparison YAML files. Use these for demos and review workflows. |
| `tests/` | Unit, integration, metadata, packaging, and report tests. |
| `tests/data/` | Test fixtures. These are allowed to be narrower and more surgical than packaged demos. |
| `scripts/` | Repository-maintenance helpers; Performance Comparison commands live under ppar.performance_comparison.cli. |
| `docs/` | Durable project documentation and design notes. |
| `_demo_output/` | Generated demo/report output. This is intentionally ignored by Git. |

## README Files

| File | Audience | Contents |
| --- | --- | --- |
| [`README.md`](../README.md) | New users and maintainers | Package overview, installation, bundled demo commands, performance-comparison smoke tests, and project checks. |
| [`ppar/demos/data/axys/README.md`](../ppar/demos/data/axys/README.md) | Demo reviewers | Packaged Axys scenario matrix, XLSX workbook commands, data/YAML descriptions, and expected workbook outputs. |
| [`tests/data/axys/README.md`](../tests/data/axys/README.md) | Test authors | Test fixture layout, expected test workflows, and fixture-specific notes. |
| Snapshot README files under `ppar/demos/data/axys/*/README.md` | Fixture readers | Small notes about individual packaged snapshot directories. |
| Snapshot README files under `tests/data/axys/*/README.md` | Fixture readers | Small notes about individual test snapshot directories. |

## Scripts

The `scripts/` directory contains repository-maintenance helpers. Performance
Comparison command implementations live in `ppar.performance_comparison.cli` and
can be run from a source checkout with `./.venv/bin/python -m <module>`.

| Command Module Or Script | Purpose | Common Use |
| --- | --- | --- |
| `scripts/check_project.py` | Runs project checks. | `./.venv/bin/python scripts/check_project.py --quick` |
| `ppar.performance_comparison.cli.report_bundle` | Writes HTML, CSV, manifest, and optional XLSX workbook artifacts for a comparison YAML. | Generate review bundles and workbooks. |
| `ppar.performance_comparison.cli.validate_bundle` | Validates a generated report bundle. | Check that expected artifacts and manifest references exist. |
| `ppar.performance_comparison.cli.validate_demo_matrix` | Validates packaged Axys demo scenarios. | Prove demo fixtures still cover documented scenarios. |
| `ppar.performance_comparison.cli.validate_config` | Validates a comparison YAML file. | Catch YAML setup issues before generating reports. |
| `ppar.performance_comparison.cli.html_report` | Writes a standalone HTML report. | Lower-level report smoke testing. |
| `ppar.performance_comparison.cli.report` | Writes a standalone Markdown report. | Lower-level report smoke testing. |
| `scripts/render_readme_images.py` | Regenerates README image artifacts. | Documentation image maintenance. |

## Installed Commands

The installed command names are declared in `pyproject.toml`.

| Command | Purpose |
| --- | --- |
| `ppar-analytics-demo` | Runs the core analytics demo. |
| `ppar-axys-analytics-demo` | Runs the Axys analytics demo. |
| `ppar-performance-comparison-demo` | Runs the packaged performance comparison demo and writes a bundle under `_demo_output/`. |
| `ppar-performance-comparison-html-report` | Writes a standalone HTML performance comparison report. |
| `ppar-performance-comparison-report` | Writes a standalone Markdown performance comparison report. |
| `ppar-performance-comparison-report-bundle` | Writes a portable performance comparison review bundle. |
| `ppar-performance-comparison-validate-bundle` | Validates a generated performance comparison review bundle. |
| `ppar-performance-comparison-validate-config` | Validates a performance comparison YAML file. |
| `ppar-performance-comparison-validate-demo-matrix` | Validates packaged Axys demo scenario coverage. |

## Demo Data

Packaged demos and test fixtures intentionally live in different places:

- `ppar/demos/data/axys/` is for packaged examples that users and reviewers can
  run. These fixtures should be understandable, documented, and stable enough
  to explain from workbook/report output.
- `tests/data/axys/` is for tests. These fixtures can be smaller, more
  specialized, and shaped around edge cases.

The packaged Axys YAML files are split by role. The workbook demos are intended
for user-facing review; the validation fixtures are scenario coverage inputs
for tests and validators.

| Role | YAML | Snapshot A | Snapshot B | Use |
| --- | --- | --- | --- | --- |
| Workbook demo | `ppar_performance_comparison.yaml` | `axys_a` | `axys_b` | Clean baseline with no expected findings. |
| Workbook demo | `ppar_performance_comparison_restatement.yaml` | `axys_a` | `axys_b_restatement` | Controlled single restatement. |
| Workbook demo | `ppar_performance_comparison_restatement_transaction_rules.yaml` | `axys_a` | `axys_b_restatement` | Same data with explicit transaction rules. |
| Workbook demo | `ppar_performance_comparison_full_spec.yaml` | `axys_full_spec_a` | `axys_full_spec_b` | Compact strict-attribution demo with all supported causal policies. |
| Validation fixture | `ppar_performance_comparison_multi_restatement.yaml` | `axys_a` | `axys_b_multi_restatement` | Multiple portfolios/periods, context rows, and residual coverage. |
| Validation fixture | `ppar_performance_comparison_modified_dietz.yaml` | `axys_modified_dietz_a` | `axys_modified_dietz_b` | External-flow Modified Dietz cross-check diagnostics. |
| Validation fixture | `ppar_performance_comparison_policy_gap_demo.yaml` | `axys_a` | `axys_b_multi_restatement` | Missing-YAML-specification coverage. |
| Validation fixture | `ppar_performance_comparison_suppressed.yaml` | `axys_a` | `axys_b_restatement` | Suppressed-finding coverage. |

For exact XLSX workbook commands and expected outputs, use
[`ppar/demos/data/axys/README.md`](../ppar/demos/data/axys/README.md).

## Common Workflows

### Run The Main Performance Comparison Demo

```bash
./.venv/bin/python -m ppar.demos.performance_comparison_demo
./.venv/bin/python -m ppar.performance_comparison.cli.validate_bundle \
  _demo_output/performance_comparison_bundle
```

Open `_demo_output/performance_comparison_bundle/report.html` first for this
non-workbook smoke test.

### Generate An XLSX Workbook Bundle

```bash
./.venv/bin/python -m ppar.performance_comparison.cli.report_bundle \
  ppar/demos/data/axys/ppar_performance_comparison_multi_restatement.yaml \
  _demo_output/workbooks/multi_restatement \
  --include-workbook
```

Use the full Axys workbook command list in
[`ppar/demos/data/axys/README.md`](../ppar/demos/data/axys/README.md) when you
want to compare the packaged workbook scenarios. When `--include-workbook` is
used, start review in `report.xlsx`; use `report.html` when you want the same
review model in a browser.

### Validate The Packaged Demo Matrix

```bash
./.venv/bin/python -m ppar.performance_comparison.cli.validate_demo_matrix
```

This is a scenario-coverage check. It is not just a bundle validator; it proves
that packaged fixtures still demonstrate the documented review situations.

### Validate A Generated Bundle

```bash
./.venv/bin/python -m ppar.performance_comparison.cli.validate_bundle \
  _demo_output/workbooks/multi_restatement
```

Use this after generating report/workbook output.

### Validate A Comparison YAML

```bash
./.venv/bin/python -m ppar.performance_comparison.cli.validate_config \
  ppar/demos/data/axys/ppar_performance_comparison_multi_restatement.yaml
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
- Demo matrix validation checks that packaged demo scenarios still explain the
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
./.venv/bin/python -m ppar.performance_comparison.cli.validate_demo_matrix
```

## Generated Output

Generated output normally belongs under `_demo_output/`. Report bundles include:

- `report.html`: browser-friendly review report.
- `report.xlsx`: primary Excel reviewer artifact when `--include-workbook` is
  used.
- `needs_review_summary.csv`: changed periods and next actions.
- `findings.csv`: complete finding-level audit trail.
- `manifest.json`: machine-readable artifact inventory.
- `README.md`: generated bundle handoff notes.

The generated bundle README is not source documentation. It describes one
specific output directory after a bundle is written.

The workbook separates review concerns into the `Portfolio Differences` sheet,
`Security Differences` sheet, `Underlying Causes` sheet, `Reported Performance
Checks` sheet, `Context` sheet, and `Raw Audit Trail` sheet. Explained amounts
appear on `Underlying Causes` sheet rows when ppar has a defensible input-level
explanation. The `Required YAML Setup` column is `None` for rows that are
already explainable and otherwise names the YAML fields or unsupported impact
method blocking attribution.

## Suggested Consolidation Rules

Before moving files, prefer clarifying the map:

1. Keep the root README short enough to answer "what is this project and how do
   I run the main workflows?"
2. Keep detailed packaged demo instructions in `ppar/demos/data/axys/README.md`.
3. Keep deep design rationale in `docs/performance_comparison_design.md`.
4. Keep session notes in `docs/work_session_notes.md`, but treat them as
   temporary working context.
5. Add new scripts only when they serve automation or a distinct workflow.
   Otherwise, prefer improving this guide or the existing script help text.
