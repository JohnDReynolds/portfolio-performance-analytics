# Repository Guide

This guide is the orientation layer for the repository. Use it when the root
README, demo fixtures, commands, validators, and tests start to feel like too
many disconnected entry points.

## Start Here

| Need | Start With | Why |
| --- | --- | --- |
| Package overview | [`README.md`](../README.md) | Top-level project description, installation, public commands, and user-facing outputs. |
| Architecture map | [`docs/architecture.md`](architecture.md) | Compact map of the installed command surface, package boundaries, data flow, setup data, and report boundary. |
| Performance Analytics demo refresh | [`docs/analytics_demo_refresh.md`](analytics_demo_refresh.md) | How to regenerate Mega-Cap demo data, update the README story, and refresh README images. |
| PPAR roadmap | [`docs/roadmap.md`](roadmap.md) | Central forward-looking plan for Performance Auditing, Performance Analytics, onboarding, reports, and demo-data guardrails. |
| Performance Auditing concepts | [`docs/performance_comparison_design.md`](performance_comparison_design.md) | Deep notes for the Performance Comparison engine inside Performance Auditing, including YAML vocabulary, report bundle structure, and implementation status. |
| Axys/APX export shape | [`docs/axysapx_common_core_export.md`](axysapx_common_core_export.md) | Starter Axys/APX export template and field-reference notes. |
| Packaged Axys/APX demos | [`ppar/setup_templates/axysapx_performance_comparison/README.md`](../ppar/setup_templates/axysapx_performance_comparison/README.md) | User-facing Axys/APX Performance Auditing and Performance Analytics demo inputs. |
| Test Axys/APX configs | [`tests/data/axys/README.md`](../tests/data/axys/README.md) | Synthetic Axys/APX snapshots, test-focused comparison YAML files, and validation matrix fixtures. |
| Historical checkpoint notes | [`docs/performance_comparison_checkpoint_notes.md`](performance_comparison_checkpoint_notes.md) | Working notes from earlier implementation sessions; useful context, not durable product documentation. |

## Directory Map

| Path | Purpose |
| --- | --- |
| `ppar/` | Installable package code. |
| `ppar/analytics/` | Core analytics engine, analytics column schema, attribution, contribution, and risk calculations. |
| `ppar/axys/` | Axys/APX-specific ingestion and normalization support. |
| `ppar/performance_comparison/` | Internal Performance Comparison engine used by Performance Auditing: loaders, comparison logic, explanation tables, report writers, and workbook export. |
| `ppar/setup_templates/` | Packaged setup inputs and tutorial scripts shipped with source distributions and wheels. |
| `ppar/setup_templates/axysapx_performance_comparison/` | Packaged user-facing Axys/APX Performance Auditing snapshots and YAML files. Use these for demos and review workflows. |
| `tests/` | Unit, integration, metadata, packaging, and report tests. |
| `tests/data/` | Test fixtures. These are allowed to be narrower and more surgical than packaged demos. |
| `scripts/` | Repository-maintenance helpers; internal Performance Auditing commands live under `ppar.performance_comparison.cli`. |
| `docs/` | Durable project documentation and design notes. |
| `docs/images/` | Durable documentation images. README-rendered analytics assets live under `docs/images/readme/`. |
| `_demo_output/` | Generated demo/report output. This is intentionally ignored by Git. |

## README Files

| File | Audience | Contents |
| --- | --- | --- |
| [`README.md`](../README.md) | New users | Package overview, installation, public setup commands, and user-facing outputs. |
| [`docs/architecture.md`](architecture.md) | Maintainers | Compact architecture map for commands, package boundaries, data flow, setup data, and report boundaries. |
| [`docs/analytics_demo_refresh.md`](analytics_demo_refresh.md) | Maintainers | Performance Analytics demo data-generation, story, and README image refresh workflow. |
| [`docs/roadmap.md`](roadmap.md) | Maintainers | Central future-work roadmap for Performance Auditing, Performance Analytics, onboarding, reports, and demo-data guardrails. |
| [`docs/performance_comparison_design.md`](performance_comparison_design.md) | Maintainers | Deep design reference, YAML vocabulary, implementation status, and open design issues. |
| [`ppar/setup_templates/axysapx_performance_comparison/README.md`](../ppar/setup_templates/axysapx_performance_comparison/README.md) | Demo reviewers | Packaged Axys/APX data/YAML descriptions, setup-installed Python runners, and expected workbook outputs. |
| [`tests/data/axys/README.md`](../tests/data/axys/README.md) | Test authors | Synthetic Axys/APX snapshots, test-only comparison YAML files, and validation matrix fixtures. |
| Snapshot README files under `tests/data/axys/snapshots/*/README.md` | Test authors | Small notes about individual synthetic snapshot directories. |

## Scripts

The `scripts/` directory contains source-checkout maintenance helpers. Those
helpers may be included in source distributions for maintainers, but they are
not installed-package demo workflows. Performance Auditing command
implementations live in `ppar.performance_comparison.cli` and can be run from a
source checkout with `./.venv/bin/python -m <module>`. The package name still
uses `performance_comparison` because that is the internal engine for the
Performance Comparison sub-feature.

| Command Module Or Script | Purpose | Common Use |
| --- | --- | --- |
| `scripts/check_release_candidate.py` | Runs the maintained release-candidate demo, setup, and health-check sequence. | `./.venv/bin/python scripts/check_release_candidate.py` |
| `scripts/check_project.py` | Runs project checks. | `./.venv/bin/python scripts/check_project.py --quick` |
| `scripts/check_performance_comparison_demo_health.py` | Runs packaged Performance Auditing demo guardrails. | `./.venv/bin/python scripts/check_performance_comparison_demo_health.py` |
| `scripts/render_readme_images.py` | Regenerates README images from packaged Mega-Cap analytics demo files. | Documentation image maintenance after analytics demo refresh. |
| `scripts/operational_demo_data/rebuild_performance_comparison_demo_data.py` | Rebuilds and audits packaged performance-comparison demo accounting. | Refresh scenario-derived `holdings.csv` plus derived `secperf.csv`/`portperf.csv` and verify fixture consistency after demo-data edits. |
| `scripts/audit_performance_comparison_demo_data.py` | Wrapper around the packaged demo-data audit. | Backward-compatible audit command. |
| `ppar.performance_comparison.cli.report_bundle` | Writes HTML, CSV, manifest, and optional XLSX workbook artifacts for a comparison YAML. | Generate review bundles and workbooks. |
| `ppar.performance_comparison.cli.validate_bundle` | Validates a generated report bundle. | Check artifacts plus CSV/HTML/XLSX semantic parity and normalized integrity metadata. |
| `ppar.performance_comparison.cli.validate_demo_matrix` | Validates Performance Comparison scenario fixtures. | Prove validation fixtures still cover documented scenarios. |
| `ppar.performance_comparison.cli.validate_config` | Validates a comparison YAML file. | Catch source-data and YAML setup issues before generating reports. |

## Maintainer Checks

Useful source-checkout checks:

```bash
./.venv/bin/python scripts/check_project.py --quick
./.venv/bin/python scripts/check_performance_comparison_demo_health.py
./.venv/bin/python scripts/render_readme_images.py
```

After major Audit or performance-sensitive changes, also run the maintained 500x
regression gate:

```bash
./.venv/bin/python scripts/check_scale.py --scale 500
```

Do not recalibrate its warning or failure limits merely because a run fails. Treat a
failure as a regression investigation unless a deliberate benchmark-policy change is
reviewed separately.

To refresh only the Performance Auditing README screenshots:

```bash
./.venv/bin/python scripts/render_readme_images.py --only performance-comparison
```

## Installed Commands

The installed command names are declared in `pyproject.toml`.

| Command | Purpose |
| --- | --- |
| `ppar` | Top-level user command for setup, Performance Auditing, and Performance Analytics. |

The source-checkout smoke path uses setup-generated scripts:

```bash
./.venv/bin/python scripts/check_performance_comparison_demo_health.py --skip-rebuild-audit
```

## Demo Data

Packaged demos and test configuration fixtures intentionally live in different
places:

- `ppar/setup_templates/axysapx_analytics/` is the public analytics setup seed
  copied by `ppar setup`.
- `ppar/setup_templates/axysapx_performance_comparison/` is the public
  performance-comparison setup seed copied by `ppar setup` and is also used
  for packaged examples that users and reviewers can run.
- `ppar/setup_templates/generic_analytics/` is maintainer/demo infrastructure. It
  feeds README marketing images, analytics regression tests, and operational
  demo-data derivation; it is not advertised as the primary onboarding path.
- `tests/data/axys/` is for synthetic Axys/APX snapshots, test-only comparison YAML
  files, and validation matrix fixtures.

The packaged Axys/APX YAML file is limited to user-facing marketing and onboarding
workflows. Broader scenario coverage lives under `tests/data/axys`.

| Role | YAML | Snapshot A | Snapshot B | Use |
| --- | --- | --- | --- | --- |
| Workbook demos | `axysapx_performance_comparison.yaml` | `snapshot_a` | `snapshot_b` | Shared portfolio/security demo spec. Setup-generated scripts select the primary review level. |

For the maintained XLSX workbook smoke paths and expected output, use
[`ppar/setup_templates/axysapx_performance_comparison/README.md`](../ppar/setup_templates/axysapx_performance_comparison/README.md).
For the validation fixture matrix, use
[`tests/data/axys/README.md`](../tests/data/axys/README.md).

### Packaged Axys/APX Performance Comparison Maintenance

Treat the packaged Axys/APX performance-comparison demo as a small accounting
lab, not as hand-edited CSV examples. The durable source-of-truth files are:

| File | Owns |
| --- | --- |
| `scripts/operational_demo_data/derive_operational_demo_data.py` | Base portfolios, selected securities, baseline holdings, baseline transactions, and synthetic Axys/APX-style performance rows. |
| `scripts/operational_demo_data/performance_comparison_transaction_scenarios.csv` | Snapshot B transaction adjustments and inserted transaction rows. |
| `scripts/operational_demo_data/performance_comparison_holding_scenarios.csv` | Snapshot B explicit holding restatements, including split-processing, accrual, valuation, and maintainer-only cost context. |
| `scripts/operational_demo_data/performance_comparison_scenario_calendar.csv` | Operational map from physical scenario rows to reviewer story periods. |
| `scripts/operational_demo_data/performance_comparison_scenario_inventory.csv` | Validated semantic contract that keeps each source period within the two-independent-change review target and protects report outcomes plus carry-forward behavior. |
| `scripts/operational_demo_data/performance_comparison_period_split_plan.csv` | Empty split backlog. Add rows only when a future scenario makes a period too crowded again. |
| `scripts/operational_demo_data/rebuild_performance_comparison_demo_data.py` | Applies transaction scenarios, derives transaction-driven holdings, rebuilds `secperf.csv`/`portperf.csv`, strips internal fields from packaged CSVs, and audits drift. |
| `ppar/setup_templates/axysapx_performance_comparison/axysapx_performance_comparison.yaml` | User-facing interpretation contract: file names, column mappings, transaction rules, field roles, reconstruction settings, and report level. |
| `ppar/setup_templates/axysapx_performance_comparison/README.md` | User-facing demo story and setup guidance. |
| `ppar.performance_comparison.cli.validate_demo_matrix` | Test-only scenario coverage guardrail. It does not define the packaged demo story. |

When changing packaged demo behavior, edit the generator/scenario source first,
then run:

```bash
./.venv/bin/python scripts/operational_demo_data/derive_operational_demo_data.py
./.venv/bin/python scripts/operational_demo_data/rebuild_performance_comparison_demo_data.py --write
./.venv/bin/python scripts/operational_demo_data/rebuild_performance_comparison_demo_data.py
./.venv/bin/python -m ppar.performance_comparison.cli.validate_demo_matrix
```

Use `--write` only when you intend to rewrite tracked packaged CSV assets.
Otherwise, run the rebuild script without `--write` to audit drift.

Current packaged scenario inventory:

| Scenario | Packaged story |
| --- | --- |
| `AAPL` price and buy rows | Equity valuation and buy-transaction amount examples. |
| `MSFT` sale rows | Sale amount, quantity, price, and commission evidence. |
| `CASHUSD` `wd`, `li`, and `lo` rows | Context-gated external cash-flow examples. |
| `CASHUSD` `dp` row | Fee-like expense classified from special-security context. |
| `JPM` `dv` and `rc` rows | Dividend income plus context-gated return of capital. |
| `CVNA` split row | Central split-factor context that explains corrected quantity and related market value. |
| `TSLA` `ss`/`cs` rows | Disclosed synthetic short-sale and cover-short lifecycle using real May 2026 prices. |
| `91282Y2Y1` `in` row | Ordinary interest income plus related holding/accrual evidence. |
| `36225MBS1` `pd` row | MBS principal paydown with portfolio-cash destination evidence. |
| `91282Y5Y1` `by`/`pa` and `sl`/`sa` rows | Paired fixed-income trade and accrued-interest settlement examples. |
| `91282Y5Y1` cost-only row | Maintainer-only fixture context proving stripped cost cannot become a Modified Dietz cause. |

## Common Workflows

### Run Release-Candidate Checks

Use the release-candidate script instead of pasting the long demo refresh and
health-check command list into a shell. The everyday check is:

```bash
./.venv/bin/python scripts/check_release_candidate.py
```

The default path is intentionally deterministic: it audits packaged demo data,
checks generated Axys/APX extract-availability docs, builds and validates
portfolio/security report bundles under `_demo_output/`, runs setup-generated
smoke scripts in a temporary site directory, validates the scenario matrix, and
then runs the 500x Analytics/Audit scale regression and project check. It does
not contact Yahoo for generic analytics data generation and does not rewrite
tracked packaged CSV assets. The 500x phase is intentionally a hard gate and may
take several minutes.

`scripts/audit_scale_baseline_500x.json` records the current machine-dependent
500x timing, memory, phase, row-count, and output-size observations. It is an
optimization reference, not permission to adjust the established scale gate.

For an on-demand larger-input stress check, run:

```bash
./.venv/bin/python scripts/check_scale.py --scale 1000
```

The 1000x Audit fixture copies the full input volume but limits snapshot changes
to the `BALANCED` portfolio. This keeps synthetic reviewer output below the
unchanged 100,000-row production ceiling. The combined command retains the
established Analytics large-site workload at 500x because the new level targets
Audit input scaling. It does not replace the 500x release-candidate gate. Its
separately measured runtime caps are 85x for a warning and 95x for a failure;
the fully changed 10x through 500x workloads use the measured `1 + scale / 7`
growth curve with 5% warning and 10% failure margins.

By default, subcommand output is captured and only printed if a command fails.
Use `--verbose` when you want the full underlying command output.

Use these opt-in switches only when the intent is explicit:

```bash
./.venv/bin/python scripts/check_release_candidate.py --build

./.venv/bin/python scripts/check_release_candidate.py \
  --include-generic-data-generation

./.venv/bin/python scripts/check_release_candidate.py \
  --write-packaged-assets
```

`--build` adds a package-build check. `--include-generic-data-generation` runs
the Yahoo-dependent generic analytics candidate-data generator.
`--write-packaged-assets` lets the operational performance-comparison rebuild
script update tracked packaged CSV assets after intentional demo-data edits.

### Run Setup-Generated Smoke Scripts

The preferred source-checkout smoke path creates a temporary setup workspace and
runs the Python scripts copied by `ppar setup`. This proves the same Python
examples that users see in their local setup folder:

```bash
./.venv/bin/python -m ppar.cli setup /tmp/ppar_smoke_site --include-generic-analytics
./.venv/bin/python /tmp/ppar_smoke_site/analytics/run_analytics.py
./.venv/bin/python /tmp/ppar_smoke_site/audit/run_audit.py
./.venv/bin/python /tmp/ppar_smoke_site/generic_analytics/run_generic_analytics.py
./.venv/bin/python -m ppar.performance_comparison.cli.validate_bundle \
  /tmp/ppar_smoke_site/audit/output/portfolio
./.venv/bin/python -m ppar.performance_comparison.cli.validate_bundle \
  /tmp/ppar_smoke_site/audit/output/security
```

Open the generated `portfolio_audit.xlsx` or `security_audit.xlsx` when present.
The matching HTML audit is generated by default. Use `--no-xlsx-output` for
HTML-only output, `--no-html-output` for XLSX-only output, or both for promoted
CSV-only review files. Keep the CSV artifacts for supplementary diagnostics and
audit traceability.

For the full packaged-demo guardrail pass, run:

```bash
./.venv/bin/python scripts/check_performance_comparison_demo_health.py
```

This consolidates the operational rebuild drift audit, extract-availability
appendix check, setup-generated script execution, bundle validation, and
packaged scenario-matrix validation.

### Generate A Custom Review Bundle

```bash
./.venv/bin/python -m ppar.performance_comparison.cli.report_bundle \
  ppar/setup_templates/axysapx_performance_comparison/axysapx_performance_comparison.yaml \
  _demo_output/custom_portfolio \
  --comparison-level portfolio \
  --include-workbook \
  --require-causal-attribution
```

Use this lower-level command when you want to choose a YAML file or output
directory yourself. Use `--comparison-level security` with the same YAML when
you want a security-level bundle. The public user-facing path is `ppar setup`
followed by `ppar audit`.

Code that needs to inspect the generated report-bundle handoff surface should
use `ppar.performance_comparison.report_bundle_contract()`. That helper returns
the portfolio/security audit filenames, required artifact keys, manifest keys,
review entrypoints, review-summary keys, Modified Dietz review basis, and review
vocabulary keys. It also declares the normalization version and the exact
timestamp/package metadata excluded from repeat-run equivalence.

For Python integrations, prefer the package-root workflow helpers:
`compare_snapshots()`, `write_performance_comparison_report_bundle()`,
`write_performance_comparison_review_workbook()`, `report_bundle_contract()`,
and `report_bundle_validation_issues()`. More specialized policy and
evidence-pack helpers, such as fixed-income boundaries, backlog gates,
transaction boundary registry data, transaction summaries, and source-data
contract validation, are intentionally direct-submodule imports rather than
package-root exports.

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
  ppar/setup_templates/axysapx_performance_comparison/axysapx_performance_comparison.yaml
```

Use this before report generation when you are editing YAML.
By default, it rejects changed source-data fields that lack additive,
evidence-only, or suppression YAML. Use `--allow-incomplete-yaml` only for
diagnostic checks of intentionally incomplete fixtures.

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
- Deterministic financial property and metamorphic tests protect calculation
  identities, sign behavior, scaling behavior, and nonfinite-input boundaries.
- Metadata tests protect packaging, public exports, and documentation promises.
- Bundle validation checks generated artifact structure.
- Demo matrix validation checks that test-only scenario fixtures still explain the
  intended review cases.
- YAML validation checks configuration completeness and vocabulary before a
  comparison is run.
- The 500x scale regression protects the established large-input runtime contract.

Production report runs retain data-dependent accounting invariants, including
no-lost-difference, no-double-counting, period-boundary, lineage, and explained-value
reconciliation checks. Repository-wide structural checks and deliberately mutated
failure probes belong in tests or batch validators instead of every user run.

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

## Release Readiness

Before tagging or publishing, keep the release check small and concrete. The
current maintained release check is:

```bash
./.venv/bin/python scripts/check_release_candidate.py --build
```

Use this pre-publish checklist after the maintainer decides to release:

1. Confirm the working tree is clean.
2. Confirm `pyproject.toml` is the only package-version authority. The public
   `ppar.__version__` value is read from installed package metadata; do not edit
   a second source constant.
3. Decide whether the current version is still right for the release.
4. Confirm the local release tag points at the intended final release commit:

   ```bash
   git rev-parse --short HEAD
   git rev-parse --short v0.1.5
   git ls-remote --tags origin v0.1.5
   ```

   If the local tag is stale and the remote tag does not exist, retag locally
   before pushing:

   ```bash
   git tag -f v0.1.5 HEAD
   ```

5. Build fresh artifacts:

   ```bash
   rm -rf dist
   ./.venv/bin/python -m build --wheel --sdist --no-isolation --outdir dist
   ./.venv/bin/python -m twine check dist/ppar-0.1.5-py3-none-any.whl dist/ppar-0.1.5.tar.gz
   ```

6. Inspect the wheel:
   - only the `ppar` console script is exposed;
   - Axys/APX analytics and performance-comparison starter files are included;
   - `_demo_output`, `scripts`, `tests`, `docs`, and obsolete demo paths are not
     present in the wheel.
7. Install the wheel into a temporary environment and run:

   ```bash
   ppar --help
   ppar setup /tmp/ppar_release_site
   ppar analytics /tmp/ppar_release_site/analytics
   ppar audit /tmp/ppar_release_site/audit
   ```
8. Push the branch and tag only after the checks above match the intended
   release commit.

Do not move, create, or push a release tag until the version and release commit
are explicit.

## Generated Output

Generated output normally belongs under `_demo_output/`:

- `_demo_output/generic_analytics`: Core analytics demo HTML/PNG artifacts.
- `_demo_output/axysapx_analytics`: Axys/APX-backed analytics demo HTML artifacts.
- `_demo_output/performance_comparison_portfolio`: Portfolio comparison review bundle.
- `_demo_output/performance_comparison_security`: Security comparison review bundle.

Performance comparison report bundles include these visible files by default:

- `portfolio_audit.xlsx` or `security_audit.xlsx`: primary Excel reviewer artifact.
- `portfolio_audit.html` or `security_audit.html`: browser-friendly review report.
- `source_detail.csv`: reviewer-friendly finding-level audit trail.
- `audit_support.zip`: complete validated supporting bundle, including findings,
  lineage, diagnostics, manifest metadata, and CSV counterparts of visible sheets.
- `README.md`: generated bundle handoff notes.

`--no-xlsx-output` suppresses XLSX, and `--no-html-output` suppresses HTML.
Supplying both promotes the three canonical review-table CSVs alongside
`source_detail.csv`; the complete audit evidence remains in `audit_support.zip`.

`--expand-all-supporting-files` replaces `audit_support.zip` with the equivalent
`supporting_files/` directory for integration and detailed troubleshooting.

The generated bundle README is not source documentation. It describes one
specific output directory after a bundle is written.

Both portfolio and security workbooks start with the `Performance Differences`
sheet, then use `Performance Difference Causes` and
`source_detail.csv` for the
normal review flow. Explained amounts appear on `Performance Difference Causes`
sheet rows when ppar has a defensible input-level explanation. The action
sheets use an `Explanation` column written for reviewers; the raw audit sheet
and bundle CSV artifacts keep lower-level fields for troubleshooting. Optional
reconstruction diagnostics can add `Reconstruction Summary`,
`Return Reconstruction Checks`, and `Security Return Checks` sheets, but normal
demo output excludes them by default.

## Suggested Consolidation Rules

Before moving files, prefer clarifying the map:

1. Keep the root README short enough to answer "what is this project and how do
   I run the main workflows?"
2. Keep detailed packaged demo instructions in `ppar/setup_templates/axysapx_performance_comparison/README.md`.
3. Keep future performance-comparison plans in
   `docs/roadmap.md`.
4. Keep deep design rationale in `docs/performance_comparison_design.md`.
5. Keep checkpoint/session notes in
   `docs/performance_comparison_checkpoint_notes.md`, but treat them as
   temporary working context.
6. Add new scripts only when they serve automation or a distinct workflow.
   Otherwise, prefer improving this guide or the existing script help text.
