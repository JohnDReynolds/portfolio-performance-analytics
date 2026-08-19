# Maintainer Guide

This guide covers the repeatable workflows used to maintain, validate, and
release PPAR. It assumes the reader already understands the product and system
shape.

## Orientation

Start with the [documentation index](README.md) for document ownership, the
[root README](../README.md) for the installed-user surface, and the
[architecture map](architecture.md) for package, data-flow, configuration, and
report boundaries. This guide begins where those orientation documents stop.

Shared maintenance priorities belong here rather than in a separate portfolio
roadmap:

- keep `ppar setup`, the Audit-focused root README, and setup-installed
  documentation aligned around the primary Audit first-run path;
- keep the explicit `ppar setup --analytics` path and Analytics README aligned
  without adding Analytics to general onboarding;
- keep package contents, public commands, generated artifact names, and release
  records synchronized with executable contracts and tests;
- keep inexpensive financial, conservation, lineage, and explanation-
  reconciliation invariants enabled in production;
- run the maintained 500x scale check after major cross-cutting, reporting,
  Audit, safety-net, or performance changes; and
- validate the complete setup and report workflow on Windows before claiming
  Windows support.

## Scripts

The `scripts/` directory contains source-checkout maintenance helpers. Those
helpers may be included in source distributions for maintainers, but they are
not installed-package demo workflows. Audit command
implementations live in `ppar.audit.cli` and can be run from a
source checkout with `./.venv/bin/python -m <module>`.

| Command Module Or Script | Purpose | Common Use |
| --- | --- | --- |
| `scripts/check_release_candidate.py` | Runs the maintained release-candidate demo, setup, and health-check sequence. | `./.venv/bin/python scripts/check_release_candidate.py` |
| `scripts/check_project.py` | Runs project checks. | `./.venv/bin/python scripts/check_project.py --quick` |
| `scripts/check_audit_demo_health.py` | Runs packaged Audit demo guardrails. | `./.venv/bin/python scripts/check_audit_demo_health.py` |
| `scripts/render_readme_images.py` | Regenerates README images from packaged Mega-Cap analytics demo files. | Documentation image maintenance after analytics demo refresh. |
| `scripts/operational_demo_data/rebuild_audit_demo_data.py` | Rebuilds and audits packaged Audit demo accounting. | Refresh scenario-derived `holdings.csv` plus derived `secperf.csv`/`portperf.csv` and verify fixture consistency after demo-data edits. |
| `scripts/check_audit_demo_data.py` | Checks the packaged Audit demo data. | Focused demo-data validation. |
| `ppar.audit.cli.validate_bundle` | Validates a generated report bundle. | Check artifacts plus CSV/HTML/XLSX semantic parity and normalized integrity metadata. |
| `scripts/validate_demo_matrix.py` | Validates Performance Comparison scenario fixtures. | Prove validation fixtures still cover documented scenarios. |
| `ppar.audit.cli.validate_config` | Validates an Audit YAML file. | Catch source-data and YAML setup issues before generating reports. |

## Maintainer Checks

Useful source-checkout checks:

```bash
./.venv/bin/python scripts/check_project.py --quick
./.venv/bin/python scripts/check_audit_demo_health.py
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

To refresh only the Audit README screenshots:

```bash
./.venv/bin/python scripts/render_readme_images.py --only audit
```

## Installed Commands

The installed command names are declared in `pyproject.toml`.

| Command | Purpose |
| --- | --- |
| `ppar` | Audit-focused setup and onboarding command; the maintained `analytics` subcommand remains directly callable. |

The source-checkout smoke path uses setup-generated scripts:

```bash
./.venv/bin/python scripts/check_audit_demo_health.py --skip-rebuild-audit
```

## Demo Data

Packaged demos and test configuration fixtures intentionally live in different
places:

- `ppar/setup_templates/axys_apx_audit/` is the public Audit workspace source
  copied directly by default `ppar setup` and is also used
  for packaged examples that users and reviewers can run.
- `ppar/setup_templates/axys_apx_analytics/` is the maintained Analytics
  workspace source copied directly by `ppar setup --analytics`.
- `ppar/setup_templates/generic_analytics/` is the optional, vendor-neutral
  Generic Analytics workspace source copied by `ppar setup
  --generic-analytics`. Its data also feeds README marketing images, analytics
  regression tests, and operational demo-data derivation. Audit remains the
  default onboarding path.
- `tests/data/axys/` is for synthetic Axys/APX snapshots, test-only Audit YAML
  files, and validation matrix fixtures.

The packaged Axys/APX YAML file is limited to user-facing marketing and onboarding
workflows. Broader scenario coverage lives under `tests/data/axys`.

| Role | YAML | Snapshot A | Snapshot B | Use |
| --- | --- | --- | --- | --- |
| Workbook demos | `axys_apx_audit.yaml` | `snapshot_a` | `snapshot_b` | Shared portfolio/security demo spec. Setup-generated scripts select the primary review level. |

For the maintained XLSX workbook smoke paths and expected output, use the
[packaged Audit demo guide](audit/packaged_demo.md).
For the validation fixture matrix, use
[`tests/data/axys/README.md`](../tests/data/axys/README.md).

### Packaged Axys/APX Audit Demo Maintenance

Treat the packaged Axys/APX Audit demo as a small accounting
lab, not as hand-edited CSV examples. The durable source-of-truth files are:

| File | Owns |
| --- | --- |
| `scripts/demo_support/market_data.py` | Shared cached market-history loading, refresh, price lookup, and total-return reconciliation for maintained Analytics and Audit demos. |
| `scripts/operational_demo_data/derive_operational_demo_data.py` | Base portfolios, selected securities, baseline holdings, baseline transactions, and synthetic Axys/APX-style performance rows. |
| `scripts/operational_demo_data/holdings.py` | Operational-demo holding construction plus trade, split, cash, and price roll-forward validation. |
| `scripts/operational_demo_data/refresh_audit_market_baseline.py` | Shared-cache market marks, transaction-led Snapshot A quantity/cash roll-forwards, and market-sensitive scenario calibration. |
| `scripts/operational_demo_data/audit_transaction_scenarios.csv` | Snapshot B transaction adjustments and inserted transaction rows. |
| `scripts/operational_demo_data/audit_holding_scenarios.csv` | Snapshot B explicit holding restatements, including split-processing, accrual, valuation, and maintainer-only cost context. |
| `scripts/operational_demo_data/audit_scenario_calendar.csv` | Operational map from physical scenario rows to reviewer story periods. |
| `scripts/operational_demo_data/audit_scenario_inventory.csv` | Validated semantic contract that keeps each source period within the two-independent-change review target and protects report outcomes plus carry-forward behavior. |
| `scripts/operational_demo_data/rebuild_audit_demo_data.py` | Applies transaction scenarios, derives transaction-driven holdings, rebuilds `secperf.csv`/`portperf.csv`, strips internal fields from packaged CSVs, and audits drift. |
| `ppar/setup_templates/axys_apx_audit/axys_apx_audit.yaml` | User-facing interpretation contract: file names, column mappings, transaction rules, field roles, reconstruction settings, and report level. |
| `ppar/setup_templates/axys_apx_audit/README.md` | Canonical README copied into every generated Audit workspace. |
| `docs/audit/packaged_demo.md` | Maintainer-facing demo story, expected output, and accounting-scenario guidance. |
| `scripts/validate_demo_matrix.py` | Test-only scenario coverage guardrail. It does not define the packaged demo story. |

When changing packaged demo behavior, edit the generator/scenario source first,
then run:

```bash
./.venv/bin/python -m scripts.operational_demo_data.derive_operational_demo_data
./.venv/bin/python -m scripts.operational_demo_data.refresh_audit_market_baseline
./.venv/bin/python scripts/operational_demo_data/rebuild_audit_demo_data.py --write
./.venv/bin/python scripts/operational_demo_data/rebuild_audit_demo_data.py
./.venv/bin/python scripts/validate_demo_matrix.py
```

The market-baseline command is a no-write preview by default. Use `--write`
only when intentionally promoting new dated prices, trades, and roll-forwards;
use `--refresh-market-history` only when the shared local CSV should contact
yFinance and replace its cached observations. Normal repeat construction reads
the cache without network access.

The shared return gate compares close-plus-dividend/split calculations with
adjusted-close returns. Differences above 0.02 percentage points (2 basis
points) are recorded as warnings; differences above 0.10 percentage points
(10 basis points) stop generation and report the security, period, source
values, and calculated returns. Split records also
drive direct quantity roll-forward checks rather than relying on the return
tolerance alone.

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

The default path is intentionally deterministic. It invokes the consolidated
Audit demo-health check, generates both maintained Audit reports through one
`ppar audit` site run under `_demo_output/audit/`, smoke-tests the
setup-generated Analytics scripts, and then runs the 500x Analytics/Audit scale regression
and project check. It does not contact Yahoo for generic analytics
data generation and does not rewrite tracked packaged CSV assets. The 500x
phase is intentionally a hard gate and may take several minutes.

`scripts/audit_scale_baseline_500x.json` records the current machine-dependent
500x timing, memory, phase, row-count, and output-size observations. It is an
optimization reference, not permission to adjust the established scale gate.
The 500x Audit timing gate compares the full workload with a 100x timing
reference. Expected growth is 5.00x, with a warning above 5.25x and failure
above 5.50x. The separate 1x run remains the financial and output-equivalence
reference, so the timing method does not weaken result validation.

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
the fully changed 10x through 100x workloads use the measured
`1 + scale / 7.64` growth curve with 5% warning and 10% failure margins. The
fully changed 500x workload uses the 100x reference and caps described above.

By default, subcommand output is captured and only printed if a command fails.
Use `--verbose` when you want the full underlying command output.

Use these opt-in switches only when the intent is explicit:

```bash
./.venv/bin/python scripts/check_release_candidate.py --build

./.venv/bin/python scripts/check_release_candidate.py --refresh-images

./.venv/bin/python scripts/check_release_candidate.py \
  --include-generic-data-generation

./.venv/bin/python scripts/check_release_candidate.py \
  --write-packaged-assets
```

`--build` regenerates `PPAR.pdf` from the current `README.md`, builds the wheel
and all README images from current generated reports, runs the complete
release-candidate sequence, installs a temporary wheel outside the source
checkout, exercises the installed Audit and Analytics workflows, validates both
Audit bundles, and writes the final Twine-validated wheel and source distribution
under `dist/`. It cleans the release-generated `_demo_output` directories first.
This is the single command used to create a distributable release candidate.
`--refresh-images` regenerates the README PNG/JPG assets as well as `PPAR.pdf`.
`--include-generic-data-generation` runs the Yahoo-dependent generic analytics
candidate-data generator.
`--write-packaged-assets` lets the operational Audit rebuild
script update tracked packaged CSV assets after intentional demo-data edits.

### Tracked Generated Assets And Archives

`PPAR.pdf` is a tracked release asset generated from `README.md`, not an
independent documentation source. Refresh it with `scripts/render_readme_pdf.py`
or the release-candidate `--build`/`--refresh-images` workflow, and commit it
only with the source README change that required the refresh.

Files under `docs/archive/` and product-specific `archive/` directories are
frozen provenance. Do not revise their conclusions or status; only add an
archival banner, repair a link after a move, or add a new clearly indexed
historical snapshot. Current guidance must live outside an archive.

### Run Setup-Generated Smoke Scripts

The preferred source-checkout smoke path creates separate temporary Audit,
Axys/APX Analytics, and Generic Analytics workspaces and runs the Python scripts
copied by `ppar setup`. This proves the same Python examples that users see in
their local workspace:

```bash
./.venv/bin/python -m ppar.cli setup /tmp/my_ppar_audit
./.venv/bin/python /tmp/my_ppar_audit/run_audit.py
./.venv/bin/python -m ppar.cli setup \
  /tmp/my_ppar_analytics \
  --analytics
./.venv/bin/python /tmp/my_ppar_analytics/run_analytics.py
./.venv/bin/python -m ppar.cli setup \
  /tmp/my_ppar_generic_analytics \
  --generic-analytics
./.venv/bin/python /tmp/my_ppar_generic_analytics/run_generic_analytics.py
./.venv/bin/python -m ppar.audit.cli.validate_bundle \
  /tmp/my_ppar_audit/output/portfolio
./.venv/bin/python -m ppar.audit.cli.validate_bundle \
  /tmp/my_ppar_audit/output/security
```

Open the generated `portfolio_audit.xlsx` or `security_audit.xlsx` when present.
The maintained `audit:` run-settings section enables the matching HTML audit.
Use `--html-only` for HTML-only output, `--xlsx-only` for XLSX-only output, or
`--csv-only` for promoted CSV-only review files for one run. Keep the CSV
artifacts for supplementary diagnostics and audit traceability.

For the full packaged-demo guardrail pass, run:

```bash
./.venv/bin/python scripts/check_audit_demo_health.py
```

This consolidates the operational rebuild drift audit, extract-availability
appendix check, setup-generated script execution, bundle validation, and
packaged scenario-matrix validation.

### Generate Maintained Audit Demo Reports

```bash
./.venv/bin/python -m ppar.cli setup \
  _demo_output/audit_workspace \
  --overwrite
./.venv/bin/python -m ppar.cli audit \
  _demo_output/audit_workspace \
  --output-directory _demo_output/audit
```

This is the same site-level implementation used by `ppar audit` and the
setup-generated `run_audit.py`. It writes both available report levels in one
atomic site run. Python integrations that intentionally need one lower-level
bundle can use `compare_snapshots()` and `write_audit_report_bundle()`.

Code that needs to inspect the generated report-bundle handoff surface should
use `ppar.audit.report_bundle_contract()`. That helper returns
the portfolio/security audit filenames, required artifact keys, manifest keys,
review entrypoints, review-summary keys, Modified Dietz review basis, and review
vocabulary keys. It also declares the normalization version and the exact
timestamp/package metadata excluded from repeat-run equivalence.

For Python integrations, prefer the package-root workflow helpers:
`compare_snapshots()`, `write_audit_report_bundle()`,
`write_audit_review_workbook()`, `report_bundle_contract()`,
and `report_bundle_validation_issues()`. More specialized policy and
evidence-pack helpers, such as transaction summaries and source-data contract
validation, are intentionally direct-submodule imports rather than package-root exports.
Transaction code-family research and fixture groupings live in
`scripts/transaction_policy_evidence.py`; installed product code does not import
them.

### Validate The Packaged Demo Matrix

```bash
./.venv/bin/python scripts/validate_demo_matrix.py
```

This is a scenario-coverage check. It is not just a bundle validator; it proves
that packaged fixtures still demonstrate the documented review situations.

### Validate A Generated Bundle

```bash
./.venv/bin/python -m ppar.audit.cli.validate_bundle \
  _demo_output/audit/portfolio
```

Use this after generating report/workbook output.

### Validate An Audit YAML

```bash
./.venv/bin/python -m ppar.audit.cli.validate_config \
  ppar/setup_templates/axys_apx_audit/axys_apx_audit.yaml
```

Use this before report generation when you are editing YAML.
It rejects changed source-data fields that lack additive, evidence-only, or
suppression YAML. During development, maintainers can call the lower-level
Python validation API with `require_complete_yaml_setup=False` to inspect an
intentionally incomplete fixture; normal commands always enforce complete
setup.

### Run Project Checks

```bash
./.venv/bin/python scripts/check_project.py --quick
```

Install the tested maintainer environment with the repository constraint set:

```bash
./.venv/bin/python -m pip install \
  --constraint constraints/ci.txt \
  --editable ".[analytics,dev]"
```

`constraints/ci.txt` pins the Python 3.11.9 and 3.12.1 CI and
release-candidate environments. Update it deliberately alongside dependency
upgrades; normal package installs continue to use the compatible lower bounds
in `pyproject.toml`.

GitHub Actions runs `scripts/check_project.py --build` on Ubuntu with both
supported Python versions. Those jobs verify tests, static analysis, package
building, and installed-wheel smoke behavior across the compatibility matrix.
They do not run the machine-sensitive 500x timing gate. The unchanged 500x
hard gate remains part of the local `scripts/check_release_candidate.py --build`
command and must pass before tagging or publishing a release.

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
  tests.test_audit_report \
  tests.test_audit_cli \
  tests.test_audit_workbook_contract
```

When changing demo data or YAML, also run:

```bash
./.venv/bin/python scripts/check_audit_demo_data.py
./.venv/bin/python scripts/operational_demo_data/rebuild_audit_demo_data.py
./.venv/bin/python scripts/validate_demo_matrix.py
```

## Release Readiness

Before tagging or publishing, create the complete release candidate with:

```bash
./.venv/bin/python scripts/check_release_candidate.py --build
```

The command stops at the first failure. On success, `dist/` contains exactly one
Twine-validated wheel and one Twine-validated source distribution. Use this
pre-publish checklist after the maintainer decides to release:

1. Confirm the working tree is clean.
2. Confirm `pyproject.toml` is the only package-version authority. The public
   `ppar.__version__` value is read from installed package metadata; do not edit
   a second source constant.
3. Decide whether the current version is still right for the release.
4. Read the selected version from the package metadata, then confirm the local
   release tag points at the intended final release commit:

   ```bash
   PPAR_RELEASE_VERSION=$(./.venv/bin/python -c \
     'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')
   git rev-parse --short HEAD
   git rev-parse --short "v${PPAR_RELEASE_VERSION}"
   git ls-remote --tags origin "v${PPAR_RELEASE_VERSION}"
   ```

   If the local tag is stale and the remote tag does not exist, retag locally
   before pushing:

   ```bash
   git tag -f "v${PPAR_RELEASE_VERSION}" HEAD
   ```

5. Run `./.venv/bin/python scripts/check_release_candidate.py --build` and
   confirm that it reports the two release artifacts under `dist/`.
6. Inspect the resulting wheel:
   - only the `ppar` console script is exposed;
   - Axys/APX Analytics and Audit workspace files are included;
   - `_demo_output`, `scripts`, `tests`, `docs`, and obsolete demo paths are not
     present in the wheel.
7. Confirm that the release-candidate `--build` check installed a wheel into
   a temporary environment and successfully ran:

   ```bash
   ppar --help
   ppar setup /tmp/my_ppar_audit
   ppar audit /tmp/my_ppar_audit
   ppar setup /tmp/my_ppar_analytics --analytics
   ppar analytics /tmp/my_ppar_analytics
   ```
8. Push the branch and tag only after the checks above match the intended
   release commit.

Do not move, create, or push a release tag until the version and release commit
are explicit.

## Generated Output

Generated output normally belongs under `_demo_output/`:

- `_demo_output/generic_analytics`: Core analytics demo HTML/PNG artifacts.
- `_demo_output/axys_apx_analytics`: Axys/APX-backed analytics demo HTML artifacts.
- `_demo_output/audit/portfolio`: Portfolio Audit review bundle.
- `_demo_output/audit/security`: Security Audit review bundle.

Audit report bundles include these visible files by default:

- `portfolio_audit.xlsx` or `security_audit.xlsx`: primary Excel reviewer artifact.
- `portfolio_audit.html` or `security_audit.html`: browser-friendly review report.
- `source_detail.csv`: reviewer-friendly finding-level audit trail.
- `audit_support.zip`: complete validated supporting bundle, including findings,
  lineage, diagnostics, manifest metadata, and CSV counterparts of visible sheets.
- `README.md`: generated bundle handoff notes.

The workspace YAML enables `audit.xlsx_output` and `audit.html_output`.
`--html-only`, `--xlsx-only`, and `--csv-only` select a nonstandard output
format for one run. CSV-only output promotes the four canonical review-table
CSVs alongside `source_detail.csv`; the complete audit evidence remains in
`audit_support.zip`.

Extract `audit_support.zip` when an integration or detailed investigation needs
the individual supporting CSV and JSON files. Extraction does not regenerate or
change the Audit results.

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
