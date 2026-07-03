# Axys/APX Performance Comparison Quick Start

Use this starter kit when your first target is Axys/APX-style performance
comparison. The goal is to get a vanilla Modified Dietz report running first,
then adjust the YAML only where your site extracts prove a different treatment.

## Starter Files

Keep these files together while onboarding a site:

- `axys_performance_comparison.yaml`: main sample YAML and first file to copy.
- `axys_column_mappings.yaml`: normalized column mapping for the demo extracts.
- `demo_extract_availability.yaml`: packaged-demo extract guardrails.
- `axys_full_spec_a/` and `axys_full_spec_b/`: sample Snapshot A/B CSV files.

Do not copy only the YAML unless you also edit its relative paths. The sample
YAML expects the schema file and Snapshot A/B folders to live beside it.

Recommended starter layout:

```text
axys_site_start/
  axys_site_start.yaml
  axys_column_mappings.yaml
  demo_extract_availability.yaml
  axys_full_spec_a/
  axys_full_spec_b/
  _site_output/
```

Use the starter kit as a working copy, not as a library reference. Edit the
copied files in `axys_site_start/`; leave the packaged files unchanged.

## What To Edit First

| File | Edit first | Usually leave alone at first |
| --- | --- | --- |
| `axys_site_start.yaml` | `snapshots.a.path`, `snapshots.b.path`, `comparison.level`, and filenames under `files:` if your extracts use different names. | Transaction rules, impact methods, reconstruction settings, and tolerances. |
| `axys_column_mappings.yaml` | Column names when your local extracts use different headers. | Normalized field names unless the product input contract changes. |
| `demo_extract_availability.yaml` | Do not edit for a first run. | Keep as a packaged-demo guardrail; local production contracts can be documented later. |
| Snapshot folders | Replace the demo CSVs with the site's Snapshot A/B extracts. | Keep native transaction codes and security identifiers case-sensitive. |

## Step 1: Confirm The Source Files

Purpose: prove that the site can provide the core files needed for Modified
Dietz comparison.

Do this:

- Prepare Snapshot A and Snapshot B directories.
- Include portfolio performance, security performance, holdings, and
  transactions files.
- Keep source transaction codes and security identifiers in their native case.

Success criterion: each snapshot has the same core file set, and each file has
the required portfolio, security, date, return, value, or transaction fields.

## Step 2: Start From The Vanilla Axys YAML

Purpose: begin from the conservative packaged Axys/APX setup instead of writing
rules from scratch.

Do this:

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

For a real site, replace `axys_full_spec_a/` and `axys_full_spec_b/` with the
site's Snapshot A/B folders, or edit `snapshots.a.path` and `snapshots.b.path`
to match the site's folder names. Edit the schema reference only if the local
column-mapping file has a different name.

Success criterion: the copied YAML still names all four core datasets and points
to your local Snapshot A and Snapshot B directories. Keep the comparison level
set to `portfolio` for the first report.

Vanilla means the smallest conservative Axys/APX path that can produce a
Modified Dietz comparison. The packaged demo also includes a few richer,
reviewer-facing examples such as fixed-income `pa`/`sa` accrued-interest
adjuncts and external-cash `li`/`lo` rows. Keep those rules unchanged until the
site has evidence that its extracts use the same codes differently.

## Step 3: Validate Before Generating Reports

Purpose: catch missing files, missing fields, unknown transaction codes, and
unsafe Axys/APX code-only assumptions before the reviewer sees a workbook.

Do this:

```bash
./.venv/bin/python -m ppar.performance_comparison.cli.validate_config \
  ./axys_site_start/axys_site_start.yaml
```

Success criterion: validation passes with no missing required datasets, no
missing required source-data columns, and no observed transaction codes without
YAML rules.

## Step 4: Keep Core Transaction Types Vanilla

Purpose: get the common transaction families working before site-specific
variants are added.

Do this:

- Keep `by`, `sl`, `dv`, `in`, fixed-income `pa`/`sa`, fee-like `dp`, and
  external-cash `li`, `lo`, and `wd` close to the packaged YAML at first.
- Keep ambiguous-flow enforcement enabled for `dp`, `li`, `lo`, and `wd`.
- Add local overrides only when your extracts provide source/destination,
  special-security, REP, or reviewed site evidence that proves the treatment.

Success criterion: core transaction rows load with clear categories such as
`buy`, `sell`, `income`, `fee_expense`, or `external_flow`; ambiguous rows fail
validation instead of being guessed.

## Step 5: Generate The First Portfolio Report

Purpose: create the main reviewer artifact for portfolio-period Modified Dietz
performance differences.

Do this:

```bash
./.venv/bin/python -m ppar.performance_comparison.cli.report_bundle \
  ./axys_site_start/axys_site_start.yaml \
  ./axys_site_start/_site_output/performance_comparison_portfolio \
  --include-workbook
```

Success criterion: `report.xlsx`, `report.html`, `manifest.json`, and support
CSVs are created in
`./axys_site_start/_site_output/performance_comparison_portfolio`.

## Step 6: Generate The First Security Report

Purpose: review security-period return differences with the same source data
and YAML.

Do this:

Copy the portfolio YAML and change only the comparison level:

```bash
cp ./axys_site_start/axys_site_start.yaml ./axys_site_start/axys_site_security.yaml
```

```yaml
comparison:
  name: Axys performance comparison demo
  level: security
```

Then generate the security bundle:

```bash
./.venv/bin/python -m ppar.performance_comparison.cli.report_bundle \
  ./axys_site_start/axys_site_security.yaml \
  ./axys_site_start/_site_output/performance_comparison_security \
  --include-workbook
```

Success criterion: `report.xlsx`, `report.html`, `manifest.json`, and support
CSVs are created in
`./axys_site_start/_site_output/performance_comparison_security`.

## Step 7: Review In This Order

Purpose: keep the first review focused on performance differences, not raw-data
wandering.

Do this:

1. Open `report.xlsx`.
2. Start with `Performance Differences`.
3. Use `Performance Difference Causes` to review rows that ppar can connect to
   Modified Dietz inputs.
4. Use `Raw Audit Trail` only when you need the detailed finding rows, match
   status, or review-only support.
5. Use `manifest.json` to confirm the YAML path, extract contract, and generated
   artifact list.

Success criterion: a reviewer can identify fully explained, partly explained,
and unexplained periods without leaving the report bundle.

## Step 8: Iterate Site Overrides Carefully

Purpose: adapt the vanilla setup without turning transaction-code guesses into
performance explanations.

Do this:

- Add one transaction-code override at a time.
- Re-run `validate_config` after every YAML change.
- Rebuild both portfolio and security reports.
- Keep cost-basis handling best-efforts; Modified Dietz explanations depend on
  beginning value, ending value, and dated flows.

Success criterion: every new site rule has a clear source-data reason, and the
report remains understandable to a reviewer who did not write the YAML.

## Demo Commands

Use the packaged demos when you want the marketing/onboarding sample exactly as
shipped:

```bash
./.venv/bin/python -m ppar.demos.performance_comparison_portfolio_demo
./.venv/bin/python -m ppar.demos.performance_comparison_security_demo
```

Those commands use `axys_performance_comparison.yaml` and the packaged sample
CSV files. For a client site, copy the YAML and point it at the site's extracts
instead of editing the packaged demo in place.
