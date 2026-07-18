# Performance Comparison Demo Source Contract

This contract defines how the packaged Audit demo source files
should be interpreted. It is intentionally narrower than the Axys/APX reference
manuals. The demo exists to answer one product question:

```text
Why did reported performance change from Snapshot A to Snapshot B?
```

The packaged CSV files are normalized demo extracts. They are not official Axys/APX
native schemas, not universal IMEX layouts, and not claims about how every
Axys/APX installation stores data internally.
The demo is formula-focused; it is not a full accounting-system export.

## Governing References

Use the Axys/APX reference chapters as evidence boundaries:

- [`Chapter_05_Transactions.md`](../axys_apx/reference/Chapter_05_Transactions.md)
- [`Chapter_06_Holdings.md`](../axys_apx/reference/Chapter_06_Holdings.md)
- [`Chapter_07_Cash.md`](../axys_apx/reference/Chapter_07_Cash.md)
- [`Chapter_10_Performance.md`](../axys_apx/reference/Chapter_10_Performance.md)
- [`Chapter_15_Data_Dictionary.md`](../axys_apx/reference/Chapter_15_Data_Dictionary.md)
- [`contracts/demo_extract_availability.md`](../axys_apx/contracts/demo_extract_availability.md)
  for field-by-field IMEX/REP availability confidence.
- [`contracts/transaction_semantics_matrix.md`](../axys_apx/contracts/transaction_semantics_matrix.md)
  for implementation-facing transaction-code treatment, required evidence, and
  coverage status.
- [Archived evidence-pack review](archive/performance_comparison_evidence_pack_review.md)
  for the earlier checkpoint inventory and decision provenance.

When those references strongly imply common transaction-code meaning, the demo
may use that evidence. When the references mark native storage, exact IMEX object
names, or vendor calculation behavior as Unknown, this demo must not present
those details as verified Axys/APX facts.

## Current Packaged Files

| Demo file | Demo role | Contract boundary |
| --- | --- | --- |
| `transactions.csv` | Normalized transaction extract used to explain changed cash flows, security-level flows, income, fees, and review-only transaction evidence. | Transaction codes should use Axys/APX-style observed codes when the reference material supports them. YAML remains the explicit interpretation contract. |
| `holdings.csv` | Normalized point-in-time holdings and valuation extract. | Native Axys/APX holding storage is Unknown. Public equities use dated yFinance market observations; synthetic fixed-income identifiers use disclosed BIL/SHY/IEI/MBB proxies. Quantities roll from opening positions through trades and explicit split scenarios. |
| `secref.csv` | Optional snapshot-specific security reference used only to qualify Data Issues populations. | Exact source case is preserved. Security-master fields do not enter performance calculations, and local field names, code dictionaries, and historical classification behavior must be validated. |
| `portperf.csv` | Reported portfolio-period performance target used for comparison. | `portperf` is a normalized demo dataset name, not a verified native Axys/APX object name. |
| `secperf.csv` | Reported security-period performance target used for comparison. | `secperf` is a normalized demo dataset name, not a verified native Axys/APX object name. |
| `splits.csv` | Optional security-level split factors used as review evidence and by `large_price_variation` normalization. | The factor is the new-shares-per-old-share multiplier on its effective date. Same-date price observations are treated as post-split; local date/factor meaning must be validated. |

## Market-Data Provenance

Analytics and Audit use one maintainer cache at
`_demo_output/demo_market_data/yfinance_market_history.csv`. It is refreshed
separately from deterministic demo construction and is not an Axys/APX source
file. The normalized cache preserves:

- Yahoo's dividend-unadjusted, split-normalized source `Close`;
- reconstructed contemporaneous closes used for holdings and trades;
- adjusted closes used for total returns;
- reported cash dividends and stock splits; and
- Yahoo symbol and repair provenance.

Cash holdings remain at 1.00. The demo's CUSIP-like fixed-income rows are still
synthetic instruments; their dated price/return behavior comes from BIL (T-bill),
SHY (short Treasury), IEI (intermediate Treasury), and MBB (agency MBS) proxies.
Controlled errors, including the CVNA split-processing story, nonpositive-price
examples, and isolated ALPHA/GOOGL stale `PRICE` field, remain deliberately
synthetic and are identified as review scenarios rather than market facts. The
GOOGL market value remains derived from the real dated observation; only the
later displayed source price is replaced after baseline refresh. The packaged-
equity variation gate retains its 28-day threshold and requires exactly this
one named ALPHA/GOOGL exception in each snapshot; every other long unchanged
equity-price run still fails.

The packaged `common_stock_20_percent` named rule uses unmodified real AVGO
holding prices. It reports six rows across the two snapshots: ALPHA and INCOME
each show the full April 309.51-to-417.43 movement, while BALANCED's segmented
April 1–10 period shows 309.51-to-371.55, or 20.04 percent under the implemented
minimum-price denominator. This executable result supersedes the earlier rough
20.4 percent description. No AVGO price or market value is injected for this
scenario.

## Extraction Requirement Labels

User-facing extraction guidance uses only three labels:

- **Required**: needed to make Fully Explained possible in the ordinary
  portfolio comparison.
- **Required only when applicable**: needed for a named feature or data
  condition, such as security comparison, multi-currency, separately stated
  accrued income, or ambiguous transaction codes.
- **Optional**: safe to omit; absence alone does not prevent Fully Explained.

The exhaustive field checklist lives in comments under `files:` in the
packaged comparison YAML and in the generated
[PPAR Axys/APX Extract Requirements and Source Guidance](../axys_apx/contracts/demo_extract_availability.md).
Internal scenario/rebuild fields are a fixture-maintenance boundary, not a
fourth extraction category. They must not leak into user-facing extracts.

## Minimum Source-Data Contract (Runtime Structural Minimum)

This section describes loader validation after a dataset is configured. It is
not the user-facing Fully Explained extraction checklist above. A field can be
structurally optional to the loader yet still be required to calculate a
particular explanation.

The comparison YAML must identify the source datasets ppar is allowed to read
from both Snapshot A and Snapshot B. The selected comparison level determines
the target performance dataset:

| Dataset | Required when | Required normalized columns |
| --- | --- | --- |
| `portfolio_performance` | `comparison.level` is `portfolio`. | `portfolio_id`, `from_date`, `thru_date`, `portfolio_return` |
| `security_performance` | `comparison.level` is `security`, or `security_return_reconstruction` is configured. | `portfolio_id`, `security_id`, `from_date`, `thru_date`, `security_return` |
| `holdings` | Portfolio or security return reconstruction is configured, or holding fields are used as performance explanations. | `portfolio_id`, `security_id`, `holding_date` |
| `transactions` | Portfolio or security return reconstruction is configured, or transaction fields are used as performance explanations. | `portfolio_id`, `security_id`, `transaction_date` |
| `fx_rates` | Optional evidence links a rate change to a counted base-currency value. | `from_currency`, `to_currency`, `rate_date`, `fx_rate`; add `portfolio_id` and `local_exposure` for report linkage |
| `security_reference` | A Data Issues `only` or `exclude` filter references `security_reference.*`. | `security_id`; each referenced qualifier column must also be present and nonblank for relevant source rows |
| `splits` | Optional; when present, `large_price_variation` uses it to normalize earlier prices to the performance-period ending share basis. | `security_id`, `split_date`, `split_factor`; factor must be finite and strictly positive for the enabled rule |

Currency basis follows the normalized field name and dataset scope. In
holdings and transactions, unqualified monetary fields use the row
`currency`; `base_` fields use portfolio `base_currency`. Portfolio/security
performance monetary fields are inherently base-currency values and remain
unprefixed. `fx_rates.fx_rate` is `to_currency` units per one `from_currency`
unit. PPAR does not create parallel `local_` names because row currency is the
detailed-data default.

Modified Dietz never treats an explicitly foreign unqualified value as base
currency. Foreign holdings, accrued income, and transactions must supply the
applicable `base_market_value`, `base_accrued`, or `base_amount` before that
amount can be counted. Cash balances are holdings, not a separate dataset.

Normalized FX rows must provide nonblank pair currencies and dates and a finite,
strictly positive rate. Pair/date rows must also be unique within the available
`rate_source` and `rate_type` provenance. These are ppar input-integrity rules,
not claims about Axys quote conventions or native FX storage.

Security-reference rows are snapshot-specific enrichment only. Their
`security_id` values must be nonblank and unique with exact source case. A
reference-qualified Data Issues check stops when the file, exact-case join row,
referenced column, or referenced value is unavailable; it does not silently
broaden or empty the reviewed population.

All supplied currency values are normalized to uppercase three-letter codes.
For a foreign row, a nonzero `holdings.market_value`, `holdings.accrued`, or
`transactions.amount` requires its explicit base-currency counterpart before
comparison. When row and base currency are the same, supplied local/base values
must agree. A portfolio-specific FX quote must use portfolio `base_currency` as
its `to_currency`.

Configured required datasets must exist in both snapshots. If a required source
file is missing, if a required normalized column cannot be resolved from the
source header and schema aliases, or if a required source column is ambiguous,
ppar stops before producing a report. Optional datasets may be omitted; if a
configured optional file exists, its required columns are still validated before
that dataset contributes evidence.

Return reconstruction tightens the contract. When portfolio or security return
reconstruction is configured, `holdings` and `transactions` are required
formula-source datasets, not optional evidence files. Modified Dietz also
requires the configured reconstruction timing, day-count, inclusion,
flow-category, income-category, return-basis, and sign-convention YAML fields.

## Field Role Contract

The executable field-role registry and comparison configuration own field-level
accounting treatment. The stable design boundary is documented in
[`performance_comparison_design.md`](performance_comparison_design.md), while
the generated
[extract requirements](../axys_apx/contracts/demo_extract_availability.md) own
the user-facing field checklist. This contract adds only two demo-specific
rules:

- packaged Axys/APX transaction rows intentionally omit a stable transaction ID
  because the current vendor evidence does not establish one as a typical
  REP/IMEX output; and
- settlement date remains review evidence unless an explicit configured rule
  makes it performance-relevant.

Unknown fields remain fail-closed and require an explicit field role before they
can participate in comparison or explanation.

## Scenario Preservation Contract

Every intentional transaction, holding, and multi-currency demo story is named
in `scripts/operational_demo_data/audit_scenario_calendar.csv`
and independently protected in
`audit_scenario_inventory.csv` in the same directory. The
inventory is a complete semantic contract rather than a list of names. It
protects economic meaning, portfolio, actual source period, reviewer story
period, scenario family, primary security, expected report disposition/status,
independent economic-change identity, and carry-forward treatment. The rebuild
audit checks those declarations against the scenario calendar, fixture input
dates, and generated report tables. Removing, moving, replacing, or changing a
scenario's outcome therefore requires an explicit, separately reviewable
inventory change; matching only old transaction-type counts is not sufficient.

The source-period budget counts independent economic changes, not physical CSV
rows. Paired fixed-income settlement legs, the TSLA short/cover cycle, and the
INCOME holding/interest pair each count as one economic story. A carried
beginning-value effect remains visible in its later report period but does not
pretend to be a new source change in that later period.

The March `BALANCED0403` contribution restatement uses the standalone
`BALANCED_CONTRIBUTION` portfolio. This preserves a real `li` external-flow and
Modified Dietz explanation without carrying that cash difference into later
BALANCED periods.

No beginning value, ending value, flow, income, or related source evidence may
be removed from reviewer-facing causes merely because it originated in an
earlier period. If it participates directly or indirectly in Modified Dietz, it
remains visible. Beginning-value rows carried from the preceding period are
explicitly labeled as inherited. Demo stories that would otherwise introduce
two unrelated new changes in one period use separate periods within an existing
portfolio. For example, BALANCED March isolates the AAPL valuation mark while
BALANCED's first May period retains the CVNA split and all inherited
beginning-value effects.

This rule is protected independently from arithmetic reconciliation. Every
changed Modified Dietz formula component must be represented in `Performance
Difference Causes`, even if no more granular source row is available in that
period. Separately, the counted cause rows must sum to `Explained Difference`,
and every `Fully Explained` period must reconcile to its reported performance
difference. Report generation fails on either violation.

## Cash-Balance Policy

`CASHUSD`, `CASHEUR`, and `CASHGBP` are normalized demo cash-balance holdings.
They should not be described
as native Axys/APX cash storage. When a transaction amount affects cash but the
source row does not prove the exact cash security, user-facing explanations may
refer to the changed `holdings` cash-balance.

The packaged demo maps USD, EUR, and GBP transaction effects explicitly to
those three cash securities. Foreign rows carry local currency plus explicit
portfolio-base values; this normalized contract does not assert native Axys/APX
field names or cash-account storage.

## Date Policy

The demo treats `FROM_DATE` and `THRU_DATE` as inclusive. The beginning holdings
date for a period is the calendar day immediately before `FROM_DATE`; the ending
holdings date is `THRU_DATE`.

Performance periods may not be reversed or overlap. Changed dated evidence is
checked against the periods for its portfolio. Multiple matches are a source-
contract error. Historical evidence outside an assigned formula boundary may
remain visible for carry-forward review but cannot own an explained amount. A
prior-day holding or FX value assigned as the beginning boundary is a Modified
Dietz input and may own `Performance Difference Explained`.

For transaction rows, `TRANSACTION_DATE` is the economic as-of date used by the
demo return-reconstruction rules. It represents the trade date or ex-date for
the event. `SETTLE_DATE` or pay date can be useful context, but it is not the
default performance date in this demo contract.

## Transaction-Code Policy

Transaction semantics are required YAML. A source transaction code can
participate only when the comparison configuration has a matching rule and any
required context. The machine-readable
[`transaction_semantics_matrix.yaml`](../axys_apx/contracts/transaction_semantics_matrix.yaml)
owns code categories, evidence requirements, coverage, and fixtures; this
document does not repeat that matrix.

Reviewer artifacts preserve the source code exactly as supplied. Normalization
is permitted only for explicit semantic classification and must not become a
general source-identifier equality rule.

Ambiguous `dp`, `li`, `lo`, and `wd` rows require reviewed source/destination,
special-security, report-semantic, or equivalent local-contract evidence. Code
alone is insufficient. When the available IMEX extract lacks that context, use
a reviewed REP/custom-report source or stop before classification.

## Fixed-Income Transaction Boundary

The packaged demo currently uses four proved fixed-income Modified Dietz input
families:

- `in` transaction rows for ordinary bond or cash interest that is treated as
  performance income; and
- `holdings.accrued` changes as additive beginning/end valuation inputs when
  present in holdings extracts; and
- context-gated `pa`/`sa` transaction rows as fixed-income accrued-interest
  adjuncts paired with 91282Y5Y1 buy/sell transactions; and
- context-gated `pd` principal-paydown rows as MBS/amortizing-security
  principal flow with portfolio-cash destination evidence.

That boundary is deliberate. Modified Dietz needs beginning value, ending
value, and dated external cash flows. It does not need ppar to rebuild an
amortization/accretion engine, bond principal schedule, yield calculation, or
tax-lot ledger before ordinary interest and configured accrued value can be
used as formula inputs.

The packaged `pa`/`sa` rows are deliberately narrower than code-only semantics.
They are accepted only because one scenario source derives the accrued-interest
transaction rows, cash movement, quantity-driven holding value/accrual rows,
`secperf.csv`, `portperf.csv`, and reviewer-facing report output together. The
rebuild path keeps these rows outside portfolio external-flow weighting, and
the YAML rules require fixed-income context before classifying them.

The packaged demo still does not use `ai` rows, and it does not treat `pd` as a
safe code-only default. Those codes need more than the Axys/APX short code
before ppar can classify them: margin or bond/principal context, cash movement,
amount sign, and local mapping or REP/report evidence. Test-only candidate
override profiles cover explicit Modified Dietz treatments for `ai`, `pa`,
`sa`, and `pd`; the packaged demo promotes only context-gated `pa`/`sa` accrued
interest and MBS `pd` principal paydown stories.

The default runtime guard uses
`ppar/setup_templates/axys_apx_audit/demo_extract_availability.yaml`. A site comparison YAML
can point to a local contract when its validated IMEX/REP extract layout differs:

```yaml
extract_contract:
  path: site_extract_contract.yaml
  enforce_ambiguous_axys_flows: true
```

`enforce_ambiguous_axys_flows` defaults to `true`. Setting it to `false` is an
explicit opt-out and should be reserved for a locally reviewed workflow.
The test-only `local_opt_out` site variant documents that boundary: code-only
ambiguous rows may classify from YAML only when the site explicitly accepts that
local risk.

Use
[`axys_apx/contracts/templates/site_extract_contract.yaml`](../axys_apx/contracts/templates/site_extract_contract.yaml)
as the starter. Copy it beside the comparison YAML, remove fields the local
extract does not expose, and keep only fields validated from IMEX, REP, a custom
report, or another reviewed source.

For an operator-facing setup pass, use the
[`Site Extract Readiness Checklist`](site_extract_readiness_checklist.md). It
summarizes the IMEX-context path, REP/report fallback path, code-only failure
mode, and bundle-manifest handoff evidence.

Local extract contracts are validated by `validate_config`. For ambiguous-flow
runtime guards, a contract must include `datasets.transactions.csv.columns`.
Each listed transaction column must map to a supported transaction alias and
must define boolean `requires_context_for_semantics` and `blocking_if_missing`
flags. When ambiguous-flow enforcement is enabled, at least one transaction
column must have both flags set to `true`.

## Site Extract Contract Setup

For a real Axys/APX site, start by validating which extract shape is available:

| Site extract shape | Contract/test expectation |
| --- | --- |
| IMEX transaction rows include source/destination and special-security context. | Keep ambiguous-flow enforcement on and classify `li`, `lo`, `dp`, and `wd` only with matching conditional YAML rules that inspect those fields. |
| REP/report or custom-report rows include reviewed category and sign semantics. | Keep ambiguous-flow enforcement on and mark the reviewed semantic fields as the blocking context in the site contract; source semantics can mark ambiguous codes as external, neutral, or performance-affecting. |
| IMEX rows expose only transaction code, amount, date, portfolio, and security. | Treat the extract as insufficient for ambiguous external-flow classification; use REP/report/custom-report evidence before running comparison. |

Use this onboarding sequence:

1. Ask the Axys/APX administrator or report owner for a transaction extract column
   list, including source/destination, special-security, and report-semantic
   fields when available.
2. Compare that list to the template profiles:
   - [`site_extract_contract_imex_context.yaml`](../axys_apx/contracts/templates/site_extract_contract_imex_context.yaml)
     when IMEX exposes source/destination and special-security context;
   - [`site_extract_contract_rep_semantics.yaml`](../axys_apx/contracts/templates/site_extract_contract_rep_semantics.yaml)
     when REP or a custom report exposes reviewed category/sign semantics;
   - [`site_extract_contract.yaml`](../axys_apx/contracts/templates/site_extract_contract.yaml)
     as the broad starter before trimming fields to the local extract.
3. Keep only fields the local extract really contains, then reference the
   contract from comparison YAML with `extract_contract.path`.
4. Run `validate_config` before generating reports. A code-only IMEX extract
   with `dp`, `li`, `lo`, or `wd` should fail until richer IMEX context,
   REP/report semantics, or another reviewed source is available.

The small fixtures in `tests/data/axys/site_variants/` pin these shapes. New
demo data and tests should grow from those fixtures until all documented Axys/APX
transaction types have explicit expected treatment, including external-flow,
non-external transfer, sweep, fee/expense, income, corporate-action,
correction/cancellation, and review-only cases.

Use
[`axys_apx/contracts/transaction_semantics_matrix.md`](../axys_apx/contracts/transaction_semantics_matrix.md)
as the human-readable checklist for expanding that coverage; the adjacent YAML
contract remains authoritative.

Corporate actions remain conservative. A split row can be shown as review
evidence, but it should not explain reported performance unless a future
supported rule explicitly implements that behavior.

## What This Contract Excludes

The packaged demo should not attempt to model:

- tax-lot accounting;
- cost-basis methodology;
- full settlement accounting;
- vendor-specific recalculation behavior not visible in the source extracts;
- universal Axys/APX native field names or object names;
- every possible Axys/APX transaction variant.

Those topics can be documented in the Axys/APX reference material, but the
performance-comparison demo should include them only when they directly help
answer why reported performance changed from Snapshot A to Snapshot B.
