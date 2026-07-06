# Performance Comparison Demo Source Contract

This contract defines how the packaged performance-comparison demo source files
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

- [`Chapter_05_Transactions.md`](axys-apx-reference/reference/Chapter_05_Transactions.md)
- [`Chapter_06_Holdings.md`](axys-apx-reference/reference/Chapter_06_Holdings.md)
- [`Chapter_07_Cash.md`](axys-apx-reference/reference/Chapter_07_Cash.md)
- [`Chapter_10_Performance.md`](axys-apx-reference/reference/Chapter_10_Performance.md)
- [`Chapter_15_Data_Dictionary.md`](axys-apx-reference/reference/Chapter_15_Data_Dictionary.md)
- [`contracts/demo_extract_availability.md`](axys-apx-reference/contracts/demo_extract_availability.md)
  for field-by-field IMEX/REP availability confidence.
- [`contracts/transaction_semantics_matrix.md`](axys-apx-reference/contracts/transaction_semantics_matrix.md)
  for implementation-facing transaction-code treatment, required evidence, and
  coverage status.
- [Boundary snapshot](performance_comparison_transaction_boundary_snapshot.md)
  for the current reviewer-facing coverage and backlog summary.
- [Evidence-pack review](performance_comparison_evidence_pack_review.md)
  for the current commit-preparation inventory.

When those references strongly imply common transaction-code meaning, the demo
may use that evidence. When the references mark native storage, exact IMEX object
names, or vendor calculation behavior as Unknown, this demo must not present
those details as verified Axys/APX facts.

## Current Packaged Files

| Demo file | Demo role | Contract boundary |
| --- | --- | --- |
| `transactions.csv` | Normalized transaction extract used to explain changed cash flows, security-level flows, income, fees, and review-only transaction evidence. | Transaction codes should use Axys/APX-style observed codes when the reference material supports them. YAML remains the explicit interpretation contract. |
| `holdings.csv` | Normalized point-in-time holdings and valuation extract. | Native Axys/APX holding storage is Unknown. This file represents the demo value source used for beginning and ending holdings. |
| `portperf.csv` | Reported portfolio-period performance target used for comparison. | `portperf` is a normalized demo dataset name, not a verified native Axys/APX object name. |
| `secperf.csv` | Reported security-period performance target used for comparison. | `secperf` is a normalized demo dataset name, not a verified native Axys/APX object name. |

## Field Boundary Taxonomy

The packaged demo keeps four field groups separate:

- **Mandatory product inputs**: fields ppar needs to compare Modified Dietz
  performance, such as period dates, beginning/ending market value, return,
  holdings value, and transaction amount when transaction rules classify the
  row.
- **Realistic packaged-demo fields**: Axys/APX-style or report-style fields that
  are defensible in the local research corpus and useful for the packaged
  review story, such as transaction dates, security identifiers, transaction
  codes, quantity, price, commission, accrued interest, and portfolio or
  security performance values.
- **Optional local-enrichment fields**: fields ppar can use when a client site
  can provide them, but that the packaged demo does not assume as typical Axys/APX
  output. Stable transaction IDs are the main example.
- **Internal scenario/rebuild fields**: deterministic fixture handles used by
  demo-generation scripts. These may include `TRANSACTION_ID`, but they must
  not leak into the user-facing packaged Axys/APX transaction CSVs.

## Minimum Source-Data Contract

The comparison YAML must identify the source datasets ppar is allowed to read
from both Snapshot A and Snapshot B. The selected comparison level determines
the target performance dataset:

| Dataset | Required when | Required normalized columns |
| --- | --- | --- |
| `portfolio_performance` | `comparison.level` is `portfolio`. | `portfolio_id`, `from_date`, `thru_date`, `portfolio_return` |
| `security_performance` | `comparison.level` is `security`, or `security_return_reconstruction` is configured. | `portfolio_id`, `security_id`, `from_date`, `thru_date`, `security_return` |
| `holdings` | Portfolio or security return reconstruction is configured, or holding fields are used as performance explanations. | `portfolio_id`, `security_id`, `holding_date` |
| `transactions` | Portfolio or security return reconstruction is configured, or transaction fields are used as performance explanations. | `portfolio_id`, `security_id`, `transaction_date` |
| `cash` | Cash rows are used as source-data evidence. | `portfolio_id`, `cash_date` |
| `fx_rates` | FX-rate rows are used as source-data evidence. | `from_currency`, `to_currency`, `rate_date`, `fx_rate` |

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

| Source field family | Performance-comparison role |
| --- | --- |
| `holdings.market_value` | Performance input for beginning and ending value. |
| `holdings.accrued` | Performance input when configured as a valuation/accrual amount. |
| `transactions.amount` | Performance input for external flows, security-level buy/sell flows, income, fee, and expense examples when YAML classifies the transaction code and any required transaction context. |
| `transactions.security_type`, `transactions.source_destination_type`, `transactions.source_destination_symbol`, `transactions.special_security_type`, `transactions.special_security_symbol` | Context used by conditional YAML transaction rules. These fields help classify ambiguous Axys/APX-style transaction codes; they are not themselves performance-cause fields. |
| `holdings.quantity`, `holdings.price` | Supporting inputs for changed holdings value. |
| `transactions.quantity`, `transactions.price`, `transactions.commission` | Supporting inputs for changed `transactions.amount`; not a formula for `transactions.amount`. |
| `portperf.gain_loss`, `secperf.gain_loss` | Reported performance-extract context. These fields are defensible as report-style gain/loss components, but the local corpus does not prove native Axys/APX IMEX performance object names or calculation basis. They must not be described as recomputed tax-lot or accounting-ledger values. |
| `transactions.settle_date` | Review evidence unless a future explicit settlement-date rule makes it performance-relevant. |

The packaged Axys/APX demo deliberately omits `transactions.transaction_id`.
Stable transaction IDs are supported by ppar and remain the strongest matching
path when a site can provide them, but the local Axys/APX research corpus does
not prove a durable native transaction identifier as typical Axys/APX REP/IMEX
output. The packaged demo therefore exercises the conservative no-ID path.

## Axys/APX Demo Completion Gate

The packaged Axys/APX demo is the accepted future seed for an Axys/APX vendor YAML
preset. The gate below records the standard used to freeze the demo without
turning it into hidden runtime product policy too early.

The demo can be considered complete enough to seed `vendor: axys` only when:

- packaged CSV fields are limited to mandatory product inputs, realistic
  packaged-demo fields, and documented local-enrichment examples;
- internal scenario/rebuild fields, including fixture transaction identifiers,
  do not appear in user-facing packaged CSVs;
- the comparison YAML has stable transaction semantics, field-impact methods,
  return-reconstruction settings, and extract-contract behavior;
- every visible changed source-data field in default reports has additive,
  evidence-only, or suppression treatment after YAML validation;
- the portfolio and security demos both include intentional Fully Explained,
  Partly Explained, and Unexplained examples with reviewer-facing wording that
  tells a coherent Modified Dietz story;
- ambiguous Axys/APX-style `dp`, `li`, `lo`, and `wd` handling still requires
  reviewed context or a documented local extract contract;
- current report bundles, manifests, review summaries, and package-resource
  boundaries validate from a source checkout and installed-package style
  resource path; and
- report HTML content is intentionally changed only when the reviewer story
  changes, and those changes are called out in phase summaries.

This gate has passed for the packaged Axys/APX demo scope. Vendor presets still
remain design-only until implementation. A future preset must expand to
inspectable resolved YAML, preserve site overrides, and document that
`vendor: axys` means "ppar's versioned Axys/APX preset semantics," not universal
Axys/APX behavior.

## Axys/APX Demo Freeze Decision Packet

Freezing the packaged Axys/APX demo as a future preset seed was a product decision,
not just a validation result. The packaged demo is accepted as the future
`vendor: axys` seed with these boundaries as versioned preset semantics:

- the packaged transaction families are `by`, `sl`, `dv`, `in`,
  fixed-income accrued-interest `pa`/`sa`, fee-like `dp`, external-cash `li`,
  external-cash `lo`, and external-cash `wd`;
- `;` remains a guardrail-only YAML rule until a realistic packaged story
  justifies promoting it;
- ambiguous Axys/APX-style `dp`, `li`, `lo`, and `wd` rows remain context-gated and
  must not be classified from transaction code alone;
- packaged transaction rows intentionally omit stable transaction identifiers
  and exercise conservative no-ID matching;
- the fee-like `dp` example assumes net-of-fees reported performance;
- the portfolio and security reports intentionally include Fully Explained,
  Partly Explained, and Unexplained examples;
- settlement-date and unsupported corporate-action differences remain review
  evidence unless a future explicit rule changes their treatment; and
- `vendor: axys` must continue to mean ppar's versioned Axys/APX preset semantics,
  not a claim about universal Axys/APX behavior.

Freeze-packet evidence map:

| Accepted boundary | Current evidence |
| --- | --- |
| Packaged transaction families are `by`, `sl`, `dv`, `in`, fixed-income accrued-interest `pa`/`sa`, fee-like `dp`, external-cash `li`, external-cash `lo`, and external-cash `wd`. | `snapshot_a/transactions.csv`, `snapshot_b/transactions.csv`, packaged Axys/APX README coverage table, comparison YAML transaction-rule comments, fixed-income boundary helpers, and demo data audit tests. |
| `;` remains a guardrail-only YAML rule. | Packaged transaction CSVs contain no `;` rows; the comparison YAML keeps the rule as defensive/reserved semantics; site-variant and matrix tests cover non-packaged corporate-action behavior. |
| Ambiguous `dp`, `li`, `lo`, and `wd` rows remain context-gated. | Packaged transaction CSVs include source/destination and special-security context columns; the packaged extract contract requires those context fields; `validate_config` reports ambiguous-flow enforcement as enabled. |
| Packaged transaction rows omit stable transaction identifiers. | Packaged transaction CSV headers omit `TRANSACTION_ID`; the README and field boundary taxonomy document stable IDs as optional local-enrichment fields. |
| Fee-like `dp` assumes net-of-fees reported performance. | The comparison YAML and packaged README both state the net-of-fees assumption and warn that gross-of-fees performance needs a separate return-basis policy. |
| Report examples intentionally include Fully Explained, Partly Explained, and Unexplained cases. | The packaged README describes the controlled restatement story; generated portfolio and security report bundles are rebuilt and validated by `scripts/check_performance_comparison_demo_health.py`. |
| Settlement-date and unsupported corporate-action differences remain review evidence. | The field-role contract classifies `transactions.settle_date` as review evidence, while the transaction semantics matrix and demo matrix tests keep unsupported corporate actions outside additive Modified Dietz treatment. |
| `vendor: axys` means ppar's versioned Axys/APX preset semantics, not universal Axys/APX behavior. | The source contract, roadmap, and vendor preset design docs use this boundary and keep preset implementation blocked until the demo is explicitly frozen. |

If any of these boundaries changes later, the correct action is to update the
packaged demo, source contract, YAML, and report story before changing the
preset semantics.

## Cash-Balance Policy

`CASH_USD` is a normalized demo cash-balance holding. It should not be described
as native Axys/APX cash storage. When a transaction amount affects cash but the
source row does not prove the exact cash security, user-facing explanations may
refer to the changed `holdings` cash-balance.

The packaged demo currently uses one USD cash-balance holding. A future
multi-currency demo should make cash-account mapping explicit before assigning
transaction effects to a specific cash security.

## Date Policy

The demo treats `FROM_DATE` and `THRU_DATE` as inclusive. The beginning holdings
date for a period is the calendar day immediately before `FROM_DATE`; the ending
holdings date is `THRU_DATE`.

For transaction rows, `TRANSACTION_DATE` is the economic as-of date used by the
demo return-reconstruction rules. It represents the trade date or ex-date for
the event. `SETTLE_DATE` or pay date can be useful context, but it is not the
default performance date in this demo contract.

## Transaction-Code Policy

Transaction semantics are required YAML. A transaction code may appear in source
files only when the comparison YAML has a matching `transaction_rules` entry.
Missing transaction rules are a hard stop for the packaged demo.

The source transaction code and the normalized transaction category are distinct:

| Source code | YAML category | Demo meaning |
| --- | --- | --- |
| `by` | `buy` | Security purchase. |
| `sl` | `sell` | Security sale. |
| `dv` | `income` | Dividend or dividend-like income. |
| `in` | `income` | Interest or income-like receipt. |
| `pa` | `fee_expense` | Purchase accrued interest when fixed-income context confirms the packaged paired-trade case. |
| `sa` | `income` | Sale accrued interest when fixed-income context confirms the packaged paired-trade case. |
| `dp` | `fee_expense` | Fee-like debit when special-security context confirms the packaged demo fee case. |
| `li` | `external_flow` or `transfer` | Defensive rule for Axys/APX-style long-in examples; external party context is required before treating it as an external flow. |
| `lo` | `external_flow` or `transfer` | Defensive rule for Axys/APX-style long-out examples; external party context is required before treating it as an external flow. |
| `wd` | `external_flow` or `transfer` | External withdrawal only when cash security and source/destination context confirm the packaged demo cash-withdrawal case. |
| `;` | `corporate_action` | Reserved for future split/journal/other-style evidence; user-facing split examples should use real historical corporate actions. |

Reviewer-facing explanations should preserve the source code as it appears in
the source file. For example, a changed buy transaction should display `by:`,
not `BY:` or `buy:`.

The comparison code may normalize transaction-code strings only for explicit
semantic classification, such as matching YAML `transaction_rules`, checking
whether a code belongs to a reviewed boundary family, or deriving a normalized
`Transaction Category`. That normalization is not a general equality rule for
source identifiers, and it must not rewrite reviewer handoff artifacts such as
`observed_codes`, workbook explanations, or audit-trail transaction-code values.

The Axys/APX transaction reference documents support using short Axys/APX-style
codes for common examples such as buys, sells, dividends, interest, fees, and
withdrawals where the evidence is strong enough for demo purposes. Code alone is
not treated as a complete accounting system. YAML supplies the role ppar is
allowed to use for performance comparison.

Ambiguous Axys/APX-style codes are deliberately stricter. The packaged demo does
not infer external-flow treatment from `li`, `lo`, `dp`, or `wd` by code alone.
If an IMEX export omits the source/destination or special-security fields needed
to match the YAML rule, the transaction loader stops before applying broad YAML
classification. A richer REP/report extract, custom report, or local-discovery
source is the preferred next design option when IMEX cannot provide that
context.

Having the context columns is necessary but not sufficient. The row values must
match a reviewed conditional rule, or the source must provide reviewed
category/sign semantics, before ppar treats an ambiguous Axys/APX code as an
external flow, transfer, fee/expense, or performance transaction.

The packaged contribution and deliver-out examples are modeled as new inserted
transaction scenarios, not as numeric mutations of unrelated existing
transactions. The contribution uses an Axys/APX-style `li` row on `CASH_USD` with
`SRC_DEST_TYPE=$pty`, `SRC_DEST_SYMBOL=$cash`, positive `AMOUNT`, zero
quantity/price/commission, and explicit YAML semantics that classify it as an
external capital inflow from that context. The deliver-out uses the same
context standard with an Axys/APX-style `lo` row and negative `AMOUNT`, so it proves
the opposite external-flow direction without treating `lo` as safe by code
alone.

## Fixed-Income Transaction Boundary

The packaged demo currently uses three proved fixed-income Modified Dietz input
families:

- `in` transaction rows for ordinary bond or cash interest that is treated as
  performance income; and
- `holdings.accrued` changes as valuation/performance evidence when the
  comparison YAML includes accrued interest as a return input; and
- context-gated `pa`/`sa` transaction rows as fixed-income accrued-interest
  adjuncts paired with TNOTE5Y buy/sell transactions.

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

The packaged demo still does not use `ai` or `pd` rows. Those codes need more
than the Axys/APX short code before ppar can classify them: margin or bond/principal
context, cash movement, amount sign, and local mapping or REP/report evidence.
Test-only candidate override profiles cover explicit Modified Dietz treatments
for `ai`, `pa`, `sa`, and `pd`, but only `pa`/`sa` have been promoted into the
packaged demo, and only for the paired fixed-income accrued-interest story.

The default runtime guard uses
`ppar/setup_templates/axysapx_performance_comparison/demo_extract_availability.yaml`. A site comparison YAML
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
[`axys-apx-reference/contracts/templates/site_extract_contract.yaml`](axys-apx-reference/contracts/templates/site_extract_contract.yaml)
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
   - [`site_extract_contract_imex_context.yaml`](axys-apx-reference/contracts/templates/site_extract_contract_imex_context.yaml)
     when IMEX exposes source/destination and special-security context;
   - [`site_extract_contract_rep_semantics.yaml`](axys-apx-reference/contracts/templates/site_extract_contract_rep_semantics.yaml)
     when REP or a custom report exposes reviewed category/sign semantics;
   - [`site_extract_contract.yaml`](axys-apx-reference/contracts/templates/site_extract_contract.yaml)
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
[`axys-apx-reference/contracts/transaction_semantics_matrix.md`](axys-apx-reference/contracts/transaction_semantics_matrix.md)
as the checklist for expanding that coverage. Use
[boundary snapshot](performance_comparison_transaction_boundary_snapshot.md)
as the compact release-readiness view of the same boundary.

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
