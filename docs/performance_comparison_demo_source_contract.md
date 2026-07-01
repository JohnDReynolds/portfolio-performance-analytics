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

## Governing References

Use the Axys/APX reference chapters as evidence boundaries:

- [`Chapter_05_Transactions.md`](axys-apx-reference/Chapter_05_Transactions.md)
- [`Chapter_06_Holdings.md`](axys-apx-reference/Chapter_06_Holdings.md)
- [`Chapter_07_Cash.md`](axys-apx-reference/Chapter_07_Cash.md)
- [`Chapter_10_Performance.md`](axys-apx-reference/Chapter_10_Performance.md)
- [`Chapter_15_Data_Dictionary.md`](axys-apx-reference/Chapter_15_Data_Dictionary.md)
- [`Appendix_Demo_Extract_Availability.md`](axys-apx-reference/Appendix_Demo_Extract_Availability.md)
  for field-by-field IMEX/REP availability confidence.
- [`Appendix_Transaction_Semantics_Matrix.md`](axys-apx-reference/Appendix_Transaction_Semantics_Matrix.md)
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
| `security_master.csv` | Normalized security-reference context. | Security-reference changes are context unless a future supported rule makes a field performance-relevant. |

## Field Role Contract

| Source field family | Performance-comparison role |
| --- | --- |
| `holdings.market_value` | Performance input for beginning and ending value. |
| `holdings.accrued` | Performance input when configured as a valuation/accrual amount. |
| `transactions.amount` | Performance input for external flows, security-level buy/sell flows, income, fee, and expense examples when YAML classifies the transaction code and any required transaction context. |
| `transactions.security_type`, `transactions.source_destination_type`, `transactions.source_destination_symbol`, `transactions.special_security_type`, `transactions.special_security_symbol` | Context used by conditional YAML transaction rules. These fields help classify ambiguous Axys-style transaction codes; they are not themselves performance-cause fields. |
| `holdings.quantity`, `holdings.price` | Supporting inputs for changed holdings value. |
| `transactions.quantity`, `transactions.price`, `transactions.commission` | Supporting inputs for changed `transactions.amount`; not a formula for `transactions.amount`. |
| `holdings.cost` | Other data difference. Cost changes are useful review evidence but do not explain reported performance in this demo contract. |
| `transactions.settle_date` | Other data difference unless a future explicit settlement-date rule makes it performance-relevant. |
| security-reference fields | Other data differences or suppressed context unless explicitly supported by a future rule. |

The packaged Axys demo deliberately omits `transactions.transaction_id`.
Stable transaction IDs are supported by ppar and remain the strongest matching
path when a site can provide them, but the local Axys/APX research corpus does
not prove a durable native transaction identifier as typical Axys REP/IMEX
output. The packaged demo therefore exercises the conservative no-ID path.

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
| `dp` | `fee_expense` | Fee-like debit when special-security context confirms the packaged demo fee case. |
| `li` | `external_flow` or `transfer` | Defensive rule for Axys-style long-in examples; external party context is required before treating it as an external flow. |
| `lo` | `external_flow` or `transfer` | Defensive rule for Axys-style long-out examples; external party context is required before treating it as an external flow. |
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

Ambiguous Axys-style codes are deliberately stricter. The packaged demo does
not infer external-flow treatment from `li`, `lo`, `dp`, or `wd` by code alone.
If an IMEX export omits the source/destination or special-security fields needed
to match the YAML rule, the transaction loader stops before applying broad YAML
classification. A richer REP/report extract, custom report, or local-discovery
source is the preferred next design option when IMEX cannot provide that
context.

Having the context columns is necessary but not sufficient. The row values must
match a reviewed conditional rule, or the source must provide reviewed
category/sign semantics, before ppar treats an ambiguous Axys code as an
external flow, transfer, fee/expense, or performance transaction.

The packaged contribution is modeled as a new inserted transaction scenario,
not as a numeric mutation of an unrelated existing transaction. It uses an
Axys-style `li` row on `CASH_USD` with `SRC_DEST_TYPE=$pty`,
`SRC_DEST_SYMBOL=$cash`, positive `AMOUNT`, zero quantity/price/commission, and
explicit YAML semantics that classify it as an external capital inflow from
that context.

## Fixed-Income Transaction Boundary

The packaged demo currently uses two proved fixed-income Modified Dietz inputs:

- `in` transaction rows for ordinary bond or cash interest that is treated as
  performance income; and
- `holdings.accrued` changes as valuation/performance evidence when the
  comparison YAML includes accrued interest as a return input.

That boundary is deliberate. Modified Dietz needs beginning value, ending
value, and dated external cash flows. It does not need ppar to rebuild an
amortization/accretion engine, bond principal schedule, yield calculation, or
tax-lot ledger before ordinary interest and configured accrued value can be
used as formula inputs.

The packaged demo deliberately does not use `ai`, `pa`, `sa`, or `pd` rows yet.
Those codes need more than the Axys short code before ppar can classify them:
bond/accrual context, cash offset, amount sign, principal or quantity movement,
and local mapping or REP/report evidence. Until those fixtures exist, `ai`,
`pa`, and `sa` remain accrued-interest backlog cases, and `pd` remains a
principal-paydown boundary case rather than ordinary income.

This is a data-quality guard, not a claim that those Axys transaction types are
unsupported forever. They should enter test-only data first. A packaged example
should wait until holdings, accrued interest, cash, security performance,
portfolio performance, and reviewer-facing report output all derive from one
coherent fixed-income story.

The default runtime guard uses
`ppar/demos/data/axys/demo_extract_availability.yaml`. A site comparison YAML
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
[`axys-apx-reference/templates/site_extract_contract.yaml`](axys-apx-reference/templates/site_extract_contract.yaml)
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

For a real Axys site, start by validating which extract shape is available:

| Site extract shape | Contract/test expectation |
| --- | --- |
| IMEX transaction rows include source/destination and special-security context. | Keep ambiguous-flow enforcement on and classify `li`, `lo`, `dp`, and `wd` only with matching conditional YAML rules that inspect those fields. |
| REP/report or custom-report rows include reviewed category and sign semantics. | Keep ambiguous-flow enforcement on and mark the reviewed semantic fields as the blocking context in the site contract; source semantics can mark ambiguous codes as external, neutral, or performance-affecting. |
| IMEX rows expose only transaction code, amount, date, portfolio, and security. | Treat the extract as insufficient for ambiguous external-flow classification; use REP/report/custom-report evidence before running comparison. |

Use this onboarding sequence:

1. Ask the Axys administrator or report owner for a transaction extract column
   list, including source/destination, special-security, and report-semantic
   fields when available.
2. Compare that list to the template profiles:
   - [`site_extract_contract_imex_context.yaml`](axys-apx-reference/templates/site_extract_contract_imex_context.yaml)
     when IMEX exposes source/destination and special-security context;
   - [`site_extract_contract_rep_semantics.yaml`](axys-apx-reference/templates/site_extract_contract_rep_semantics.yaml)
     when REP or a custom report exposes reviewed category/sign semantics;
   - [`site_extract_contract.yaml`](axys-apx-reference/templates/site_extract_contract.yaml)
     as the broad starter before trimming fields to the local extract.
3. Keep only fields the local extract really contains, then reference the
   contract from comparison YAML with `extract_contract.path`.
4. Run `validate_config` before generating reports. A code-only IMEX extract
   with `dp`, `li`, `lo`, or `wd` should fail until richer IMEX context,
   REP/report semantics, or another reviewed source is available.

The small fixtures in `tests/data/axys/site_variants/` pin these shapes. New
demo data and tests should grow from those fixtures until all documented Axys
transaction types have explicit expected treatment, including external-flow,
non-external transfer, sweep, fee/expense, income, corporate-action,
correction/cancellation, and review-only cases.

Use
[`axys-apx-reference/Appendix_Transaction_Semantics_Matrix.md`](axys-apx-reference/Appendix_Transaction_Semantics_Matrix.md)
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
