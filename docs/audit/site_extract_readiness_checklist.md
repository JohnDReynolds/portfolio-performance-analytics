# Site Extract Readiness Checklist

Use this checklist before running performance comparison against a site-specific
Axys/APX extract. The goal is to confirm the source files expose enough evidence
for Modified Dietz formula roles without turning ppar into a local accounting
system.

## Required Review

- Confirm the comparison YAML points to the intended snapshot directories and
  transaction file.
- Run `validate_config` before generating a report bundle.
- Keep `extract_contract.enforce_ambiguous_axys_flows: true` unless a local
  reviewer has approved a documented opt-out.
- Confirm every transaction code in performance-relevant data has either a
  recognized source category, a recognized transaction code, or explicit
  `transaction_rules`.

## Transaction Code And Context Case

New or reviewed site contracts should declare a positive integer `version` and
may require exact native-case semantics:

```yaml
extract_contract:
  path: site_extract_contract.yaml
  enforce_ambiguous_axys_flows: true
  transaction_semantics_case: exact
```

With `exact`, YAML transaction-rule keys and native `when` values match source
text after whitespace trimming without changing case. `by` and `BY`, or `Desk`
and `DESK`, can therefore have different explicitly reviewed meanings. Built-in
code-only compatibility inference is disabled, so an unmatched uppercase code
remains unknown; uppercase does not mean cancellation. The referenced extract
contract must contain `version: 1` or another positive integer.

Existing comparison YAML that omits `transaction_semantics_case` retains its
established case-insensitive rule behavior. This compatibility default does not
establish a source contract for cancellation or for new site mappings.

## IMEX With Context Fields

Use an IMEX-context contract when the transaction extract includes source fields
that conditional `transaction_rules` can inspect. For ambiguous Axys/APX-style
`li`, `lo`, `ti`, `dp`, and `wd` rows, useful context usually includes:

- transaction security type;
- source/destination type;
- source/destination symbol;
- special-security type;
- special-security symbol.

Having the columns is only the first gate. The row values still need reviewed
conditional rules before ppar treats an ambiguous code as an external flow,
transfer, fee/expense, or performance transaction.

## REP/Report Semantic Fallback

Use a REP/report or custom-report source when IMEX cannot expose enough row
context. The fallback source should carry reviewed semantics directly, such as:

- normalized transaction category;
- cash-flow sign;
- performance-flow sign;
- return-basis treatment for fees or expenses when relevant.

Do not weaken the `li`/`lo`/`ti`/`dp`/`wd` guardrail only because IMEX is sparse.
Prefer a richer local extract that proves the Modified Dietz formula role.

## Optional Deliver-In Original-Cost Review

Sites may opt into `deliver_in_original_cost_incomplete` when a reviewed local
population should carry both an original-cost amount and original-cost date.
The YAML `only` population must name transaction code, security type,
source/destination type, and source/destination symbol. Enabling the check also
requires normalized `original_cost` and `original_cost_date` columns in both
snapshot transaction extracts; aliases such as `ORIG_COST` and
`ORIG_COST_DATE` are accepted.

The check reports absence only. Zero cost counts as supplied, and ppar neither
calculates cost basis nor concludes that Axys/APX substituted trade-date market
value. Keep the check disabled when the extract cannot distinguish a blank
field from an unavailable column. These are optional input fields and do not
add columns to Audit reports.

## Tested Candidate Override Profiles

Some Axys/APX transaction families are plausible but not safe enough for a
future core `vendor: axys` preset. ppar keeps these as tested candidate
override profiles: copy/adapt examples that show how a site can classify rows
after local evidence proves the Modified Dietz role.

These profiles are not universal Axys/APX rules. Code-only rows still stay
`unknown`. Enable a profile only after confirming the same context fields and
business meaning in the site's IMEX, REP, custom-report, or other reviewed
source.

| Profile | Codes | Candidate Modified Dietz treatment | Required local evidence |
| --- | --- | --- | --- |
| `fixed_income_accruals` | `pa`, `sa` | Purchase accrued interest as fee/expense; sale accrued interest as income. The packaged demo includes one paired 91282Y5Y1 example, while this profile remains useful for local override/onboarding variants. | Bond or accrued-interest context, amount sign, settlement context, and local mapping or REP/report treatment. |
| `ai_margin_interest` | `ai` | Margin or negative interest as fee/expense. The packaged demo includes one context-gated example; this profile remains a local onboarding example. | Margin or negative-interest context, amount sign, and local mapping or REP/report treatment. |
| `alternate_fee_context` | `dp` with `epus expense` context | Fee or expense under an explicit local rule. `epus` is not treated as a standalone transaction code. | Token role, fee symbol, amount sign, and local mapping or REP/report treatment. |
| `rc_return_of_capital` | `rc` | Return of capital as performance income. | Return-of-capital context, security identity, amount sign, and local mapping or REP/report treatment. |
| `pd_principal_paydown` | `pd` | Principal paydown as performance income. | Bond/MBS/ABS principal-paydown context, cash movement, amount sign, and local mapping or REP/report treatment. |
| `short_side_trades` | `ss`, `cs` | Short sale as performance sell; cover short as performance buy. | Short/security type, cash or margin context, amount/quantity signs, and local mapping or REP/report treatment. |

Cost basis, principal, factor, and amortization details may make demo data or
site data look more internally realistic. They are best-efforts construction
context for these profiles, not requirements for ppar's Modified Dietz
calculation or explanation unless a future formula surface explicitly uses
those fields.

## Code-Only Failure Mode

A code-only extract is expected to fail when it contains ambiguous Axys/APX flow
codes and lacks the blocking context columns named in the extract contract. This
is a useful source-quality signal. Fix the extract, add a REP/report fallback,
or document a reviewed local opt-out before treating the rows as performance
inputs.

## Reviewed Local Opt-Out

The local opt-out exists for reviewed site workflows that already know how to
interpret code-only ambiguous rows outside ppar's extract contract. It should be
rare and explicit:

```yaml
extract_contract:
  enforce_ambiguous_axys_flows: false
```

When this setting is false, conditional context-column guards do not protect
`li`, `lo`, `ti`, `dp`, and `wd` rows. The site's YAML rules and review process carry
the classification risk.

## Handoff Evidence

Generated report bundles include `manifest.json`. When the bundle writer knows
the comparison YAML path, the manifest records:

- comparison YAML path;
- extract-contract path;
- ambiguous-flow enforcement status;
- required transaction context columns;
- observed transaction codes;
- transaction semantics-source counts.

Use that manifest with the README and CSV artifacts when handing a review bundle
to another reviewer.
