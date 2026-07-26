# Axys Site Variant Fixtures

These fixtures model small site-specific extract shapes used by contract tests.
They are not native Axys schemas. Each variant focuses on ambiguous Axys
transaction codes whose performance treatment cannot be trusted from code alone.
Candidate override profiles prove copy/adapt YAML examples for local
onboarding; they do not promote the transaction code into a packaged Axys demo
default or future versioned Axys/APX Audit profile.

| Variant | Purpose |
| --- | --- |
| `alternate_fee_context` | A `dp` row with `epus expense` special-security context classifies as a fee only through an explicit site rule; `epus` is not promoted as a standalone transaction code. |
| `ai_margin_interest` | The documented `caus` / `margin` special-security pair plus `$pth` / `$cash` context qualifies a negative `ai` row only through an explicit YAML rule; code-only treatment remains unknown. |
| `exact_case_rules` | A versioned site contract gives `by` and `BY` distinct, explicitly configured economic meanings and requires exact native context values; uppercase is not treated as cancellation. |
| `fixed_income_accruals` | Bond accrued-interest rows classify `pa` and `sa` only through explicit YAML rules; code-only treatment remains unknown. |
| `imex_context` | IMEX-style transaction rows include source/destination and special-security context, so conditional YAML can classify `li`, `lo`, `dp`, and `wd`. |
| `imex_code_only` | Code-only IMEX-style `li`/`lo`/`dp`/`ti`/`wd` rows intentionally omit the required context and must fail before YAML can classify ambiguous external flows. |
| `local_opt_out` | Code-only ambiguous rows classify only because `enforce_ambiguous_axys_flows` is explicitly disabled; this models reviewed local-risk behavior, not the default path. |
| `pd_principal_paydown` | Fixed-income `pd` rows use the documented `$pty` / `$cash` context and blank special-security fields; principal/cost mechanics remain outside this Modified Dietz fixture. |
| `rc_return_of_capital` | Equity `rc` rows use the documented `$pty` / `$cash` context and blank special-security fields; cost-basis treatment remains outside this Modified Dietz fixture. |
| `rep_semantics` | REP/report-style rows carry already-reviewed category and sign semantics, so a site contract can use those fields as the blocking classification context. |
| `review_only_actions` | REP/report-style rows mark correction/reversal-like rows and a synthetic corporate-action marker as neutral review evidence, not formula inputs. |
| `short_side_trades` | Short-security rows classify lowercase `ss` and `cs` as performance sell/buy rows only through explicit YAML rules; uppercase cancellation evidence remains separate. |

The `imex_context` and `rep_semantics` contract files intentionally match the
documented onboarding profiles in
`docs/axys_apx/contracts/templates/site_extract_contract_imex_context.yaml` and
`docs/axys_apx/contracts/templates/site_extract_contract_rep_semantics.yaml`.

For all candidate override profiles, cost basis, principal, factor, and
amortization details are best-efforts demo-construction context unless a future
formula surface explicitly uses those fields. The tests in this directory focus
on Modified Dietz transaction category and sign/flow semantics.
