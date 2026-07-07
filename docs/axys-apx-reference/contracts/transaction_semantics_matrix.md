# Transaction Semantics Matrix Contract

This contract is the implementation-facing transaction matrix for ppar Axys/APX
demo and test work. It translates the observed transaction-code evidence in the
reference chapters into conservative performance-comparison expectations.

It is not an official Axys or APX transaction-code manual. The local reference
corpus repeatedly shows that code-only interpretation is unsafe. Treat this file
as a seed contract for demo data, tests, site extract contracts, and local
mapping review.

The machine-readable companion is
[`transaction_semantics_matrix.yaml`](transaction_semantics_matrix.yaml). Tests use
that YAML to check demo transaction rules and fixture coverage.

## Document Role

Use this Markdown file as a human-readable companion to the YAML contract. Do
not expand it into a second evidence archive or a parallel reader-facing
transaction-code manual. The intended ownership is:

- `../reference/Chapter_05_Transactions.md` holds the reader-facing transaction
  semantics, confidence labels, and open questions.
- `../evidence/Research_05_Transactions.md` preserves source evidence and
  research history.
- `transaction_semantics_matrix.yaml` is the implementation contract used by
  tests, demo fixtures, and validation logic.
- This file explains the YAML at a practical level for maintainers.

When this file needs more explanation than a compact contract can comfortably
hold, put the evidence and reasoning in the research archive and summarize the
reader-facing conclusion in Chapter 05.

## Evidence Boundary

Primary local references:

- [`Chapter_05_Transactions.md`](../reference/Chapter_05_Transactions.md)
- [`evidence/Research_05_Transactions.md`](../evidence/Research_05_Transactions.md)
- [`Chapter_07_Cash.md`](../reference/Chapter_07_Cash.md)
- [`evidence/Research_07_Cash.md`](../evidence/Research_07_Cash.md)
- [`demo_extract_availability.md`](demo_extract_availability.md)
- [`performance_comparison_demo_source_contract.md`](../../performance_comparison_demo_source_contract.md)

Important boundary rules:

- A code can be observed without being proven as a universal native Axys/APX
  code.
- `li`, `lo`, `dp`, and `wd` require context before ppar may treat them as
  external flows or non-external flows.
- Uppercase cancellation/delete behavior is observed in integration workflows,
  but native universality remains Unknown.
- `epus` and `exus` are observed fee/expense classification tokens, but sources
  disagree on whether they are transaction codes, labels, security types, or
  special-security terms.
- Preserve native transaction-code and security-identifier case. Some observed
  integration translation logic is case-insensitive, but ppar matching and
  equality checks should not become case-insensitive unless an explicit site
  mapping says so.

## Field Meanings

| Field | Meaning |
| --- | --- |
| `ppar category` | Normalized transaction category used by performance comparison. |
| `Cash sign` | Direction of cash/economic movement from the portfolio perspective when known. |
| `Performance flow` | Whether the row is an external capital flow, a return/performance event, neutral transfer, correction, or unknown. |
| `Required evidence` | Minimum source evidence needed before ppar should assign the treatment. |
| `IMEX confidence` | Confidence that a practical IMEX-style extract can provide enough fields for this treatment. |
| `REP confidence` | Confidence that a REP/report/custom-report extract can provide enough fields for this treatment. |
| `Coverage status` | Primary ppar demo/test status bucket. It may summarize the strongest or most important coverage path rather than every fixture. |
| `fixtures` | Machine-readable YAML list of packaged-demo or site-variant fixtures where the code is covered. Use this for exact coverage locations. |
| `coverage_notes` | Machine-readable YAML rationale explaining the fixture coverage, backlog boundary, or context-only treatment. |

## Core Observed Code Matrix

| Code / token | Observed meaning | Default ppar category | Cash sign | Performance flow | Required evidence | IMEX confidence | REP confidence | Coverage status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `by` | Buy; reinvestment buy leg; non-cash deposit paired leg. | `buy` | Negative for normal purchase; context-dependent for paired/reinvestment rows. | `performance` for security purchase; context-dependent for reinvestment pairs. | Code, security, amount/quantity signs; reinvestment context if paired. | Medium | Medium | Covered in packaged demo and tests for normal buy. | Observed in Axys/APX integration evidence. |
| `sl` | Sell; positive closure leg. | `sell` | Positive for normal sale. | `performance` | Code, security, amount/quantity signs. | Medium | Medium | Covered in packaged demo and tests for normal sell. | Observed in APX and integration evidence; Axys native status remains not fully verified. |
| `ss` | Short sale. | `sell` or `unknown pending review` | Positive or collateral-like, depending on short-account treatment. | `performance` or `unknown` | Short/security type, cash/margin/short symbols, amount/quantity signs. | Low to Medium | Medium | Partially covered by a test-only site variant. | `site_variants/short_side_trades` proves explicit short-security YAML classification as a performance sell; code-only treatment remains unknown and packaged-demo use still needs a coherent short-account scenario. |
| `cs` | Cover short. | `buy` or `unknown pending review` | Negative or collateral-like, depending on short-account treatment. | `performance` or `unknown` | Short/security type, cash/margin/short symbols, amount/quantity signs. | Low to Medium | Medium | Partially covered by a test-only site variant. | `site_variants/short_side_trades` proves explicit short-security YAML classification as a performance buy; code-only treatment remains unknown, and uppercase `CS` cancellation evidence stays separate. |
| `dv` | Dividend, dividend-paying security income, reinvested-dividend leg. | `income` | Positive unless withholding/negative-dividend context says otherwise. | `performance` | Code, security/income context, amount sign, reinvestment pair context when present. | Medium | Medium | Covered in packaged demo and tests for normal income. | Reinvestment may require paired `dv` and `by` rows. |
| `in` | Income, interest, cash-security dividend/income. | `income` | Positive unless negative-interest context says otherwise. | `performance` | Code, security type, amount sign, income/accrual context. | Medium | Medium | Covered in packaged demo and tests for normal interest/income. | Cash-security income and bond income need separate examples. |
| `ai` | Negative interest or margin interest. | `fee_expense` or `income` pending local mapping. | Usually negative for margin/negative interest; context-dependent. | `performance` | Security type, margin symbols, amount sign, local mapping or REP/report semantics. | Medium | Medium | Partially covered by a test-only site variant. | `site_variants/ai_margin_interest` proves explicit margin-context YAML classification; code-only treatment remains unknown and packaged-demo use still needs local mapping or REP/report evidence. |
| `sa` | Sale accrued interest / sell-side accrued interest. | `income` or `fee_expense` pending local mapping. | Positive or context-dependent. | `performance` or `unknown` | Bond/accrual context, accrued-interest amount, settlement date, local mapping or REP/report semantics. | Medium | Medium | Partially covered by packaged demo and a test-only site variant. | Packaged demo includes sale accrued interest only with explicit fixed-income context and paired bond-trade scenario support; code-only treatment remains unknown. |
| `pa` | Purchase accrued interest / buy-side accrued interest. | `fee_expense` or `income` pending local mapping. | Negative or context-dependent. | `performance` or `unknown` | Bond/accrual context, accrued-interest amount, settlement date, local mapping or REP/report semantics. | Medium | Medium | Partially covered by packaged demo and a test-only site variant. | Packaged demo includes purchase accrued interest only with explicit fixed-income context and paired bond-trade scenario support; code-only treatment remains unknown. |
| `rc` | Return of capital; ByAllAccounts Axys/APX mapping evidence points to portfolio-cash destination context. | `income` or `corporate_action` pending local policy. | Usually positive cash; cost-basis handling is separate from the Modified Dietz role. | `performance` or `review_only` depending on configured methodology. | Code, security, amount sign, cash destination, return-of-capital context, and local mapping or report treatment. | Medium | Medium | Partially covered by packaged demo and a test-only site variant. | Packaged demo covers `rc` only with explicit equity/security and return-of-capital context; code-only treatment remains unknown and cost-basis treatment stays best-efforts demo-construction context. |
| `pd` | Principal paydown / bond-security return-of-capital transaction; ByAllAccounts mapping evidence points to portfolio-cash destination context. | `sell`-like security flow or `corporate_action` pending local policy. | Usually positive cash; principal/cost mechanics are separate from the Modified Dietz role. | `performance` or `review_only` depending on configured methodology. | Bond/MBS/ABS security type, principal/paydown context, cash movement, local mapping or REP/report treatment. | Medium | Medium | Partially covered by packaged demo and a test-only site variant. | Packaged demo covers `pd` only with MBS/amortizing-security, principal-paydown, and portfolio-cash destination context, using sell-like security-flow mechanics for Modified Dietz; code-only treatment remains unknown. Principal, factor, cost-basis, and amortization handling stay best-efforts demo-construction context. |
| `li` | Deliver in, transfer in, credit, deposit, positive movement. | `external_flow` or `transfer` | Positive when external inflow; none/neutral for internal transfer. | `external` or `neutral` | Source/destination type and symbol, security type, amount/quantity signs, local mapping or reviewed REP semantics. | Medium when context fields available; insufficient when code-only. | Medium to High | Covered in packaged demo and site-variant tests for external and transfer cases. | Packaged demo covers a contextual external cash contribution; code-only `li` must remain unsafe. |
| `lo` | Deliver out, transfer out, debit, withdrawal, negative movement. | `external_flow` or `transfer` | Negative when external outflow; none/neutral for internal transfer. | `external` or `neutral` | Source/destination type and symbol, security type, amount/quantity signs, local mapping or reviewed REP semantics. | Medium when context fields available; insufficient when code-only. | Medium to High | Covered in packaged demo and site-variant tests for external and transfer cases. | Packaged demo covers a contextual external cash deliver-out; code-only `lo` must remain unsafe. |
| `dp` | Cash-security buy, tax, fee, recordkeeping, investment expense, service charge, debit-like case. | `fee_expense`, `transfer`, or `unknown pending review` | Negative for fee/expense; neutral for sweep/internal movement. | `performance`, `neutral`, or `unknown` | Special security type/symbol, source/destination context, fee/tax/sweep symbols, local mapping. | Medium when context fields available; insufficient when code-only. | Medium to High | Covered in packaged demo and site-variant tests for fee and sweep/transfer cases. | Do not classify as external flow by code alone. |
| `wd` | Cash-security sell / withdrawal-like cash-security movement. | `external_flow`, `transfer`, or `sell` | Negative for external withdrawal; neutral for sweep/internal movement; context-dependent for cash-security sell. | `external`, `neutral`, `performance`, or `unknown` | Cash security, source/destination type and symbol, sweep/cash symbols, amount sign, local mapping or reviewed REP semantics. | Medium when context fields available; insufficient when code-only. | Medium to High | Covered in packaged demo and site-variant tests for external and sweep/transfer cases. | Code-only `wd` must remain unsafe; do not infer client withdrawal from the code name. |
| `;` | Journal, Other, Split marker in integration tables. | `corporate_action` or `unknown pending review` | Usually none or context-dependent. | `neutral`, `review_only`, or `unknown` | Corporate-action/report context, split details, local mapping. | Low | Medium | Backlog; not currently included in the user-facing packaged comparison demo. | Treat as integration behavior, not official native code. |
| `epus` | Management fee / fee-related token in conversion evidence. | `fee_expense` when confirmed as fee context. | Negative | `performance` | Must confirm whether token appears as transaction code, label, security type, or special-security field; fee symbol such as `expense` or `custfee`. | Low to Medium | Medium | Not yet covered as a standalone transaction-code fixture. | Prefer using as special-security context until local evidence proves otherwise. |
| `exus` | Expense / fee-related token in conversion and special-security evidence. | `fee_expense` when confirmed as expense context. | Negative | `performance` | Must confirm whether token appears as transaction code, label, security type, or special-security field; fee symbol such as `expense` or `custfee`. | Low to Medium | Medium | Covered as special-security context for `dp` fee examples. | Prefer using as special-security context until local evidence proves otherwise. |

## Pairing and Cancellation Matrix

| Pattern | Observed meaning | ppar treatment | Required evidence | Coverage status | Notes |
| --- | --- | --- | --- | --- | --- |
| `BY`, `SL`, `SS`, `CS` uppercase examples | Delete/cancellation or blotter-delete logic in integration workflows. | `correction/reversal` or `unknown pending review`; do not treat as new economic event until linked to original. | Original transaction link or enough matching fields to identify the reversal target. | Partially documented; not yet covered by performance demo fixtures. | Universality across native Axys/APX versions is Unknown. |
| `dp` / `wd` pair | Cash sweep or intra-account cash journal candidate. | `transfer` / `neutral` when context proves internal sweep. | Matching account/date/amount/security/source-destination fields and sweep/cash symbols. | Covered in site-variant tests at row-classification level. | Integration tools may remove such pairs before import. |
| `li` / `lo` pair | Transfer-in/transfer-out or internal movement candidate. | `transfer` / `neutral` unless external party context proves external flow. | Source/destination type and symbol, security type, quantity/amount signs, account mapping. | Covered in site-variant tests at row-classification level; inserted cash-flow behavior is test-only. | Can also represent true capital movement. |
| `ti` / `to` pair | Observed opposite cash-journal pair in cash research. | `unknown pending review` until supported by transaction-code evidence. | Local mapping, source/destination fields, matching pair evidence. | Not yet covered. | Mentioned as AIA pair logic; exact ppar treatment unresolved. |
| `si` / `so` pair | Observed opposite cash-journal pair in cash research. | `unknown pending review` until supported by transaction-code evidence. | Local mapping, source/destination fields, matching pair evidence. | Not yet covered. | Mentioned as AIA pair logic; exact ppar treatment unresolved. |
| `tr` / `ts` pair | Observed opposite cash-journal pair in cash research. | `unknown pending review` until supported by transaction-code evidence. | Local mapping, source/destination fields, matching pair evidence. | Not yet covered. | Mentioned as AIA pair logic; exact ppar treatment unresolved. |
| `dv` + `by` reinvestment pair | Reinvested dividend represented as income plus buy. | Income leg is `performance`; buy leg is `performance`; avoid double-counting as external flow. | Pairing evidence such as dividend-wash symbol, dates, offsetting amounts, security, and local mapping. | Partially covered by a test-only reconstruction fixture. | Double-count guards are covered; full pair matching still needs dividend-wash context before packaged-demo inclusion. |

## Snapshot Matching Boundary

Performance-comparison reports should match transactions only when the evidence
is strong enough to avoid invented linkage:

1. Prefer a trustworthy source transaction identifier when one is available.
2. Without an identifier, allow strict fallback matching only for a unique
   one-to-one row on both sides using portfolio, trade date, security
   identifier, and transaction code.
3. Do not fuzzy-match transactions by amount, near date, partial description,
   similar security, or best-score logic.
4. If multiple same-day candidates exist, or if a transaction was added,
   deleted, or moved to a different trade date, leave the rows unmatched unless
   site-specific evidence proves linkage.
5. Keep unmatched rows reportable. They can still explain Modified Dietz
   numerator or weighted-flow effects without pretending that Snapshot A and
   Snapshot B rows are the same transaction.

## External-Flow Decision Rules

Use this order before assigning `external_flow`:

1. Detect corrections, cancellations, deletes, and reversal-like rows.
2. Apply explicit site mapping and reviewed REP/report semantics.
3. For IMEX rows, require the context fields declared in the site extract
   contract before classifying ambiguous codes. Context-field presence alone is
   not enough; the row values must match a reviewed conditional rule.
4. Treat `li` and `lo` as external flows only when source/destination and
   security context prove an external party movement.
5. Treat `wd` as an external withdrawal only when cash security and
   source/destination context prove external cash out.
6. Treat `dp` as fee/expense, sweep/transfer, or unknown based on
   special-security and source/destination context; do not treat it as an
   external flow by default.
7. Treat trade, income, fee, corporate-action, and principal-event rows as
   performance or review evidence according to explicit YAML/source semantics,
   not external capital flows.

## Coverage Backlog

Future demo data and tests should add explicit examples for:

- short sale and cover short: `ss`, `cs`;
- packaged-demo promotion for `ai` only after coherent margin-interest,
  holdings, cash, and performance scenarios exist;
- real-world split/corporate-action example using an actual historical split
  date and security when the demo period is designed to support it;
- reinvested dividend pair: `dv` + `by` with dividend-wash context;
- uppercase cancellation/reversal examples with original-transaction matching;
- cash-journal pair families beyond the current `dp`/`wd`, `li`/`lo` examples:
  `ti`/`to`, `si`/`so`, and `tr`/`ts`;
- local-review fixtures where code and context are present but treatment remains
  unknown until site mapping is supplied.

The goal is not to invent native Axys/APX semantics. The goal is to make ppar's
classification assumptions visible, testable, and easy to override when a real
site extract proves different behavior.
