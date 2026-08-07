# PPAR Audit Workspace

The packaged Axys/APX demonstration data is the source for a generated PPAR
Audit workspace. Run it first, then replace the CSVs with your own approved
Axys/APX exports.

## First Run

```bash
ppar setup ./my_ppar_audit
ppar audit ./my_ppar_audit
```

Open the files printed by the Audit command.

Output goes here:

- `output/portfolio/portfolio_audit.xlsx`
- `output/security/security_audit.xlsx`

When you are ready to use your own data, open `my_ppar_audit/README.md` and
follow the `Customizing` section. Existing files are kept unless you pass
`--overwrite`.

## Workspace Layout

Setup copies the Audit workspace files into:

```text
my_ppar_audit/
  README.md
  ppar.yaml
  run_audit.py
  snapshot_a/
  snapshot_b/
```

There is one packaged Axys/APX audit YAML file. Portfolio and security reports
use the same source snapshots and choose the review level at runtime.

Audit includes Performance Comparison, which explains changed
reported performance, and Data Issues, which flags suspicious source-data
relationships.

## Details

The packaged CSV files follow the
[Performance Comparison Demo Source Contract](demo_source_contract.md).
They are normalized demo extracts, not official Axys/APX native schemas.
The audit YAML comments and extract-availability contract use three practical
extraction labels: **Required**, **Required only when applicable**, and
**Optional**. Required is intentionally narrow: data needed to make Fully
Explained possible. For field-by-field requirements and IMEX/REP availability
confidence, see
[PPAR Axys/APX Extract Requirements and Source Guidance](../axys_apx/contracts/demo_extract_availability.md).

The demo includes normalized USD-base multi-currency examples: `SAP.DE` and
`CASHEUR` in EUR, `SHEL.L` and `CASHGBP` in GBP, and `CASHUSD` in USD. Local
amounts/market values are paired with explicit base-currency values. A changed
EUR dividend and a changed GBP/USD rate reach Explained Difference separately.
The FX estimate is a ppar screening formula using an explicit unchanged local
exposure; it is not a claim about proprietary Axys/APX calculation mechanics.

The packaged fixture uses dated public-market observations instead of constant
equity marks. The demonstration data is generated from a maintained yFinance
cache: adjusted closes support total returns, reconstructed
contemporaneous closes support holdings and trades, and reported dividends and
splits provide independent reconciliation evidence. Cash remains at 1.00. The
CUSIP-like fixed-income examples are synthetic and use disclosed BIL, SHY, IEI,
and MBB market proxies. Their accrued-income balances use consistent synthetic
per-unit rates, with one explicit 91282Y5Y1 mismatch retained for review. The
common-stock baseline does not invent dividends for non-dividend-paying
securities; the JPM and AAPL examples use the documented real dates and rates.
Intentional Snapshot A errors remain clearly bounded Audit/Data Issues scenarios
rather than claims about the public-market source.

Setup creates `ppar.yaml` in the workspace root. It keeps normal presentation,
review-policy, and diagnostic settings in one `audit:` section. Save those
reproducible choices in YAML. The public command accepts only straightforward
one-run changes to the output location, title, format, and supporting-file layout.
Audit validation checks minimum required datasets, required normalized columns,
complete YAML treatment for changed source-data fields, and the strict Data Issues check
contract before report bundles are written. Unknown Data Issues issue types, unsupported
per-check keys, malformed filters, non-Boolean enablement, invalid tolerances, and
an enabled conservative check without its required `only` population stop with the
exact YAML path. Mandatory portfolio/security market-value continuity remains active
when optional Data Issues checks are disabled.
The user-facing `ppar audit` command explicitly runs the portfolio view and, when
`secperf.csv` is available in both snapshots, the security view. The workspace
therefore does not contain a misleading single `comparison.level`. Lower-level
single-view tools must select their level explicitly. Required datasets use
their standard filenames when their `files.*` keys are omitted. Genuinely
optional evidence remains explicitly configured so file presence alone cannot
expand the findings or accounting-policy surface. Audit YAML still requires all
six comparison tolerances. Omitted extract-contract settings use the packaged
contract, ambiguous-flow enforcement, and exact-case matching. Configuring
`transactions`, `holdings`, or `fx_rates` requires the complete corresponding
transaction, holding/price, or FX impact-policy block.
The opt-in `large_price_variation` check instead uses a strict nonempty list of
uniquely identified rules. Each rule can use scalar-or-list exact-match filters,
and requires an explicitly configured decimal minimum variation. Every
established period with at least two comparable positive observations is
eligible. The check combines linked boundary holdings with inclusive trade-date
transaction prices and uses optional `splits.csv` factors to put prices on the
period-ending share basis. The packaged common-stock rule uses a 30 percent
tolerance and real dated observations without injecting a price or market value.
The opt-in `deliver_in_original_cost_incomplete` check reviews only an explicit
transaction-code, security-type, and source/destination population. It requires
both original-cost source columns in both snapshots, treats zero cost as
supplied, and does not calculate cost basis or claim that a source-system
fallback occurred. The packaged YAML retains a disabled, fully scoped example,
but the primary demo transactions deliberately omit those optional columns so
they are not mistaken for Performance Comparison requirements. Focused tests
cover the enabled check end to end.
Optional `secmast.csv` fields can qualify Data Issues populations through
`security_master.*` filters. Those joins and values preserve exact source
case and fail closed when required reference evidence is missing; they do not
change performance calculations. The workspace includes only `Security Symbol`,
`Security Type`, and `Asset Class Code`, because those are the only reference
fields used by its active rules. A direct `files.*.columns.security_id` mapping
takes precedence over automatic inference. Otherwise, mapping both
`security_type` and `security_symbol`
constructs compact normalized identifiers such as `csusAAPL` from type followed
by symbol. An advanced `security_id` section can set a separator such as `_` to
produce `csus_AAPL`. Axys/APX security
types are typically four characters; PPAR preserves the configured source value
and rejects any observed component pairs that would construct the same key. Site
extracts may add other supported reference qualifiers when the YAML explicitly
uses them.
The same validation is available through
`ppar.audit.cli.validate_config` when maintainers need to
check a YAML file without writing reports; its success summary lists effective
Data Issues checks and the master-switch policy.

| Role | YAML |
| --- | --- |
| Workbook demos | `axys_apx_audit.yaml` |

## Standard Audit Output

Run the audit once:

```bash
ppar audit ./my_ppar_audit
```

When the configured security-performance files are available, the command
creates both report bundles:

- `my_ppar_audit/output/portfolio/portfolio_audit.xlsx`
- `my_ppar_audit/output/portfolio/portfolio_audit.html`
- `my_ppar_audit/output/portfolio/source_detail.csv`
- `my_ppar_audit/output/portfolio/audit_support.zip`
- `my_ppar_audit/output/security/security_audit.xlsx`
- `my_ppar_audit/output/security/security_audit.html`
- `my_ppar_audit/output/security/source_detail.csv`
- `my_ppar_audit/output/security/audit_support.zip`

If `files.security_performance` is unavailable, the portfolio bundle is still
created and the command reports that security output was skipped. Other
security-generation failures remain fatal.

The workspace `audit:` settings generate XLSX and HTML. For one run, generate
HTML without XLSX with:

```bash
ppar audit ./my_ppar_audit --html-only
```

Use `--xlsx-only` for XLSX without HTML. Use `--csv-only` to create an audit
with `performance_differences.csv`,
`performance_difference_causes.csv`, `data_issues.csv`, and
`source_detail.csv` at each report root.

`source_detail.csv` always stays at the report root and is never duplicated in
`audit_support.zip`. Extract `audit_support.zip` when the remaining supporting
CSV and JSON files are needed individually. Extraction does not regenerate or
change the Audit results.

Open `portfolio_audit.xlsx` or `security_audit.xlsx` for review, use the matching
HTML audit for browser review, and keep the CSV artifacts for
supporting detail and traceability. The
report is designed for review, not for raw data export. It separates performance
differences from identifiable input differences and other evidence:

- `Performance Differences` sheet: one row per portfolio period with a performance
  difference in the portfolio demo.
- `Performance Differences` sheet: one row per security-period return difference in
  the security demo. Review keys include `Portfolio`, `From Date`, `Thru Date`,
  and `Security`.
- `Performance Difference Causes` sheet: input rows such as holdings, transactions,
  and FX rates. `B - A Difference` shows the raw input-value difference, and
  `Performance Difference Explained` appears only when ppar can calculate a
  defensible performance explanation. A changed FX rate is supporting evidence;
  the linked `holdings.base_market_value` or `transactions.base_amount` row carries
  the counted amount.
  Yellow cells are included in explained performance difference. Gold cells are
  possible causes for remaining unexplained differences.
- `Data Issues` sheet: consistency checks across the union of Snapshot A
  and Snapshot B. The packaged demo includes focused examples for holdings and
  transaction price ranges, duplicate transactions, a narrowly scoped
  nonpositive holding price, a reference-scoped nonpositive transaction price,
  an observed-date stale GOOGL holding price,
  an exact-case transaction-versus-reference security-type mismatch,
  dividend-rate mismatches, fixed-income
  accrued-interest transaction-rate mismatches, missing dividends, and
  holdings.accrued rate mismatches. Both nonpositive-price examples, the stale
  price, and the classification mismatch are present in both snapshots and do
  not create A-versus-B performance differences.
- Optional reconstruction diagnostics can add `Reconstruction Summary`,
  `Return Reconstruction Checks`, and `Security Return Checks` sheets for
  implementation review, but normal demo output excludes them by default.
- Review-only supporting rows remain in `source_detail.csv`. Transaction
  quantity, price, and commission rows may also appear on
  `Performance Difference Causes` when they support a changed
  `transactions.amount`.
- `source_detail.csv`: reviewer-friendly finding rows used to build the reports.
  Transaction match status appears here for audit and troubleshooting; the
  separate `transaction_matching_diagnostics.csv` artifact is row-identity audit
  support rather than a main review sheet. Source detail always remains at the
  report root and is not stored in `audit_support.zip`.
- `audit_support.zip` contains `supporting_files/cause_lineage.csv`, the
  machine-readable trace from every
  report cause back to its source finding fingerprint. This is primarily for
  integration and invariant validation, not normal reviewer workflow.

Data used:

- Snapshot A: `snapshot_a`
- Snapshot B: `snapshot_b`
- Files: Axys/APX-style portfolio performance, security performance,
  transactions, holdings, and optional security master.
- `secmast.csv` is snapshot-specific Data Issues enrichment. The packaged
  nonpositive-price, stale-price, classification-mismatch, and fixed-income
  rate checks use reviewed `security_master` qualifiers without affecting
  Modified Dietz, choosing an authoritative classification, or assigning
  transaction semantics.
- `portperf.csv.BASE_CURRENCY` is authoritative for each portfolio. PPAR fills
  missing row-level base currency from it and rejects contradictory holdings or
  transaction/cash values. Security-performance rows do not repeat currency metadata.
- Currency names follow one rule: unqualified monetary fields in holdings,
  transactions, and cash use the row `CURRENCY`; `BASE_` fields use portfolio
  `BASE_CURRENCY`; portfolio/security performance monetary fields are already
  base-currency values and remain unprefixed. `FX_RATE` is `TO_CURRENCY` units
  per one `FROM_CURRENCY` unit. PPAR does not create `LOCAL_` duplicates.
- Foreign values cannot silently enter Modified Dietz. A nonzero foreign
  `MKT_VAL`, `ACCRUED`, transaction `AMOUNT`, or cash value needs its explicit
  `BASE_` counterpart before it can be counted.
- Scope: three operational portfolios (`ALPHA`, `BALANCED`, and `INCOME`), six
  monthly periods, ten mega-cap equities, one CVNA split-processing example,
  `CASHUSD`, `912797AA1`, `91282Y2Y1`, and `91282Y5Y1`. `ALPHA` is the closest match
  to the maintained Mega-Cap Alpha portfolio; `BALANCED` and `INCOME` reuse the
  same securities with larger cash/fixed-income sleeves. CVNA appears only in
  the BALANCED portfolio as a small corporate-action processing example tied to
  a May 2026 5-for-1 split.
- YAML: includes transaction semantics; standard field roles supply the common
  performance-input, input-component, and context treatment.
- YAML: maps source transaction codes (`by`, `sl`, `dv`, `ai`, `in`, `pa`, `pd`,
  `sa`, `rc`, `dp`, `li`, `ti`, `lo`, and `wd`) to normalized categories such as
  `buy`, `sell`, `income`, `fee_expense`, `external_flow`, and `corporate_action`.
  Reviewer-facing explanations preserve the source code rather than uppercasing
  or replacing it with the category.
- Packaged transaction rows intentionally use only the small user-facing set
  `by`, `sl`, `ss`, `cs`, `dv`, `ai`, `in`, `pa`, `pd`, `sa`, `rc`, `dp`, `li`,
  `ti`, `lo`, and `wd`. The
  packaged `pa` and `sa` rows appear only as fixed-income accrued-interest
  adjuncts paired with 91282Y5Y1 buy/sell rows. The packaged `rc` row appears only
  as an equity/security return-of-capital row with portfolio-cash destination
  context. The packaged `pd` row appears only as an MBS principal-paydown row
  with fixed-income and portfolio-cash destination context. The public mapping
  rows leave Special Security Type / Symbol blank for both codes; the demo does
  the same rather than inserting readable event labels into native-looking
  fields.
  The packaged `li` row is a plain external cash contribution with
  external-party context, and the packaged `lo` row is an external cash
  deliver-out with the same context standard. More ambiguous `li`/`lo` transfer
  cases and synthetic corporate-action rows live in test-only fixtures until a
  realistic packaged story and evidence trail justify adding them here.
- Packaged transaction rows omit `TRANSACTION_ID`. ppar supports stable
  transaction IDs when a local extract provides them, but the packaged Axys/APX
  demo uses the more realistic conservative no-ID path by default. Internal
  scenario/rebuild files may still use deterministic transaction IDs as fixture
  handles; those IDs are not packaged as user-facing Axys/APX transaction fields.
- Current transaction coverage by home:

  | Home | Transaction families |
  | --- | --- |
  | Packaged demo rows | `by`, `sl`, short-side `ss`/`cs`, ordinary and reinvested `dv`, contextual margin-interest `ai`, `in`, fixed-income accrued-interest `pa`/`sa`, equity/security return-of-capital `rc`, MBS principal-paydown `pd`, fee/withholding-like `dp`, external-security `ti`, external-cash `li`, external-cash `lo`, and external-cash `wd`. |
  | Packaged split-factor rows | CVNA `splits.csv` row in Snapshot B, used as context evidence for central split processing. |
  | YAML branches reserved for runtime guards | Non-packaged conditional branches for ambiguous flow codes. |
  | Test-only fixtures | Neutral review-only `;` markers, internal-transfer `li`/`lo` site variants, `dp`/`wd` site variants, the alternate `dp` plus `epus expense` context, and local-override examples for context-dependent codes. |
  | Evidence-blocked backlog | Code-only `ai`/`ti`, standalone `epus`, uppercase reversal rows, and additional corporate actions until source evidence and accounting policy are strong enough. |

  The packaged fixed-income story is intentionally narrow: ordinary 91282Y2Y1
  interest uses an `in` transaction row, 36225MBS1 `pd` principal-paydown uses
  MBS plus portfolio-cash destination context, accrued-interest
  restatement uses `holdings.accrued`, and 91282Y5Y1 `pa`/`sa` rows are packaged
  only with paired fixed-income trade context. The packaged demo does not infer
  accrued-interest, margin-interest, or principal-paydown treatment from code
  alone.
- Real site extracts should keep ambiguous-flow enforcement enabled. IMEX is
  sufficient only when transaction rows include source/destination and
  special-security context for `dp`, `li`, `lo`, `ti`, and `wd`; otherwise use a REP,
  custom report, or reviewed source that supplies transaction category and
  cash/performance sign semantics.
- YAML: includes explicit `portfolio_return_reconstruction` settings for
  Modified Dietz diagnostic checks.
- YAML: includes explicit `security_return_reconstruction` settings for
  security-level Modified Dietz diagnostic checks.
- YAML: treats fee-like `dp` transactions as performance-impacting because this packaged
  fixture assumes the reported returns are net of fees. For gross-of-fees
  performance, fees would need a different return-basis policy.

Expected workbook:

- Changed ALPHA, BALANCED, and INCOME periods in the portfolio demo
  `Performance Differences` sheet.
- Changed security-period returns in the security demo `Performance Differences`
  sheet.
- `Performance Difference Causes` sheet should show understandable additive transaction
  amount, holding market value, holding accrued, and weighted price examples.
  It should also show supporting input-component rows without assigning them
  `Performance Difference Explained` values when those rows help explain a
  counted source-data difference.
- The packaged demo intentionally includes fully explained, partly explained,
  and unexplained periods so reviewers can see each status in both report
  families.
- Optional reconstruction diagnostics should show where source-derived Modified
  Dietz returns agree with reported return differences and where they do not.
- The controlled restatement includes:
  - a fully explained ALPHA period with AAPL price/security-return changes and
    `CASHUSD` holding changes;
  - an ALPHA `wd` external-withdrawal amount restatement visible in the return
    reconstruction check;
  - a fully explained ALPHA period with a changed AAPL `by` transaction amount,
    changed AAPL holding quantity/market value, and related transaction
    quantity, price, and commission support rows;
  - a fully explained ALPHA period with an inserted `lo` row on `CASHUSD` for
    an external cash deliver-out;
  - a fully explained BALANCED period with a changed MSFT `sl` transaction
    amount and related quantity, price, and commission support rows;
  - a fully explained BALANCED period with a JPM `dv` gross-dividend amount
    change and a separate contextual `dp` withholding-expense change under the
    demo's after-tax/net-performance assumption;
  - a fully explained BALANCED period with a context-scoped JPM `ti`
    external-security deliver-in whose quantity, value, and holding effect
    agree without making `ti` a code-only rule;
  - a context-gated BALANCED period with a JPM `rc` return-of-capital row;
  - a fully explained BALANCED period with an inserted `li` row on `CASHUSD`
    for an external cash contribution;
  - a fully explained BALANCED period where Snapshot A missed the CVNA
    5-for-1 split quantity adjustment while both snapshots use split-adjusted
    ending prices, so Snapshot B corrects CVNA ending quantity and market value;
  - a fully explained INCOME period with a larger fee-like `dp` transaction and
    a separate context-gated negative `ai` margin-interest correction,
    classified from their source context with matching lower `CASHUSD` ending
    value;
  - two isolated INCOME periods showing a missed/late AAPL `dv` row: Snapshot B
    adds the real 2026-05-14 payable-date dividend with a matched `dvwash` `by`
    reinvestment leg and removes Snapshot A's later 2026-05-23 version;
  - a partly explained BALANCED period where the same AAPL price correction and
    standalone MSFT holding market-value correction explain part of the
    reported portfolio and MSFT security return differences, leaving an
    intentional residual for reviewer triage;
  - a fully explained INCOME period with the same AAPL price correction plus
    91282Y2Y1 `in` interest, 36225MBS1 `pd` principal-paydown cash/principal
    movement, market-value and accrued-interest changes, and related 91282Y2Y1
    quantity evidence;
  - a partly explained INCOME period where paired 91282Y5Y1 `by`/`pa` and
    `sl`/`sa` fixed-income trade/accrued-interest settlement rows affect the
    cash/performance inputs, while separate quantity-driven holding value and
    accrued-value rows remain visible as holding inputs. Incomplete or
    overlapping estimates still require reviewer triage.

Why: this is the most focused workbook for understanding the causal-attribution
model. It keeps the data small and transaction semantics explicit while still
distinguishing identifiable input causes from reported-performance diagnostics.

After generating the workbook demo bundle, validate it with:

```bash
./.venv/bin/python -m ppar.audit.cli.validate_bundle \
  my_ppar_audit/output/portfolio
./.venv/bin/python -m ppar.audit.cli.validate_bundle \
  my_ppar_audit/output/security
```

For a full packaged-demo health pass from a source checkout, maintainers can use
the consolidated maintenance script:

```bash
./.venv/bin/python scripts/check_audit_demo_health.py
```

That script is not part of the installed-package demo workflow. It runs the
rebuild drift audit, extract-availability contract check, portfolio/security
bundle generation, bundle validation, and packaged scenario matrix validation.

## YAML Policy Decision Guide

Use the YAML policy blocks to state what ppar is allowed to treat as an
explanation. The values are intentionally explicit because different vendors
can use the same-looking fields with different sign, timing, denominator, or
accounting conventions.

The workbook uses a small field-role model:

| Role | Typical fields | Workbook treatment |
| --- | --- | --- |
| `performance_input` | `holdings.market_value`, `holdings.base_market_value`, `holdings.accrued`, `holdings.base_accrued`, `transactions.amount`, `transactions.base_amount` | Additive rows on the `Performance Difference Causes` sheet when enough inputs are available. An unqualified detailed value is counted only when its row currency equals base currency; otherwise its explicit `base_` counterpart is counted. |
| `input_component` | `holdings.quantity`, `holdings.price`, `fx_rates.fx_rate`, transaction quantity/price/commission | Shown beside related performance inputs when useful, or kept in `source_detail.csv` as support for the related performance input. |
| `reported_performance_component` | portfolio return and security return | Compared as the reported results being explained; not treated as root-cause input differences. |
| `context` | explicitly classified fields such as holding cost | Kept in `source_detail.csv` as review context. |

An unknown compared field does not silently become context. PPAR stops until
the field receives an explicit accounting role, and a YAML suppression cannot
bypass that classification decision. Impact-policy requirements are derived
from the role table above.

Missing transaction semantics are still a hard stop for user-facing bundle
generation because transaction amount attribution depends on transaction-code
classification. Blank `Performance Difference Explained` cells mean the row is
review-only or not currently additively estimated, not that YAML setup is
missing.

## Method Coverage Goal

Each supported public YAML impact method should have at least one user-facing
demo example or test-only validation fixture, plus one validator assertion.
Tests can still cover narrow edge cases, but the packaged demos should stay
focused on reviewer-facing workflows.

The supported string vocabulary is summarized in
`docs/audit/performance_comparison_design.md`. The package code backs those strings
with enums, but YAML examples intentionally show the plain string values users
edit.

The portfolio fixture is intentionally action-oriented. It contains ALPHA,
BALANCED, and INCOME operational portfolios with changed portfolio/security
periods so reviewers can see the most understandable causal-attribution bases:

- transaction amount over beginning market value
- holding market value over beginning market value
- holding accrued over beginning market value
- price delta over snapshot A price, weighted by snapshot A security weight

The packaged demo is expected to be internally consistent: visible portfolio
and security performance differences should be fully explained by source-data
differences in the workbook. Test-only fixtures can still cover unresolved or
policy-gap scenarios, but the user-facing demo should not depend on accidental
internal inconsistency.

From a source checkout, maintainers keep the derived `secperf.csv` and
`portperf.csv` files aligned with:

```bash
./.venv/bin/python scripts/operational_demo_data/rebuild_audit_demo_data.py
```

The default mode audits the checked-in files without writing. Add `--write`
after intentional fixture edits to recompute security beginning weights,
security contributions, and portfolio performance rows from security
performance rows.

Current public YAML targets are intentionally narrow:

- `extract_contract`: optionally overrides the packaged extract contract and
  fail-closed runtime guards. When the section is omitted, PPAR uses the
  packaged contract and enforces ambiguous Axys/APX flow context. Transaction
  rules always match native source values by exact case. The packaged contract can enforce
  context-field presence before ambiguous Axys/APX `li`, `lo`, `ti`, `dp`, or
  `wd` rows can be classified by YAML rules. Use
  `docs/axys_apx/contracts/templates/site_extract_contract.yaml` as a starting
  point when a real site needs a local contract. Unmatched codes remain unknown;
  uppercase does not acquire cancellation meaning.
- `transaction_rules`: authoritatively classifies matching transaction rows for
  amount attribution. A complete matching rule overrides any recognized
  category/sign labels carried by the source row; when no rule matches, usable
  source semantics remain available.
  Ambiguous Axys/APX-style `li`, `lo`, `ti`, `dp`, and `wd` examples require matching
  transaction-context fields before they are treated as external flows or
  fee/expense rows. Fixed-income `pa`/`sa` accrued-interest adjuncts require
  fixed-income context and paired-trade support in the packaged demo. The
  packaged `pd` row requires amortizing MBS effects and portfolio-cash
  destination evidence. The `rc` and `pd` rules do not rely on fabricated
  special-security values. Code-only `ai` remains unknown even
  though the packaged demo now has one context-gated margin-interest example.
- `splits`: optional security-level split-factor evidence. The packaged CVNA
  row is intentionally not a transaction. It supports the review story that
  Snapshot B has central split-factor evidence and corrected holdings quantity,
  while Snapshot A is missing that factor.
- `transaction_amount_delta_over_return_denominator`: explicit amount-impact
  method used after `transaction_rules` mark a transaction code as
  performance-affecting. It is no longer supplied by an internal default.
- `transaction_impact_methods.external_flow`: optional `modified_dietz`
  cross-checks for external-flow transactions.
- suppression rules: remove known, intentionally ignored differences from the
  active review while retaining them in the source detail.

The `Performance Difference Causes` worksheet retains its established columns.
Every transaction-associated row starts `Explanation` with its native source
transaction code and a colon, including the context-gated `ai` and `ti`
scenarios.
