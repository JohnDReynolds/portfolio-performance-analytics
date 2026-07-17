# PPAR Axys/APX Extract Requirements and Source Guidance

Repository: AXYS / APX Reference Repository
Scope: `ppar/setup_templates/axys_apx_audit/snapshot_a` and
`ppar/setup_templates/axys_apx_audit/snapshot_b`
Status: Draft confidence matrix generated from the packaged YAML contract.

<!-- GENERATED FROM ppar/setup_templates/axys_apx_audit/demo_extract_availability.yaml. -->
<!-- Run scripts/render_demo_extract_availability.py after editing the YAML. -->

---

## Purpose

This contract estimates how likely each packaged Axys/APX demo dataset and column is to be obtainable from an Axys/APX installation through IMEX and/or REP-style report extracts.

The machine-readable source of truth is `ppar/setup_templates/axys_apx_audit/demo_extract_availability.yaml`. Tests verify that the YAML covers every packaged comparison demo CSV header and that this contract is current.

The packaged demo files are normalized demo extracts. They are not official Axys/APX schemas, not universal IMEX profiles, and not claims that every Axys/APX site can export every field with these exact names.

The two packaged snapshots currently use the same file layouts:

- `holdings.csv`
- `portperf.csv`
- `secperf.csv`
- `transactions.csv`
- `fx_rates.csv`
- `splits.csv`

## Confidence Labels

| Label | Meaning |
|---|---|
| High | Strongly likely to be obtainable from a typical Axys/APX environment. |
| Medium / High | Between medium and high; likely available, but still report/profile dependent. |
| Medium | Plausible, but site configuration, report choice, or export profile must be validated. |
| Medium / Unknown | Between medium and unknown; evidence mentions the area, but not the exact object/field. |
| Low / Medium | Between low and medium; possible and plausible, but not proven as a standard field. |
| Low | Possible through customization or site-specific configuration, but not enough to rely on. |
| Unknown | Not established by the local Axys/APX reference corpus. |

## Evidence Boundaries

The confidence ratings below rely on these local reference conclusions:

- `Chapter_12_Imex.md` establishes IMEX / Import-Export as an Axys/APX import/export mechanism, but says the complete native object and field dictionaries are not available in the supplied source material.
- `Chapter_12_Imex.md` documents Axys/APX CI import workflows for transactions, positions, prices, security information, Trade Blotter context, and selected transaction translation fields.
- `Chapter_13_Rep.md` establishes REP / RepLang / REP32 as a report-driven extraction path and distinguishes it from IMEX.
- `Chapter_10_Performance.md` says `portperf` and `secperf` should be treated as normalized/local names unless a live IMEX object, report output, or vendor manual confirms native names.
- `docs/axys_apx/axys_apx_common_core_export.md` is a starter reference only. It proposes common field aliases but does not override the more conservative chapter confidence boundaries.

## Interpretation Rules

Use this matrix as an implementation planning aid:

- **IMEX confidence** asks whether the value is likely obtainable from a structured Axys/APX IMEX-style export or adjacent import/export workflow.
- **REP confidence** asks whether the value is likely obtainable from a standard or custom REP/Replang report extract.
- A high confidence rating does not mean the exact demo column name exists in Axys/APX. It means the underlying value is likely available.
- Performance fields are rated more conservatively for IMEX because the local reference corpus does not contain an official performance IMEX object dictionary.
- Transaction source/destination and special-security fields are rated as integration-context fields. The corpus supports them in transaction translation workflows, but not as guaranteed columns in every posted-transaction export.

## What You Need to Extract

This is the user-facing checklist for making **Fully Explained** possible. It intentionally uses only three labels:

- **Required**: needed for the ordinary portfolio comparison.
- **Required only when applicable**: needed only for the named feature or data condition.
- **Optional**: safe to omit; its absence alone does not prevent Fully Explained.

These labels describe extraction needs. They are separate from PPAR's internal validation and evidence-role rules.

| Dataset | Dataset requirement | Why / likely source | Required fields (when dataset applies) | Required only when applicable | Optional fields |
|---|---|---|---|---|---|
| `holdings.csv` | Required | Beginning and ending base-currency values provide the Modified Dietz valuation inputs used to explain a changed return. Likely source: IMEX positions/holdings export or a REP appraisal report. | `PORT`, `SEC`, `HOLDING_DATE`, `MKT_VAL` | `CURRENCY` — Required for holdings whose local currency differs from the portfolio base currency.<br>`BASE_CURRENCY` — Required for multi-currency portfolios unless a validated portfolio-level source supplies it.<br>`BASE_MKT_VAL` — Required when MKT_VAL is local-currency rather than portfolio-base value.<br>`ACCRUED` — Required when accrued income is stated separately from MKT_VAL and affects beginning or ending value. | `QTY`, `PRICE` |
| `portperf.csv` | Required | The portfolio, period, and reported return define the performance difference PPAR must explain. Likely source: REP performance report preferred; a native performance IMEX object is not verified by the current evidence. | `PORTFOLIO_CODE`, `FROM_DATE`, `THRU_DATE`, `PORT_RETURN` | `BASE_CURRENCY` — Required when the portfolio contains holdings or transactions outside its reporting currency. | `BEGIN_MV`, `END_MV`, `FLOW`, `INCOME`, `GAIN_LOSS` |
| `secperf.csv` | Required only when applicable | Required only when the user wants security-level differences to reach Fully Explained; portfolio-only audit does not need this file. Likely source: REP security-performance or attribution report preferred; a native performance IMEX object is not verified by the current evidence. | `PORTFOLIO_CODE`, `SECURITY_ID`, `FROM_DATE`, `THRU_DATE`, `SEC_RETURN` | None | `BEGIN_MV`, `END_MV`, `BEGIN_WEIGHT`, `INCOME`, `GAIN_LOSS`, `CONTRIBUTION` |
| `transactions.csv` | Required | Dated, classified amounts are needed to explain changed external flows, income, fees, and security activity. Likely source: IMEX transaction export first; use REP/custom output or another reviewed source when IMEX omits transaction-semantics context. | `PORT`, `TRANSACTION_DATE`, `SEC`, `TRAN`, `AMOUNT` | `SEC_TYPE` — Required by the packaged guard when ambiguous DP, LI, LO, or WD codes can appear.<br>`SRC_DEST_TYPE` — Required by the packaged guard when ambiguous DP, LI, LO, or WD codes can appear.<br>`SRC_DEST_SYMBOL` — Required by the packaged guard when ambiguous DP, LI, LO, or WD codes can appear.<br>`SPECIAL_SEC_TYPE` — Required by the packaged guard when ambiguous DP, LI, LO, or WD codes can appear.<br>`SPECIAL_SEC_SYMBOL` — Required by the packaged guard when ambiguous DP, LI, LO, or WD codes can appear.<br>`CURRENCY` — Required for transaction amounts stated in a currency other than portfolio base currency.<br>`BASE_CURRENCY` — Required for multi-currency portfolios unless the authoritative portfolio-performance row supplies it.<br>`BASE_AMOUNT` — Required when AMOUNT is local-currency rather than the portfolio-base amount used by return reconstruction. | `SETTLE_DATE`, `QTY`, `PRICE`, `COMMISSION` |
| `fx_rates.csv` | Optional | Optional supporting evidence that can link a rate change to a counted base-currency value. Likely source: Local discovery is required; use a validated REP extract, FX/price source, or other controlled rate source. | `PORT`, `FROM_CURRENCY`, `TO_CURRENCY`, `RATE_DATE`, `FX_RATE`, `LOCAL_EXPOSURE` | `RATE_SOURCE` — Required when pair and date do not uniquely identify one controlled rate series.<br>`RATE_TYPE` — Required when pair and date do not uniquely identify one controlled rate convention. | None |
| `splits.csv` | Optional | Split factors add review context but do not directly enter the current Modified Dietz explanation formula. Likely source: Direct split.inf or local split-factor export; use REP/custom output only when it exposes equivalent factors. | None | None | `SEC`, `SPLIT_DATE`, `SPLIT_FACTOR` |


## Availability Matrix

### `holdings.csv`

| Dataset | Demo column | Normalized meaning | IMEX confidence | REP confidence | Evidence basis | Open questions | Comments |
|---|---|---|---|---|---|---|---|
| holdings | `PORT` | Portfolio/account identifier. | High | High | Position/appraisal evidence in Chapter_06_Holdings.md, Chapter_12_Imex.md, Chapter_13_Rep.md, and axys_apx_common_core_export.md. | Confirm exact local IMEX profile and field name. | Core portfolio/client identifier in position and appraisal workflows. |
| holdings | `SEC` | Security identifier. | High | High | Position/appraisal evidence in Chapter_06_Holdings.md, Chapter_12_Imex.md, Chapter_13_Rep.md, and axys_apx_common_core_export.md. | Confirm exact local IMEX profile and field name. | Security identifiers are supported by CI/security-resolution and report evidence. |
| holdings | `HOLDING_DATE` | As-of date for holdings. | High | High | Position/appraisal evidence in Chapter_06_Holdings.md, Chapter_12_Imex.md, Chapter_13_Rep.md, and axys_apx_common_core_export.md. | Confirm exact local IMEX profile and field name. | Position exports and appraisal reports are inherently as-of-date based. |
| holdings | `CURRENCY` | Local currency of the holding row. | Unknown | Low / Medium | The research supports multi-currency concepts but does not prove this exact export field or normalized layout. | Confirm the local source, field label, currency basis, and date semantics. | Normalized PPAR multi-currency demo field; exact Axys/APX source and label require local validation. |
| holdings | `BASE_CURRENCY` | Portfolio reporting currency used for the holding row. | Unknown | Low / Medium | The research supports multi-currency concepts but does not prove this exact export field or normalized layout. | Confirm the local source, field label, currency basis, and date semantics. | Normalized PPAR multi-currency demo field; exact Axys/APX source and label require local validation. |
| holdings | `QTY` | Quantity, shares, or units. | High | High | Position/appraisal evidence in Chapter_06_Holdings.md, Chapter_12_Imex.md, Chapter_13_Rep.md, and axys_apx_common_core_export.md. | Confirm exact local IMEX profile and field name. | Quantity is directly supported by transaction and appraisal report evidence. |
| holdings | `PRICE` | Market price used for valuation. | High | High | Position/appraisal evidence in Chapter_06_Holdings.md, Chapter_12_Imex.md, Chapter_13_Rep.md, and axys_apx_common_core_export.md. | Confirm exact local IMEX profile and field name. | Price import/export evidence and appraisal/report labels support availability. |
| holdings | `MKT_VAL` | Market value. | High | High | Position/appraisal evidence in Chapter_06_Holdings.md, Chapter_12_Imex.md, Chapter_13_Rep.md, and axys_apx_common_core_export.md. | Confirm exact local IMEX profile and field name. | Market value appears in report evidence and is a core position/appraisal value. |
| holdings | `BASE_MKT_VAL` | Holding market value translated to portfolio base currency. | Unknown | Low / Medium | The research supports multi-currency concepts but does not prove this exact export field or normalized layout. | Confirm the local source, field label, currency basis, and date semantics. | Normalized PPAR multi-currency demo field; exact Axys/APX source and label require local validation. |
| holdings | `ACCRUED` | Accrued income or accrued interest. | Medium | Medium | Fixed-income/accrual handling is documented as performance-sensitive; common-core holdings reference treats accrued income as plausible. | Confirm accrued-income field, date basis, and fixed-income treatment. | Common for fixed-income holdings, but exact export/report field requires validation. |


### `portperf.csv`

| Dataset | Demo column | Normalized meaning | IMEX confidence | REP confidence | Evidence basis | Open questions | Comments |
|---|---|---|---|---|---|---|---|
| portfolio performance | `END_MV` | Ending market value. | Medium | High | Chapter_10_Performance.md treats portfolio performance as a report/extract boundary and prefers REP for report-tie values. | Confirm whether a local IMEX performance object exists or use REP output. | Likely available through performance/appraisal reports; exact IMEX object remains unknown. |
| portfolio performance | `FLOW` | Period external flow. | Low / Medium | Medium | Performance chapter and report evidence support the concept, but the local corpus does not prove a native IMEX performance object/field. | Confirm methodology, gross/net basis, and stored-vs-report-calculated behavior. | More likely through reports when displayed or calculated; IMEX object/field is unproven. |
| portfolio performance | `INCOME` | Period income component. | Low / Medium | Medium | Performance chapter and report evidence support the concept, but the local corpus does not prove a native IMEX performance object/field. | Confirm methodology, gross/net basis, and stored-vs-report-calculated behavior. | Likely reportable, but basis and inclusion rules require validation. |
| portfolio performance | `GAIN_LOSS` | Period gain/loss component. | Low / Medium | Medium | Performance chapter and report evidence support the concept, but the local corpus does not prove a native IMEX performance object/field. | Confirm methodology, gross/net basis, and stored-vs-report-calculated behavior. | Report evidence supports gain/loss labels; exact performance component is report-dependent. |
| portfolio performance | `PORTFOLIO_CODE` | Portfolio/account identifier. | High | High | Chapter_10_Performance.md treats portfolio performance as a report/extract boundary and prefers REP for report-tie values. | Confirm whether a local IMEX performance object exists or use REP output. | Core identifier. |
| portfolio performance | `FROM_DATE` | Period start date. | Medium | High | Chapter_10_Performance.md treats portfolio performance as a report/extract boundary and prefers REP for report-tie values. | Confirm whether a local IMEX performance object exists or use REP output. | Report parameters reliably provide the period; structured export field names require validation. |
| portfolio performance | `THRU_DATE` | Period end date. | Medium | High | Chapter_10_Performance.md treats portfolio performance as a report/extract boundary and prefers REP for report-tie values. | Confirm whether a local IMEX performance object exists or use REP output. | Report parameters reliably provide the period; structured export field names require validation. |
| portfolio performance | `BEGIN_MV` | Beginning market value. | Medium | High | Chapter_10_Performance.md treats portfolio performance as a report/extract boundary and prefers REP for report-tie values. | Confirm whether a local IMEX performance object exists or use REP output. | Likely reportable; exact IMEX object/field remains unproven. |
| portfolio performance | `PORT_RETURN` | Portfolio return. | Medium / Unknown | High | Performance chapter and report evidence support the concept, but the local corpus does not prove a native IMEX performance object/field. | Confirm methodology, gross/net basis, and stored-vs-report-calculated behavior. | Performance-history IMEX fields are not established; REP is preferred for report-tie values. |
| portfolio performance | `BASE_CURRENCY` | Authoritative portfolio reporting currency for the performance row. | Unknown | Low / Medium | The research supports multi-currency concepts but does not prove this exact export field or normalized layout. | Confirm the local source, field label, currency basis, and date semantics. | Normalized PPAR multi-currency demo field; exact Axys/APX source and label require local validation. |


### `secperf.csv`

| Dataset | Demo column | Normalized meaning | IMEX confidence | REP confidence | Evidence basis | Open questions | Comments |
|---|---|---|---|---|---|---|---|
| security performance | `END_MV` | Ending market value for security row. | Medium | High | Chapter_10_Performance.md treats portfolio performance as a report/extract boundary and prefers REP for report-tie values. | Confirm whether values come from security performance, appraisal, or custom REP output. | Likely available through security performance or appraisal-style reports. |
| security performance | `INCOME` | Security-period income. | Low / Medium | Medium | Performance chapter and report evidence support the concept, but the local corpus does not prove a native IMEX performance object/field. | Confirm methodology, contribution basis, and stored-vs-report-calculated behavior. | Reportable in some layouts; methodology and inclusion rules require validation. |
| security performance | `GAIN_LOSS` | Security-period gain/loss. | Low / Medium | Medium | Performance chapter and report evidence support the concept, but the local corpus does not prove a native IMEX performance object/field. | Confirm methodology, contribution basis, and stored-vs-report-calculated behavior. | Plausible as report output, but exact component semantics are report-dependent. |
| security performance | `PORTFOLIO_CODE` | Portfolio/account identifier. | High | High | Chapter_10_Performance.md treats portfolio performance as a report/extract boundary and prefers REP for report-tie values. | Confirm whether values come from security performance, appraisal, or custom REP output. | Core identifier. |
| security performance | `SECURITY_ID` | Security identifier. | High | High | Chapter_10_Performance.md treats portfolio performance as a report/extract boundary and prefers REP for report-tie values. | Confirm whether values come from security performance, appraisal, or custom REP output. | Core security-row identifier. |
| security performance | `FROM_DATE` | Period start date. | Medium | High | Chapter_10_Performance.md treats portfolio performance as a report/extract boundary and prefers REP for report-tie values. | Confirm whether values come from security performance, appraisal, or custom REP output. | Report parameter/output availability is likely; structured export names require validation. |
| security performance | `THRU_DATE` | Period end date. | Medium | High | Chapter_10_Performance.md treats portfolio performance as a report/extract boundary and prefers REP for report-tie values. | Confirm whether values come from security performance, appraisal, or custom REP output. | Report parameter/output availability is likely; structured export names require validation. |
| security performance | `BEGIN_WEIGHT` | Beginning portfolio weight. | Low / Medium | Medium | Performance chapter and report evidence support the concept, but the local corpus does not prove a native IMEX performance object/field. | Confirm methodology, contribution basis, and stored-vs-report-calculated behavior. | Likely reportable in performance/attribution reports, but not proven as an IMEX field. |
| security performance | `BEGIN_MV` | Beginning market value for security row. | Medium | High | Chapter_10_Performance.md treats portfolio performance as a report/extract boundary and prefers REP for report-tie values. | Confirm whether values come from security performance, appraisal, or custom REP output. | Likely reportable; exact IMEX field remains unproven. |
| security performance | `SEC_RETURN` | Security return. | Medium / Unknown | High | Performance chapter and report evidence support the concept, but the local corpus does not prove a native IMEX performance object/field. | Confirm methodology, contribution basis, and stored-vs-report-calculated behavior. | Performance-history IMEX fields are not established; REP is preferred for report-tie values. |
| security performance | `CONTRIBUTION` | Security contribution to return. | Low / Medium | Medium / High | Performance chapter and report evidence support the concept, but the local corpus does not prove a native IMEX performance object/field. | Confirm methodology, contribution basis, and stored-vs-report-calculated behavior. | Contribution reports are supported, but exact Axys/APX security-contribution export requires validation. |


### `transactions.csv`

| Dataset | Demo column | Normalized meaning | IMEX confidence | REP confidence | Evidence basis | Open questions | Comments |
|---|---|---|---|---|---|---|---|
| transactions | `PORT` | Portfolio/account identifier. | High | High | Transaction import/export and report evidence in Chapter_05_Transactions.md, Chapter_12_Imex.md, and Chapter_13_Rep.md. | Confirm exact posted-transaction export fields and sign convention. | Core transaction/account field. |
| transactions | `TRANSACTION_DATE` | Trade or economic date. | High | High | Transaction import/export and report evidence in Chapter_05_Transactions.md, Chapter_12_Imex.md, and Chapter_13_Rep.md. | Confirm exact posted-transaction export fields and sign convention. | Trade date appears in report evidence and transaction import workflows. |
| transactions | `SETTLE_DATE` | Settlement or pay date. | Medium | High | Transaction import/export and report evidence in Chapter_05_Transactions.md, Chapter_12_Imex.md, and Chapter_13_Rep.md. | Confirm exact posted-transaction export fields and sign convention. | Settle date appears in report-output evidence; IMEX availability is profile-dependent. |
| transactions | `SEC` | Security identifier. | High | High | Transaction import/export and report evidence in Chapter_05_Transactions.md, Chapter_12_Imex.md, and Chapter_13_Rep.md. | Confirm exact posted-transaction export fields and sign convention. | Core transaction/security field. |
| transactions | `TRAN` | Transaction code or type. | Medium | Medium | Transaction import/export and report evidence in Chapter_05_Transactions.md, Chapter_12_Imex.md, and Chapter_13_Rep.md. | Confirm exact posted-transaction export fields and sign convention. | Transaction type/code is supported in translation/import evidence; report exposure depends on layout. |
| transactions | `SEC_TYPE` | Security type for transaction security. | Medium | Low / Medium | Observed CI/Axys/APX transaction translation context in Chapter_05_Transactions.md and Chapter_12_Imex.md; required for ambiguous external-flow handling. | Confirm whether local IMEX exposes this context on posted transaction exports.<br>Use REP, a custom report, or another source if IMEX cannot expose it. | Central to Axys/APX security resolution; transaction-row availability depends on export/report design. |
| transactions | `SRC_DEST_TYPE` | Source/destination type for ambiguous flows. | Medium | Low / Medium | Observed CI/Axys/APX transaction translation context in Chapter_05_Transactions.md and Chapter_12_Imex.md; required for ambiguous external-flow handling. | Confirm whether local IMEX exposes this context on posted transaction exports.<br>Use REP, a custom report, or another source if IMEX cannot expose it. | Observed transaction translation/integration field, not guaranteed in every posted export. |
| transactions | `SRC_DEST_SYMBOL` | Source/destination symbol for ambiguous flows. | Medium | Low / Medium | Observed CI/Axys/APX transaction translation context in Chapter_05_Transactions.md and Chapter_12_Imex.md; required for ambiguous external-flow handling. | Confirm whether local IMEX exposes this context on posted transaction exports.<br>Use REP, a custom report, or another source if IMEX cannot expose it. | Observed transaction translation/integration field, not guaranteed in every posted export. |
| transactions | `SPECIAL_SEC_TYPE` | Special security type for fee or special handling. | Medium | Low | Observed CI/Axys/APX transaction translation context in Chapter_05_Transactions.md and Chapter_12_Imex.md; required for ambiguous external-flow handling. | Confirm whether local IMEX exposes this context on posted transaction exports.<br>Use REP, a custom report, or another source if IMEX cannot expose it. | Supported in integration evidence; standard report exposure is uncertain. |
| transactions | `SPECIAL_SEC_SYMBOL` | Special security symbol for fee or special handling. | Medium | Low | Observed CI/Axys/APX transaction translation context in Chapter_05_Transactions.md and Chapter_12_Imex.md; required for ambiguous external-flow handling. | Confirm whether local IMEX exposes this context on posted transaction exports.<br>Use REP, a custom report, or another source if IMEX cannot expose it. | Supported in integration evidence; standard report exposure is uncertain. |
| transactions | `CURRENCY` | Local currency of the transaction amount. | Unknown | Low / Medium | The research supports multi-currency concepts but does not prove this exact export field or normalized layout. | Confirm the local source, field label, currency basis, and date semantics. | Normalized PPAR multi-currency demo field; exact Axys/APX source and label require local validation. |
| transactions | `BASE_CURRENCY` | Portfolio reporting currency for the transaction. | Unknown | Low / Medium | The research supports multi-currency concepts but does not prove this exact export field or normalized layout. | Confirm the local source, field label, currency basis, and date semantics. | Normalized PPAR multi-currency demo field; exact Axys/APX source and label require local validation. |
| transactions | `QTY` | Transaction quantity. | High | High | Transaction import/export and report evidence in Chapter_05_Transactions.md, Chapter_12_Imex.md, and Chapter_13_Rep.md. | Confirm exact posted-transaction export fields and sign convention. | Quantity appears in transaction and report evidence. |
| transactions | `PRICE` | Transaction price. | High | High | Transaction import/export and report evidence in Chapter_05_Transactions.md, Chapter_12_Imex.md, and Chapter_13_Rep.md. | Confirm exact posted-transaction export fields and sign convention. | Price appears in transaction and report evidence. |
| transactions | `AMOUNT` | Net amount, proceeds, or cash amount. | High | Medium / High | Transaction import/export and report evidence in Chapter_05_Transactions.md, Chapter_12_Imex.md, and Chapter_13_Rep.md. | Confirm exact posted-transaction export fields and sign convention. | Core transaction amount is likely available, though report labels can vary. |
| transactions | `BASE_AMOUNT` | Transaction amount translated to portfolio base currency. | Unknown | Low / Medium | The research supports multi-currency concepts but does not prove this exact export field or normalized layout. | Confirm the local source, field label, currency basis, and date semantics. | Normalized PPAR multi-currency demo field; exact Axys/APX source and label require local validation. |
| transactions | `COMMISSION` | Commission. | Medium | Medium | Transaction import/export and report evidence in Chapter_05_Transactions.md, Chapter_12_Imex.md, and Chapter_13_Rep.md. | Confirm exact posted-transaction export fields and sign convention. | Supported in CI parameter evidence; report/profile availability should be validated. |


### `fx_rates.csv`

| Dataset | Demo column | Normalized meaning | IMEX confidence | REP confidence | Evidence basis | Open questions | Comments |
|---|---|---|---|---|---|---|---|
| FX rates | `PORT` | Portfolio/account identifier for exposure linkage. | Unknown | Low / Medium | The research supports multi-currency concepts but does not prove this exact export field or normalized layout. | Confirm the local source, field label, currency basis, and date semantics. | Normalized PPAR multi-currency demo field; exact Axys/APX source and label require local validation. |
| FX rates | `FROM_CURRENCY` | Local currency converted by the rate. | Unknown | Low / Medium | The research supports multi-currency concepts but does not prove this exact export field or normalized layout. | Confirm the local source, field label, currency basis, and date semantics. | Normalized PPAR multi-currency demo field; exact Axys/APX source and label require local validation. |
| FX rates | `TO_CURRENCY` | Portfolio base currency produced by the rate. | Unknown | Low / Medium | The research supports multi-currency concepts but does not prove this exact export field or normalized layout. | Confirm the local source, field label, currency basis, and date semantics. | Normalized PPAR multi-currency demo field; exact Axys/APX source and label require local validation. |
| FX rates | `RATE_DATE` | Effective date of the FX rate. | Unknown | Low / Medium | The research supports multi-currency concepts but does not prove this exact export field or normalized layout. | Confirm the local source, field label, currency basis, and date semantics. | Normalized PPAR multi-currency demo field; exact Axys/APX source and label require local validation. |
| FX rates | `FX_RATE` | Units of base currency per unit of local currency. | Unknown | Low / Medium | The research supports multi-currency concepts but does not prove this exact export field or normalized layout. | Confirm the local source, field label, currency basis, and date semantics. | Normalized PPAR multi-currency demo field; exact Axys/APX source and label require local validation. |
| FX rates | `RATE_SOURCE` | Provenance label for the normalized FX rate. | Unknown | Low / Medium | The research supports multi-currency concepts but does not prove this exact export field or normalized layout. | Confirm the local source, field label, currency basis, and date semantics. | Normalized PPAR multi-currency demo field; exact Axys/APX source and label require local validation. |
| FX rates | `RATE_TYPE` | Rate convention such as closing or average. | Unknown | Low / Medium | The research supports multi-currency concepts but does not prove this exact export field or normalized layout. | Confirm the local source, field label, currency basis, and date semantics. | Normalized PPAR multi-currency demo field; exact Axys/APX source and label require local validation. |
| FX rates | `LOCAL_EXPOSURE` | Local-currency exposure explicitly linked to the portfolio and rate. | Unknown | Low / Medium | The research supports multi-currency concepts but does not prove this exact export field or normalized layout. | Confirm the local source, field label, currency basis, and date semantics. | Normalized PPAR multi-currency demo field; exact Axys/APX source and label require local validation. |


### `splits.csv`

| Dataset | Demo column | Normalized meaning | IMEX confidence | REP confidence | Evidence basis | Open questions | Comments |
|---|---|---|---|---|---|---|---|
| split factors | `SEC` | Security identifier for the split-factor row. | Medium / High | Medium | Chapter_09_Corporate_Actions.md and Research_09_Corporate_Actions.md identify Axys `split.inf` as securities splits data and cite consultant date/symbol/factor merge evidence. | Confirm whether the local site can export split.inf or an equivalent split-factor report. | Axys split-file research supports security-level split records; exact local export field names must be validated. |
| split factors | `SPLIT_DATE` | Effective date for the split factor. | Medium / High | Medium | Chapter_09_Corporate_Actions.md and Research_09_Corporate_Actions.md cite AdventGuru/Kevin Shea logical `SplitDate` evidence from exported split.inf files. | Confirm local date field name and whether the date is effective, ex-date, or another local split date basis. | Consultant split-file merge evidence uses a date field for exported split records. |
| split factors | `SPLIT_FACTOR` | Share multiplier or split factor. | Medium / High | Medium | Chapter_09_Corporate_Actions.md and Research_09_Corporate_Actions.md cite AdventGuru/Kevin Shea logical `SplitFactor` evidence from exported split.inf files. | Confirm whether the local factor convention is multiplier, ratio text, inverse factor, or another representation. | Consultant split-file merge evidence uses a factor field for exported split records; exact reverse-split convention is unknown. |


## Candidate Name Mapping

These names are candidate aliases for local discovery. They are not assertions that the packaged demo headers are official Axys/APX IMEX or REP names.

| Label | Meaning |
|---|---|
| Inferred Alias | Likely field/report aliases for the underlying value, but not proven as official native Axys/APX names by the local corpus. |
| Report Label Inferred | Likely report labels or report-style aliases; structured IMEX names still need local confirmation. |
| Normalized Demo Only | A normalized demo-system name. Do not treat it as a native Axys/APX or REP label. |

### `holdings.csv` Name Candidates

| Dataset | Demo column | Candidate Axys/APX export names | Candidate report labels | Name confidence | Notes |
|---|---|---|---|---|---|
| holdings | `PORT` | PORT, Portfolio, Portfolio Code, Account | Portfolio, Account | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| holdings | `SEC` | SEC, Security, Symbol, Security ID | Security, Symbol, Security ID | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| holdings | `HOLDING_DATE` | Date, As Of Date, Holding Date | As Of Date, Report Date | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| holdings | `CURRENCY` | Currency, Local Currency | Currency, Local Currency | Inferred Alias | Normalized demo name only; confirm the exact local Axys/APX label. |
| holdings | `BASE_CURRENCY` | Base Currency, Portfolio Currency | Base Currency, Reporting Currency | Inferred Alias | Normalized demo name only; confirm the exact local Axys/APX label. |
| holdings | `QTY` | QTY, Quantity, Shares, Units | Quantity, Shares, Units | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| holdings | `PRICE` | PRICE, Price, Market Price | Price, Market Price | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| holdings | `MKT_VAL` | MKT_VAL, Market Value, MarketVal | Market Value | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| holdings | `BASE_MKT_VAL` | Base Market Value, Reporting Market Value | Base Market Value, Reporting Market Value | Inferred Alias | Normalized demo name only; confirm the exact local Axys/APX label. |
| holdings | `ACCRUED` | ACCRUED, Accrued Interest, Accrued Income | Accrued Interest, Accrued Income | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |


### `portperf.csv` Name Candidates

| Dataset | Demo column | Candidate Axys/APX export names | Candidate report labels | Name confidence | Notes |
|---|---|---|---|---|---|
| portfolio performance | `END_MV` | Ending Market Value, End Market Value | Ending Market Value, End MV | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| portfolio performance | `FLOW` | External Flow, Flow, Net Flow | Flow, External Flow, Net Contributions/Withdrawals | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| portfolio performance | `INCOME` | Income | Income | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| portfolio performance | `GAIN_LOSS` | Gain/Loss, Gain Loss | Gain/Loss, Realized/Unrealized Gain/Loss | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| portfolio performance | `PORTFOLIO_CODE` | PORT, Portfolio, Portfolio Code, Account | Portfolio, Account | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| portfolio performance | `FROM_DATE` | From Date, Start Date | From Date, Beginning Date | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| portfolio performance | `THRU_DATE` | Thru Date, Through Date, End Date | Thru Date, Ending Date | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| portfolio performance | `BEGIN_MV` | Beginning Market Value, Begin Market Value | Beginning Market Value, Begin MV | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| portfolio performance | `PORT_RETURN` | Return, Portfolio Return | Portfolio Return, Total Return | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| portfolio performance | `BASE_CURRENCY` | Base Currency, Portfolio Currency | Base Currency, Reporting Currency | Inferred Alias | Normalized demo name only; confirm the exact local Axys/APX label. |


### `secperf.csv` Name Candidates

| Dataset | Demo column | Candidate Axys/APX export names | Candidate report labels | Name confidence | Notes |
|---|---|---|---|---|---|
| security performance | `END_MV` | Ending Market Value, End Market Value | Ending Market Value, End MV | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| security performance | `INCOME` | Income | Income | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| security performance | `GAIN_LOSS` | Gain/Loss, Gain Loss | Gain/Loss, Realized/Unrealized Gain/Loss | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| security performance | `PORTFOLIO_CODE` | PORT, Portfolio, Portfolio Code, Account | Portfolio, Account | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| security performance | `SECURITY_ID` | SEC, Security, Symbol, Security ID | Security, Symbol, Security ID | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| security performance | `FROM_DATE` | From Date, Start Date | From Date, Beginning Date | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| security performance | `THRU_DATE` | Thru Date, Through Date, End Date | Thru Date, Ending Date | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| security performance | `BEGIN_WEIGHT` | Beginning Weight, Begin Weight | Beginning Weight, Begin Wt | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| security performance | `BEGIN_MV` | Beginning Market Value, Begin Market Value | Beginning Market Value, Begin MV | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| security performance | `SEC_RETURN` | Return, Security Return | Security Return, Total Return | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| security performance | `CONTRIBUTION` | Contribution, Contribution to Return | Contribution, Contribution to Return | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |


### `transactions.csv` Name Candidates

| Dataset | Demo column | Candidate Axys/APX export names | Candidate report labels | Name confidence | Notes |
|---|---|---|---|---|---|
| transactions | `PORT` | PORT, Portfolio, Account | Portfolio, Account | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| transactions | `TRANSACTION_DATE` | Trade Date, Transaction Date | Trade Date, Transaction Date | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| transactions | `SETTLE_DATE` | Settle Date, Settlement Date, Pay Date | Settle Date, Settlement Date | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| transactions | `SEC` | SEC, Security, Symbol, Security ID | Security, Symbol, Security ID | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| transactions | `TRAN` | Transaction Code, Tran, Transaction Type | Transaction Code, Transaction Type | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| transactions | `SEC_TYPE` | Security Type, Sec Type | Security Type | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| transactions | `SRC_DEST_TYPE` | Source/Destination Type, Src Dest Type | Source/Destination Type | Normalized Demo Only | Normalized demo header retained for ppar consistency; native Axys/APX naming is not established by the local corpus. |
| transactions | `SRC_DEST_SYMBOL` | Source/Destination Symbol, Src Dest Symbol | Source/Destination Symbol | Normalized Demo Only | Normalized demo header retained for ppar consistency; native Axys/APX naming is not established by the local corpus. |
| transactions | `SPECIAL_SEC_TYPE` | Special Security Type, Special Sec Type | Special Security Type | Normalized Demo Only | Normalized demo header retained for ppar consistency; native Axys/APX naming is not established by the local corpus. |
| transactions | `SPECIAL_SEC_SYMBOL` | Special Security Symbol, Special Sec Symbol | Special Security Symbol | Normalized Demo Only | Normalized demo header retained for ppar consistency; native Axys/APX naming is not established by the local corpus. |
| transactions | `CURRENCY` | Currency, Transaction Currency | Currency, Transaction Currency | Inferred Alias | Normalized demo name only; confirm the exact local Axys/APX label. |
| transactions | `BASE_CURRENCY` | Base Currency, Portfolio Currency | Base Currency, Reporting Currency | Inferred Alias | Normalized demo name only; confirm the exact local Axys/APX label. |
| transactions | `QTY` | QTY, Quantity, Shares, Units | Quantity, Shares, Units | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| transactions | `PRICE` | PRICE, Price | Price | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| transactions | `AMOUNT` | Amount, Net Amount, Cash Amount | Amount, Net Amount, Cash Amount | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| transactions | `BASE_AMOUNT` | Base Amount, Reporting Amount | Base Amount, Reporting Amount | Inferred Alias | Normalized demo name only; confirm the exact local Axys/APX label. |
| transactions | `COMMISSION` | Commission, Commissions | Commission | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |


### `fx_rates.csv` Name Candidates

| Dataset | Demo column | Candidate Axys/APX export names | Candidate report labels | Name confidence | Notes |
|---|---|---|---|---|---|
| FX rates | `PORT` | Portfolio, Account | Portfolio, Account | Inferred Alias | Normalized demo name only; confirm the exact local Axys/APX label. |
| FX rates | `FROM_CURRENCY` | From Currency, Local Currency | From Currency, Local Currency | Inferred Alias | Normalized demo name only; confirm the exact local Axys/APX label. |
| FX rates | `TO_CURRENCY` | To Currency, Base Currency | To Currency, Base Currency | Inferred Alias | Normalized demo name only; confirm the exact local Axys/APX label. |
| FX rates | `RATE_DATE` | Rate Date, Price Date | Rate Date, As Of Date | Inferred Alias | Normalized demo name only; confirm the exact local Axys/APX label. |
| FX rates | `FX_RATE` | FX Rate, Exchange Rate | FX Rate, Exchange Rate | Inferred Alias | Normalized demo name only; confirm the exact local Axys/APX label. |
| FX rates | `RATE_SOURCE` | Rate Source, Price Source | Rate Source, Price Source | Inferred Alias | Normalized demo name only; confirm the exact local Axys/APX label. |
| FX rates | `RATE_TYPE` | Rate Type, Price Type | Rate Type, Price Type | Inferred Alias | Normalized demo name only; confirm the exact local Axys/APX label. |
| FX rates | `LOCAL_EXPOSURE` | Local Exposure, Local Market Value | Local Exposure, Local Market Value | Inferred Alias | Normalized demo name only; confirm the exact local Axys/APX label. |


### `splits.csv` Name Candidates

| Dataset | Demo column | Candidate Axys/APX export names | Candidate report labels | Name confidence | Notes |
|---|---|---|---|---|---|
| split factors | `SEC` | SEC, SplitSymbol, Security, Symbol | Security, Symbol, Split Symbol | Inferred Alias | `SplitSymbol` is consultant-derived logical evidence, not a verified official Axys header. |
| split factors | `SPLIT_DATE` | SPLIT_DATE, SplitDate, Effective Date | Split Date, Effective Date | Inferred Alias | `SplitDate` is consultant-derived logical evidence, not a verified official Axys header. |
| split factors | `SPLIT_FACTOR` | SPLIT_FACTOR, SplitFactor, Factor | Split Factor, Factor | Inferred Alias | `SplitFactor` is consultant-derived logical evidence, not a verified official Axys header. |


## Source Strategy Matrix

This matrix translates availability confidence into implementation guidance about where each value is most likely to come from. Extraction requirements are defined only by the three-category checklist above; internal runtime guard flags are intentionally not shown here.

| Label | Meaning |
|---|---|
| IMEX preferred | Use IMEX first when the local profile exposes the field. |
| REP preferred | Use REP/report output first because report-tie behavior matters. |
| IMEX or REP | Either IMEX or REP is acceptable after local field validation. |
| IMEX then REP cross-check | Use IMEX as the primary feed and REP as a report-output cross-check. |
| Local discovery required | Do not assume a source; validate local IMEX/REP/custom-report support first. |

### `holdings.csv` Source Strategy

| Dataset | Demo column | Preferred source | Fallback source | Notes |
|---|---|---|---|---|
| holdings | `PORT` | IMEX or REP | REP preferred | Holdings reconstruction can use either validated position/appraisal IMEX output or an appraisal-style REP extract. |
| holdings | `SEC` | IMEX or REP | REP preferred | Holdings reconstruction can use either validated position/appraisal IMEX output or an appraisal-style REP extract. |
| holdings | `HOLDING_DATE` | IMEX or REP | REP preferred | Holdings reconstruction can use either validated position/appraisal IMEX output or an appraisal-style REP extract. |
| holdings | `CURRENCY` | Local discovery required | REP preferred | Validate the local extract and currency basis against a report sample before use. |
| holdings | `BASE_CURRENCY` | Local discovery required | REP preferred | Validate the local extract and currency basis against a report sample before use. |
| holdings | `QTY` | IMEX or REP | REP preferred | Holdings reconstruction can use either validated position/appraisal IMEX output or an appraisal-style REP extract. |
| holdings | `PRICE` | IMEX or REP | REP preferred | Holdings reconstruction can use either validated position/appraisal IMEX output or an appraisal-style REP extract. |
| holdings | `MKT_VAL` | IMEX or REP | REP preferred | Holdings reconstruction can use either validated position/appraisal IMEX output or an appraisal-style REP extract. |
| holdings | `BASE_MKT_VAL` | Local discovery required | REP preferred | Validate the local extract and currency basis against a report sample before use. |
| holdings | `ACCRUED` | IMEX or REP | Local discovery required | Accrued income can affect performance reconciliation, so validate the field before running accrual-sensitive comparisons. |


### `portperf.csv` Source Strategy

| Dataset | Demo column | Preferred source | Fallback source | Notes |
|---|---|---|---|---|
| portfolio performance | `END_MV` | REP preferred | Local discovery required | Prefer report output because the local corpus does not prove native performance IMEX object names or calculation basis. |
| portfolio performance | `FLOW` | REP preferred | Local discovery required | Prefer report output because the local corpus does not prove native performance IMEX object names or calculation basis. |
| portfolio performance | `INCOME` | REP preferred | Local discovery required | Prefer report output because the local corpus does not prove native performance IMEX object names or calculation basis. |
| portfolio performance | `GAIN_LOSS` | REP preferred | Local discovery required | Prefer report output because the local corpus does not prove native performance IMEX object names or calculation basis. |
| portfolio performance | `PORTFOLIO_CODE` | IMEX or REP | REP preferred | Portfolio identifier must be present regardless of the extract source. |
| portfolio performance | `FROM_DATE` | REP preferred | Local discovery required | Prefer report output because the local corpus does not prove native performance IMEX object names or calculation basis. |
| portfolio performance | `THRU_DATE` | REP preferred | Local discovery required | Prefer report output because the local corpus does not prove native performance IMEX object names or calculation basis. |
| portfolio performance | `BEGIN_MV` | REP preferred | Local discovery required | Prefer report output because the local corpus does not prove native performance IMEX object names or calculation basis. |
| portfolio performance | `PORT_RETURN` | REP preferred | Local discovery required | Prefer report output because the local corpus does not prove native performance IMEX object names or calculation basis. |
| portfolio performance | `BASE_CURRENCY` | Local discovery required | REP preferred | Validate the local extract and currency basis against a report sample before use. |


### `secperf.csv` Source Strategy

| Dataset | Demo column | Preferred source | Fallback source | Notes |
|---|---|---|---|---|
| security performance | `END_MV` | REP preferred | Local discovery required | Prefer report output because the local corpus does not prove native performance IMEX object names or calculation basis. |
| security performance | `INCOME` | REP preferred | Local discovery required | Prefer report output because the local corpus does not prove native performance IMEX object names or calculation basis. |
| security performance | `GAIN_LOSS` | REP preferred | Local discovery required | Prefer report output because the local corpus does not prove native performance IMEX object names or calculation basis. |
| security performance | `PORTFOLIO_CODE` | IMEX or REP | REP preferred | Portfolio and security identifiers must be present regardless of the extract source. |
| security performance | `SECURITY_ID` | IMEX or REP | REP preferred | Portfolio and security identifiers must be present regardless of the extract source. |
| security performance | `FROM_DATE` | REP preferred | Local discovery required | Prefer report output because the local corpus does not prove native performance IMEX object names or calculation basis. |
| security performance | `THRU_DATE` | REP preferred | Local discovery required | Prefer report output because the local corpus does not prove native performance IMEX object names or calculation basis. |
| security performance | `BEGIN_WEIGHT` | REP preferred | Local discovery required | Attribution-style values are report-sensitive; do not assume IMEX availability. |
| security performance | `BEGIN_MV` | REP preferred | Local discovery required | Prefer report output because the local corpus does not prove native performance IMEX object names or calculation basis. |
| security performance | `SEC_RETURN` | REP preferred | Local discovery required | Prefer report output because the local corpus does not prove native performance IMEX object names or calculation basis. |
| security performance | `CONTRIBUTION` | REP preferred | Local discovery required | Attribution-style values are report-sensitive; do not assume IMEX availability. |


### `transactions.csv` Source Strategy

| Dataset | Demo column | Preferred source | Fallback source | Notes |
|---|---|---|---|---|
| transactions | `PORT` | IMEX then REP cross-check | REP preferred | Transaction core fields are IMEX-suitable, with REP useful for report-facing cross-checks and sign-convention validation. |
| transactions | `TRANSACTION_DATE` | IMEX then REP cross-check | REP preferred | Transaction core fields are IMEX-suitable, with REP useful for report-facing cross-checks and sign-convention validation. |
| transactions | `SETTLE_DATE` | IMEX then REP cross-check | REP preferred | Transaction core fields are IMEX-suitable, with REP useful for report-facing cross-checks and sign-convention validation. |
| transactions | `SEC` | IMEX then REP cross-check | REP preferred | Transaction core fields are IMEX-suitable, with REP useful for report-facing cross-checks and sign-convention validation. |
| transactions | `TRAN` | IMEX then REP cross-check | REP preferred | Transaction core fields are IMEX-suitable, with REP useful for report-facing cross-checks and sign-convention validation. |
| transactions | `SEC_TYPE` | Local discovery required | REP preferred | Required for ambiguous Axys/APX flow semantics. If IMEX does not expose this context, stop and use REP, a custom report, or another local source. |
| transactions | `SRC_DEST_TYPE` | Local discovery required | REP preferred | Required for ambiguous Axys/APX flow semantics. If IMEX does not expose this context, stop and use REP, a custom report, or another local source. |
| transactions | `SRC_DEST_SYMBOL` | Local discovery required | REP preferred | Required for ambiguous Axys/APX flow semantics. If IMEX does not expose this context, stop and use REP, a custom report, or another local source. |
| transactions | `SPECIAL_SEC_TYPE` | Local discovery required | REP preferred | Required for ambiguous Axys/APX flow semantics. If IMEX does not expose this context, stop and use REP, a custom report, or another local source. |
| transactions | `SPECIAL_SEC_SYMBOL` | Local discovery required | REP preferred | Required for ambiguous Axys/APX flow semantics. If IMEX does not expose this context, stop and use REP, a custom report, or another local source. |
| transactions | `CURRENCY` | Local discovery required | REP preferred | Validate the local extract and currency basis against a report sample before use. |
| transactions | `BASE_CURRENCY` | Local discovery required | REP preferred | Validate the local extract and currency basis against a report sample before use. |
| transactions | `QTY` | IMEX then REP cross-check | REP preferred | Transaction core fields are IMEX-suitable, with REP useful for report-facing cross-checks and sign-convention validation. |
| transactions | `PRICE` | IMEX then REP cross-check | REP preferred | Transaction core fields are IMEX-suitable, with REP useful for report-facing cross-checks and sign-convention validation. |
| transactions | `AMOUNT` | IMEX then REP cross-check | REP preferred | Transaction core fields are IMEX-suitable, with REP useful for report-facing cross-checks and sign-convention validation. |
| transactions | `BASE_AMOUNT` | Local discovery required | REP preferred | Validate the local extract and currency basis against a report sample before use. |
| transactions | `COMMISSION` | IMEX then REP cross-check | REP preferred | Transaction core fields are IMEX-suitable, with REP useful for report-facing cross-checks and sign-convention validation. |


### `fx_rates.csv` Source Strategy

| Dataset | Demo column | Preferred source | Fallback source | Notes |
|---|---|---|---|---|
| FX rates | `PORT` | Local discovery required | REP preferred | Validate the local extract and currency basis against a report sample before use. |
| FX rates | `FROM_CURRENCY` | Local discovery required | REP preferred | Validate the local extract and currency basis against a report sample before use. |
| FX rates | `TO_CURRENCY` | Local discovery required | REP preferred | Validate the local extract and currency basis against a report sample before use. |
| FX rates | `RATE_DATE` | Local discovery required | REP preferred | Validate the local extract and currency basis against a report sample before use. |
| FX rates | `FX_RATE` | Local discovery required | REP preferred | Validate the local extract and currency basis against a report sample before use. |
| FX rates | `RATE_SOURCE` | Local discovery required | REP preferred | Validate the local extract and currency basis against a report sample before use. |
| FX rates | `RATE_TYPE` | Local discovery required | REP preferred | Validate the local extract and currency basis against a report sample before use. |
| FX rates | `LOCAL_EXPOSURE` | Local discovery required | REP preferred | Validate the local extract and currency basis against a report sample before use. |


### `splits.csv` Source Strategy

| Dataset | Demo column | Preferred source | Fallback source | Notes |
|---|---|---|---|---|
| split factors | `SEC` | IMEX or REP | Local discovery required | Prefer a direct split.inf/exported split-factor source; use REP/custom reports only if they expose security-level split factors. |
| split factors | `SPLIT_DATE` | IMEX or REP | Local discovery required | Validate the local date basis before comparing split factors to holdings and prices. |
| split factors | `SPLIT_FACTOR` | IMEX or REP | Local discovery required | Treat split factors as context evidence explaining holdings changes, not as cash-flow transactions. |


## Practical Extraction Guidance

| Need | Preferred path | Reason |
|---|---|---|
| Holdings values for return reconstruction | IMEX or REP | Holdings/positions are relatively well-supported by both export and appraisal/report paths. |
| Transaction amount, quantity, price, and commission | IMEX first, REP as cross-check | Transaction import/export evidence is strong for core transaction data; REP can validate user-visible report output. |
| Ambiguous external-flow classification for `li`, `lo`, `dp`, `wd`-style rows | IMEX only if source/destination and special-security context is exposed; otherwise REP/custom report or another source is required | Code alone is not enough for all cases. The required context fields must be proven locally. |
| Portfolio and security reported returns | REP/report extract preferred | The local reference corpus treats performance IMEX objects and fields as Unknown/validate-locally. |
| Security classifications | IMEX or security-master report extract | Asset class, sector, country, currency, and industry are plausible, but history/timing must be validated. |

## Local Validation Checklist

Before treating a site extract as equivalent to these demo files, collect:

1. IMEX profile or REP report name.
2. Axys/APX version and client/reporting tool version.
3. Exact field names and aliases used in the export.
4. Report parameters, date basis, portfolio list, currency basis, and gross/net return basis.
5. A paired report/export sample for at least one portfolio and period.
6. Transaction examples for `li`, `lo`, `dp`, and `wd` with source/destination and special-security context.
7. Evidence whether performance fields are stored values, report-calculated values, or export-calculated values.

## Related References

- [Chapter_05_Transactions.md](../reference/Chapter_05_Transactions.md)
- [Chapter_06_Holdings.md](../reference/Chapter_06_Holdings.md)
- [Chapter_10_Performance.md](../reference/Chapter_10_Performance.md)
- [Chapter_12_Imex.md](../reference/Chapter_12_Imex.md)
- [Chapter_13_Rep.md](../reference/Chapter_13_Rep.md)
- [Chapter_15_Data_Dictionary.md](../reference/Chapter_15_Data_Dictionary.md)
- [axys_apx_common_core_export.md](../axys_apx_common_core_export.md)
- [demo_source_contract.md](../../audit/demo_source_contract.md)
