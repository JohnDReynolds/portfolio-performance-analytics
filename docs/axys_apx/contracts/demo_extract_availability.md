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
- `secmast.csv`
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
| `holdings.csv` | Required | Beginning and ending base-currency values provide the Modified Dietz valuation inputs used to explain a changed return. Likely source: IMEX positions/holdings export or a REP appraisal report. | `Portfolio Code`, `Security Symbol`, `Security Type`, `Holding Date`, `Market Value` | `Currency Code` — Required for holdings whose local currency differs from the portfolio base currency.<br>`Base Currency` — Required for multi-currency portfolios unless a validated portfolio-level source supplies it.<br>`Base Market Value` — Required when MKT_VAL is local-currency rather than portfolio-base value.<br>`Accrued Income` — Required when accrued income is stated separately from MKT_VAL and affects beginning or ending value. | `Quantity`, `Price` |
| `portperf.csv` | Required | The portfolio, period, and reported return define the performance difference PPAR must explain. Likely source: REP performance report preferred; a native performance IMEX object is not verified by the current evidence. | `Portfolio Code`, `From Date`, `Thru Date`, `Portfolio Return` | `Base Currency` — Required when the portfolio contains holdings or transactions outside its reporting currency. | None |
| `secperf.csv` | Required only when applicable | Required only when the user wants security-level differences to reach Fully Explained; portfolio-only audit does not need this file. Likely source: REP security-performance or attribution report preferred; a native performance IMEX object is not verified by the current evidence. | `Portfolio Code`, `Security Symbol`, `Security Type`, `From Date`, `Thru Date`, `Security Return` | None | None |
| `transactions.csv` | Required | Dated, classified amounts are needed to explain changed external flows, income, fees, and security activity. Likely source: IMEX transaction export first; use REP/custom output or another reviewed source when IMEX omits transaction-semantics context. | `Portfolio Code`, `Transaction Date`, `Security Symbol`, `Security Type`, `Transaction Code`, `Amount` | `Transaction Security Type` — Required by the packaged guard when ambiguous DP, LI, LO, TI, or WD codes can appear.<br>`Source/Destination Type` — Required by the packaged guard when ambiguous DP, LI, LO, TI, or WD codes can appear.<br>`Source/Destination Symbol` — Required by the packaged guard when ambiguous DP, LI, LO, TI, or WD codes can appear.<br>`Special Security Type` — Required by the packaged guard when ambiguous DP, LI, LO, TI, or WD codes can appear.<br>`Special Security Symbol` — Required by the packaged guard when ambiguous DP, LI, LO, TI, or WD codes can appear.<br>`Currency Code` — Required for transaction amounts stated in a currency other than portfolio base currency.<br>`Base Currency` — Required for multi-currency portfolios unless the authoritative portfolio-performance row supplies it.<br>`Base Amount` — Required when AMOUNT is local-currency rather than the portfolio-base amount used by the performance calculation. | `Settlement Date`, `Quantity`, `Price`, `Commission` |
| `secmast.csv` | Required only when applicable | Security master values qualify Data Issues populations without changing performance calculations. Likely source: IMEX security-information export or another reviewed security-master source extract. | `Security Symbol`, `Security Type` | `Asset Class Code` — Required when a Data Issues filter references security_master.asset_class_code. | None |
| `splits.csv` | Optional | Split factors add review context but do not directly enter the current Modified Dietz explanation formula. Likely source: Direct split.inf or local split-factor export; use REP/custom output only when it exposes equivalent factors. | `Security Symbol`, `Security Type` | None | `Split Date`, `Split Factor` |


## Availability Matrix

### `holdings.csv`

| Dataset | Demo column | Normalized meaning | IMEX confidence | REP confidence | Evidence basis | Open questions | Comments |
|---|---|---|---|---|---|---|---|
| holdings | `Portfolio Code` | Portfolio/account identifier. | High | High | Position/appraisal evidence in Chapter_06_Holdings.md, Chapter_12_Imex.md, Chapter_13_Rep.md, and axys_apx_common_core_export.md. | Confirm exact local IMEX profile and field name. | Core portfolio/client identifier in position and appraisal workflows. |
| holdings | `Security Symbol` | Security identifier. | High | High | Position/appraisal evidence in Chapter_06_Holdings.md, Chapter_12_Imex.md, Chapter_13_Rep.md, and axys_apx_common_core_export.md. | Confirm exact local IMEX profile and field name. | Security identifiers are supported by CI/security-resolution and report evidence. |
| holdings | `Security Type` | Security-type component paired with Security Symbol to construct PPAR security_id. | High | Medium | Chapter_04_Security_Master.md and Research_04_Security_Master.md verify security-type exports and mixed-case examples in integration workflows. | Confirm the local security-type field, value dictionary, case rules, and version applicability. | Symbol alone is not unique; the reviewed type/symbol pair defines source security identity. |
| holdings | `Holding Date` | As-of date for holdings. | High | High | Position/appraisal evidence in Chapter_06_Holdings.md, Chapter_12_Imex.md, Chapter_13_Rep.md, and axys_apx_common_core_export.md. | Confirm exact local IMEX profile and field name. | Position exports and appraisal reports are inherently as-of-date based. |
| holdings | `Accrued Income` | Accrued income or accrued interest. | Medium | Medium | Fixed-income/accrual handling is documented as performance-sensitive; common-core holdings reference treats accrued income as plausible. | Confirm accrued-income field, date basis, and fixed-income treatment. | Common for fixed-income holdings, but exact export/report field requires validation. |
| holdings | `Base Currency` | Portfolio reporting currency used for the holding row. | Unknown | Low / Medium | The research supports multi-currency concepts but does not prove this exact export field or normalized layout. | Confirm the local source, field label, currency basis, and date semantics. | Normalized PPAR multi-currency demo field; exact Axys/APX source and label require local validation. |
| holdings | `Base Market Value` | Holding market value translated to portfolio base currency. | Unknown | Low / Medium | The research supports multi-currency concepts but does not prove this exact export field or normalized layout. | Confirm the local source, field label, currency basis, and date semantics. | Normalized PPAR multi-currency demo field; exact Axys/APX source and label require local validation. |
| holdings | `Currency Code` | Local currency of the holding row. | Unknown | Low / Medium | The research supports multi-currency concepts but does not prove this exact export field or normalized layout. | Confirm the local source, field label, currency basis, and date semantics. | Normalized PPAR multi-currency demo field; exact Axys/APX source and label require local validation. |
| holdings | `Market Value` | Market value. | High | High | Position/appraisal evidence in Chapter_06_Holdings.md, Chapter_12_Imex.md, Chapter_13_Rep.md, and axys_apx_common_core_export.md. | Confirm exact local IMEX profile and field name. | Market value appears in report evidence and is a core position/appraisal value. |
| holdings | `Price` | Market price used for valuation. | High | High | Position/appraisal evidence in Chapter_06_Holdings.md, Chapter_12_Imex.md, Chapter_13_Rep.md, and axys_apx_common_core_export.md. | Confirm exact local IMEX profile and field name. | Price import/export evidence and appraisal/report labels support availability. |
| holdings | `Quantity` | Quantity, shares, or units. | High | High | Position/appraisal evidence in Chapter_06_Holdings.md, Chapter_12_Imex.md, Chapter_13_Rep.md, and axys_apx_common_core_export.md. | Confirm exact local IMEX profile and field name. | Quantity is directly supported by transaction and appraisal report evidence. |


### `portperf.csv`

| Dataset | Demo column | Normalized meaning | IMEX confidence | REP confidence | Evidence basis | Open questions | Comments |
|---|---|---|---|---|---|---|---|
| portfolio performance | `Portfolio Code` | Portfolio/account identifier. | High | High | Chapter_10_Performance.md treats portfolio performance as a report/extract boundary and prefers REP for report-tie values. | Confirm whether a local IMEX performance object exists or use REP output. | Core identifier. |
| portfolio performance | `From Date` | Period start date. | Medium | High | Chapter_10_Performance.md treats portfolio performance as a report/extract boundary and prefers REP for report-tie values. | Confirm whether a local IMEX performance object exists or use REP output. | Report parameters reliably provide the period; structured export field names require validation. |
| portfolio performance | `Thru Date` | Period end date. | Medium | High | Chapter_10_Performance.md treats portfolio performance as a report/extract boundary and prefers REP for report-tie values. | Confirm whether a local IMEX performance object exists or use REP output. | Report parameters reliably provide the period; structured export field names require validation. |
| portfolio performance | `Portfolio Return` | Portfolio return. | Medium / Unknown | High | Performance chapter and report evidence support the concept, but the local corpus does not prove a native IMEX performance object/field. | Confirm methodology, gross/net basis, and stored-vs-report-calculated behavior. | Performance-history IMEX fields are not established; REP is preferred for report-tie values. |
| portfolio performance | `Base Currency` | Authoritative portfolio reporting currency for the performance row. | Unknown | Low / Medium | The research supports multi-currency concepts but does not prove this exact export field or normalized layout. | Confirm the local source, field label, currency basis, and date semantics. | Normalized PPAR multi-currency demo field; exact Axys/APX source and label require local validation. |


### `secperf.csv`

| Dataset | Demo column | Normalized meaning | IMEX confidence | REP confidence | Evidence basis | Open questions | Comments |
|---|---|---|---|---|---|---|---|
| security performance | `Portfolio Code` | Portfolio/account identifier. | High | High | Chapter_10_Performance.md treats portfolio performance as a report/extract boundary and prefers REP for report-tie values. | Confirm whether values come from security performance, appraisal, or custom REP output. | Core identifier. |
| security performance | `Security Symbol` | Security identifier. | High | High | Chapter_10_Performance.md treats portfolio performance as a report/extract boundary and prefers REP for report-tie values. | Confirm whether values come from security performance, appraisal, or custom REP output. | Core security-row identifier. |
| security performance | `Security Type` | Security-type component paired with Security Symbol to construct PPAR security_id. | High | Medium | Chapter_04_Security_Master.md and Research_04_Security_Master.md verify security-type exports and mixed-case examples in integration workflows. | Confirm the local security-type field, value dictionary, case rules, and version applicability. | Symbol alone is not unique; the reviewed type/symbol pair defines source security identity. |
| security performance | `From Date` | Period start date. | Medium | High | Chapter_10_Performance.md treats portfolio performance as a report/extract boundary and prefers REP for report-tie values. | Confirm whether values come from security performance, appraisal, or custom REP output. | Report parameter/output availability is likely; structured export names require validation. |
| security performance | `Thru Date` | Period end date. | Medium | High | Chapter_10_Performance.md treats portfolio performance as a report/extract boundary and prefers REP for report-tie values. | Confirm whether values come from security performance, appraisal, or custom REP output. | Report parameter/output availability is likely; structured export names require validation. |
| security performance | `Security Return` | Security return. | Medium / Unknown | High | Performance chapter and report evidence support the concept, but the local corpus does not prove a native IMEX performance object/field. | Confirm methodology, contribution basis, and stored-vs-report-calculated behavior. | Performance-history IMEX fields are not established; REP is preferred for report-tie values. |


### `transactions.csv`

| Dataset | Demo column | Normalized meaning | IMEX confidence | REP confidence | Evidence basis | Open questions | Comments |
|---|---|---|---|---|---|---|---|
| transactions | `Portfolio Code` | Portfolio/account identifier. | High | High | Transaction import/export and report evidence in Chapter_05_Transactions.md, Chapter_12_Imex.md, and Chapter_13_Rep.md. | Confirm exact posted-transaction export fields and sign convention. | Core transaction/account field. |
| transactions | `Security Symbol` | Security identifier. | High | High | Transaction import/export and report evidence in Chapter_05_Transactions.md, Chapter_12_Imex.md, and Chapter_13_Rep.md. | Confirm exact posted-transaction export fields and sign convention. | Core transaction/security field. |
| transactions | `Security Type` | Security-type component paired with Security Symbol to construct PPAR security_id. | High | Medium | Chapter_04_Security_Master.md and Research_04_Security_Master.md verify security-type exports and mixed-case examples in integration workflows. | Confirm the local security-type field, value dictionary, case rules, and version applicability. | Symbol alone is not unique; the reviewed type/symbol pair defines source security identity. |
| transactions | `Transaction Date` | Trade or economic date. | High | High | Transaction import/export and report evidence in Chapter_05_Transactions.md, Chapter_12_Imex.md, and Chapter_13_Rep.md. | Confirm exact posted-transaction export fields and sign convention. | Trade date appears in report evidence and transaction import workflows. |
| transactions | `Amount` | Net amount, proceeds, or cash amount. | High | Medium / High | Transaction import/export and report evidence in Chapter_05_Transactions.md, Chapter_12_Imex.md, and Chapter_13_Rep.md. | Confirm exact posted-transaction export fields and sign convention. | Core transaction amount is likely available, though report labels can vary. |
| transactions | `Base Amount` | Transaction amount translated to portfolio base currency. | Unknown | Low / Medium | The research supports multi-currency concepts but does not prove this exact export field or normalized layout. | Confirm the local source, field label, currency basis, and date semantics. | Normalized PPAR multi-currency demo field; exact Axys/APX source and label require local validation. |
| transactions | `Base Currency` | Portfolio reporting currency for the transaction. | Unknown | Low / Medium | The research supports multi-currency concepts but does not prove this exact export field or normalized layout. | Confirm the local source, field label, currency basis, and date semantics. | Normalized PPAR multi-currency demo field; exact Axys/APX source and label require local validation. |
| transactions | `Commission` | Commission. | Medium | Medium | Transaction import/export and report evidence in Chapter_05_Transactions.md, Chapter_12_Imex.md, and Chapter_13_Rep.md. | Confirm exact posted-transaction export fields and sign convention. | Supported in CI parameter evidence; report/profile availability should be validated. |
| transactions | `Currency Code` | Local currency of the transaction amount. | Unknown | Low / Medium | The research supports multi-currency concepts but does not prove this exact export field or normalized layout. | Confirm the local source, field label, currency basis, and date semantics. | Normalized PPAR multi-currency demo field; exact Axys/APX source and label require local validation. |
| transactions | `Price` | Transaction price. | High | High | Transaction import/export and report evidence in Chapter_05_Transactions.md, Chapter_12_Imex.md, and Chapter_13_Rep.md. | Confirm exact posted-transaction export fields and sign convention. | Price appears in transaction and report evidence. |
| transactions | `Quantity` | Transaction quantity. | High | High | Transaction import/export and report evidence in Chapter_05_Transactions.md, Chapter_12_Imex.md, and Chapter_13_Rep.md. | Confirm exact posted-transaction export fields and sign convention. | Quantity appears in transaction and report evidence. |
| transactions | `Settlement Date` | Settlement or pay date. | Medium | High | Transaction import/export and report evidence in Chapter_05_Transactions.md, Chapter_12_Imex.md, and Chapter_13_Rep.md. | Confirm exact posted-transaction export fields and sign convention. | Settle date appears in report-output evidence; IMEX availability is profile-dependent. |
| transactions | `Source/Destination Symbol` | Source/destination symbol for ambiguous flows. | Medium | Low / Medium | Observed CI/Axys/APX transaction translation context in Chapter_05_Transactions.md and Chapter_12_Imex.md; required for ambiguous external-flow handling. | Confirm whether local IMEX exposes this context on posted transaction exports.<br>Use REP, a custom report, or another source if IMEX cannot expose it. | Observed transaction translation/integration field, not guaranteed in every posted export. |
| transactions | `Source/Destination Type` | Source/destination type for ambiguous flows. | Medium | Low / Medium | Observed CI/Axys/APX transaction translation context in Chapter_05_Transactions.md and Chapter_12_Imex.md; required for ambiguous external-flow handling. | Confirm whether local IMEX exposes this context on posted transaction exports.<br>Use REP, a custom report, or another source if IMEX cannot expose it. | Observed transaction translation/integration field, not guaranteed in every posted export. |
| transactions | `Special Security Symbol` | Axys/APX symbol half of a special type/symbol transaction pair. | Medium | Low | Observed CI/Axys/APX examples use symbols such as `custfee`, `expense`, `with`, and `margin` paired with four-character special security types. | Confirm whether local IMEX exposes this context on posted transaction exports.<br>Use REP, a custom report, or another source if IMEX cannot expose it. | Supported in integration evidence; standard report exposure is uncertain. |
| transactions | `Special Security Type` | Axys/APX security-type half of a special type/symbol transaction pair. | Medium | Low | Observed CI/Axys/APX examples use four-character types such as `exus`, `epus`, and `caus`; reviewed `rc`/`pd` rows leave the pair blank. | Confirm whether local IMEX exposes this context on posted transaction exports.<br>Use REP, a custom report, or another source if IMEX cannot expose it.<br>Do not substitute free-form event labels for native type identifiers. | Supported in integration evidence; standard report exposure is uncertain. |
| transactions | `Transaction Code` | Transaction code or type. | Medium | Medium | Transaction import/export and report evidence in Chapter_05_Transactions.md, Chapter_12_Imex.md, and Chapter_13_Rep.md. | Confirm exact posted-transaction export fields and sign convention. | Transaction type/code is supported in translation/import evidence; report exposure depends on layout. |
| transactions | `Transaction Security Type` | Security type for transaction security. | Medium | Low / Medium | Observed CI/Axys/APX transaction translation context in Chapter_05_Transactions.md and Chapter_12_Imex.md; required for ambiguous external-flow handling. | Confirm whether local IMEX exposes this context on posted transaction exports.<br>Use REP, a custom report, or another source if IMEX cannot expose it. | Central to Axys/APX security resolution; transaction-row availability depends on export/report design. |


### `secmast.csv`

| Dataset | Demo column | Normalized meaning | IMEX confidence | REP confidence | Evidence basis | Open questions | Comments |
|---|---|---|---|---|---|---|---|
| security master | `Security Symbol` | Exact-case security identifier used to enrich holdings or transaction rows. | High | High | Chapter_04_Security_Master.md and Research_04_Security_Master.md support security-information exports and security-level report labels, while the formal native key remains unknown. | Confirm the local identifier field, exact-case behavior, uniqueness, and relationship to security type. | Security symbol or identifier is central to reviewed security-information and report workflows. |
| security master | `Security Type` | Product security-type code used to qualify Data Issues populations. | High | Medium | Chapter_04_Security_Master.md and Research_04_Security_Master.md verify security-type exports and mixed-case examples in integration workflows. | Confirm the local security-type field, value dictionary, case rules, and version applicability. | Security type is operationally paired with symbol in reviewed integration workflows, but the complete native dictionary is unknown. |
| security master | `Asset Class Code` | Site security-master asset-class code used to qualify Data Issues populations. | Medium | High | Chapter_04_Security_Master.md, Chapter_11_Classifications.md, and their evidence ledgers support asset-class export labels and classification behavior. | Confirm the local asset-class code field, code dictionary, effective-date behavior, and reclassification policy. | Asset-class labels are supported in export/report workflows, but exact native field names and histories remain site dependent. |


### `splits.csv`

| Dataset | Demo column | Normalized meaning | IMEX confidence | REP confidence | Evidence basis | Open questions | Comments |
|---|---|---|---|---|---|---|---|
| split factors | `Security Symbol` | Security identifier for the split-factor row. | Medium / High | Medium | Chapter_09_Corporate_Actions.md and Research_09_Corporate_Actions.md identify Axys `split.inf` as securities splits data and cite consultant date/symbol/factor merge evidence. | Confirm whether the local site can export split.inf or an equivalent split-factor report. | Axys split-file research supports security-level split records; exact local export field names must be validated. |
| split factors | `Security Type` | Security-type component paired with Security Symbol to construct PPAR security_id. | High | Medium | Chapter_04_Security_Master.md and Research_04_Security_Master.md verify security-type exports and mixed-case examples in integration workflows. | Confirm the local security-type field, value dictionary, case rules, and version applicability. | Symbol alone is not unique; the reviewed type/symbol pair defines source security identity. |
| split factors | `Split Date` | Effective date for the split factor. | Medium / High | Medium | Chapter_09_Corporate_Actions.md and Research_09_Corporate_Actions.md cite AdventGuru/Kevin Shea logical `SplitDate` evidence from exported split.inf files. | Confirm local date field name and whether the date is effective, ex-date, or another local split date basis. | Consultant split-file merge evidence uses a date field for exported split records. |
| split factors | `Split Factor` | Share multiplier or split factor. | Medium / High | Medium | Chapter_09_Corporate_Actions.md and Research_09_Corporate_Actions.md cite AdventGuru/Kevin Shea logical `SplitFactor` evidence from exported split.inf files. | Confirm whether the local factor convention is multiplier, ratio text, inverse factor, or another representation. | Consultant split-file merge evidence uses a factor field for exported split records; exact reverse-split convention is unknown. |


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
| holdings | `Portfolio Code` | PORT, Portfolio, Portfolio Code, Account | Portfolio, Account | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| holdings | `Security Symbol` | SEC, Security, Symbol, Security ID | Security, Symbol, Security ID | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| holdings | `Security Type` | SECURITY_TYPE, SEC_TYPE, Security Type, Sec Type Code, Type | Security Type, Sec Type, Type | Inferred Alias | Candidate aliases and labels are workflow evidence, not a complete native field dictionary. |
| holdings | `Holding Date` | Date, As Of Date, Holding Date | As Of Date, Report Date | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| holdings | `Accrued Income` | ACCRUED, Accrued Interest, Accrued Income | Accrued Interest, Accrued Income | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| holdings | `Base Currency` | Base Currency, Portfolio Currency | Base Currency, Reporting Currency | Inferred Alias | Normalized demo name only; confirm the exact local Axys/APX label. |
| holdings | `Base Market Value` | Base Market Value, Reporting Market Value | Base Market Value, Reporting Market Value | Inferred Alias | Normalized demo name only; confirm the exact local Axys/APX label. |
| holdings | `Currency Code` | Currency, Local Currency | Currency, Local Currency | Inferred Alias | Normalized demo name only; confirm the exact local Axys/APX label. |
| holdings | `Market Value` | MKT_VAL, Market Value, MarketVal | Market Value | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| holdings | `Price` | PRICE, Price, Market Price | Price, Market Price | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| holdings | `Quantity` | QTY, Quantity, Shares, Units | Quantity, Shares, Units | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |


### `portperf.csv` Name Candidates

| Dataset | Demo column | Candidate Axys/APX export names | Candidate report labels | Name confidence | Notes |
|---|---|---|---|---|---|
| portfolio performance | `Portfolio Code` | PORT, Portfolio, Portfolio Code, Account | Portfolio, Account | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| portfolio performance | `From Date` | From Date, Start Date | From Date, Beginning Date | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| portfolio performance | `Thru Date` | Thru Date, Through Date, End Date | Thru Date, Ending Date | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| portfolio performance | `Portfolio Return` | Return, Portfolio Return | Portfolio Return, Total Return | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| portfolio performance | `Base Currency` | Base Currency, Portfolio Currency | Base Currency, Reporting Currency | Inferred Alias | Normalized demo name only; confirm the exact local Axys/APX label. |


### `secperf.csv` Name Candidates

| Dataset | Demo column | Candidate Axys/APX export names | Candidate report labels | Name confidence | Notes |
|---|---|---|---|---|---|
| security performance | `Portfolio Code` | PORT, Portfolio, Portfolio Code, Account | Portfolio, Account | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| security performance | `Security Symbol` | SEC, Security, Symbol, Security ID | Security, Symbol, Security ID | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| security performance | `Security Type` | SECURITY_TYPE, SEC_TYPE, Security Type, Sec Type Code, Type | Security Type, Sec Type, Type | Inferred Alias | Candidate aliases and labels are workflow evidence, not a complete native field dictionary. |
| security performance | `From Date` | From Date, Start Date | From Date, Beginning Date | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| security performance | `Thru Date` | Thru Date, Through Date, End Date | Thru Date, Ending Date | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| security performance | `Security Return` | Return, Security Return | Security Return, Total Return | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |


### `transactions.csv` Name Candidates

| Dataset | Demo column | Candidate Axys/APX export names | Candidate report labels | Name confidence | Notes |
|---|---|---|---|---|---|
| transactions | `Portfolio Code` | PORT, Portfolio, Account | Portfolio, Account | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| transactions | `Security Symbol` | SEC, Security, Symbol, Security ID | Security, Symbol, Security ID | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| transactions | `Security Type` | SECURITY_TYPE, SEC_TYPE, Security Type, Sec Type Code, Type | Security Type, Sec Type, Type | Inferred Alias | Candidate aliases and labels are workflow evidence, not a complete native field dictionary. |
| transactions | `Transaction Date` | Trade Date, Transaction Date | Trade Date, Transaction Date | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| transactions | `Amount` | Amount, Net Amount, Cash Amount | Amount, Net Amount, Cash Amount | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| transactions | `Base Amount` | Base Amount, Reporting Amount | Base Amount, Reporting Amount | Inferred Alias | Normalized demo name only; confirm the exact local Axys/APX label. |
| transactions | `Base Currency` | Base Currency, Portfolio Currency | Base Currency, Reporting Currency | Inferred Alias | Normalized demo name only; confirm the exact local Axys/APX label. |
| transactions | `Commission` | Commission, Commissions | Commission | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| transactions | `Currency Code` | Currency, Transaction Currency | Currency, Transaction Currency | Inferred Alias | Normalized demo name only; confirm the exact local Axys/APX label. |
| transactions | `Price` | PRICE, Price | Price | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| transactions | `Quantity` | QTY, Quantity, Shares, Units | Quantity, Shares, Units | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| transactions | `Settlement Date` | Settle Date, Settlement Date, Pay Date | Settle Date, Settlement Date | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| transactions | `Source/Destination Symbol` | Source/Destination Symbol, Src Dest Symbol | Source/Destination Symbol | Normalized Demo Only | Normalized demo header retained for ppar consistency; native Axys/APX naming is not established by the local corpus. |
| transactions | `Source/Destination Type` | Source/Destination Type, Src Dest Type | Source/Destination Type | Normalized Demo Only | Normalized demo header retained for ppar consistency; native Axys/APX naming is not established by the local corpus. |
| transactions | `Special Security Symbol` | Special Security Symbol, Special Sec Symbol | Special Security Symbol | Normalized Demo Only | Normalized demo header retained for ppar consistency. Reviewed symbols are distinct from the type and may be descriptive; the exact native extract heading remains unverified. |
| transactions | `Special Security Type` | Special Security Type, Special Sec Type | Special Security Type | Normalized Demo Only | Normalized demo header retained for ppar consistency. Reviewed translation tables establish the paired concept but not this exact extract heading; values must remain source-faithful security-type identifiers. |
| transactions | `Transaction Code` | Transaction Code, Tran, Transaction Type | Transaction Code, Transaction Type | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| transactions | `Transaction Security Type` | Security Type, Sec Type | Security Type | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |


### `secmast.csv` Name Candidates

| Dataset | Demo column | Candidate Axys/APX export names | Candidate report labels | Name confidence | Notes |
|---|---|---|---|---|---|
| security master | `Security Symbol` | SECURITY_ID, SEC, Security, Symbol, Security Symbol | Security, Symbol, Security ID | Inferred Alias | Candidate aliases only; confirm the exact local IMEX profile or REP field. |
| security master | `Security Type` | SECURITY_TYPE, SEC_TYPE, Security Type, Sec Type Code, Type | Security Type, Sec Type, Type | Inferred Alias | Candidate aliases and labels are workflow evidence, not a complete native field dictionary. |
| security master | `Asset Class Code` | ASSET_CLASS_CODE, ASSET_CLASS, Asset Class Code, Asset Class | Asset Class Code, Asset Class | Inferred Alias | The normalized demo name is plausible but must be mapped to the local export or report label. |


### `splits.csv` Name Candidates

| Dataset | Demo column | Candidate Axys/APX export names | Candidate report labels | Name confidence | Notes |
|---|---|---|---|---|---|
| split factors | `Security Symbol` | SEC, SplitSymbol, Security, Symbol | Security, Symbol, Split Symbol | Inferred Alias | `SplitSymbol` is consultant-derived logical evidence, not a verified official Axys header. |
| split factors | `Security Type` | SECURITY_TYPE, SEC_TYPE, Security Type, Sec Type Code, Type | Security Type, Sec Type, Type | Inferred Alias | Candidate aliases and labels are workflow evidence, not a complete native field dictionary. |
| split factors | `Split Date` | SPLIT_DATE, SplitDate, Effective Date | Split Date, Effective Date | Inferred Alias | `SplitDate` is consultant-derived logical evidence, not a verified official Axys header. |
| split factors | `Split Factor` | SPLIT_FACTOR, SplitFactor, Factor | Split Factor, Factor | Inferred Alias | `SplitFactor` is consultant-derived logical evidence, not a verified official Axys header. |


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
| holdings | `Portfolio Code` | IMEX or REP | REP preferred | Holdings reconstruction can use either validated position/appraisal IMEX output or an appraisal-style REP extract. |
| holdings | `Security Symbol` | IMEX or REP | REP preferred | Holdings reconstruction can use either validated position/appraisal IMEX output or an appraisal-style REP extract. |
| holdings | `Security Type` | IMEX then REP cross-check | REP preferred | Preserve the exact source value and validate the site-specific code dictionary before filtering. |
| holdings | `Holding Date` | IMEX or REP | REP preferred | Holdings reconstruction can use either validated position/appraisal IMEX output or an appraisal-style REP extract. |
| holdings | `Accrued Income` | IMEX or REP | Local discovery required | Accrued income can affect performance reconciliation, so validate the field before running accrual-sensitive comparisons. |
| holdings | `Base Currency` | Local discovery required | REP preferred | Validate the local extract and currency basis against a report sample before use. |
| holdings | `Base Market Value` | Local discovery required | REP preferred | Validate the local extract and currency basis against a report sample before use. |
| holdings | `Currency Code` | Local discovery required | REP preferred | Validate the local extract and currency basis against a report sample before use. |
| holdings | `Market Value` | IMEX or REP | REP preferred | Holdings reconstruction can use either validated position/appraisal IMEX output or an appraisal-style REP extract. |
| holdings | `Price` | IMEX or REP | REP preferred | Holdings reconstruction can use either validated position/appraisal IMEX output or an appraisal-style REP extract. |
| holdings | `Quantity` | IMEX or REP | REP preferred | Holdings reconstruction can use either validated position/appraisal IMEX output or an appraisal-style REP extract. |


### `portperf.csv` Source Strategy

| Dataset | Demo column | Preferred source | Fallback source | Notes |
|---|---|---|---|---|
| portfolio performance | `Portfolio Code` | IMEX or REP | REP preferred | Portfolio identifier must be present regardless of the extract source. |
| portfolio performance | `From Date` | REP preferred | Local discovery required | Prefer report output because the local corpus does not prove native performance IMEX object names or calculation basis. |
| portfolio performance | `Thru Date` | REP preferred | Local discovery required | Prefer report output because the local corpus does not prove native performance IMEX object names or calculation basis. |
| portfolio performance | `Portfolio Return` | REP preferred | Local discovery required | Prefer report output because the local corpus does not prove native performance IMEX object names or calculation basis. |
| portfolio performance | `Base Currency` | Local discovery required | REP preferred | Validate the local extract and currency basis against a report sample before use. |


### `secperf.csv` Source Strategy

| Dataset | Demo column | Preferred source | Fallback source | Notes |
|---|---|---|---|---|
| security performance | `Portfolio Code` | IMEX or REP | REP preferred | Portfolio and security identifiers must be present regardless of the extract source. |
| security performance | `Security Symbol` | IMEX or REP | REP preferred | Portfolio and security identifiers must be present regardless of the extract source. |
| security performance | `Security Type` | IMEX then REP cross-check | REP preferred | Preserve the exact source value and validate the site-specific code dictionary before filtering. |
| security performance | `From Date` | REP preferred | Local discovery required | Prefer report output because the local corpus does not prove native performance IMEX object names or calculation basis. |
| security performance | `Thru Date` | REP preferred | Local discovery required | Prefer report output because the local corpus does not prove native performance IMEX object names or calculation basis. |
| security performance | `Security Return` | REP preferred | Local discovery required | Prefer report output because the local corpus does not prove native performance IMEX object names or calculation basis. |


### `transactions.csv` Source Strategy

| Dataset | Demo column | Preferred source | Fallback source | Notes |
|---|---|---|---|---|
| transactions | `Portfolio Code` | IMEX then REP cross-check | REP preferred | Transaction core fields are IMEX-suitable, with REP useful for report-facing cross-checks and sign-convention validation. |
| transactions | `Security Symbol` | IMEX then REP cross-check | REP preferred | Transaction core fields are IMEX-suitable, with REP useful for report-facing cross-checks and sign-convention validation. |
| transactions | `Security Type` | IMEX then REP cross-check | REP preferred | Preserve the exact source value and validate the site-specific code dictionary before filtering. |
| transactions | `Transaction Date` | IMEX then REP cross-check | REP preferred | Transaction core fields are IMEX-suitable, with REP useful for report-facing cross-checks and sign-convention validation. |
| transactions | `Amount` | IMEX then REP cross-check | REP preferred | Transaction core fields are IMEX-suitable, with REP useful for report-facing cross-checks and sign-convention validation. |
| transactions | `Base Amount` | Local discovery required | REP preferred | Validate the local extract and currency basis against a report sample before use. |
| transactions | `Base Currency` | Local discovery required | REP preferred | Validate the local extract and currency basis against a report sample before use. |
| transactions | `Commission` | IMEX then REP cross-check | REP preferred | Transaction core fields are IMEX-suitable, with REP useful for report-facing cross-checks and sign-convention validation. |
| transactions | `Currency Code` | Local discovery required | REP preferred | Validate the local extract and currency basis against a report sample before use. |
| transactions | `Price` | IMEX then REP cross-check | REP preferred | Transaction core fields are IMEX-suitable, with REP useful for report-facing cross-checks and sign-convention validation. |
| transactions | `Quantity` | IMEX then REP cross-check | REP preferred | Transaction core fields are IMEX-suitable, with REP useful for report-facing cross-checks and sign-convention validation. |
| transactions | `Settlement Date` | IMEX then REP cross-check | REP preferred | Transaction core fields are IMEX-suitable, with REP useful for report-facing cross-checks and sign-convention validation. |
| transactions | `Source/Destination Symbol` | Local discovery required | REP preferred | Required for ambiguous Axys/APX flow semantics. If IMEX does not expose this context, stop and use REP, a custom report, or another local source. |
| transactions | `Source/Destination Type` | Local discovery required | REP preferred | Required for ambiguous Axys/APX flow semantics. If IMEX does not expose this context, stop and use REP, a custom report, or another local source. |
| transactions | `Special Security Symbol` | Local discovery required | REP preferred | Required for ambiguous Axys/APX flow semantics. If IMEX does not expose this context, stop and use REP, a custom report, or another local source. |
| transactions | `Special Security Type` | Local discovery required | REP preferred | Required for ambiguous Axys/APX flow semantics. If IMEX does not expose this context, stop and use REP, a custom report, or another local source. |
| transactions | `Transaction Code` | IMEX then REP cross-check | REP preferred | Transaction core fields are IMEX-suitable, with REP useful for report-facing cross-checks and sign-convention validation. |
| transactions | `Transaction Security Type` | Local discovery required | REP preferred | Required for ambiguous Axys/APX flow semantics. If IMEX does not expose this context, stop and use REP, a custom report, or another local source. |


### `secmast.csv` Source Strategy

| Dataset | Demo column | Preferred source | Fallback source | Notes |
|---|---|---|---|---|
| security master | `Security Symbol` | IMEX then REP cross-check | REP preferred | Preserve exact source case and reconcile the identifier to the holdings and transaction extracts before using reference qualifiers. |
| security master | `Security Type` | IMEX then REP cross-check | REP preferred | Preserve the exact source value and validate the site-specific code dictionary before filtering. |
| security master | `Asset Class Code` | IMEX then REP cross-check | REP preferred | Tie the code to the same snapshot as the source rows so later reclassification does not silently change an audit population. |


### `splits.csv` Source Strategy

| Dataset | Demo column | Preferred source | Fallback source | Notes |
|---|---|---|---|---|
| split factors | `Security Symbol` | IMEX or REP | Local discovery required | Prefer a direct split.inf/exported split-factor source; use REP/custom reports only if they expose security-level split factors. |
| split factors | `Security Type` | IMEX then REP cross-check | REP preferred | Preserve the exact source value and validate the site-specific code dictionary before filtering. |
| split factors | `Split Date` | IMEX or REP | Local discovery required | Validate the local date basis before comparing split factors to holdings and prices. |
| split factors | `Split Factor` | IMEX or REP | Local discovery required | Treat split factors as context evidence explaining holdings changes, not as cash-flow transactions. |


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
