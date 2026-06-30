# Appendix - Demo Extract Availability

Repository: AXYS / APX Reference Repository
Scope: `ppar/demos/data/axys/axys_full_spec_a` and
`ppar/demos/data/axys/axys_full_spec_b`
Status: Draft confidence matrix generated from the packaged YAML contract.

<!-- GENERATED FROM ppar/demos/data/axys/demo_extract_availability.yaml. -->
<!-- Run scripts/render_demo_extract_availability.py after editing the YAML. -->

---

## Purpose

This appendix estimates how likely each packaged Axys demo dataset and column is to be obtainable from an Axys installation through IMEX and/or REP-style report extracts.

The machine-readable source of truth is `ppar/demos/data/axys/demo_extract_availability.yaml`. Tests verify that the YAML covers every packaged full-spec Axys demo CSV header and that this appendix is current.

The packaged demo files are normalized demo extracts. They are not official Axys schemas, not universal IMEX profiles, and not claims that every Axys site can export every field with these exact names.

The two full-spec snapshots currently use the same file layouts:

- `holdings.csv`
- `portperf.csv`
- `sec_ref.csv`
- `secperf.csv`
- `transactions.csv`

## Confidence Labels

| Label | Meaning |
|---|---|
| High | Strongly likely to be obtainable from a typical Axys environment. |
| Medium / High | Between medium and high; likely available, but still report/profile dependent. |
| Medium | Plausible, but site configuration, report choice, or export profile must be validated. |
| Medium / Unknown | Between medium and unknown; evidence mentions the area, but not the exact object/field. |
| Low / Medium | Between low and medium; possible and plausible, but not proven as a standard field. |
| Low | Possible through customization or site-specific configuration, but not enough to rely on. |
| Unknown | Not established by the local Axys/APX reference corpus. |

## Evidence Boundaries

The confidence ratings below rely on these local reference conclusions:

- `Chapter_12_Imex.md` establishes IMEX / Import-Export as an Axys import/export mechanism, but says the complete native object and field dictionaries are not available in the supplied source material.
- `Chapter_12_Imex.md` documents Axys CI import workflows for transactions, positions, prices, security information, Trade Blotter context, and selected transaction translation fields.
- `Chapter_13_Rep.md` establishes REP / RepLang / REP32 as a report-driven extraction path and distinguishes it from IMEX.
- `Chapter_10_Performance.md` says `portperf` and `secperf` should be treated as normalized/local names unless a live IMEX object, report output, or vendor manual confirms native names.
- `docs/axys_common_core_export.md` is a starter reference only. It proposes common field aliases but does not override the more conservative chapter confidence boundaries.

## Interpretation Rules

Use this matrix as an implementation planning aid:

- **IMEX confidence** asks whether the value is likely obtainable from a structured Axys IMEX-style export or adjacent import/export workflow.
- **REP confidence** asks whether the value is likely obtainable from a standard or custom REP/Replang report extract.
- A high confidence rating does not mean the exact demo column name exists in Axys. It means the underlying value is likely available.
- Performance fields are rated more conservatively for IMEX because the local reference corpus does not contain an official performance IMEX object dictionary.
- Transaction source/destination and special-security fields are rated as integration-context fields. The corpus supports them in transaction translation workflows, but not as guaranteed columns in every posted-transaction export.

## Availability Matrix

### `holdings.csv`

| Dataset | Demo column | Normalized meaning | IMEX confidence | REP confidence | Evidence basis | Open questions | Comments |
|---|---|---|---|---|---|---|---|
| holdings | `PORT` | Portfolio/account identifier. | High | High | Position/appraisal evidence in Chapter_06_Holdings.md, Chapter_12_Imex.md, Chapter_13_Rep.md, and axys_common_core_export.md. | Confirm exact local IMEX profile and field name. | Core portfolio/client identifier in position and appraisal workflows. |
| holdings | `SEC` | Security identifier. | High | High | Position/appraisal evidence in Chapter_06_Holdings.md, Chapter_12_Imex.md, Chapter_13_Rep.md, and axys_common_core_export.md. | Confirm exact local IMEX profile and field name. | Security identifiers are supported by CI/security-resolution and report evidence. |
| holdings | `HOLDING_DATE` | As-of date for holdings. | High | High | Position/appraisal evidence in Chapter_06_Holdings.md, Chapter_12_Imex.md, Chapter_13_Rep.md, and axys_common_core_export.md. | Confirm exact local IMEX profile and field name. | Position exports and appraisal reports are inherently as-of-date based. |
| holdings | `QTY` | Quantity, shares, or units. | High | High | Position/appraisal evidence in Chapter_06_Holdings.md, Chapter_12_Imex.md, Chapter_13_Rep.md, and axys_common_core_export.md. | Confirm exact local IMEX profile and field name. | Quantity is directly supported by transaction and appraisal report evidence. |
| holdings | `PRICE` | Market price used for valuation. | High | High | Position/appraisal evidence in Chapter_06_Holdings.md, Chapter_12_Imex.md, Chapter_13_Rep.md, and axys_common_core_export.md. | Confirm exact local IMEX profile and field name. | Price import/export evidence and appraisal/report labels support availability. |
| holdings | `MKT_VAL` | Market value. | High | High | Position/appraisal evidence in Chapter_06_Holdings.md, Chapter_12_Imex.md, Chapter_13_Rep.md, and axys_common_core_export.md. | Confirm exact local IMEX profile and field name. | Market value appears in report evidence and is a core position/appraisal value. |
| holdings | `COST` | Cost or cost basis. | Medium | High | Report-output cost evidence and common-core holdings reference; IMEX depends on the position, tax-lot, or lot-summary profile. | Confirm whether local holdings export exposes cost or requires lot/report extraction. | Cost appears in report-output evidence; IMEX availability depends on position/lot profile. |
| holdings | `ACCRUED` | Accrued income or accrued interest. | Medium | Medium | Fixed-income/accrual handling is documented as performance-sensitive; common-core holdings reference treats accrued income as plausible. | Confirm accrued-income field, date basis, and fixed-income treatment. | Common for fixed-income holdings, but exact export/report field requires validation. |


### `portperf.csv`

| Dataset | Demo column | Normalized meaning | IMEX confidence | REP confidence | Evidence basis | Open questions | Comments |
|---|---|---|---|---|---|---|---|
| portfolio performance | `END_MV` | Ending market value. | Medium | High | Chapter_10_Performance.md treats portfolio performance as a report/extract boundary and prefers REP for report-tie values. | Confirm whether a local IMEX performance object exists or use REP output. | Likely available through performance/appraisal reports; exact IMEX object remains unknown. |
| portfolio performance | `FLOW` | Period external flow. | Low / Medium | Medium | Performance chapter and report evidence support the concept, but the local corpus does not prove a native IMEX performance object/field. | Confirm methodology, gross/net basis, and stored-vs-report-calculated behavior. | More likely through reports when displayed or calculated; IMEX object/field is unproven. |
| portfolio performance | `INCOME` | Period income component. | Low / Medium | Medium | Performance chapter and report evidence support the concept, but the local corpus does not prove a native IMEX performance object/field. | Confirm methodology, gross/net basis, and stored-vs-report-calculated behavior. | Likely reportable, but basis and inclusion rules require validation. |
| portfolio performance | `GAIN_LOSS` | Period gain/loss component. | Low / Medium | Medium | Performance chapter and report evidence support the concept, but the local corpus does not prove a native IMEX performance object/field. | Confirm methodology, gross/net basis, and stored-vs-report-calculated behavior. | Report evidence supports gain/loss labels; exact performance component is report-dependent. |
| portfolio performance | `PORTFOLIO_CODE` | Portfolio/account identifier. | High | High | Chapter_10_Performance.md treats portfolio performance as a report/extract boundary and prefers REP for report-tie values. | Confirm whether a local IMEX performance object exists or use REP output. | Core identifier. |
| portfolio performance | `PORTFOLIO_NAME` | Portfolio/account display name. | Medium | High | Chapter_10_Performance.md treats portfolio performance as a report/extract boundary and prefers REP for report-tie values. | Confirm whether a local IMEX performance object exists or use REP output. | Likely available from account/report output; exact IMEX field is site-specific. |
| portfolio performance | `FROM_DATE` | Period start date. | Medium | High | Chapter_10_Performance.md treats portfolio performance as a report/extract boundary and prefers REP for report-tie values. | Confirm whether a local IMEX performance object exists or use REP output. | Report parameters reliably provide the period; structured export field names require validation. |
| portfolio performance | `THRU_DATE` | Period end date. | Medium | High | Chapter_10_Performance.md treats portfolio performance as a report/extract boundary and prefers REP for report-tie values. | Confirm whether a local IMEX performance object exists or use REP output. | Report parameters reliably provide the period; structured export field names require validation. |
| portfolio performance | `BEGIN_MV` | Beginning market value. | Medium | High | Chapter_10_Performance.md treats portfolio performance as a report/extract boundary and prefers REP for report-tie values. | Confirm whether a local IMEX performance object exists or use REP output. | Likely reportable; exact IMEX object/field remains unproven. |
| portfolio performance | `PORT_RETURN` | Portfolio return. | Medium / Unknown | High | Performance chapter and report evidence support the concept, but the local corpus does not prove a native IMEX performance object/field. | Confirm methodology, gross/net basis, and stored-vs-report-calculated behavior. | Performance-history IMEX fields are not established; REP is preferred for report-tie values. |


### `sec_ref.csv`

| Dataset | Demo column | Normalized meaning | IMEX confidence | REP confidence | Evidence basis | Open questions | Comments |
|---|---|---|---|---|---|---|---|
| security master | `SECURITY_ID` | Security identifier. | High | High | Security-master and classification evidence in Chapter_04_Security_Master.md, Chapter_12_Imex.md, Chapter_13_Rep.md, and axys_common_core_export.md. | Confirm exact local security-master export aliases. | Security information is supported in IMEX/CI evidence and reports. |
| security master | `SECURITY_NAME` | Security display name or description. | High | High | Security-master and classification evidence in Chapter_04_Security_Master.md, Chapter_12_Imex.md, Chapter_13_Rep.md, and axys_common_core_export.md. | Confirm exact local security-master export aliases. | Security description/name is a common security-master and report field. |
| security master | `ASSET_CLASS_CODE` | Asset class classification. | Medium | Medium | Security-master and classification evidence in Chapter_04_Security_Master.md, Chapter_12_Imex.md, Chapter_13_Rep.md, and axys_common_core_export.md. | Confirm exact security-master export field and whether classification history is as-of-date aware. | Conversion evidence mentions asset classes; exact field names and history behavior require validation. |
| security master | `SECTOR_CODE` | Sector classification. | Medium | Medium | Security-master and classification evidence in Chapter_04_Security_Master.md, Chapter_12_Imex.md, Chapter_13_Rep.md, and axys_common_core_export.md. | Confirm exact security-master export field and whether classification history is as-of-date aware. | Conversion and reporting evidence support sector use; exact source and timing require validation. |
| security master | `COUNTRY_CODE` | Country classification. | Medium | Medium | Security-master and classification evidence in Chapter_04_Security_Master.md, Chapter_12_Imex.md, Chapter_13_Rep.md, and axys_common_core_export.md. | Confirm exact security-master export field and whether classification history is as-of-date aware. | Axys reporting by country is supported at product level; exact source requires validation. |
| security master | `CURRENCY_CODE` | Security/local currency code. | Medium | Medium | Security-master and classification evidence in Chapter_04_Security_Master.md, Chapter_12_Imex.md, Chapter_13_Rep.md, and axys_common_core_export.md. | Confirm exact security-master export field and whether classification history is as-of-date aware. | Common in security and multicurrency contexts, but site configuration matters. |
| security master | `INDUSTRY_CODE` | Industry classification. | Medium | Medium | Security-master and classification evidence in Chapter_04_Security_Master.md, Chapter_12_Imex.md, Chapter_13_Rep.md, and axys_common_core_export.md. | Confirm exact security-master export field and whether classification history is as-of-date aware. | Conversion evidence mentions industries; exact field source requires validation. |


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
| security performance | `CONTRIBUTION` | Security contribution to return. | Low / Medium | Medium / High | Performance chapter and report evidence support the concept, but the local corpus does not prove a native IMEX performance object/field. | Confirm methodology, contribution basis, and stored-vs-report-calculated behavior. | Contribution reports are supported, but exact Axys security-contribution export requires validation. |


### `transactions.csv`

| Dataset | Demo column | Normalized meaning | IMEX confidence | REP confidence | Evidence basis | Open questions | Comments |
|---|---|---|---|---|---|---|---|
| transactions | `TRANSACTION_ID` | Stable transaction identifier. | Low / Medium | Low / Medium | Stable transaction ID is a demo/reconciliation convenience; durable native ID is not proven by the local corpus. | Confirm whether local Axys exports a stable posted transaction identifier. | A site may have IDs or row keys, but a durable posted-transaction ID is not proven. |
| transactions | `PORT` | Portfolio/account identifier. | High | High | Transaction import/export and report evidence in Chapter_05_Transactions.md, Chapter_12_Imex.md, and Chapter_13_Rep.md. | Confirm exact posted-transaction export fields and sign convention. | Core transaction/account field. |
| transactions | `TRANSACTION_DATE` | Trade or economic date. | High | High | Transaction import/export and report evidence in Chapter_05_Transactions.md, Chapter_12_Imex.md, and Chapter_13_Rep.md. | Confirm exact posted-transaction export fields and sign convention. | Trade date appears in report evidence and transaction import workflows. |
| transactions | `SETTLE_DATE` | Settlement or pay date. | Medium | High | Transaction import/export and report evidence in Chapter_05_Transactions.md, Chapter_12_Imex.md, and Chapter_13_Rep.md. | Confirm exact posted-transaction export fields and sign convention. | Settle date appears in report-output evidence; IMEX availability is profile-dependent. |
| transactions | `SEC` | Security identifier. | High | High | Transaction import/export and report evidence in Chapter_05_Transactions.md, Chapter_12_Imex.md, and Chapter_13_Rep.md. | Confirm exact posted-transaction export fields and sign convention. | Core transaction/security field. |
| transactions | `TRAN` | Transaction code or type. | Medium | Medium | Transaction import/export and report evidence in Chapter_05_Transactions.md, Chapter_12_Imex.md, and Chapter_13_Rep.md. | Confirm exact posted-transaction export fields and sign convention. | Transaction type/code is supported in translation/import evidence; report exposure depends on layout. |
| transactions | `SEC_TYPE` | Security type for transaction security. | Medium | Low / Medium | Observed CI/Axys transaction translation context in Chapter_05_Transactions.md and Chapter_12_Imex.md; required for ambiguous external-flow handling. | Confirm whether local IMEX exposes this context on posted transaction exports.; Use REP, a custom report, or another source if IMEX cannot expose it. | Central to Axys security resolution; transaction-row availability depends on export/report design. |
| transactions | `SRC_DEST_TYPE` | Source/destination type for ambiguous flows. | Medium | Low / Medium | Observed CI/Axys transaction translation context in Chapter_05_Transactions.md and Chapter_12_Imex.md; required for ambiguous external-flow handling. | Confirm whether local IMEX exposes this context on posted transaction exports.; Use REP, a custom report, or another source if IMEX cannot expose it. | Observed transaction translation/integration field, not guaranteed in every posted export. |
| transactions | `SRC_DEST_SYMBOL` | Source/destination symbol for ambiguous flows. | Medium | Low / Medium | Observed CI/Axys transaction translation context in Chapter_05_Transactions.md and Chapter_12_Imex.md; required for ambiguous external-flow handling. | Confirm whether local IMEX exposes this context on posted transaction exports.; Use REP, a custom report, or another source if IMEX cannot expose it. | Observed transaction translation/integration field, not guaranteed in every posted export. |
| transactions | `SPECIAL_SEC_TYPE` | Special security type for fee or special handling. | Medium | Low | Observed CI/Axys transaction translation context in Chapter_05_Transactions.md and Chapter_12_Imex.md; required for ambiguous external-flow handling. | Confirm whether local IMEX exposes this context on posted transaction exports.; Use REP, a custom report, or another source if IMEX cannot expose it. | Supported in integration evidence; standard report exposure is uncertain. |
| transactions | `SPECIAL_SEC_SYMBOL` | Special security symbol for fee or special handling. | Medium | Low | Observed CI/Axys transaction translation context in Chapter_05_Transactions.md and Chapter_12_Imex.md; required for ambiguous external-flow handling. | Confirm whether local IMEX exposes this context on posted transaction exports.; Use REP, a custom report, or another source if IMEX cannot expose it. | Supported in integration evidence; standard report exposure is uncertain. |
| transactions | `QTY` | Transaction quantity. | High | High | Transaction import/export and report evidence in Chapter_05_Transactions.md, Chapter_12_Imex.md, and Chapter_13_Rep.md. | Confirm exact posted-transaction export fields and sign convention. | Quantity appears in transaction and report evidence. |
| transactions | `PRICE` | Transaction price. | High | High | Transaction import/export and report evidence in Chapter_05_Transactions.md, Chapter_12_Imex.md, and Chapter_13_Rep.md. | Confirm exact posted-transaction export fields and sign convention. | Price appears in transaction and report evidence. |
| transactions | `AMOUNT` | Net amount, proceeds, or cash amount. | High | Medium / High | Transaction import/export and report evidence in Chapter_05_Transactions.md, Chapter_12_Imex.md, and Chapter_13_Rep.md. | Confirm exact posted-transaction export fields and sign convention. | Core transaction amount is likely available, though report labels can vary. |
| transactions | `COMMISSION` | Commission. | Medium | Medium | Transaction import/export and report evidence in Chapter_05_Transactions.md, Chapter_12_Imex.md, and Chapter_13_Rep.md. | Confirm exact posted-transaction export fields and sign convention. | Supported in CI parameter evidence; report/profile availability should be validated. |


## Candidate Name Mapping

These names are candidate aliases for local discovery. They are not assertions that the packaged demo headers are official Axys IMEX or REP names.

| Label | Meaning |
|---|---|
| Inferred Alias | Likely field/report aliases for the underlying value, but not proven as official native Axys names by the local corpus. |
| Report Label Inferred | Likely report labels or report-style aliases; structured IMEX names still need local confirmation. |
| Normalized Demo Only | A normalized demo-system name. Do not treat it as a native Axys or REP label. |

### `holdings.csv` Name Candidates

| Dataset | Demo column | Candidate Axys/export names | Candidate report labels | Name confidence | Notes |
|---|---|---|---|---|---|
| holdings | `PORT` | PORT, Portfolio, Portfolio Code, Account | Portfolio, Account | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| holdings | `SEC` | SEC, Security, Symbol, Security ID | Security, Symbol, Security ID | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| holdings | `HOLDING_DATE` | Date, As Of Date, Holding Date | As Of Date, Report Date | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| holdings | `QTY` | QTY, Quantity, Shares, Units | Quantity, Shares, Units | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| holdings | `PRICE` | PRICE, Price, Market Price | Price, Market Price | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| holdings | `MKT_VAL` | MKT_VAL, Market Value, MarketVal | Market Value | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| holdings | `COST` | COST, Cost, Cost Basis | Cost, Cost Basis | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| holdings | `ACCRUED` | ACCRUED, Accrued Interest, Accrued Income | Accrued Interest, Accrued Income | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |


### `portperf.csv` Name Candidates

| Dataset | Demo column | Candidate Axys/export names | Candidate report labels | Name confidence | Notes |
|---|---|---|---|---|---|
| portfolio performance | `END_MV` | Ending Market Value, End Market Value | Ending Market Value, End MV | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| portfolio performance | `FLOW` | External Flow, Flow, Net Flow | Flow, External Flow, Net Contributions/Withdrawals | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| portfolio performance | `INCOME` | Income | Income | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| portfolio performance | `GAIN_LOSS` | Gain/Loss, Gain Loss | Gain/Loss, Realized/Unrealized Gain/Loss | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| portfolio performance | `PORTFOLIO_CODE` | PORT, Portfolio, Portfolio Code, Account | Portfolio, Account | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| portfolio performance | `PORTFOLIO_NAME` | Portfolio Name, Account Name | Portfolio Name, Account Name | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| portfolio performance | `FROM_DATE` | From Date, Start Date | From Date, Beginning Date | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| portfolio performance | `THRU_DATE` | Thru Date, Through Date, End Date | Thru Date, Ending Date | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| portfolio performance | `BEGIN_MV` | Beginning Market Value, Begin Market Value | Beginning Market Value, Begin MV | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| portfolio performance | `PORT_RETURN` | Return, Portfolio Return | Portfolio Return, Total Return | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |


### `sec_ref.csv` Name Candidates

| Dataset | Demo column | Candidate Axys/export names | Candidate report labels | Name confidence | Notes |
|---|---|---|---|---|---|
| security master | `SECURITY_ID` | SEC, Security, Symbol, Security ID | Security, Symbol, Security ID | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| security master | `SECURITY_NAME` | Security Name, Description, Security Description | Security Name, Description | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| security master | `ASSET_CLASS_CODE` | Asset Class, Asset Class Code | Asset Class | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| security master | `SECTOR_CODE` | Sector, Sector Code | Sector | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| security master | `COUNTRY_CODE` | Country, Country Code | Country | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| security master | `CURRENCY_CODE` | Currency, Currency Code, Local Currency | Currency, Local Currency | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| security master | `INDUSTRY_CODE` | Industry, Industry Code | Industry | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |


### `secperf.csv` Name Candidates

| Dataset | Demo column | Candidate Axys/export names | Candidate report labels | Name confidence | Notes |
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

| Dataset | Demo column | Candidate Axys/export names | Candidate report labels | Name confidence | Notes |
|---|---|---|---|---|---|
| transactions | `TRANSACTION_ID` | Transaction ID, Row ID | Transaction ID, Reference | Normalized Demo Only | Stable native transaction ID is not proven; use a local durable key if available. |
| transactions | `PORT` | PORT, Portfolio, Account | Portfolio, Account | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| transactions | `TRANSACTION_DATE` | Trade Date, Transaction Date | Trade Date, Transaction Date | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| transactions | `SETTLE_DATE` | Settle Date, Settlement Date, Pay Date | Settle Date, Settlement Date | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| transactions | `SEC` | SEC, Security, Symbol, Security ID | Security, Symbol, Security ID | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| transactions | `TRAN` | Transaction Code, Tran, Transaction Type | Transaction Code, Transaction Type | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| transactions | `SEC_TYPE` | Security Type, Sec Type | Security Type | Report Label Inferred | Report-style label candidates are more credible than native IMEX names for this field. |
| transactions | `SRC_DEST_TYPE` | Source/Destination Type, Src Dest Type | Source/Destination Type | Normalized Demo Only | Normalized demo header retained for ppar consistency; native Axys naming is not established by the local corpus. |
| transactions | `SRC_DEST_SYMBOL` | Source/Destination Symbol, Src Dest Symbol | Source/Destination Symbol | Normalized Demo Only | Normalized demo header retained for ppar consistency; native Axys naming is not established by the local corpus. |
| transactions | `SPECIAL_SEC_TYPE` | Special Security Type, Special Sec Type | Special Security Type | Normalized Demo Only | Normalized demo header retained for ppar consistency; native Axys naming is not established by the local corpus. |
| transactions | `SPECIAL_SEC_SYMBOL` | Special Security Symbol, Special Sec Symbol | Special Security Symbol | Normalized Demo Only | Normalized demo header retained for ppar consistency; native Axys naming is not established by the local corpus. |
| transactions | `QTY` | QTY, Quantity, Shares, Units | Quantity, Shares, Units | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| transactions | `PRICE` | PRICE, Price | Price | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| transactions | `AMOUNT` | Amount, Net Amount, Cash Amount | Amount, Net Amount, Cash Amount | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |
| transactions | `COMMISSION` | Commission, Commissions | Commission | Inferred Alias | Candidate aliases only; confirm against the local IMEX profile, REP output, or vendor field dictionary. |


## Source Strategy Matrix

This matrix translates availability confidence into implementation guidance. `Blocking if missing` means ppar should not silently proceed for that field in a workflow that depends on the corresponding dataset.

| Label | Meaning |
|---|---|
| IMEX preferred | Use IMEX first when the local profile exposes the field. |
| REP preferred | Use REP/report output first because report-tie behavior matters. |
| IMEX or REP | Either IMEX or REP is acceptable after local field validation. |
| IMEX then REP cross-check | Use IMEX as the primary feed and REP as a report-output cross-check. |
| Local discovery required | Do not assume a source; validate local IMEX/REP/custom-report support first. |

### `holdings.csv` Source Strategy

| Dataset | Demo column | Preferred source | Fallback source | Context required | Blocking if missing | Notes |
|---|---|---|---|---|---|---|
| holdings | `PORT` | IMEX or REP | REP preferred | No | Yes | Holdings reconstruction can use either validated position/appraisal IMEX output or an appraisal-style REP extract. |
| holdings | `SEC` | IMEX or REP | REP preferred | No | Yes | Holdings reconstruction can use either validated position/appraisal IMEX output or an appraisal-style REP extract. |
| holdings | `HOLDING_DATE` | IMEX or REP | REP preferred | No | Yes | Holdings reconstruction can use either validated position/appraisal IMEX output or an appraisal-style REP extract. |
| holdings | `QTY` | IMEX or REP | REP preferred | No | Yes | Holdings reconstruction can use either validated position/appraisal IMEX output or an appraisal-style REP extract. |
| holdings | `PRICE` | IMEX or REP | REP preferred | No | Yes | Holdings reconstruction can use either validated position/appraisal IMEX output or an appraisal-style REP extract. |
| holdings | `MKT_VAL` | IMEX or REP | REP preferred | No | Yes | Holdings reconstruction can use either validated position/appraisal IMEX output or an appraisal-style REP extract. |
| holdings | `COST` | REP preferred | Local discovery required | No | No | Useful as context, but ppar does not require cost to attribute return differences. |
| holdings | `ACCRUED` | IMEX or REP | Local discovery required | No | Yes | Accrued income can affect performance reconciliation, so validate the field before running accrual-sensitive comparisons. |


### `portperf.csv` Source Strategy

| Dataset | Demo column | Preferred source | Fallback source | Context required | Blocking if missing | Notes |
|---|---|---|---|---|---|---|
| portfolio performance | `END_MV` | REP preferred | Local discovery required | No | Yes | Prefer report output because the local corpus does not prove native performance IMEX object names or calculation basis. |
| portfolio performance | `FLOW` | REP preferred | Local discovery required | No | Yes | Prefer report output because the local corpus does not prove native performance IMEX object names or calculation basis. |
| portfolio performance | `INCOME` | REP preferred | Local discovery required | No | Yes | Prefer report output because the local corpus does not prove native performance IMEX object names or calculation basis. |
| portfolio performance | `GAIN_LOSS` | REP preferred | Local discovery required | No | Yes | Prefer report output because the local corpus does not prove native performance IMEX object names or calculation basis. |
| portfolio performance | `PORTFOLIO_CODE` | IMEX or REP | REP preferred | No | Yes | Portfolio identifier must be present regardless of the extract source. |
| portfolio performance | `PORTFOLIO_NAME` | REP preferred | IMEX or REP | No | No | Useful context, but not required for core performance attribution. |
| portfolio performance | `FROM_DATE` | REP preferred | Local discovery required | No | Yes | Prefer report output because the local corpus does not prove native performance IMEX object names or calculation basis. |
| portfolio performance | `THRU_DATE` | REP preferred | Local discovery required | No | Yes | Prefer report output because the local corpus does not prove native performance IMEX object names or calculation basis. |
| portfolio performance | `BEGIN_MV` | REP preferred | Local discovery required | No | Yes | Prefer report output because the local corpus does not prove native performance IMEX object names or calculation basis. |
| portfolio performance | `PORT_RETURN` | REP preferred | Local discovery required | No | Yes | Prefer report output because the local corpus does not prove native performance IMEX object names or calculation basis. |


### `sec_ref.csv` Source Strategy

| Dataset | Demo column | Preferred source | Fallback source | Context required | Blocking if missing | Notes |
|---|---|---|---|---|---|---|
| security master | `SECURITY_ID` | IMEX or REP | REP preferred | No | Yes | Use whichever local security-master export or report provides stable identifiers and classification timing. |
| security master | `SECURITY_NAME` | IMEX or REP | Local discovery required | No | No | Use whichever local security-master export or report provides stable identifiers and classification timing. |
| security master | `ASSET_CLASS_CODE` | IMEX or REP | Local discovery required | No | No | Use whichever local security-master export or report provides stable identifiers and classification timing. |
| security master | `SECTOR_CODE` | IMEX or REP | Local discovery required | No | No | Use whichever local security-master export or report provides stable identifiers and classification timing. |
| security master | `COUNTRY_CODE` | IMEX or REP | Local discovery required | No | No | Use whichever local security-master export or report provides stable identifiers and classification timing. |
| security master | `CURRENCY_CODE` | IMEX or REP | Local discovery required | No | No | Use whichever local security-master export or report provides stable identifiers and classification timing. |
| security master | `INDUSTRY_CODE` | IMEX or REP | Local discovery required | No | No | Use whichever local security-master export or report provides stable identifiers and classification timing. |


### `secperf.csv` Source Strategy

| Dataset | Demo column | Preferred source | Fallback source | Context required | Blocking if missing | Notes |
|---|---|---|---|---|---|---|
| security performance | `END_MV` | REP preferred | Local discovery required | No | Yes | Prefer report output because the local corpus does not prove native performance IMEX object names or calculation basis. |
| security performance | `INCOME` | REP preferred | Local discovery required | No | Yes | Prefer report output because the local corpus does not prove native performance IMEX object names or calculation basis. |
| security performance | `GAIN_LOSS` | REP preferred | Local discovery required | No | Yes | Prefer report output because the local corpus does not prove native performance IMEX object names or calculation basis. |
| security performance | `PORTFOLIO_CODE` | IMEX or REP | REP preferred | No | Yes | Portfolio and security identifiers must be present regardless of the extract source. |
| security performance | `SECURITY_ID` | IMEX or REP | REP preferred | No | Yes | Portfolio and security identifiers must be present regardless of the extract source. |
| security performance | `FROM_DATE` | REP preferred | Local discovery required | No | Yes | Prefer report output because the local corpus does not prove native performance IMEX object names or calculation basis. |
| security performance | `THRU_DATE` | REP preferred | Local discovery required | No | Yes | Prefer report output because the local corpus does not prove native performance IMEX object names or calculation basis. |
| security performance | `BEGIN_WEIGHT` | REP preferred | Local discovery required | No | No | Attribution-style values are report-sensitive; do not assume IMEX availability. |
| security performance | `BEGIN_MV` | REP preferred | Local discovery required | No | Yes | Prefer report output because the local corpus does not prove native performance IMEX object names or calculation basis. |
| security performance | `SEC_RETURN` | REP preferred | Local discovery required | No | Yes | Prefer report output because the local corpus does not prove native performance IMEX object names or calculation basis. |
| security performance | `CONTRIBUTION` | REP preferred | Local discovery required | No | No | Attribution-style values are report-sensitive; do not assume IMEX availability. |


### `transactions.csv` Source Strategy

| Dataset | Demo column | Preferred source | Fallback source | Context required | Blocking if missing | Notes |
|---|---|---|---|---|---|---|
| transactions | `TRANSACTION_ID` | Local discovery required | IMEX then REP cross-check | No | No | A stable native ID is not proven; ppar may need a deterministic local row key. |
| transactions | `PORT` | IMEX then REP cross-check | REP preferred | No | Yes | Transaction core fields are IMEX-suitable, with REP useful for report-facing cross-checks and sign-convention validation. |
| transactions | `TRANSACTION_DATE` | IMEX then REP cross-check | REP preferred | No | Yes | Transaction core fields are IMEX-suitable, with REP useful for report-facing cross-checks and sign-convention validation. |
| transactions | `SETTLE_DATE` | IMEX then REP cross-check | REP preferred | No | No | Transaction core fields are IMEX-suitable, with REP useful for report-facing cross-checks and sign-convention validation. |
| transactions | `SEC` | IMEX then REP cross-check | REP preferred | No | Yes | Transaction core fields are IMEX-suitable, with REP useful for report-facing cross-checks and sign-convention validation. |
| transactions | `TRAN` | IMEX then REP cross-check | REP preferred | No | Yes | Transaction core fields are IMEX-suitable, with REP useful for report-facing cross-checks and sign-convention validation. |
| transactions | `SEC_TYPE` | Local discovery required | REP preferred | Yes | Yes | Required for ambiguous Axys flow semantics. If IMEX does not expose this context, stop and use REP, a custom report, or another local source. |
| transactions | `SRC_DEST_TYPE` | Local discovery required | REP preferred | Yes | Yes | Required for ambiguous Axys flow semantics. If IMEX does not expose this context, stop and use REP, a custom report, or another local source. |
| transactions | `SRC_DEST_SYMBOL` | Local discovery required | REP preferred | Yes | Yes | Required for ambiguous Axys flow semantics. If IMEX does not expose this context, stop and use REP, a custom report, or another local source. |
| transactions | `SPECIAL_SEC_TYPE` | Local discovery required | REP preferred | Yes | Yes | Required for ambiguous Axys flow semantics. If IMEX does not expose this context, stop and use REP, a custom report, or another local source. |
| transactions | `SPECIAL_SEC_SYMBOL` | Local discovery required | REP preferred | Yes | Yes | Required for ambiguous Axys flow semantics. If IMEX does not expose this context, stop and use REP, a custom report, or another local source. |
| transactions | `QTY` | IMEX then REP cross-check | REP preferred | No | Yes | Transaction core fields are IMEX-suitable, with REP useful for report-facing cross-checks and sign-convention validation. |
| transactions | `PRICE` | IMEX then REP cross-check | REP preferred | No | Yes | Transaction core fields are IMEX-suitable, with REP useful for report-facing cross-checks and sign-convention validation. |
| transactions | `AMOUNT` | IMEX then REP cross-check | REP preferred | No | Yes | Transaction core fields are IMEX-suitable, with REP useful for report-facing cross-checks and sign-convention validation. |
| transactions | `COMMISSION` | IMEX then REP cross-check | REP preferred | No | No | Transaction core fields are IMEX-suitable, with REP useful for report-facing cross-checks and sign-convention validation. |


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

- [Chapter_05_Transactions.md](Chapter_05_Transactions.md)
- [Chapter_06_Holdings.md](Chapter_06_Holdings.md)
- [Chapter_10_Performance.md](Chapter_10_Performance.md)
- [Chapter_12_Imex.md](Chapter_12_Imex.md)
- [Chapter_13_Rep.md](Chapter_13_Rep.md)
- [Chapter_15_Data_Dictionary.md](Chapter_15_Data_Dictionary.md)
- [../axys_common_core_export.md](../axys_common_core_export.md)
- [../performance_comparison_demo_source_contract.md](../performance_comparison_demo_source_contract.md)
