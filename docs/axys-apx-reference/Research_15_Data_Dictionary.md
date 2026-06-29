# Research_15_Data_Dictionary.md

# Axys / APX Data Dictionary Research

## 1. Research Status

| Item | Value |
|---|---|
| Repository chapter supported | `Chapter_15_Data_Dictionary.md` |
| Research file | `Research_15_Data_Dictionary.md` |
| Governing specification | `AXYS_APX_REFERENCE_BLUEPRINT.md`, Version 2.0 |
| Research basis | Supplied blueprint only |
| External vendor documentation supplied | No |
| IMEX export samples supplied | No |
| REP report samples supplied | No |
| Existing chapter supplied | No |
| Existing research supplied | No |
| Overall completeness | Incomplete by design; factual field-level content cannot be derived from the blueprint alone |
| Recommended next step | Add vendor documentation, IMEX object definitions, sample IMEX exports, REP report outputs, and existing repository research chapters before writing the final chapter |

**Research conclusion:** The supplied blueprint is sufficient to define the required standards, scope, chapter structure, confidence labels, and field-dictionary format. It is not sufficient to populate an authoritative Axys/APX data dictionary with actual field names, file names, report names, object names, or processing behavior.

**Confidence:** Verified.

---

## 2. Governing Editorial Requirements Extracted from Blueprint

The blueprint defines the repository as a technical reference for SS&C Axys and SS&C APX. It requires implementation-oriented documentation covering architecture, accounting data, IMEX, REP, reports, file layouts, data fields, processing behavior, version differences, and implementation quirks.

**Confidence:** Verified.

The blueprint requires technical statements to be classified as one of:

| Confidence Label | Meaning for this research file |
|---|---|
| Verified | Directly supported by supplied source material |
| High Confidence | Strongly supported by supplied source material or repeated supplied evidence |
| Medium Confidence | Plausible but not fully supported by supplied evidence |
| Unknown | Not supported by supplied source material; must not be invented |

**Confidence:** Verified.

The blueprint explicitly instructs that unsupported information should be marked **Unknown**, and that invented certainty is not acceptable.

**Confidence:** Verified.

The blueprint requires Axys and APX to be documented separately whenever their behavior differs.

**Confidence:** Verified.

The blueprint prefers vendor documentation, IMEX exports, REP reports, production observations, consultant documentation, examples, and tables over general explanations.

**Confidence:** Verified.

The blueprint defines the standard field dictionary format as:

| Field | Description | Axys | APX | IMEX | REP | Confidence |
|------|-------------|------|-----|------|-----|------------|

**Confidence:** Verified.

---

## 3. Research Scope for Chapter 15

Chapter 15 should function as a cross-repository data dictionary. It should not replace detailed subject chapters such as Security Master, Transactions, Holdings, Cash, Pricing, Corporate Actions, Performance, Classifications, IMEX, REP, or Reports. Instead, it should consolidate field names, meanings, source locations, confidence levels, and cross-system mappings.

**Confidence:** High Confidence.

Reason: This follows from the repository structure and from the blueprint's field dictionary standard, but the blueprint does not explicitly define Chapter 15's internal scope beyond the chapter title.

---

## 4. Evidence Inventory

| Evidence Type | Supplied? | Usable for Field Dictionary? | Notes | Confidence |
|---|---:|---:|---|---|
| Blueprint | Yes | Partially | Defines standards, required sections, confidence labels, and field dictionary format | Verified |
| Vendor Axys documentation | No | No | Needed for authoritative Axys fields, files, reports, and behavior | Verified |
| Vendor APX documentation | No | No | Needed for authoritative APX fields, database objects, reports, and behavior | Verified |
| IMEX object definitions | No | No | Needed to identify exportable objects and field names | Verified |
| IMEX export samples | No | No | Needed to verify actual field names, order, values, and quirks | Verified |
| REP report definitions | No | No | Needed to verify report names, columns, and parameter behavior | Verified |
| REP output samples | No | No | Needed to verify displayed fields and report-specific calculation behavior | Verified |
| Existing repository research chapters | No | No | Needed to consolidate cross-chapter fields | Verified |
| Production observations | No | No | Needed to classify implementation quirks as Verified or High Confidence | Verified |
| Consultant documentation | No | No | Could support High Confidence claims if supplied | Verified |

---

## 5. Data Dictionary Design Requirements

The data dictionary should contain factual entries only. Each entry should identify:

| Required Element | Purpose | Confidence |
|---|---|---|
| Field name | The literal field, column, tag, report column, or export column name | Verified when sourced from export/report/doc |
| Description | Meaning of the field in Axys/APX usage | Verified or High Confidence only when documented |
| Axys support | Whether Axys uses or exposes the field | Unknown until sourced |
| APX support | Whether APX uses or exposes the field | Unknown until sourced |
| IMEX support | Whether the field appears in an IMEX object/export | Unknown until sourced |
| REP support | Whether the field appears in a REP report/output | Unknown until sourced |
| Data type | Text, date, numeric, decimal, code, Boolean, etc. | Unknown until sourced |
| Required/optional | Whether field is required for imports/exports/reports | Unknown until sourced |
| Key role | Identifier, foreign key, classification, metric, date, amount, quantity, code, etc. | Unknown until sourced |
| Source chapter | Chapter where the field is explained in detail | Unknown until chapter exists |
| Notes/quirks | Known behavior, version differences, or implementation caveats | Unknown until sourced |
| Confidence | Verified, High Confidence, Medium Confidence, or Unknown | Verified requirement from blueprint |

---

## 6. Recommended Chapter 15 Structure

The blueprint standard chapter template allows:

1. Overview
2. Axys
3. APX
4. IMEX
5. REP
6. Data Model
7. Common Fields
8. Examples
9. Known Issues / Quirks
10. References
11. Unknowns

For Chapter 15, the following adapted structure is recommended:

| Section | Purpose | Confidence |
|---|---|---|
| Overview | Explain that Chapter 15 is a consolidated cross-reference for field names and meanings | High Confidence |
| Dictionary Conventions | Define confidence labels, field naming, aliases, and source precedence | High Confidence |
| Axys Data Fields | Axys-specific fields, files, and reports | Unknown pending sources |
| APX Data Fields | APX-specific fields, tables, views, and reports | Unknown pending sources |
| IMEX Fields | Export/import object fields and sample layouts | Unknown pending sources |
| REP Fields | Report fields, report columns, and parameter-dependent columns | Unknown pending sources |
| Common Field Families | Security identifiers, portfolio identifiers, dates, quantities, prices, cash, classifications, performance fields | High Confidence as an organizing scheme; field names Unknown |
| Field Dictionary Tables | Canonical tables using blueprint format | Verified format |
| Alias / Synonym Map | Same concept under different field names across Axys, APX, IMEX, and REP | Unknown pending sources |
| Required Field Sets | Minimal sets for common use cases such as security master, transactions, holdings, performance, pricing, classifications | Unknown pending sources |
| Examples | Sample field dictionary rows from verified exports/reports | Unknown pending sources |
| Known Issues / Quirks | Implementation notes, version differences, naming inconsistencies, type conversions | Unknown pending sources |
| References | Source list | Verified for blueprint only |
| Unknowns | Explicit gaps | Verified as required by blueprint |

---

## 7. Source Precedence Model

When multiple sources disagree, Chapter 15 should prefer evidence in this order:

| Rank | Source Type | Rationale | Confidence Impact |
|---:|---|---|---|
| 1 | Actual IMEX export from a production/test system | Verifies real field names and exported values | Verified for that environment/version |
| 2 | Actual REP output from a production/test system | Verifies displayed/reportable columns | Verified for that environment/version |
| 3 | Vendor documentation | Authoritative but may differ by version or configuration | Verified if versioned |
| 4 | Existing repository research with citations | Consolidated but should trace back to source | High Confidence or Verified depending source |
| 5 | Consultant documentation | Useful but should be checked against extracts | High Confidence or Medium Confidence |
| 6 | User/production observations | Useful for quirks and behavior | High Confidence when detailed and repeatable |
| 7 | General domain knowledge | Not sufficient for field names | Unknown or Medium Confidence only |

**Confidence:** High Confidence.

Reason: The blueprint prefers vendor documentation, IMEX exports, REP reports, production observations, consultant documentation, examples, and tables; the precise ranking above is a recommended implementation detail.

---

## 8. Axys Data Dictionary Research

### 8.1 Axys Field Inventory

No Axys field inventory was supplied.

| Field | Description | Axys | APX | IMEX | REP | Confidence |
|------|-------------|------|-----|------|-----|------------|
| Unknown | Axys security identifier fields | Unknown | Unknown | Unknown | Unknown | Unknown |
| Unknown | Axys portfolio/account identifier fields | Unknown | Unknown | Unknown | Unknown | Unknown |
| Unknown | Axys transaction date fields | Unknown | Unknown | Unknown | Unknown | Unknown |
| Unknown | Axys trade date fields | Unknown | Unknown | Unknown | Unknown | Unknown |
| Unknown | Axys settlement date fields | Unknown | Unknown | Unknown | Unknown | Unknown |
| Unknown | Axys quantity/share fields | Unknown | Unknown | Unknown | Unknown | Unknown |
| Unknown | Axys price fields | Unknown | Unknown | Unknown | Unknown | Unknown |
| Unknown | Axys market value fields | Unknown | Unknown | Unknown | Unknown | Unknown |
| Unknown | Axys cost fields | Unknown | Unknown | Unknown | Unknown | Unknown |
| Unknown | Axys cash fields | Unknown | Unknown | Unknown | Unknown | Unknown |
| Unknown | Axys income fields | Unknown | Unknown | Unknown | Unknown | Unknown |
| Unknown | Axys classification fields | Unknown | Unknown | Unknown | Unknown | Unknown |
| Unknown | Axys performance fields | Unknown | Unknown | Unknown | Unknown | Unknown |
| Unknown | Axys report column names | Unknown | Unknown | Unknown | Unknown | Unknown |

### 8.2 Axys Behavior Relevant to Data Dictionary

| Topic | Research Finding | Confidence |
|---|---|---|
| Axys stores or exposes data fields | Unknown from supplied source | Unknown |
| Axys file-level field names | Unknown from supplied source | Unknown |
| Axys IMEX exportable fields | Unknown from supplied source | Unknown |
| Axys REP report columns | Unknown from supplied source | Unknown |
| Axys processing behavior affecting fields | Unknown from supplied source | Unknown |
| Axys version differences affecting fields | Unknown from supplied source | Unknown |
| Axys implementation quirks affecting fields | Unknown from supplied source | Unknown |

---

## 9. APX Data Dictionary Research

### 9.1 APX Field Inventory

No APX field inventory was supplied.

| Field | Description | Axys | APX | IMEX | REP | Confidence |
|------|-------------|------|-----|------|-----|------------|
| Unknown | APX security identifier fields | Unknown | Unknown | Unknown | Unknown | Unknown |
| Unknown | APX portfolio/account identifier fields | Unknown | Unknown | Unknown | Unknown | Unknown |
| Unknown | APX transaction date fields | Unknown | Unknown | Unknown | Unknown | Unknown |
| Unknown | APX trade date fields | Unknown | Unknown | Unknown | Unknown | Unknown |
| Unknown | APX settlement date fields | Unknown | Unknown | Unknown | Unknown | Unknown |
| Unknown | APX quantity/share fields | Unknown | Unknown | Unknown | Unknown | Unknown |
| Unknown | APX price fields | Unknown | Unknown | Unknown | Unknown | Unknown |
| Unknown | APX market value fields | Unknown | Unknown | Unknown | Unknown | Unknown |
| Unknown | APX cost fields | Unknown | Unknown | Unknown | Unknown | Unknown |
| Unknown | APX cash fields | Unknown | Unknown | Unknown | Unknown | Unknown |
| Unknown | APX income fields | Unknown | Unknown | Unknown | Unknown | Unknown |
| Unknown | APX classification fields | Unknown | Unknown | Unknown | Unknown | Unknown |
| Unknown | APX performance fields | Unknown | Unknown | Unknown | Unknown | Unknown |
| Unknown | APX report column names | Unknown | Unknown | Unknown | Unknown | Unknown |

### 9.2 APX Behavior Relevant to Data Dictionary

| Topic | Research Finding | Confidence |
|---|---|---|
| APX stores or exposes data fields | Unknown from supplied source | Unknown |
| APX database objects | Unknown from supplied source | Unknown |
| APX IMEX exportable fields | Unknown from supplied source | Unknown |
| APX REP report columns | Unknown from supplied source | Unknown |
| APX processing behavior affecting fields | Unknown from supplied source | Unknown |
| APX version differences affecting fields | Unknown from supplied source | Unknown |
| APX implementation quirks affecting fields | Unknown from supplied source | Unknown |

---

## 10. IMEX Data Dictionary Research

### 10.1 IMEX Object Inventory

No IMEX object list, field list, or export sample was supplied.

| IMEX Object | Expected Content | Axys | APX | Field List Supplied? | Confidence |
|---|---|---|---|---:|---|
| Unknown | Security master export/import object | Unknown | Unknown | No | Unknown |
| Unknown | Transaction export/import object | Unknown | Unknown | No | Unknown |
| Unknown | Holdings/positions export/import object | Unknown | Unknown | No | Unknown |
| Unknown | Cash export/import object | Unknown | Unknown | No | Unknown |
| Unknown | Pricing export/import object | Unknown | Unknown | No | Unknown |
| Unknown | Corporate action export/import object | Unknown | Unknown | No | Unknown |
| Unknown | Classification export/import object | Unknown | Unknown | No | Unknown |
| Unknown | Portfolio performance export/import object | Unknown | Unknown | No | Unknown |
| Unknown | Security performance export/import object | Unknown | Unknown | No | Unknown |

### 10.2 IMEX Field Requirements

| Topic | Research Finding | Confidence |
|---|---|---|
| IMEX field names | Unknown from supplied source | Unknown |
| IMEX object names | Unknown from supplied source | Unknown |
| IMEX field ordering | Unknown from supplied source | Unknown |
| IMEX delimiters/file format | Unknown from supplied source | Unknown |
| IMEX import required fields | Unknown from supplied source | Unknown |
| IMEX export optional fields | Unknown from supplied source | Unknown |
| IMEX date formats | Unknown from supplied source | Unknown |
| IMEX numeric formats | Unknown from supplied source | Unknown |
| IMEX code fields | Unknown from supplied source | Unknown |
| IMEX validation behavior | Unknown from supplied source | Unknown |
| IMEX version differences | Unknown from supplied source | Unknown |
| IMEX quirks | Unknown from supplied source | Unknown |

---

## 11. REP Data Dictionary Research

### 11.1 REP Report Inventory

No REP report definitions or report outputs were supplied.

| REP Report | Expected Content | Axys | APX | Field/Column List Supplied? | Confidence |
|---|---|---|---|---:|---|
| Unknown | Security master / security reference report | Unknown | Unknown | No | Unknown |
| Unknown | Transaction report | Unknown | Unknown | No | Unknown |
| Unknown | Holdings/positions report | Unknown | Unknown | No | Unknown |
| Unknown | Cash report | Unknown | Unknown | No | Unknown |
| Unknown | Pricing report | Unknown | Unknown | No | Unknown |
| Unknown | Corporate action report | Unknown | Unknown | No | Unknown |
| Unknown | Classification report | Unknown | Unknown | No | Unknown |
| Unknown | Portfolio performance report | Unknown | Unknown | No | Unknown |
| Unknown | Security performance report | Unknown | Unknown | No | Unknown |

### 11.2 REP Field Requirements

| Topic | Research Finding | Confidence |
|---|---|---|
| REP report names | Unknown from supplied source | Unknown |
| REP report column names | Unknown from supplied source | Unknown |
| REP report parameters affecting output | Unknown from supplied source | Unknown |
| REP output formats | Unknown from supplied source | Unknown |
| REP field aliases | Unknown from supplied source | Unknown |
| REP calculated fields | Unknown from supplied source | Unknown |
| REP stored vs recalculated values | Unknown from supplied source | Unknown |
| REP version differences | Unknown from supplied source | Unknown |
| REP quirks | Unknown from supplied source | Unknown |

---

## 12. Common Field Families for Future Research

The following field families are recommended for Chapter 15 organization. The family names are organizational categories, not verified Axys/APX field names.

| Field Family | Purpose | Actual Field Names Supplied? | Confidence |
|---|---|---:|---|
| Security identifiers | Identify securities/instruments across files, exports, and reports | No | High Confidence as category; field names Unknown |
| Portfolio/account identifiers | Identify portfolios/accounts/entities | No | High Confidence as category; field names Unknown |
| Dates | Trade, settlement, effective, valuation, price, performance, and report dates | No | High Confidence as category; field names Unknown |
| Transaction codes/types | Identify buys, sells, income, cash movements, corporate actions, fees, and other transaction types | No | High Confidence as category; field names Unknown |
| Quantity/shares/par | Position units and transaction units | No | High Confidence as category; field names Unknown |
| Price fields | Transaction prices, market prices, close prices, and valuation prices | No | High Confidence as category; field names Unknown |
| Market value fields | Position and account valuation fields | No | High Confidence as category; field names Unknown |
| Cost fields | Cost basis, realized/unrealized gain/loss inputs | No | High Confidence as category; field names Unknown |
| Cash fields | Cash balances and cash transactions | No | High Confidence as category; field names Unknown |
| Income fields | Dividends, interest, coupons, accruals, and income classification | No | High Confidence as category; field names Unknown |
| Classification fields | Industry, sector, country, asset class, currency, custom groupings | No | High Confidence as category; field names Unknown |
| Performance fields | Portfolio return, security return, contribution, weights, benchmark-related fields | No | High Confidence as category; field names Unknown |
| Report metadata | Report dates, run dates, parameters, basis, currency, grouping, and options | No | High Confidence as category; field names Unknown |
| Audit/control fields | Source, load date, modified date, user, status, validation flags | No | Medium Confidence as useful category; field names Unknown |

---

## 13. Proposed Canonical Field Dictionary Template

This is the minimum template required by the blueprint:

| Field | Description | Axys | APX | IMEX | REP | Confidence |
|------|-------------|------|-----|------|-----|------------|
| Unknown | Placeholder until sourced | Unknown | Unknown | Unknown | Unknown | Unknown |

For Chapter 15, the following expanded template is recommended:

| Field | Alias / Display Name | Description | Field Family | Data Type | Axys | APX | IMEX Object | REP Report | Required? | Source | Notes / Quirks | Confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown |

**Confidence:** High Confidence.

Reason: The blueprint requires the seven-column field dictionary. Additional columns are useful for a data dictionary chapter, but not explicitly required by the blueprint.

---

## 14. Proposed Alias / Synonym Mapping Template

A data dictionary should track conceptual equivalence without assuming identical semantics.

| Concept | Axys Field | APX Field | IMEX Field | REP Column | Same Meaning? | Notes | Confidence |
|---|---|---|---|---|---|---|---|
| Security identifier | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown |
| Portfolio identifier | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown |
| Valuation date | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown |
| Trade date | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown |
| Settlement date | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown |
| Quantity | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown |
| Price | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown |
| Market value | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown |
| Cost | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown |
| Return | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown |
| Weight | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown |
| Contribution | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown |
| Classification | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown |

---

## 15. Required Source Material Before Final Chapter Expansion

To create an authoritative Chapter 15, the following source material is needed.

### 15.1 Minimum Useful Inputs

| Source Material | Why Needed | Priority |
|---|---|---:|
| Axys IMEX object list | Establish Axys export/import object names | 1 |
| APX IMEX object list | Establish APX export/import object names | 1 |
| Sample Axys IMEX exports | Verify actual field names and values | 1 |
| Sample APX IMEX exports | Verify actual field names and values | 1 |
| Axys REP report definitions or outputs | Verify report names and columns | 1 |
| APX REP report definitions or outputs | Verify report names and columns | 1 |
| Existing Research_04 through Research_14 files | Consolidate known field names from source-specific research | 1 |
| Vendor documentation excerpts | Confirm official field definitions and version differences | 1 |
| Version/environment notes | Separate Axys/APX and version-specific behavior | 2 |
| Consultant notes or production observations | Capture implementation quirks | 2 |

### 15.2 Ideal Inputs by Topic

| Topic | Needed Evidence |
|---|---|
| Security Master | Security master exports, report outputs, vendor docs |
| Transactions | Transaction exports, import layouts, transaction code references |
| Holdings | Holdings/positions exports and reports |
| Cash | Cash balance and cash transaction exports/reports |
| Pricing | Price master exports, price reports, currency/price source fields |
| Corporate Actions | Corporate action exports/reports and transaction linkage evidence |
| Performance | Portfolio/security performance exports and reports |
| Classifications | Classification/grouping exports, security master classification fields, report grouping samples |
| IMEX | Object list, field list, file format rules, required fields, validation behavior |
| REP | Report list, field lists, parameter behavior, output examples |
| Reports | Report names, columns, stored/recalculated behavior, grouping behavior |

---

## 16. Known Issues / Quirks

No Axys/APX-specific data dictionary quirks were supplied.

| Area | Known Issue / Quirk | Axys | APX | Confidence |
|---|---|---|---|---|
| Field naming | Whether Axys and APX use identical names for equivalent concepts | Unknown | Unknown | Unknown |
| Field aliases | Whether REP columns differ from IMEX field names | Unknown | Unknown | Unknown |
| Date semantics | Whether same-named date fields have different meanings by object/report | Unknown | Unknown | Unknown |
| Numeric precision | Decimal precision and rounding by export/report | Unknown | Unknown | Unknown |
| Stored vs calculated fields | Whether reported values are stored, recalculated, or parameter-dependent | Unknown | Unknown | Unknown |
| Version differences | Whether field names or meanings differ by product version | Unknown | Unknown | Unknown |
| Custom fields | Whether client-specific fields appear in exports/reports | Unknown | Unknown | Unknown |
| Classifications | Whether classification fields are stored directly, joined, or report-derived | Unknown | Unknown | Unknown |
| Performance fields | Whether performance fields are stored, derived, linked, or recalculated | Unknown | Unknown | Unknown |

---

## 17. Example Entries

No verified Axys/APX field names were supplied. Therefore, this research file cannot provide real populated examples without violating the blueprint.

### 17.1 Empty Example Template

| Field | Description | Axys | APX | IMEX | REP | Confidence |
|------|-------------|------|-----|------|-----|------------|
| Unknown | Unknown until sourced from vendor documentation, IMEX export, REP output, or production observation | Unknown | Unknown | Unknown | Unknown | Unknown |

### 17.2 Example of Acceptable Future Entry Format

The following row is a format example only and does not assert an actual Axys/APX field.

| Field | Description | Axys | APX | IMEX | REP | Confidence |
|------|-------------|------|-----|------|-----|------------|
| `[verified_field_name]` | `[definition from source]` | `[supported/not supported/source-specific notes]` | `[supported/not supported/source-specific notes]` | `[object/export name]` | `[report/column name]` | Verified |

**Important:** Replace bracketed placeholders only when supported by supplied evidence.

---

## 18. Unknowns

The following items remain Unknown because they were not supplied.

| Question | Status |
|---|---|
| Which Axys fields identify a security? | Unknown |
| Which APX fields identify a security? | Unknown |
| Which IMEX object exports securities? | Unknown |
| Which IMEX object exports transactions? | Unknown |
| Which IMEX object exports holdings? | Unknown |
| Which IMEX object exports cash? | Unknown |
| Which IMEX object exports pricing? | Unknown |
| Which IMEX object exports performance? | Unknown |
| Which REP reports contain security master fields? | Unknown |
| Which REP reports contain transaction fields? | Unknown |
| Which REP reports contain holdings fields? | Unknown |
| Which REP reports contain cash fields? | Unknown |
| Which REP reports contain pricing fields? | Unknown |
| Which REP reports contain performance fields? | Unknown |
| Which fields are required for imports? | Unknown |
| Which fields are optional for exports? | Unknown |
| Which fields are calculated rather than stored? | Unknown |
| Which fields have version-dependent names or meanings? | Unknown |
| Which fields differ between Axys and APX? | Unknown |
| Which report columns are aliases for IMEX fields? | Unknown |
| Which fields are client-customizable? | Unknown |
| Which fields are safe to use as persistent keys? | Unknown |
| Which fields are display-only/report-only? | Unknown |
| Which fields are required for reconciliation/audit workflows? | Unknown |

---

## 19. References

| Reference | Type | Supplied? | Notes | Confidence |
|---|---|---:|---|---|
| `AXYS_APX_REFERENCE_BLUEPRINT.md`, Version 2.0 | Repository governing specification | Yes | Defines standards, repository purpose, confidence labels, chapter list, field dictionary format, and success criteria | Verified |

---

## 20. Research Summary

This research file establishes the structure and standards for `Research_15_Data_Dictionary.md` and the eventual `Chapter_15_Data_Dictionary.md`.

Verified facts available from the supplied blueprint:

1. The repository is a permanent technical reference for SS&C Axys and SS&C APX.
2. The repository should preserve factual, implementation-oriented knowledge.
3. The repository should cover data fields, file layouts, IMEX, REP, reports, processing behavior, version differences, and quirks.
4. Technical statements should be classified as Verified, High Confidence, Medium Confidence, or Unknown.
5. Axys and APX should be separated whenever behavior differs.
6. Unsupported information should be marked Unknown.
7. The required field dictionary format is: `Field`, `Description`, `Axys`, `APX`, `IMEX`, `REP`, `Confidence`.

No actual Axys/APX field names, IMEX layouts, REP report names, report fields, file names, version differences, or quirks were supplied. Therefore, all product-specific data dictionary content remains Unknown pending additional source material.

