# Evidence Boundary: Data Dictionary

This file records the provenance boundary for
[`Chapter_15_Data_Dictionary.md`](../reference/Chapter_15_Data_Dictionary.md).
Chapter 15 is a derivative cross-reference: it consolidates literal fields,
codes, filenames, utilities, report labels, and aliases supported by the
subject chapters. It is not an independent source domain and therefore does
not require a parallel narrative research chapter.

## Evidence Ownership

| Evidence | Canonical evidence file | Reader-facing owner |
|---|---|---|
| Security identifiers, `sec.inf`, and `type.inf` | [`Research_04_Security_Master.md`](Research_04_Security_Master.md) | Chapter 04 |
| Transaction fields and codes | [`Research_05_Transactions.md`](Research_05_Transactions.md) | Chapter 05 |
| Holdings and position fields | [`Research_06_Holdings.md`](Research_06_Holdings.md) | Chapter 06 |
| Cash symbols and cash-related fields | [`Research_07_Cash.md`](Research_07_Cash.md) | Chapter 07 |
| Prices, price files, and diagnostics | [`Research_08_Pricing.md`](Research_08_Pricing.md) | Chapter 08 |
| `split.inf` and logical split fields | [`Research_09_Corporate_Actions.md`](Research_09_Corporate_Actions.md) | Chapter 09 |
| Performance labels and candidate extract fields | [`Research_10_Performance.md`](Research_10_Performance.md) | Chapter 10 |
| Classification fields and labels | [`Research_11_Classifications.md`](Research_11_Classifications.md) | Chapter 11 |
| IMEX utilities, paths, logs, and candidate normalized schemas | [`Research_12_IMEX.md`](Research_12_IMEX.md) | Chapter 12 |
| REP files and expressions | [`Research_13_REP.md`](Research_13_REP.md) | Chapter 13 |
| Report names and report-output labels | [`Research_14_Reports.md`](Research_14_Reports.md) | Chapter 14 |
| Public-source claims incorporated on 2026-07-17 | [`Public_Web_Research_2026-07-17.md`](Public_Web_Research_2026-07-17.md) | Affected subject chapters |

## Preserved Conclusions

- The repository confidence labels and the original field-dictionary format
  come from `../axys_apx_reference_blueprint.md`.
- Candidate normalized schemas are implementation design aids, not verified
  native Axys/APX field dictionaries. Their evidence and limitations are
  maintained in Research 12.
- `SplitDate`, `SplitSymbol`, and `SplitFactor` are logical labels found in
  consultant split-file processing evidence. They are not verified official
  `split.inf` headers. Their evidence is maintained in Research 09.
- Official native field dictionaries remain unavailable without versioned
  vendor documentation, live IMEX catalogs, sanitized exports, REP source, or
  APX schema/API material.

## Maintenance Rule

Do not add source-domain narrative here. Add new evidence to the owning subject
research file or to a cross-topic claim ledger, then add or update the Chapter
15 index entry. Git history retains the earlier planning narrative that was
used to design Chapter 15.
