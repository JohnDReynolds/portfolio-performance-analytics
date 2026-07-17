# Evidence Archive

This folder preserves compact source-and-claim ledgers, ownership boundaries,
focused provenance, and unresolved evidence requirements for the Axys/APX
reference.

Use these files when you need to audit why a chapter says something, chase an
Unknown, or recover detail that is too noisy for the reader-facing chapters.
Start with the chapter files in `../reference/` for normal reading.

If a research file contains the clearest current explanation of a topic, fold
that explanation into the relevant chapter and leave the research file as
provenance.

## Evidence Index

| Evidence file | Status | Supports |
|---|---|---|
| [Public_Web_Research_2026-07-17.md](Public_Web_Research_2026-07-17.md) | Claim ledger | Public evidence incorporated across Chapters 02-14 and 17. |
| [Research_02_Axys_Architecture.md](Research_02_Axys_Architecture.md) | Claim ledger | Chapter 02 |
| [Research_03_APX_Architecture.md](Research_03_APX_Architecture.md) | Claim ledger | Chapter 03 |
| [Research_04_Security_Master.md](Research_04_Security_Master.md) | Claim ledger | Chapter 04 |
| [Research_05_Transactions.md](Research_05_Transactions.md) | Claim ledger | Chapter 05 |
| [Research_06_Holdings.md](Research_06_Holdings.md) | Claim ledger | Chapter 06 |
| [Research_07_Cash.md](Research_07_Cash.md) | Claim ledger | Chapter 07 |
| [Research_08_Pricing.md](Research_08_Pricing.md) | Claim ledger | Chapter 08 |
| [Research_09_Corporate_Actions.md](Research_09_Corporate_Actions.md) | Claim ledger | Chapter 09 |
| [Research_10_Performance.md](Research_10_Performance.md) | Claim ledger | Chapter 10 |
| [Research_11_Classifications.md](Research_11_Classifications.md) | Claim ledger | Chapter 11 |
| [Research_12_IMEX.md](Research_12_IMEX.md) | Claim ledger | Chapter 12 |
| [Research_13_REP.md](Research_13_REP.md) | Claim ledger | Chapter 13 REP/RepLang evidence. |
| [Research_14_Reports.md](Research_14_Reports.md) | Claim ledger | Chapter 14 report evidence. |
| [Research_15_Data_Dictionary.md](Research_15_Data_Dictionary.md) | Ownership boundary | Derivative Chapter 15 index. |
| [Research_16_Glossary.md](Research_16_Glossary.md) | Ownership boundary | Derivative Chapter 16 glossary. |
| [Research_17_Multi_Currency.md](Research_17_Multi_Currency.md) | Claim ledger | Chapter 17 currency, FX, valuation, and performance boundaries. |
| [Research_17A_Multi_Currency_Cash_Provenance.md](Research_17A_Multi_Currency_Cash_Provenance.md) | Focused claim ledger | Chapter 17 purchase cash-provenance inference boundary. |

There is no `Research_01` file because `Chapter_01_Overview.md` is an
orientation chapter rather than a source-domain research chapter.

Chapter 17 uses two ledgers because purchase cash-bucket provenance is a narrow
inference problem within the broader currency, FX, valuation, and performance
domain. Research 17A owns only that focused inference boundary.

## Maintenance Boundary

Evidence files are compact provenance, not parallel manuals. Record new
evidence with:

- source identity and retrieval or observation date;
- extracted claim, quotation locator, or artifact locator;
- affected Axys/APX version or environment when known;
- confidence and any contradiction;
- unresolved question; and
- the chapter or contract conclusion affected by the evidence.

Reader explanations, conceptual models, worked examples, audit rules, and
implementation recommendations belong in the matching chapter or contract.
Integrate new durable conclusions into the reader-facing chapter and append
only the granular provenance needed to support them in the matching ledger.

Cross-topic public research may use a dated central ledger with stable claim
IDs. Matching topic research files should point to the relevant claim IDs and
record only the topic-specific chapter impact or unresolved contradiction.

Unknowns are canonical in the owning reader chapter. Evidence files should
record the missing source or contradiction needed to resolve an Unknown rather
than repeat the chapter's complete Unknowns table.
