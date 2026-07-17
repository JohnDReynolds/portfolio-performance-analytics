# Axys/APX Reference

This folder is a technical reference for SS&C Advent Axys/APX behavior that
matters to source-data extraction, performance comparison, and implementation
decisions.

The files are intentionally split by role. Read the chapter files first. Use
the `evidence/` folder when you need provenance or unresolved evidence. Use the
contracts and YAML files as implementation contracts, not as vendor manuals.

## Reader Path

1. Start with [Chapter_01_Overview.md](reference/Chapter_01_Overview.md) for the
   repository map, evidence rules, and current Axys/APX blockers.
2. Read the subject chapter that matches your question.
3. Check the matching `evidence/Research_*.md` file only when you need source
   evidence, dated research notes, or unresolved open questions.
4. Use the contracts and YAML files when implementing or validating ppar demo
   behavior.
5. Use [axys_apx_common_core_export.md](axys_apx_common_core_export.md) as the
   starter field-shape reference, subject to the confidence boundaries in the
   chapters and contracts.
6. Use [axys_apx_reference_blueprint.md](axys_apx_reference_blueprint.md) when
   maintaining the repository's editorial standards, evidence rules, or
   chapter structure.

## File Roles

| File group | Role | Use for | Do not use as |
|---|---|---|---|
| `axys_apx_reference_blueprint.md` | Governing editorial specification | Evidence standards, confidence labels, chapter structure, and repository workflow. | Axys/APX product evidence or implementation behavior. |
| `reference/Chapter_*.md` | Reader-facing reference | Supported conclusions, Unknowns, implementation cautions, and cross-topic navigation. | Full vendor specifications or complete source-system schemas. |
| `evidence/Research_*.md` | Evidence archive | Source-and-claim ledgers, ownership boundaries, focused provenance, and unresolved evidence requirements. | The first reading path for a new reader. |
| `contracts/*.md` | Implementation aid | Cross-cutting contracts, generated summaries, and demo/test guidance. | A replacement for the chapters or official vendor documentation. |
| `contracts/*.yaml` | Machine-readable contract | Validation, test fixtures, and structured implementation inputs. | Narrative explanation. |
| `contracts/templates/*.yaml` | Site contract examples | Starting points for site-specific extract contracts. | Guaranteed Axys/APX schemas. |
| `axys_apx_common_core_export.md` | Starter export reference | Common field shapes and candidate aliases for integration planning. | An official or guaranteed Axys/APX export specification. |

## Where to Put New Information

| Information type | Primary location | Supporting location |
|---|---|---|
| Transaction-code meanings, confidence labels, and context gates | [Chapter_05_Transactions.md](reference/Chapter_05_Transactions.md) | [Research_05_Transactions.md](evidence/Research_05_Transactions.md) |
| Transaction behavior used by ppar demos, tests, or validation | [transaction_semantics_matrix.yaml](contracts/transaction_semantics_matrix.yaml) | [transaction_semantics_matrix.md](contracts/transaction_semantics_matrix.md) |
| Split factors, corporate actions, reorganizations, and ACA/Reorg evidence | [Chapter_09_Corporate_Actions.md](reference/Chapter_09_Corporate_Actions.md) | [Research_09_Corporate_Actions.md](evidence/Research_09_Corporate_Actions.md) |
| Dataset, file, and field definitions | [Chapter_15_Data_Dictionary.md](reference/Chapter_15_Data_Dictionary.md) | Relevant `Research_*.md` file |
| Terms and conceptual vocabulary | [Chapter_16_Glossary.md](reference/Chapter_16_Glossary.md) | Relevant subject chapter |
| Multi-currency capability, FX evidence, and implementation boundaries | [Chapter_17_Multi_Currency.md](reference/Chapter_17_Multi_Currency.md) | [Research_17_Multi_Currency.md](evidence/Research_17_Multi_Currency.md) and [Research_17A_Multi_Currency_Cash_Provenance.md](evidence/Research_17A_Multi_Currency_Cash_Provenance.md) |

## Chapter Index

Use [reference/README.md](reference/README.md) as the single complete chapter
index. Chapter 01 provides system-level orientation and cross-cutting blockers;
the subject chapters own their detailed conclusions.

## Implementation Contracts

Implementation-facing files live in `contracts/` so they do not compete with
the reader-facing chapters. See [contracts/README.md](contracts/README.md) for
the contract and template index.

## Evidence Archive

The `evidence/Research_*.md` files preserve traceability as compact claim
ledgers or ownership boundaries. They are not a parallel reference manual.
Evidence important to current implementation or reader understanding should be
summarized in the relevant chapter.

## Temporary Research Intake

When a temporary research file is used as input, merge it into the matching
`evidence/Research_*.md` file first, summarize the durable conclusion in the
matching `reference/Chapter_*.md` file, and then delete the temporary file. Keep
the source filename and merge date in the evidence file so provenance remains
visible without leaving loose files in the repository root.

## Maintenance Rules

- If a chapter and contract conflict, treat that as a cleanup issue. The
  contract should summarize or operationalize the chapter, not compete with it.
- If a research file contains the clearest current explanation of a topic, fold
  that explanation into the relevant chapter and leave the research file as
  provenance.
- Integrate durable conclusions into the relevant subject section rather than
  appending dated `Update` or `Addendum` sections. When chronology matters,
  retain one compact `Research Provenance` note that points to the evidence
  archive.
- Keep Unknowns explicit. Unsupported certainty is worse than a documented gap.
- Keep implementation contracts conservative. They should describe supported
  ppar behavior, not imply official vendor methodology.
