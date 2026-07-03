# Axys/APX Reference

This folder is a technical reference for SS&C Advent Axys and APX behavior that
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

## File Roles

| File group | Role | Use for | Do not use as |
|---|---|---|---|
| `reference/Chapter_*.md` | Reader-facing reference | Supported conclusions, Unknowns, implementation cautions, and cross-topic navigation. | Full vendor specifications or complete source-system schemas. |
| `evidence/Research_*.md` | Evidence archive | Source-by-source notes, research history, provenance, and unresolved details. | The first reading path for a new reader. |
| `contracts/*.md` | Implementation aid | Cross-cutting contracts, generated summaries, and demo/test guidance. | A replacement for the chapters or official vendor documentation. |
| `contracts/*.yaml` | Machine-readable contract | Validation, test fixtures, and structured implementation inputs. | Narrative explanation. |
| `contracts/templates/*.yaml` | Site contract examples | Starting points for site-specific extract contracts. | Guaranteed Axys/APX schemas. |

## Main Reference Chapters

See [reference/README.md](reference/README.md) for the chapter index. The main
starting points are:

- [Chapter_01_Overview.md](reference/Chapter_01_Overview.md) for repository
  orientation, evidence rules, blockers, and cross-chapter navigation.
- [Chapter_05_Transactions.md](reference/Chapter_05_Transactions.md) for
  transaction codes, posting context, and source-data classification boundaries.
- [Chapter_10_Performance.md](reference/Chapter_10_Performance.md) for
  performance inputs, outputs, report labels, and methodology boundaries.
- [Chapter_15_Data_Dictionary.md](reference/Chapter_15_Data_Dictionary.md) and
  [Chapter_16_Glossary.md](reference/Chapter_16_Glossary.md) for field,
  artifact, dataset, and conceptual vocabulary.

## Implementation Contracts

Implementation-facing files live in `contracts/` so they do not compete with
the reader-facing chapters. See [contracts/README.md](contracts/README.md) for
the contract and template index.

## Evidence Archive

The `evidence/Research_*.md` files are preserved for traceability. They are not
intended to be a parallel reference manual. If evidence in that folder is
important for current implementation or reader understanding, it should also be
summarized in the relevant chapter.

## Maintenance Rules

- If a chapter and contract conflict, treat that as a cleanup issue. The
  contract should summarize or operationalize the chapter, not compete with it.
- If a research file contains the clearest current explanation of a topic, fold
  that explanation into the relevant chapter and leave the research file as
  provenance.
- Keep Unknowns explicit. Unsupported certainty is worse than a documented gap.
- Keep implementation contracts conservative. They should describe supported
  ppar behavior, not imply official vendor methodology.
