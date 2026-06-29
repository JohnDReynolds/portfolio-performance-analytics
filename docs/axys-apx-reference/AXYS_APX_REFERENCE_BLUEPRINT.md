# AXYS / APX Reference Blueprint
Version 2.0

> This document defines the editorial standards for the Axys/APX Reference Repository.
> The repository exists to document **how Axys and APX actually work**.
> It is intended for both human readers and AI coding assistants (such as Codex).

---

# 1. Purpose

This repository is a permanent technical reference for SS&C Axys and SS&C APX.

Its purpose is to preserve factual, implementation-oriented knowledge about:

- system architecture
- accounting data
- IMEX
- REP
- reports
- file layouts
- data fields
- processing behavior
- version differences
- implementation quirks

The repository is **not**:

- a software project
- a product roadmap
- a portfolio accounting textbook
- an AI workflow guide

---

# 2. Editorial Principles

## Facts First

Document supported facts.

Never invent Axys or APX behavior.

Every important technical statement should be identified as:

- Verified
- High Confidence
- Medium Confidence
- Unknown

Unknown is acceptable.

Invented certainty is not.

---

## Separate Axys and APX

Whenever behavior differs, document each system independently.

Avoid generic statements such as "the system."

---

## Prefer Evidence

Prefer:

- vendor documentation
- IMEX exports
- REP reports
- production observations
- consultant documentation
- examples
- tables

over general explanations.

---

## Preserve Unknowns

If something cannot be verified, record it as Unknown rather than guessing.

---

# 3. Intended Audience

This repository is written for:

- software developers
- consultants
- investment operations
- performance analysts
- data engineers
- AI coding assistants

Assume the reader is technically proficient.

---

# 4. Repository Structure

```text
    Chapter_01_Overview.md
    Chapter_02_Axys_Architecture.md
    Chapter_03_APX_Architecture.md
    Chapter_04_Security_Master.md
    Chapter_05_Transactions.md
    Chapter_06_Holdings.md
    Chapter_07_Cash.md
    Chapter_08_Pricing.md
    Chapter_09_Corporate_Actions.md
    Chapter_10_Performance.md
    Chapter_11_Classifications.md
    Chapter_12_IMEX.md
    Chapter_13_REP.md
    Chapter_14_Reports.md
    Chapter_15_Data_Dictionary.md
    Chapter_16_Glossary.md
    Research_01_Overview.md
    Research_02_Axys_Architecture.md
    Research_03_APX_Architecture.md
    Research_04_Security_Master.md
    Research_05_Transactions.md
    Research_06_Holdings.md
    Research_07_Cash.md
    Research_08_Pricing.md
    Research_09_Corporate_Actions.md
    Research_10_Performance.md
    Research_11_Classifications.md
    Research_12_IMEX.md
    Research_13_REP.md
    Research_14_Reports.md
    Research_15_Data_Dictionary.md
    Research_16_Glossary.md
```

---

# 5. Standard Chapter Template

Use only the sections that are applicable.

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

---

# 6. Documentation Standards

Prefer:

- tables over prose
- field dictionaries
- sample IMEX exports
- sample REP reports
- diagrams
- examples
- version differences

Avoid:

- unnecessary portfolio accounting theory
- product ideas
- speculative implementation details
- unsupported conclusions

---

# 7. Field Dictionary Standard

| Field | Description | Axys | APX | IMEX | REP | Confidence |
|------|-------------|------|-----|------|-----|------------|

---

# 8. Success Criteria

A reader should be able to answer questions such as:

- Which IMEX object exports transactions?
- Which REP report contains security performance?
- Where are classifications stored?
- Does APX store or recalculate performance?
- Which fields identify a security?
- Which fields are required?
- Which reports use stored values?
- What are the known quirks?

using only this repository.

---

# Appendix A – Standard ChatGPT Prompts

## Prompt 1 – Research

Use this example prompt to create research notes for chapter 05-Transactions.md:

```text
Please read the attached AXYS_APX_REFERENCE_BLUEPRINT.md before doing anything else.

Treat it as the governing specification for this repository.

Create extensive research that will be everything necesary for the following chapter:

Research_16_Glossary.md

The goal is to collect factual information about Axys and APX.

Focus on:

- Axys behavior
- APX behavior
- IMEX
- REP
- field names
- report names
- processing behavior
- version differences
- implementation quirks
- examples
- references

Classify technical statements as:

- Verified
- High Confidence
- Medium Confidence
- Unknown

Never invent unsupported behavior.

I know this is a big long task, so take your time.  Stop and ask me questions if you need to.

Produce a single downloadable Markdown research file suitable for the research folder.
```

---

## Prompt 2 – Write or Expand a Chapter

Upload:

- this Blueprint
- the existing chapter (if any)
- relevant research notes
- relevant sample exports or reports

Use this example prompt to create chapter 05-Transactions.md:

```text
Please read the attached AXYS_APX_REFERENCE_BLUEPRINT.md before doing anything else.

Treat it as the governing specification for this repository.

Write or expand the following repository chapter:

Chapter_16_Glossary.md

Use only the supplied research and source material.  If you feel that the supplied material is not sufficient, and you could do much better with additional resource material, then stop and tell me specifically what you need.  Do NOT give me a document and then turn right around and tell me that you can give me a better one.  Stop and ask me questions if you need to in order to produce a better document.  Conversely, if I uploaded an existing document that you think is generally sufficient, then just stop and tell me that you have no further edits.

Write as a technical reference manual.

Prefer:

- factual statements
- tables
- field dictionaries
- IMEX details
- REP details
- examples
- version differences
- known quirks

Separate Axys and APX whenever their behavior differs.

Mark unsupported information as Unknown.

Do not invent field names, transaction codes, report behavior, or implementation details.

I know this is a big long task, so take your time.

Produce a single downloadable Markdown file only.
```

---

# Appendix B – Repository Workflow

For each chapter:

1. Research the topic.
2. Save the research in the `research` folder.
3. Write or expand the repository chapter using the research.
4. Update the chapter as additional verified information becomes available.

The repository should evolve by accumulating verified knowledge, not by rewriting theory.
