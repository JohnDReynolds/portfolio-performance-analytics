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
docs/
    01-Overview.md
    02-Axys-Architecture.md
    03-APX-Architecture.md
    04-Security-Master.md
    05-Transactions.md
    06-Holdings.md
    07-Cash.md
    08-Pricing.md
    09-Corporate-Actions.md
    10-Performance.md
    11-Classifications.md
    12-IMEX.md
    13-REP.md
    14-Reports.md
    15-Data-Dictionary.md
    16-Glossary.md

research/
examples/
references/
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

Conduct research for the following chapter:

04-Security-Master.md

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

Chapter_05_Transactions.md

Use only the supplied research and source material.  If you feel that the supplied material is not sufficient, and you could do much better with additional resource material, then stop and tell me specifically what you need.

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

I know this is a big long task, so take your time.  Do NOT give me a document and then turn right around and tell me that you can give me a better one.  Stop and ask me questions if you need to in order to produce a better document.

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
