# PPAR Audit

> **Archived migration snapshot.** This v0.10 document is preserved for
> traceability and is not maintained. Use
> [`../product_constitution.md`](../product_constitution.md)
> for the current product constitution,
> [`../roadmap.md`](../roadmap.md) for the current product roadmap, and
> [`../product_specifications_index.md`](../product_specifications_index.md)
> for approved detailed specifications.

## Foundational Product Design

### Phases 1 through 3F — Doctrine, Architecture, and Evidence-Gated Product Direction

| Document field | Value |
|---|---|
| Document status | Phases 2 through 3F founder-approved; document split outline for founder review |
| Version | 0.10 |
| Date | 2026-07-17 |
| Primary audience | Founder, product leadership, engineering, implementation, and future commercial leadership |
| Writing posture | Internal and candid first; externally reusable second |
| Canonical file | `PPAR_Audit_Foundational_Product_Design.md` |
| Phase covered | Phase 1 — Product Doctrine and Boundaries; Phase 2 — Users, Workflows, and Conceptual Product Architecture; Phase 3A — Performance Change Investigation; Phase 3B — Performance Data Quality Audit; Phase 3C — Audit Readiness; Phase 3D — Executive Investigation Summary; Phase 3E — Audit Health Dashboard and Operational Intelligence; Phase 3F — Human Review and Disposition boundary |
| Supersedes | Version 0.2; its reviewed Phase 1 content is incorporated into this canonical document |
| Next phase gate | Founder review and approval of the proposed document split outline before restructuring or Phase 4 |
| Confirmed deployment doctrine | Local-first execution within the client-controlled environment is a permanent product principle |

## Change Log

### Version 0.10 — 2026-07-17

- Recorded founder approval of the evidence-gated Phase 3F boundary and the
  document split in principle.
- Added the founder principle that PPAR should maximize the value, accuracy,
  and presentation of information while recognizing that investigation
  workflow varies across users and remains too early to productize.
- Converted Human Review and Disposition from a future feature assumption into
  a distant, evidence-gated possibility. Validation may conclude that PPAR
  should provide no managed workflow.
- Added a proposed two-document outline for founder review only. No file was
  created, renamed, split, or moved, and Phase 4 remains gated.
- Preserved all twelve safety invariants. No application code was changed.

### Version 0.9 — 2026-07-17

- Recorded founder approval of all seven Phase 3E recommendations while
  preserving the recurring-use gate: approval of the design does not authorize
  history/dashboard implementation before client evidence establishes value.
- Adopted an evidence-horizon drafting discipline in response to founder
  concern that the canonical document was becoming unwieldy and specifying too
  many hypothetical details too early.
- Added a deliberately lean Phase 3F boundary brief for Human Review and
  Disposition. It records only current truth, non-negotiable product contracts,
  validation questions, and the evidence gate for later design. Workflow state
  names, schemas, storage, permissions, UI, notifications, and integrations
  remain intentionally unspecified.
- Deferred Phase 4 until the founder reviews both the Phase 3F boundary and the
  recommended consolidation of this document into a concise governing product
  design/roadmap with detailed approved specifications retained separately.
- Preserved all twelve safety invariants. No application code was changed.

### Version 0.8 — 2026-07-16

- Recorded founder approval of Phase 3D and its six recommended defaults:
  Executive Investigation Summary first in HTML/XLSX; one progressive summary
  for managers and analysts; separate performance and Data Audit attention;
  cause ranking only within one comparable review unit; affected-row/entity
  Data Audit counts rather than unvalidated incident counts; and deterministic
  narrative/actions without editable workflow or generative narrative.
- Added the Phase 3E functional specification for the Audit Health Dashboard
  and Operational Intelligence, including the conditional local history model,
  source-state identity, deduplication, metric-specific comparability, bounded
  descriptive metrics, recurrence language, management views, machine-readable
  output, failure behavior, safety mapping, and client-validation gate.
- Preserved the founder-approved deferral of history implementation until
  recurring client use establishes a serious business case. Phase 3E specifies
  the safe product boundary; it does not authorize a database, hosted service,
  composite health score, cross-client benchmarking, workflow inference, or
  application-code changes.
- Preserved all twelve safety invariants and the Phase 3F gate. No application
  code was changed.

### Version 0.7 — 2026-07-16

- Recorded founder approval of Phase 3C and its six recommended defaults:
  Ready/Qualified Ready/Blocked with Diagnostic Only outside the readiness
  statuses; one engine for explicit and automatic preflight; a small local
  versioned readiness artifact; external approval references rather than PPAR
  workflow; product/contract/configuration identity in provenance; and a
  controlled-validation versus operational-reliance distinction.
- Added the Phase 3D functional specification for the Executive Investigation
  Summary, grounded in the current three shared review tables, derived
  period-level triage, summary/supporting CSVs, review handoff metadata,
  evidence links, and generated portfolio/security bundles.
- Preserved the founder-approved Phase 3A decision to use status plus
  full-precision explained and residual amounts without an
  explanation-completeness percentage.
- Corrected an overstatement inherited from an implementation-design note:
  current HTML begins with `Performance Differences`, not a separate
  first-screen `Problems` grid. Such an executive entry layer remains future
  work.
- Preserved the local-first doctrine, all twelve safety invariants, deferred
  workflow/history infrastructure, and the Phase 3E gate. No application code
  was changed.

### Version 0.6 — 2026-07-16

- Recorded founder approval of Phase 3B and its five recommended defaults:
  concise rule-execution coverage; optional pilot rules limited to understood
  populations; fail-closed malformed rule configuration; integrated Data Audit
  placement initially; and reuse of Phase 3A prioritization until client
  evidence justifies a separate model.
- Added the Phase 3C functional specification for Audit Readiness, grounded in
  current configuration validation, source/extract contracts, structural and
  financial-integrity controls, diagnostic exceptions, bundle gates, and
  internal regression infrastructure.
- Corrected the current provenance boundary: generated bundles record the
  comparison path, source/extract context, contract versions, and semantic
  fingerprints, but do not currently embed the installed PPAR version or a
  complete configuration snapshot/fingerprint.
- Re-ran the complete project suite at the v0.6 checkpoint: all 824 tests
  passed, superseding the earlier intake note about a documentation-style
  failure.
- Preserved the local-first doctrine, all twelve safety invariants, and the
  Phase 3D gate. No application code was changed.

### Version 0.5 — 2026-07-16

- Recorded founder approval of Phase 3A and the four authorized working
  assumptions: fixed explanation precision; operational materiality used only
  for prioritization under approved policy; no explanation-completeness
  percentage initially; and portfolio reports mandatory in pilots with
  security reports selected when the case requires them.
- Added the Phase 3B functional specification for Performance Data Quality
  Audit, grounded in the current seven optional checks, mandatory continuity,
  snapshot-separated union execution, rule configuration, reviewer outputs,
  counted-cause separation, false-positive controls, and client-validation
  requirements.
- Preserved the decision to defer human-note/workflow and cross-run history
  infrastructure until a serious business case and client validation justify
  their complexity.
- Preserved all twelve safety invariants and the Phase 3C gate. No application
  code was changed.

### Version 0.4 — 2026-07-16

- Recorded founder approval of Phase 2 and authorization of its recommended
  defaults as Phase 3 working assumptions.
- Added the Phase 3A functional specification for Performance Change
  Investigation, with normative requirements, current/future status, source and
  configuration contracts, portfolio/security behavior, Modified Dietz
  attribution, explanation completeness, evidence, lineage, reviewer guidance,
  history/stability direction, acceptance criteria, and client-validation plan.
- Distinguished comparison tolerance, arithmetic/serialization precision, and
  operational materiality so future prioritization cannot silently erase a
  reportable source difference.
- Recorded the founder decision that human-note/workflow infrastructure and
  cross-run history/stability infrastructure are long-term possibilities whose
  implementation is DEFERRED until a serious business case and client
  validation justify their complexity.
- Preserved all twelve safety invariants and the Phase 3B gate. No application
  code was changed.

### Version 0.3 — 2026-07-16

- Created the intended unversioned canonical working document from the reviewed
  v0.2 snapshot.
- Re-verified Phase 1 against current project HEAD, generated portfolio and
  security evidence packs, executable contracts, tests, setup materials, and
  current-status documentation.
- Corrected the split-factor boundary, the exact Data Audit rule surface,
  manifest version, report entrypoints, transaction-contract authority, and
  capability-status labels without changing the approved product doctrine.
- Added Phase 2 actor profiles, decision rights, end-to-end workflows,
  conceptual product architecture, evidence flow, orthogonal status model,
  exception paths, acceptance assessment, and founder-review decisions.
- Preserved the Phase 3 gate. This version contains no Phase 3 functional
  specification and makes no application-code change.
- Consolidated the reviewed v0.2 content into this file and removed the
  redundant versioned snapshot after founder direction.

## Table of Contents

1. [Document Purpose and Authority](#0-document-purpose-and-authority)
2. [Executive Product Doctrine](#1-executive-product-doctrine)
3. [The Problem Domain](#2-the-problem-domain)
4. [Mission, Outcomes, and Success](#3-mission-outcomes-and-success)
5. [Primary Customer and User Hypotheses](#4-primary-customer-and-user-hypotheses)
6. [Jobs to Be Done](#5-jobs-to-be-done)
7. [Current Product Baseline](#6-current-product-baseline)
8. [Long-Term Capability Model](#7-long-term-capability-model)
9. [Product Principles](#8-product-principles)
10. [Trust, Review, and Accountability Model](#9-trust-review-and-accountability-model)
11. [Product Boundaries and Non-Goals](#10-product-boundaries-and-non-goals)
12. [Product-Claims Discipline](#11-product-claims-discipline)
13. [Terminology and Definitions](#12-terminology-and-definitions)
14. [First-Client Validation Doctrine](#13-first-client-validation-doctrine)
15. [Product Maturity Roadmap Summary](#14-product-maturity-roadmap-summary)
16. [Decisions Confirmed in Phase 1](#15-decisions-confirmed-in-phase-1)
17. [Open Decisions for Founder Review](#16-open-decisions-for-founder-review)
18. [Phase 2 — Users, Workflows, and Conceptual Product Architecture](#17-phase-2--users-workflows-and-conceptual-product-architecture)
19. [Phase 3A — Performance Change Investigation](#18-phase-3a--performance-change-investigation)
20. [Phase 3B — Performance Data Quality Audit](#19-phase-3b--performance-data-quality-audit)
21. [Phase 3C — Audit Readiness](#20-phase-3c--audit-readiness)
22. [Phase 3D — Executive Investigation Summary](#21-phase-3d--executive-investigation-summary)
23. [Phase 3E — Audit Health Dashboard and Operational Intelligence](#22-phase-3e--audit-health-dashboard-and-operational-intelligence)
24. [Phase 3F — Human Review and Disposition Boundary](#23-phase-3f--human-review-and-disposition-boundary)
25. [Appendix A — Source Register](#appendix-a--source-register)
26. [Appendix B — Representative Workbook Observations](#appendix-b--representative-workbook-observations)
27. [Appendix C — Incorporated Product-Expansion Inventory](#appendix-c--incorporated-product-expansion-inventory)
28. [Appendix D — Planned Foundational Design Sections](#appendix-d--planned-foundational-design-sections)
29. [Appendix E — Additional Implementation-Document Intake](#appendix-e--additional-implementation-document-intake)
30. [Appendix F — External Evidence](#appendix-f--external-evidence)

---

## 0. Document Purpose and Authority

### 0.1 Purpose

This document establishes the product doctrine for **PPAR Audit** and
progressively integrates the approved conceptual architecture and detailed
functional specifications that inherit it. Commercial packaging and final
release plans remain later-phase work.

It is intended to answer the foundational questions that every later product decision must inherit:

- What problem does PPAR Audit exist to solve?
- What product category is it trying to define?
- What is the narrow wedge that gives it a clear identity?
- What outcomes must the product deliver?
- What must remain subject to human judgment?
- What does the current product demonstrably do?
- Which ideas are approved directions rather than current capabilities?
- What is deliberately outside the product boundary?
- What language can be used honestly before real-client validation?

The purpose is not to make the product sound larger than it is. The purpose is to provide a durable basis for deciding what PPAR Audit should become without blurring current truth, future ambition, and attractive but unsupported ideas.

### 0.2 Authority hierarchy

The following authority order applies when sources conflict:

1. Current executable product behavior, tests, generated artifacts, and current
   machine-readable contracts
2. The current safety-invariant catalog and its maintainer-facing contract
3. The current README, setup-installed documentation, and compact architecture
   document
4. The **Current Status** and **Current Open Items** sections of the central PPAR
   roadmap
5. Current-checkpoint and explicit status notes in the deep
   performance-comparison design reference
6. Current machine-readable transaction-semantics and extract contracts
7. Explicit founder decisions in the governing handoff and this canonical
   product-design document
8. Founder-provided context concerning testing and commercial validation
9. Historical implementation journals, reviewer snapshots, rendered summaries,
   and earlier brainstorming or product-expansion drafts

Several supplied engineering documents are cumulative records: current contracts, completed status notes, historical plans, and superseded language may appear in the same file. When that occurs, an explicit current status, completion note, central-roadmap designation, or safety-invariant contract takes precedence over an older narrative paragraph.

Current product documentation and demonstrated behavior are authoritative descriptions of the present product. This document may recommend future changes, but it does not convert a recommendation into an implemented capability.

### 0.3 Source basis

This version is grounded in current project paths and generated evidence:

- `README.md`, `PPAR.pdf`, and `docs/architecture.md`
- `docs/roadmap.md`, using its current-status and current-open-items sections
- `docs/audit/performance_comparison_design.md`
- `docs/audit/performance_comparison_safety_invariants.md` and the executable
  `SAFETY_INVARIANTS` catalog
- `docs/axys_apx/contracts/transaction_semantics_matrix.yaml` and its
  rendered Markdown companion
- `docs/audit/demo_source_contract.md`,
  `docs/audit/site_extract_readiness_checklist.md`, and the current extract-contract
  implementation/templates
- the setup-installed audit README and `ppar.yaml`
- current generated portfolio and security XLSX, HTML, CSV, manifest,
  review-summary, and support-bundle artifacts under `_demo_output/`
- the executable report-bundle contract and current focused/full test evidence
- founder-provided context that the product has undergone extensive automated,
  financial-invariant, report-reconciliation, and large-dataset testing, but has
  not yet been validated against a real client’s Axys/APX exports

A source register appears in Appendix A. Observations from the representative workbook appear in Appendix B. Appendix E records how the additional implementation documents were classified and reconciled.

### 0.4 Additional implementation-document classification

| Document | Classification for this product design | How it should be used | Important caution |
|---|---|---|---|
| `docs/audit/performance_comparison_safety_invariants.md` | Current maintainer contract for enforced safety guarantees | Define current control promises, failure classes, lineage, conservation, parity, and determinism | It proves the software contract within supported scope; it does not validate a client’s mappings, exports, or accounting policy |
| `docs/roadmap.md` | Central engineering roadmap and implementation journal | Use **Current Status**, **Current Open Items**, and explicit current completion notes | Most numbered phases are historical implementation rationale, not a new product roadmap or a set of customer commitments |
| `docs/audit/performance_comparison_design.md` | Deep design/reference note with current and historical material | Use current checkpoint, explicit status notes, data contracts, field roles, matching rules, and report semantics | Earlier design paragraphs can be superseded by later implementation-status notes in the same file |
| `docs/audit/performance_comparison_transaction_boundary_snapshot.md` | Compact reviewer snapshot of transaction-semantics boundaries | Use to preserve conservative coverage and identify context-required or backlog families | The referenced machine-readable semantics matrix remains the implementation contract; narrow packaged examples do not establish universal production support |
| `docs/audit/archive/performance_comparison_evidence_pack_review.md` | Historical reviewer aid for one evidence-pack checkpoint | Use as a concise inventory of evidence-pack, extract-readiness, boundary, fixture, and validator work at that checkpoint | It explicitly is not the roadmap or the transaction-semantics contract and is not a current-status authority |

### 0.5 Design depth and evidence horizon

The canonical document governs product doctrine, current truth, major product
boundaries, approved direction, roadmap gates, and decisions that would be
costly or dangerous to reverse. It is not a substitute for client discovery or
an invitation to specify every plausible future feature.

Use this depth rule:

- **Current and near-term validated capabilities:** specify behavior deeply
  enough to preserve safety, guide implementation, and define acceptance.
- **Approved but unvalidated directions:** record the problem, value,
  non-negotiable boundaries, dependencies, and validation gate only.
- **Deferred or speculative capabilities:** do not define detailed schemas,
  state machines, permissions, UI, storage, integrations, or edge-case
  contracts until evidence makes those decisions necessary.

Detailed design should occur at the **evidence horizon**: the point at which a
real user problem, observed workflow, implementation decision, or safety risk
requires specificity. Attractive completeness beyond that horizon creates
false precision and maintenance burden.

When a later phase reaches beyond the evidence horizon, the correct output is a
short boundary brief and a validation plan, not a hypothetical functional
specification. This discipline supersedes any earlier instruction to fill every
standard specification field for a capability whose business case remains
unvalidated.

### 0.6 Capability-status language

The following labels will be used throughout the foundational design:

| Status | Meaning |
|---|---|
| **CURRENT — DEMONSTRATED** | Present in current generated artifacts or directly observed current behavior |
| **CURRENT — DOCUMENTED** | Described as current in authoritative project documentation or executable contracts |
| **CURRENT — REQUIRES CLIENT VALIDATION** | Implemented and internally tested, but not yet validated using a real client’s production-style exports and approved policy |
| **APPROVED DIRECTION** | Accepted as part of the intended product direction, but not necessarily designed in detail or implemented |
| **CANDIDATE** | A related idea that may fit the doctrine but requires prioritization and specification |
| **DEFERRED** | Potentially useful but intentionally postponed |
| **OUT OF SCOPE** | Inconsistent with the product identity or an unacceptable source of product dilution |
| **OPEN DECISION** | A material question that requires an explicit founder decision |

These labels are not release commitments.

---

# 1. Executive Product Doctrine

## 1.1 Internal category thesis

**PPAR Audit is the quality assurance layer for portfolio performance.**

It sits beside a portfolio accounting system rather than replacing it. It examines portfolio-accounting exports, compares an earlier and later view of reported data, determines what changed, quantifies supported causes of changed Modified Dietz performance, preserves supporting evidence, and directs human attention to unresolved or suspicious conditions.

The phrase **Performance Quality Assurance** is an internal category thesis. It is useful because it describes the intended long-term territory more clearly than “audit plus analytics.” It should not yet be presented as an established industry category or as a claim that PPAR provides independent assurance.

## 1.2 Core product promise

The strongest honest formulation of the promise is:

> **When reported performance changes, PPAR Audit should tell the reviewer why—or clearly identify what it cannot explain.**

This is intentionally more precise than an unqualified promise to “tell you exactly why.”

The current product explicitly supports:

- fully explained differences,
- partly explained differences, and
- unexplained differences.

A product that preserves unresolved items is more trustworthy than one that maximizes the appearance of automation. “Unexplained” is not necessarily a product failure. It is a controlled outcome when the supplied data, configured policy, or available methodology does not support a defensible conclusion.

Within the currently supported comparison surface, the implementation contract adds a second promise: reportable differences must not be silently lost or double-counted; counted causes must reconcile; unsupported treatment must fail closed or remain review evidence; and materially equivalent report formats must be derived from the same validated semantics.

## 1.3 Product wedge

The narrow commercial and product wedge is:

> **Structured investigation of changed reported portfolio performance.**

This is the center of the product identity. It converts an ad hoc spreadsheet exercise into a repeatable review process:

1. Identify the periods and securities whose reported performance changed.
2. Compare relevant source-data across two snapshots.
3. Quantify source changes that are permitted to explain Modified Dietz performance differences.
4. Separate counted causes from contextual evidence.
5. Reconcile explained amounts to reviewer-facing totals.
6. Surface the unresolved residual.
7. Preserve the source evidence required for review.

Everything else in the long-term vision must strengthen this central job.

## 1.4 Expansion path

The approved expansion path remains tightly connected to the wedge:

1. **Performance Change Investigation** — explain changed performance.
2. **Performance Data Quality Audit** — flag suspicious source-data relationships that may undermine performance quality.
3. **Audit Readiness** — identify missing, invalid, or ambiguous inputs before a full investigation.
4. **Executive Investigation Summary** — make investigation results immediately usable by management.
5. **Audit Health Dashboard** — show the current health and stability of the performance process.
6. **Operational Intelligence** — use accumulated audit history to reveal recurring weaknesses.
7. **PPAR Audit Rules Library** — encode reusable portfolio-operations knowledge as transparent rules.

This is an expansion from a reactive investigation tool toward a comprehensive operational review system. It is not permission to expand into unrelated accounting, reconciliation, analytics, or workflow categories.

## 1.5 Product identity in one paragraph

PPAR Audit is a local-first software product for investment operations and performance teams. It compares two configured snapshots of portfolio-accounting data and identifies where reported portfolio or security performance changed. It connects supported changes in holdings, transactions, FX rates, splits, and related Modified Dietz inputs to the performance difference; preserves traceable supporting evidence; distinguishes fully, partly, and unexplained results; and flags suspicious data relationships for human review. Over time, it should become the organization’s repeatable quality-assurance layer around portfolio performance without becoming the accounting book of record, a general reconciliation platform, or an independent assurance provider.

---

# 2. The Problem Domain

## 2.1 The central problem

The problem is not merely that historical performance changes.

Historical performance can change for legitimate reasons, including corrected holdings, back-dated transactions, revised prices, FX changes, split treatment, accrued-interest changes, source-file timing, or methodology differences. A changed return may be correct, incorrect, or impossible to determine from the available evidence.

The customer problem is:

> **The firm cannot explain the change quickly, quantitatively, consistently, and defensibly.**

That gap creates manual investigation work and uncertainty precisely when a performance analyst or operations manager needs a clear answer.

## 2.2 The current ad hoc process

Without a structured tool, the investigation commonly becomes a sequence of manual exports and spreadsheet comparisons:

- identify which reported returns changed,
- locate the relevant periods,
- compare transactions,
- compare holdings and prices,
- check cash flows,
- inspect corporate actions,
- review FX,
- infer relationships,
- estimate which differences affected performance,
- ask another specialist about unexplained items,
- preserve evidence in an inconsistent collection of files.

This process has several structural weaknesses:

- It is dependent on analyst experience.
- It is difficult to reproduce.
- It often mixes true causes, correlated changes, and unrelated anomalies.
- It is easy to lose the chain from a source-data change to a return effect.
- It is hard to prove that the explanation is complete.
- It usually produces weak management communication.
- It rarely creates reusable organizational knowledge.

PPAR Audit’s purpose is to replace this ad hoc process with a controlled investigation model.

## 2.3 The three strongest customer problems

### Problem 1 — Changed performance cannot be explained efficiently

A previously reported portfolio or security return changes, and the operations or performance team must determine why.

The strongest value of PPAR Audit is not that it finds differences. Generic file-comparison tools can find differences. Its differentiated value is that it connects supported source-data changes to a performance difference and separates quantified causes from evidence that still needs human judgment.

### Problem 2 — Suspicious data relationships are discovered too late

Portfolio accounting data may contain missing dividends, duplicate transactions, inconsistent prices, unusual accrued-interest relationships, split problems, or other conditions that deserve review.

These conditions may not always cause a changed return, and they may not always be errors. The operational problem is that they can remain hidden until they affect reporting, trigger a client question, or complicate an investigation.

### Problem 3 — The organization does not learn from repeated corrections

A firm may solve each investigation individually without developing visibility into:

- which portfolios change most often,
- which source categories produce the largest differences,
- which securities or transaction types recur,
- which issues remain unresolved,
- whether performance quality is improving.

The current product does not yet demonstrate a historical repository or operational-intelligence layer. This is an approved long-term direction, not a current claim.

## 2.4 Consequences of the problem

The business consequences can include:

- analyst time spent on repetitive investigation,
- reporting delays,
- escalation to senior operations or performance staff,
- inconsistent answers to internal or external questions,
- key-person dependency,
- weak evidence retention,
- reduced confidence in portfolio-accounting outputs,
- difficulty determining whether a correction is isolated or systemic.

PPAR Audit should not market unmeasured savings or liability reduction before client validation. These are problem hypotheses to be measured during pilots.

## 2.5 Triggering events

The most direct trigger is:

> “A return we previously reported has changed, and someone needs an explanation now.”

Related triggers may include:

- month-end or quarter-end review,
- discovery of a back-dated transaction,
- a revised accounting export,
- a price, FX, or corporate-action correction,
- client or portfolio-manager questions,
- a system conversion or configuration change,
- preparation for a reporting, compliance, or GIPS-related review,
- repeated unexplained historical restatements,
- a new operations leader seeking stronger controls.

The first-client strategy should prioritize prospects experiencing an active and recurring trigger rather than firms that merely find the concept interesting.

---

# 3. Mission, Outcomes, and Success

## 3.1 Mission

> **Make changed portfolio performance explainable, reviewable, and traceable to source evidence.**

The long-term mission extends beyond individual investigations:

> **Help investment organizations continuously improve the quality of reported portfolio performance.**

The first mission is the wedge. The second is the expansion.

## 3.2 Desired user outcome

After a PPAR Audit run, a qualified reviewer should be able to answer:

- What performance changed?
- By how much?
- What source-data changed?
- Which source changes are counted as causes?
- How much of the performance difference is explained?
- What remains unexplained?
- What suspicious source-data conditions deserve review?
- Where is the supporting evidence?
- What should happen next?

## 3.3 Definition of a successful audit run

A successful audit run is not defined solely by the percentage automatically explained.

Within the configured scope, a successful run should:

1. Identify the relevant changed performance periods.
2. Preserve a traceable disposition for relevant source differences.
3. Quantify only causes allowed by explicit methodology and site policy.
4. Reconcile counted cause amounts to the explained difference.
5. Reconcile displayed cause amounts after report rounding.
6. Preserve supporting source detail and lineage.
7. Surface unsupported or ambiguous items as review context.
8. Identify data-quality findings without presenting every anomaly as a proven error.
9. Fail clearly when required input or configuration is missing.
10. Produce a review package that another qualified person can follow.

## 3.4 Product north-star outcome

The north-star outcome is:

> **A trusted explanation that is faster to review than a manual investigation and more defensible than an ad hoc spreadsheet.**

The first-client pilots must measure both sides:

- **Analytical trust:** Are the findings and calculations accepted as accurate and appropriately limited?
- **Operational usefulness:** Does the review materially improve the investigation process?

## 3.5 Anti-goals and misleading metrics

The following are poor primary success metrics:

- percentage of periods labeled “Fully Explained,” without assessing correctness;
- number of rules, without assessing value or false positives;
- number of report pages;
- number of supported columns;
- number of platforms claimed;
- number of findings, since more findings may mean more noise rather than more value.

PPAR Audit must not optimize for automatic explanation at the expense of defensibility. A clearly labeled unresolved residual is preferable to a fabricated explanation.

---

# 4. Primary Customer and User Hypotheses

This section records working hypotheses. They are not yet validated by customer evidence.

## 4.1 Initial client profile

The strongest first validation partner is likely to be:

- an Axys-oriented, or possibly APX-oriented, investment manager or RIA;
- large enough to have recurring performance corrections and a dedicated operations or performance function;
- small enough to avoid enterprise procurement and long implementation cycles;
- operating with meaningful portfolio and transaction complexity;
- still relying heavily on manual reports and spreadsheets for investigations;
- able to produce the required CSV exports;
- willing to expose real edge cases and participate in mapping, review, and reconciliation;
- motivated by current pain rather than general curiosity;
- comfortable with an early implementation whose purpose includes product validation.

The first client is not merely a purchaser. It is a **validation partner**.

## 4.2 Primary daily user

**Working hypothesis:** Performance analyst, portfolio accounting analyst, or investment-operations analyst.

The primary user needs:

- a concise list of changed performance,
- quantified cause detail,
- direct access to source evidence,
- clear distinction between causes and review context,
- understandable failure messages,
- the ability to rerun an investigation reproducibly.

The primary user is expected to understand portfolio accounting and performance concepts. The product should not require that person to be a Python developer.

## 4.3 Product champion

**Working hypothesis:** Head of Performance, Director of Investment Operations, or Operations Manager.

The champion experiences the cost of manual investigations and can provide:

- access to appropriate data,
- staff participation,
- transaction-treatment decisions,
- validation of results,
- internal advocacy,
- permission for a case study if the pilot succeeds.

## 4.4 Economic buyer

**Working hypothesis:** Head of Operations, COO, or another leader responsible for investment operations, reporting quality, or operational risk.

The economic buyer is likely to care about:

- investigation time,
- reporting confidence,
- dependency on key analysts,
- repeatability,
- control evidence,
- implementation burden,
- data security,
- ongoing support.

This must be tested. In some firms, the economic buyer may be a performance leader or technology leader rather than the COO.

## 4.5 Secondary stakeholders

Potential secondary users or reviewers include:

- portfolio accounting management,
- compliance or GIPS personnel,
- client-reporting leadership,
- technology and information security,
- internal audit,
- senior investment operations management.

PPAR Audit should not assume these groups have the same needs. Phase 2 will define actor-specific workflows and outputs.

## 4.6 Likely blockers

Potential blockers include:

- information-security concerns,
- inability to produce consistent exports,
- uncertainty over transaction-code treatment,
- skepticism toward a new calculation tool,
- fear that findings will expose operational weaknesses,
- lack of an accountable reviewer,
- procurement requirements disproportionate to an early pilot,
- demands for broad platform or custom-report support,
- ambiguity about whether the product provides an “audit opinion.”

Trust, scope, and terminology must be addressed before feature breadth.

---

# 5. Jobs to Be Done

## 5.1 Primary job

> **When reported portfolio or security performance changes between two accounting snapshots, help me determine what changed, quantify the supported causes, identify the unresolved residual, and preserve evidence so I can defend the conclusion.**

This job defines the product.

## 5.2 Supporting jobs

### Identify the investigation scope

- Which portfolios changed?
- Which securities changed?
- Which dates or subperiods changed?
- Which differences are material enough to review?

### Understand source-data changes

- What changed in holdings?
- What changed in transactions?
- What changed in FX rates or split factors?
- Which changes are inherited into later periods?
- Which source rows are related?

### Separate cause from context

- Which changes are allowed to explain Modified Dietz performance?
- Which changes are relevant context but not counted?
- Which conditions are suspicious but independent of the performance difference?
- Which configuration decisions prevent an automated conclusion?

### Direct human review

- Which rows require a transaction-treatment decision?
- Which items may reflect missing source-data?
- Which items may reflect source-file timing?
- Which items may reflect vendor methodology outside the configured model?
- What should the reviewer inspect next?

### Preserve a defensible record

- Can another qualified reviewer reproduce the logic?
- Can the reviewer trace a displayed amount to source evidence?
- Are support files preserved with the report?
- Is the distinction between source fact and product inference clear?

## 5.3 Approved future jobs

The following are approved directions but not current product claims:

- Determine whether an audit is ready to run.
- Summarize the investigation for management.
- Monitor portfolio-performance stability over time.
- Identify recurring operational failure patterns.
- Track finding review, disposition, and closure.
- Compare quality trends across portfolios, sources, or reporting cycles.

## 5.4 Jobs that do not define PPAR Audit

The following jobs should not drive the product:

- calculate the official accounting book of record,
- replace custodian cash or position reconciliation,
- provide broad portfolio attribution and investment analytics,
- manage every operations workflow,
- correct data directly in the accounting system,
- provide an independent assurance opinion,
- become a general-purpose report builder.

---

# 6. Current Product Baseline

## 6.1 Distribution and execution

### Current behavior

PPAR is distributed as a Python package with an installed `ppar` command. The documented setup path creates local starter workspaces, including an `audit` directory and heavily commented YAML configuration. The current installed setup includes Axys/APX-oriented starter CSV files and mappings.

Users run the audit through a local command such as:

```bash
ppar audit <site_directory>/audit
```

### Product meaning

- **CURRENT — DOCUMENTED:** Local execution is part of the present product.
- **CURRENT — REQUIRES CLIENT VALIDATION:** Installation, operation, updating,
  and support have not yet been validated in a real client environment.
- Local-first execution within a client-controlled environment is a permanent
  product principle, not a temporary go-to-market choice.

“Local-first” describes the data-control boundary, not necessarily one physical workstation. The application may run on a client desktop, client server, or client-controlled private infrastructure, but portfolio-accounting data and calculations must not be processed by a PPAR-operated hosted service.

Local execution is commercially meaningful because client data remains under client control. It also transfers installation, dependency, access, security, update, licensing, and support considerations into the implementation model. Any future licensing, update, or support communication must be data-minimizing, disclosed, and separated from client portfolio data.

## 6.2 Input model

### Documented inputs

Performance Auditing uses two snapshots, typically an older/original snapshot and a newer/restated snapshot.

The current starter structure includes:

- portfolio performance,
- security performance,
- holdings,
- transactions,
- FX rates,
- split factors.

`splits` is a first-class optional normalized comparison dataset in the current
implementation. A split-factor row is preserved as evidence and may support the
explanation of a related holdings quantity and market-value correction. The
split-factor row itself is evidence-only: it does not own an additive Modified
Dietz explained amount and it is not one of the currently configured Data Audit
rules.

Portfolio performance is the minimum top-level comparison dataset. Other datasets may be optional unless the configuration marks them as required or a requested reconstruction/explanation path needs them. Missing optional evidence should reduce explanation depth and remain visible to the reviewer; it should not be silently treated as proof that no cause exists.

### Configuration

YAML is the principal configuration and onboarding surface. It defines:

- file locations,
- local column mappings,
- normalized field mappings,
- transaction-code treatment,
- report assumptions,
- which changed fields can explain performance,
- which changed fields are context evidence,
- which findings are suppressed.

### Product meaning

- Site configuration is not mere technical plumbing. It is part of the audit policy.
- Silent guessing is unacceptable when a missing classification can change the explanation.
- A site implementation is not complete until relevant treatment decisions are documented and approved.

## 6.3 Comparison and explanation engine

### Current documented behavior

The performance-comparison path:

1. loads Snapshot A and Snapshot B;
2. normalizes configured datasets;
3. identifies source-data differences;
4. assembles Modified Dietz evidence and explanation tables;
5. creates portfolio-level and/or security-level review packages.

Snapshot labels are intentionally neutral. Snapshot B minus Snapshot A defines the ordinary comparison direction, but neither label proves which snapshot is correct or authoritative.

Holdings, transactions, FX rates, and splits can be treated as shared source findings when both report levels are produced. Portfolio and security performance remain separate calculations.

The engine does not attempt to reconstruct a complete accounting ledger. It counts configured Modified Dietz formula inputs, presents supporting source-data evidence, and leaves unsupported or ambiguous rows as review context unless explicit policy permits otherwise.

Cash has one canonical normalized representation through holdings, such as a cash security. A separate cash dataset must not compete with holdings for the same valuation effect. This is a deliberate double-counting control, not merely an implementation preference.

### Current validation contract

The documented production safeguards include:

- source mapping and accounting-role validation,
- traceable disposition of source differences,
- conservation of counted Modified Dietz causes,
- reconciliation to reviewer-facing totals,
- displayed-value reconciliation after rounding,
- controlled failure when required treatment is missing.

A period may be labeled **Fully Explained** only when its counted causes reconcile to the explained difference and its displayed causes reconcile after report rounding.

## 6.4 Current safety-invariant contract

**Capability status: CURRENT — DOCUMENTED; CURRENT — REQUIRES CLIENT VALIDATION for real client source contracts and operating conditions**

The maintainer-facing safety contract defines twelve enforced invariants:

| ID | Current guarantee | Product meaning |
|---|---|---|
| `SN-01` | No lost differences | Every reportable source difference remains in the complete audit trail unless processing stops before safe interpretation |
| `SN-02` | No double counting | Multiple rows may describe one economic effect, but exactly one designated representation may own the explained amount |
| `SN-03` | Fully Explained arithmetic | Internal, displayed, and serialized workbook arithmetic must reconcile |
| `SN-04` | Beginning and ending continuity | Portfolio- and security-level continuity anomalies remain visible review findings |
| `SN-05` | Bidirectional source lineage | Source-backed causes retain stable links to the findings from which they were built, and persisted lineage is validated |
| `SN-06` | Currency and unit consistency | Unsafe local/base-currency or unit treatment stops the affected workflow rather than being silently coerced |
| `SN-07` | Period-boundary safety | Reversed, overlapping, ambiguous, or out-of-boundary inputs cannot silently own explained performance |
| `SN-08` | Demo scenario preservation | Packaged scenario meaning is protected against fixture drift |
| `SN-09` | Demo fixture isolation | Independent changes and carry-forward stories remain inspectable and controlled |
| `SN-10` | Report-format parity | CSV, HTML, XLSX, and internal review semantics are checked for parity |
| `SN-11` | Deterministic output | Identical inputs and configuration produce equivalent normalized artifacts apart from explicitly volatile metadata |
| `SN-12` | Fail-closed policy coverage | Unknown changed fields and incomplete impact treatment stop processing; suppression cannot substitute for classification |

The safety contract distinguishes four failure classes:

- **Internal logic error:** PPAR produced inconsistent arithmetic, lineage, evidence ownership, or artifact semantics; report generation must stop.
- **Source contract error:** supplied data or configuration cannot be interpreted safely; the affected workflow must stop with an actionable error.
- **Visible review finding:** the source may be valid but a suspicious relationship requires judgment; generate the report and display the issue without counting it automatically.
- **Demo maintenance error:** checked-in fixture intent or coverage drifted; fail maintenance/tests without converting that problem into a client finding.

These guarantees protect how PPAR processes a supported source contract. They do not prove that a client supplied every relevant file, approved the correct mapping, selected the correct accounting treatment, or produced complete source exports. Those are central first-client validation questions.

Every reportable source difference has one permitted analytical disposition:

1. `counted_cause`, or
2. `review_evidence`.

Suppression is metadata, not a third disposition, and must not erase the last representation of a reportable difference from the complete audit trail.

## 6.5 Representative performance-difference output

The current generated portfolio workbook contains three review sheets:

1. `Performance Differences`
2. `Performance Difference Causes`
3. `Data Audit Issues`

The first sheet demonstrates:

- portfolio and date scope,
- performance difference,
- explained difference,
- unexplained difference,
- status,
- reviewer comments.

It includes examples of:

- **Fully Explained**
- **Partly Explained**
- **Unexplained**

The partly explained example explicitly indicates a possible transaction cause that is not counted under the current YAML configuration. The unexplained example directs the reviewer toward possible missing data, timing differences, or vendor methodology mismatch.

This is important product behavior: configuration and evidence determine what may be counted; the product does not silently force every difference into a cause.

## 6.6 Representative cause output

The cause sheet demonstrates:

- dataset and field lineage,
- security,
- Snapshot A value,
- Snapshot B value,
- source difference,
- quantified performance difference explained,
- narrative explanation.

It also demonstrates an important distinction:

- some rows carry a quantified performance effect;
- other rows explain a related source-data relationship without being counted as an independent performance cause.

Examples include:

- changed beginning market value,
- weighted external flow,
- changed ending holdings market value,
- inherited beginning-value effects from a preceding period,
- changed transaction amount, price, quantity, or commission,
- cash-balance consequences.

This distinction must be preserved in future user interfaces and rules.

## 6.7 Data-quality findings

### Current documented and demonstrated issue types

The current packaged YAML configures seven optional Data Audit rules:

- `holdings_price_range`,
- `transactions_price_range`,
- `duplicate_transactions`,
- `dividend_rate`,
- `missing_dividend`,
- `pa_sa_rate`, and
- `holdings_accrued_rate`.

Beginning/end market-value continuity is a separate mandatory visible Data
Audit control at portfolio and security grain. It remains enabled even when the
optional checks are disabled. Split factors currently appear as comparison and
supporting evidence; split-factor-versus-quantity plausibility is a future rule
candidate, not a current Data Audit rule.

Beginning/ending market-value continuity is a mandatory visible integrity check at portfolio and security grain even when optional Data Audit checks are disabled. Optional checks run across the union of Snapshot A and Snapshot B so an issue can be shown as existing in Snapshot A, Snapshot B, or both.

The representative data-quality sheet includes:

- snapshot,
- portfolio,
- as-of date,
- dataset/field,
- security,
- issue type,
- reference value,
- observed value,
- difference,
- tolerance,
- explanation.

### Product meaning

A data-quality finding indicates a suspicious relationship worthy of review. It does not automatically prove:

- that a source value is wrong,
- that the finding caused a performance difference,
- that the same rule is appropriate at every client,
- that the issue should be corrected without human review.

The current Data Audit worksheet intentionally emphasizes observed values, differences, tolerances, and explanations rather than assigning a pseudo-precise severity score. Severity, prioritization, and recommended action are future rules-library design decisions and must be validated with real reviewers.

## 6.8 Review packages and outputs

The current documented audit outputs include:

- portfolio audit in Excel,
- portfolio audit in HTML,
- security audit in Excel,
- security audit in HTML,
- `source_detail.csv`,
- a complete persisted findings trail,
- cause-lineage evidence,
- `manifest.json` and a compact `review_summary.json`,
- supporting evidence in `audit_support.zip`,
- CSV-only primary tables when Excel and HTML are disabled.

Current generated portfolio and security workbooks each contain exactly three
normal review sheets:

1. `Performance Differences`
2. `Performance Difference Causes`
3. `Data Audit Issues`

`source_detail.csv` is a separate promoted reviewer artifact, not a workbook
sheet. The current executable report-bundle contract uses manifest version 4,
review-summary version 1, and 21 required supporting artifacts. In normal XLSX
mode, the workbook is the declared primary review entrypoint; the manifest also
declares period-triage, cause-summary, context-summary, transaction-diagnostic,
complete-audit-trail, and review-handoff entrypoints.

The report layers reuse the same validated summaries, cause candidates, formula rows, and evidence tables. This is an important trust property: presentation formats should not be separate calculations.

The evidence pack also records artifact entrypoints, source context, transaction-semantics summaries, row counts, and normalized fingerprints. These are primarily audit, validation, handoff, and integration controls; they should not overwhelm the ordinary reviewer’s first-stop workbook or HTML flow.

## 6.9 Transaction identity and semantic boundary

**Capability status: CURRENT — DOCUMENTED; CURRENT — REQUIRES CLIENT VALIDATION**

Transaction comparison is intentionally conservative:

- A stable source `transaction_id`, when available in both snapshots, is the strongest edit-pairing evidence.
- Without a stable ID, PPAR may pair rows only through an exact one-to-one singleton fallback using the same portfolio, trade date, security identifier, and native transaction code.
- Matching remains case-sensitive and does not use fuzzy dates, nearest values, amount, quantity, or price similarity.
- Ambiguous groups remain unpaired and visible rather than being forced into an edit relationship.
- A date move can appear as one dropped row and one added row; PPAR does not claim they are the same transaction without sufficient identity evidence.

Transaction-code meaning is also context dependent. Common packaged examples and test fixtures do not prove universal Axys/APX semantics. Ambiguous flow, fixed-income, capital-return, short-side, and corporate-action families require explicit source context, reviewed local mappings, or a deliberately review-only disposition. Code-only inference must remain blocked where the evidence contract says it is unsafe.

The current machine-readable matrix is the coverage authority. The packaged
demo contains ordinary `by`, `sl`, `dv`, and `in`; context-gated `li`, `lo`,
`dp`, and `wd`; fixed-income `pa`/`sa`; return-of-capital `rc`; principal-
paydown `pd`; and a disclosed synthetic `ss`/`cs` short lifecycle. Several of
those rows remain `partial` rather than universally covered because their
treatment depends on asset/source context and local policy. `ai` remains a
test-only candidate profile, `;` remains test-only review evidence, and
standalone `epus` remains backlog. The compact boundary snapshot and rendered
matrix contain older coverage language and do not override the YAML matrix.

## 6.10 Scale and safety behavior

The current documentation states that Audit stops before producing an unusably large report when a primary review table would exceed 100,000 rows. The error identifies the oversized table and major contributors so the user can reduce scope or correct upstream differences.

This is a good example of fail-safe product behavior. It should remain explicit rather than silently truncating findings.

## 6.11 Testing and validation status

### Current evidence

According to founder-provided context, the product has undergone:

- extensive automated testing,
- financial-invariant testing,
- report-reconciliation testing,
- large-dataset performance testing.

The architecture documentation also describes structural, metamorphic, artifact-parity, demo-matrix, and scale-regression tests. The additional implementation documents describe enforced conservation, lineage, currency/unit, period-boundary, report-parity, deterministic-output, and fail-closed policy controls.

### Current working-tree verification

The 2026-07-16 intake began at project HEAD `e035174` and the current working
tree. At the v0.6 checkpoint, the complete project suite ran 824 tests and all
passed. This supersedes the earlier intake note about a documentation-style
failure in the then-supplied handoff/versioned snapshot. Current generated
bundles and the executable bundle contract independently confirm manifest
version 4 and the expected report surfaces. These results remain internal
engineering evidence, not real-client validation.

### Commercial reality

The product has **not yet been validated against a real client’s Axys/APX exports**.

Therefore:

- compatibility with real client exports is not yet established;
- onboarding effort is not yet known;
- transaction-code variability is not yet characterized;
- false-positive and false-negative behavior on client data is not yet measured;
- investigation time savings are not yet measured;
- client trust is not yet demonstrated;
- production support requirements are not yet known.

This is the most important current product risk and must remain visible in roadmap and marketing decisions.

## 6.12 Current-baseline summary

| Capability | Status | Current limitation or qualification |
|---|---|---|
| Local Python audit execution | CURRENT — DOCUMENTED; CURRENT — REQUIRES CLIENT VALIDATION | Local-first is permanent doctrine; deployment and support have not yet been validated in a client-controlled production environment |
| Axys/APX starter datasets and mappings | CURRENT — DOCUMENTED; CURRENT — REQUIRES CLIENT VALIDATION | Accepted packaged-demo seed is not proof of compatibility with every Axys/APX site |
| Two-snapshot comparison | CURRENT — DEMONSTRATED; CURRENT — REQUIRES CLIENT VALIDATION | Real client snapshot variability is untested |
| Portfolio-level changed-performance review | CURRENT — DEMONSTRATED; CURRENT — REQUIRES CLIENT VALIDATION | Current generated workbook is packaged-demo evidence, not client validation |
| Security-level changed-performance review | CURRENT — DEMONSTRATED; CURRENT — REQUIRES CLIENT VALIDATION | Current generated security workbook is packaged-demo evidence, not client validation |
| Modified Dietz reconstruction/evidence and source-row explanation | CURRENT — DEMONSTRATED; CURRENT — DOCUMENTED; CURRENT — REQUIRES CLIENT VALIDATION | Coverage depends on data availability, approved configuration, and supported formula roles |
| Fully/Partly/Unexplained classification | CURRENT — DEMONSTRATED; CURRENT — REQUIRES CLIENT VALIDATION | Materiality, reviewer interpretation, and operational use need client validation |
| Complete finding trail and cause lineage | CURRENT — DOCUMENTED | Must be tested against real client source shapes and handoff practices |
| No-lost-difference, no-double-counting, and reconciliation invariants | CURRENT — DOCUMENTED | Protect supported processing; do not establish correctness of client mappings or exports |
| Currency/unit and period-boundary fail-closed controls | CURRENT — DOCUMENTED | Multi-currency and site-specific extract patterns remain commercially unvalidated |
| Deterministic output and cross-format parity | CURRENT — DOCUMENTED | Client environments and office-software variations still require implementation testing |
| Conservative transaction matching | CURRENT — DOCUMENTED | Real-site transaction identity availability and ambiguity rates are unknown |
| Technical audit-readiness/preflight controls | CURRENT — DOCUMENTED | A unified reviewer-facing readiness product experience is not yet established |
| Data-quality findings | CURRENT — DEMONSTRATED; CURRENT — REQUIRES CLIENT VALIDATION | Seven optional rules plus mandatory continuity are current; utility, tolerance calibration, and false-positive rates need client validation |
| Split-factor comparison evidence | CURRENT — DEMONSTRATED; CURRENT — REQUIRES CLIENT VALIDATION | Optional evidence-only normalized dataset; not a current Data Audit rule or counted split cause |
| Excel, HTML, CSV, manifest, and evidence-pack output | CURRENT — DEMONSTRATED; CURRENT — DOCUMENTED; CURRENT — REQUIRES CLIENT VALIDATION | Manifest v4; management usability and operational handoff require client validation |
| Large-report safety stop | CURRENT — DOCUMENTED | Practical scoping behavior requires client validation |
| Persistent audit history | APPROVED DIRECTION | No current history repository or cross-run status model |
| Audit health dashboard | APPROVED DIRECTION | Requires comparable local audit history and validated metrics |
| Reviewer workflow and disposition | CANDIDATE | Human review is current doctrine; product-managed workflow is not current |
| Unified product-facing rules library | CURRENT — DOCUMENTED; APPROVED DIRECTION | Stable codes, field roles, safety invariants, and semantics contracts are current foundations; a coherent product-facing library is not current |
| Broad non-Axys/APX compatibility | CANDIDATE | Must not be claimed without a documented, tested, commercially supportable platform contract |

---

# 7. Long-Term Capability Model

The earlier product-expansion specification is incorporated here as a
capability model. Detailed functional specifications are added progressively in
Phase 3, beginning with Performance Change Investigation in Section 18.

## 7.1 Center: Performance Change Investigation

**Capability status: CURRENT — DEMONSTRATED; CURRENT — DOCUMENTED; APPROVED DIRECTION**

This is the product’s center of gravity.

Current capability:

- identify changed performance,
- quantify supported causes,
- distinguish unresolved differences,
- preserve evidence,
- identify review items.

Approved strengthening:

- explanation completeness,
- executive root-cause summary,
- repeated-restatement timeline,
- portfolio stability analysis,
- cross-period investigation history,
- cause-category trend analysis.

### Doctrine constraint

Every enhancement must improve the speed, completeness, reviewability, or defensibility of changed-performance investigation.

### Terminology refinement

Use **explanation completeness** or **explanation coverage** for the mathematically explained portion of a difference.

Do not use **confidence score** unless a defined and validated confidence methodology exists. A period may be 99.9% explained mathematically while still relying on incorrect source mapping. Completeness and confidence are not the same concept.

## 7.2 Adjacent core: Performance Data Quality Audit

**Capability status: CURRENT — DEMONSTRATED; CURRENT — DOCUMENTED; APPROVED DIRECTION**

The data-quality component should expand through transparent audit rules tied to performance-relevant source-data.

Current demonstrated/documented areas include:

- holdings-price and transaction-price ranges,
- duplicate transactions,
- dividend-rate and missing-dividend relationships,
- purchase/sale rate consistency,
- holdings accrued-rate consistency,
- mandatory beginning/end holdings continuity.

Split factors are a current optional comparison/evidence dataset, not a current
Data Audit rule. Corporate-action quality checks are an APPROVED DIRECTION;
any specific split-plausibility rule remains a CANDIDATE until specified and
validated.

Approved rule-family directions include:

### Pricing

- zero or negative prices,
- missing prices,
- stale prices,
- extreme price changes,
- price-source changes,
- inconsistent same-day prices,
- price changes without a corresponding split or corporate action.

### Transactions

- duplicate transactions,
- invalid or unexpected codes,
- inconsistent amount, price, and quantity relationships,
- settlement/trade-date anomalies when the required fields are available,
- back-dated corrections,
- income after position disposal,
- currency mismatches,
- unusually large corrections.

### Holdings

- zero quantity with nonzero market value,
- nonzero quantity with missing price,
- quantity changes unsupported by available transactions,
- holdings that appear or disappear unexpectedly,
- inconsistent cash and market-value relationships,
- negative positions where policy indicates they are unexpected.

### Income and accrued interest

- missing expected dividends where a defensible reference exists,
- duplicate income,
- inconsistent dividend rates,
- accrued-interest rate differences,
- coupon inconsistencies when reference terms are available,
- withholding or tax anomalies when data supports them.

### Corporate actions

- split factor mismatches,
- reverse-split inconsistencies,
- missing or inconsistent spin-off treatment,
- merger or conversion-ratio anomalies,
- identifier changes that break continuity.

### Foreign exchange

- missing FX rates,
- inconsistent same-day rates,
- extreme FX changes,
- base-currency mismatches,
- changed prior-period FX rates.

### Doctrine constraint

A rule must identify:

- the factual relationship observed,
- the tolerance or policy applied,
- the evidence required,
- whether it is a cause, context, independent quality finding, or configuration issue,
- the likely false-positive conditions.

A rule catalog must not imply that every flagged condition is an error.

A focused standalone Data Auditing command or report may eventually be useful before, after, or apart from a specific performance restatement. It should reuse the same normalized data and rule implementation and must not duplicate or fork the Performance Change Investigation engine.

## 7.3 Precondition layer: Audit Readiness

**Capability status: CURRENT — DOCUMENTED; APPROVED DIRECTION**

The current product already has meaningful readiness controls. Configuration validation and report-generation guardrails can identify or block conditions such as:

- required datasets or files missing,
- required normalized columns missing,
- incomplete YAML treatment for changed fields,
- unknown accounting roles,
- unsafe ambiguous transaction semantics,
- missing transaction context required by an extract contract,
- invalid currency or base-value treatment,
- reversed or overlapping periods,
- unsupported reconstruction inputs,
- unsafe duplicate keys or ambiguous matching groups.

The approved product direction is to consolidate those controls into a reviewer-facing **Audit Readiness** experience that determines whether the supplied inputs, configuration, and requested scope are sufficient for a meaningful audit.

Additional candidate checks include:

- malformed identifiers or dates,
- missing prices or FX required for the requested scope,
- missing split data,
- incomplete period coverage,
- unsupported report scope,
- configuration assumptions awaiting client approval,
- operator guidance identifying the exact corrective action.

### Doctrine constraint

Audit Readiness should prevent false confidence. It must distinguish:

- a hard blocker,
- a warning that qualifies the result,
- a nonmaterial information item.

It should not merely rename technical errors or produce a longer list of exceptions. The product value is a clear readiness decision, source-contract explanation, and actionable next step.

## 7.4 Signature experience: Executive Investigation Summary

**Capability status: CURRENT — DOCUMENTED; APPROVED DIRECTION**

Current bundles include compact handoff metadata and a derived needs-review summary, but these are not yet the intended management experience. Every audit should eventually begin with a concise, human-readable executive investigation summary.

Candidate content:

- portfolio or security scope,
- reporting period,
- original and restated reported performance where appropriate,
- performance difference,
- explained difference,
- unexplained residual,
- explanation completeness,
- status,
- principal cause categories,
- largest quantified causes,
- highest-priority data-quality findings,
- recommended review items,
- references to detailed evidence.

### Doctrine constraint

The executive summary must not replace the evidence package. It is a navigation and communication layer over validated detail.

## 7.5 Management layer: Audit Health Dashboard

**Capability status: APPROVED DIRECTION**

The dashboard should answer:

- How stable is reported performance?
- How often are periods restated?
- How much is explained automatically?
- What remains unresolved?
- Which portfolios experience the most corrections?
- Which cause categories recur?
- Which data-quality rules fire most often?
- Is performance quality improving?

Candidate metrics include:

- restatement count and magnitude,
- explanation completeness distribution,
- unresolved residual count and age,
- portfolio stability ranking,
- largest corrections,
- most common cause categories,
- most common data-quality findings,
- period-over-period trends.

### Doctrine constraint

The dashboard requires consistent audit history. It should not be built as a superficial visualization over one workbook.

## 7.6 Learning layer: Operational Intelligence

**Capability status: APPROVED DIRECTION**

Operational Intelligence uses accumulated, comparable audit history to identify recurring patterns such as:

- frequently corrected portfolios,
- frequently corrected securities,
- recurring transaction categories,
- repeated custodian or source-file issues where source identity is available,
- recurring configuration gaps,
- common unresolved causes,
- quality trends by reporting cycle.

### Doctrine constraint

Operational Intelligence must be based on stable definitions and sufficiently comparable history. It should not infer organizational blame from incomplete data.

## 7.7 Strategic asset: PPAR Audit Rules Library

**Capability status: CURRENT — DOCUMENTED; APPROVED DIRECTION**

The current product already contains important rule-system foundations: stable finding codes, explicit field roles, YAML policy requirements, data-audit check codes, safety invariants, extract-contract safeguards, and a transaction-semantics boundary. The rules library should turn those foundations into a coherent, product-facing, versioned expression of portfolio-operations knowledge.

Each rule should eventually define:

- stable rule ID,
- name,
- category,
- business rationale,
- applicable data and asset types,
- required inputs,
- detection logic,
- tolerance and materiality,
- severity,
- finding classification,
- evidence produced,
- potential relationship to performance,
- recommended reviewer action,
- configurable parameters,
- allowed suppression behavior,
- known false-positive conditions,
- validation examples,
- test requirements,
- lifecycle status.

### Doctrine constraint

The product should expand through coherent rule families and investigation capabilities, not a proliferation of bespoke reports for individual clients.

---

# 8. Product Principles

## Principle 1 — Changed-performance explanation remains the center

The product may expand into readiness, data quality, dashboards, and operational intelligence, but those capabilities must strengthen confidence in reported performance.

A feature that has no meaningful relationship to performance quality should normally be excluded.

## Principle 2 — Evidence before assertion

Every material conclusion should be traceable to:

- the input snapshot,
- the normalized field,
- the observed source difference,
- the applicable policy or rule,
- the calculated effect where applicable.

Narrative explanations are useful only when they rest on visible evidence.

## Principle 3 — Quantitative conservation is non-negotiable

When a source-data difference is counted as a performance cause, counted amounts must reconcile to the explained difference under the configured methodology.

No reportable difference may be silently lost. Multiple physical rows may describe one economic effect, but only one designated owner may carry the explained amount. Displayed values must also reconcile after rounding. Cosmetic reporting must never weaken mathematical controls.

## Principle 4 — Do not guess silently

When required configuration is missing or a relationship is ambiguous, the product should:

- block the run,
- qualify the result, or
- surface a review item.

It should not choose a convenient accounting treatment without explicit policy.

## Principle 5 — Separate cause, context, anomaly, and configuration

These concepts must never be merged:

- **Counted cause:** quantitatively contributes to the explained performance difference.
- **Context evidence:** helps understand a relationship but is not independently counted.
- **Data-quality finding:** suspicious condition that deserves review and may be independent of the performance change.
- **Configuration issue:** missing or ambiguous policy that affects analysis.
- **Unexplained residual:** performance difference not defensibly accounted for.

This separation is foundational to product trust. Every reportable difference must remain represented as either a counted cause or review evidence; suppression may change review priority but may not erase the complete audit trail.

## Principle 6 — Human review is a designed outcome

Human review is not a temporary defect to be removed at any cost.

The product should automate deterministic work, organize evidence, and narrow uncertainty. Qualified people remain responsible for:

- approving site policy,
- interpreting ambiguous conditions,
- deciding whether a source value is wrong,
- correcting the accounting system,
- closing material findings.

## Principle 7 — Deterministic and reproducible by default

The same inputs, configuration, and product version should produce the same material result. CSV, HTML, XLSX, and internal review semantics should remain aligned because they are presentations of shared validated tables, not independent calculations.

The current implementation treats cross-format parity and repeat-run determinism as enforced safety guarantees. If probabilistic or AI-assisted functionality is ever introduced, it must be clearly separated from deterministic calculations and must not silently alter counted causes.

## Principle 8 — Configuration is controlled business policy

YAML mappings and transaction treatment affect the audit conclusion.

Configuration should therefore be:

- versioned,
- reviewable,
- attributable,
- documented,
- approved during implementation,
- included or referenced in the evidence package.

## Principle 9 — Materiality and tolerance must be explicit

Rules need transparent thresholds. The reviewer should be able to understand:

- why a difference was included,
- why a difference was suppressed,
- whether a threshold is absolute, relative, or both,
- who approved site-specific changes.

A large number of immaterial findings can destroy product usefulness.

## Principle 10 — Local-first and data-minimizing

PPAR Audit executes within a client-controlled environment and treats source files as configured site inputs rather than a PPAR-operated hidden database.

This is a permanent product principle. PPAR must not require client portfolio data to be uploaded to or processed by a PPAR-operated hosted service. Client-controlled desktops, servers, private infrastructure, and internal schedulers are compatible with the principle.

The product should minimize data movement and retain only what is necessary for the audit and evidence package. Licensing, update, and support mechanisms must be architected so they do not silently transmit portfolio data or audit evidence outside the client-controlled boundary.

## Principle 11 — Complement the system of record

PPAR Audit should analyze and explain outputs from portfolio accounting systems. It should not become the accounting book of record or write corrections back automatically.

The client’s authorized accounting process remains responsible for source corrections and official reported performance.

## Principle 12 — Reviewability over report volume

The product should optimize for:

- clear prioritization,
- navigable evidence,
- material findings,
- understandable statuses,
- concise management communication.

More rows and more tabs do not necessarily create more value.

## Principle 13 — Fail safely and visibly

Examples of correct failure behavior include:

- blocking on missing required treatment,
- stopping before unusably large report artifacts,
- identifying oversized contributors,
- preserving unexplained residuals,
- refusing unsupported scope.

Silent truncation, silent coercion, and hidden assumptions are unacceptable. Failure classes must remain distinct: internal logic errors and unsafe source contracts stop processing, while suspicious but interpretable relationships remain visible review findings.

## Principle 14 — Repeatability before breadth

The first objective is a repeatable Axys/APX-oriented implementation, not superficial support for many platforms.

Configurable normalization creates future potential. It does not establish current compatibility.

## Principle 15 — Claims follow evidence

Marketing language must not outrun:

- demonstrated product behavior,
- client validation,
- measured user outcomes,
- known coverage,
- contractual boundaries.

The strongest long-term brand will be built by being unusually precise about limitations.

## Principle 16 — Expand through reusable rules and product capabilities

Client learning should improve:

- starter mappings,
- validation,
- rules,
- documentation,
- onboarding,
- review experience.

It should not produce an accumulation of one-off scripts that cannot be reused.

## Principle 17 — Audit and Analytics have separate product identities

PPAR Audit should not be marketed as half of an “Audit and Analytics” bundle.

Performance Analytics may remain in the shared technical package, but its buyer, use case, message, and product roadmap should be treated separately.

## Principle 18 — Safety guarantees are change-controlled contracts

No product enhancement may silently weaken the meaning of no-lost-difference, no-double-counting, Fully Explained arithmetic, lineage, currency/unit safety, period-boundary safety, fail-closed policy, report parity, or deterministic output.

A change to one of those guarantees requires an explicit product and engineering decision, updated documentation, and corresponding tests.

## Principle 19 — Information quality before workflow ownership

PPAR Audit should provide as much decision-useful and accurate information as
the available evidence and supported methodology permit. How that information
is structured, prioritized, explained, and presented is part of the product,
not cosmetic packaging.

The process used to review, route, approve, correct, retain, or close that
information will differ across clients and users. PPAR should make its output
easy to use within those workflows without assuming that it should own them.

Managed workflow is a distant possibility, not an expected product layer. It
should be considered only after repeated client evidence shows a common,
material problem that cannot be handled adequately through clear evidence
packs, reproducible reruns, or the client's existing systems.

---

# 9. Trust, Review, and Accountability Model

## 9.1 Trust layers

PPAR Audit must earn trust at several layers.

### Input trust

- Were the intended files supplied?
- Are dates, identifiers, and values valid?
- Is the requested scope complete?
- Are Snapshot A and Snapshot B comparable?

### Configuration trust

- Are local fields mapped correctly?
- Are transaction codes classified correctly?
- Are external flows and other Modified Dietz roles defined?
- Are suppressions and tolerances approved?

### Calculation trust

- Are counted causes allowed under the configured policy?
- Are formula inputs assembled correctly?
- Are amounts conserved?
- Do displayed values reconcile?

### Evidence trust

- Can the reviewer trace a cause to source-data?
- Is the distinction between raw source, normalized value, calculated result, and narrative explanation visible?
- Are support files preserved?

### Review trust

- Are unresolved items obvious?
- Can a reviewer understand why the product did not count an item?
- Can findings be independently checked?
- Are materiality and severity clear?

### Operational trust

- Can the process be rerun?
- Is the version and configuration known?
- Is the implementation supportable?
- Do successive runs remain comparable?

## 9.2 Current safety-guarantee model

PPAR’s current safety guarantees operate inside a defined boundary:

```text
supplied source files + resolved configuration + supported product version
    -> normalized facts
    -> reportable differences
    -> explicit dispositions and lineage
    -> reconciled reviewer artifacts
```

The guarantees protect transformation, ownership, conservation, visibility, and presentation. They do not independently prove:

- that the client exported every relevant source record,
- that a source field means what the mapping says it means,
- that a transaction policy is correct for the client,
- that a vendor’s reported return uses the assumed methodology,
- that a suspicious relationship is an error.

This distinction is crucial to first-client validation. The pilot must test the client source contract and policy assumptions; it should not merely rerun the synthetic invariant suite.

### Failure handling

| Failure class | Required product behavior |
|---|---|
| Internal logic error | Stop generation; do not downgrade inconsistent evidence or arithmetic to a warning |
| Source contract error | Stop the affected workflow and identify the data/configuration action required |
| Visible review finding | Continue safely, display the condition, and do not count it without an explicit rule |
| Demo maintenance error | Fail internal maintenance/tests without presenting the fixture problem as a client finding |

## 9.3 Evidence hierarchy

The product should preserve the following hierarchy:

1. **Source fact** — value present in Snapshot A or Snapshot B.
2. **Normalized fact** — source value mapped into PPAR’s normalized dataset.
3. **Observed difference** — Snapshot B minus Snapshot A or another transparent relationship.
4. **Configured classification** — accounting role, rule, tolerance, or policy.
5. **Calculated effect** — Modified Dietz or other explicitly supported quantitative result.
6. **Reviewer-facing explanation** — readable narrative based on the preceding layers.
7. **Human disposition** — decision made by an authorized reviewer.

A narrative should never be treated as stronger evidence than the values and policy that produced it.

The complete audit trail must retain the lossless finding-level representation, including suppressed rows, stable logical locators, fingerprints, dispositions, and cause lineage. Reviewer-facing summaries may filter or prioritize that trail, but they may not become the only surviving representation of a reportable difference.

## 9.4 Review statuses

The current status model is valuable and should remain conceptually stable:

### Fully Explained

All material performance difference within scope is accounted for by counted causes under configured policy, and the displayed causes reconcile after rounding.

### Partly Explained

Some material performance difference is accounted for, but a residual remains or a potential cause cannot be counted under current policy.

### Unexplained

The material performance difference is not defensibly accounted for by counted causes within the supplied data and configuration.

Future workflow statuses such as `Open`, `Reviewed`, `Accepted`, `Corrected`, or `Closed` should not replace the analytical status. Analytical status and workflow status answer different questions.

## 9.5 Audit terminology and assurance boundary

The word **audit** is used here to mean a structured, evidence-preserving operational examination of changed performance and related source-data.

PPAR Audit does not currently claim to provide:

- an independent audit opinion,
- financial-statement assurance,
- a GIPS verification,
- regulatory certification,
- a guarantee that reported performance is correct,
- a guarantee that all errors have been detected.

Contracts, documentation, and marketing must state this boundary clearly. Legal review of terminology will be required before broad commercial launch.

## 9.6 Client accountability

The client remains responsible for:

- selecting and exporting the correct source-data,
- approving mappings and accounting treatment,
- determining official performance,
- deciding whether a finding is an error,
- correcting source systems,
- approving final disposition,
- determining reliance for regulatory, compliance, client-reporting, or audit purposes.

PPAR Audit is responsible for accurately performing its documented comparison, calculation, validation, and reporting behavior within configured scope.

---

# 10. Product Boundaries and Non-Goals

## 10.1 Boundary table

| Area | Product boundary | Permitted adjacent behavior |
|---|---|---|
| Portfolio accounting book of record | OUT OF SCOPE | Read exported data and explain differences |
| Official performance engine | OUT OF SCOPE as product identity | Perform diagnostic Modified Dietz calculations and evidence assembly |
| Full accounting-ledger reconstruction | OUT OF SCOPE | Count configured formula inputs and show supporting relationships |
| General cash, position, or trade reconciliation | OUT OF SCOPE | Compare performance-relevant holdings, transactions, FX, and splits between snapshots |
| Source-system correction and writeback | OUT OF SCOPE | Recommend items for client review and correction |
| Independent audit or assurance opinion | OUT OF SCOPE | Produce operational evidence packages |
| GIPS verification or compliance certification | OUT OF SCOPE | Support internal investigation and evidence review without certifying compliance |
| Broad investment analytics | Separate product identity | Audit may display performance context needed for investigation |
| Universal accounting-platform compatibility | CANDIDATE; not established as a current claim | Build from Axys/APX starter support and expand only after validated implementations |
| PPAR-operated hosted processing of client portfolio data | OUT OF SCOPE under permanent local-first doctrine | Support client-controlled desktop, server, private-infrastructure, and scheduling deployments |
| Universal transaction-code interpretation | OUT OF SCOPE | Use explicit source context, reviewed site mappings, and review-only treatment where semantics are unsafe |
| Black-box AI cause attribution | OUT OF SCOPE under current doctrine | AI may later assist navigation or summarization only if clearly separated and controlled |
| General report builder | OUT OF SCOPE | Provide purpose-built review and evidence outputs |
| Enterprise case-management system | DEFERRED | Add narrowly tailored review and disposition workflows only when justified |
| External market/reference-data service | OUT OF SCOPE initially | Use supplied or explicitly integrated reference data when a rule requires it |
| Error correction automation | OUT OF SCOPE | Identify suspicious conditions and preserve evidence |
| Guarantee of completeness | OUT OF SCOPE | State scope, coverage, and unresolved items explicitly |

## 10.2 Internal reconciliation versus reconciliation product

PPAR Audit must perform rigorous internal reconciliation:

- counted causes to explained difference,
- displayed causes after rounding,
- report formats to common validated tables.

This does not make it a general reconciliation platform. The distinction should be explicit:

> **Mathematical and evidence reconciliation are core controls inside PPAR Audit; enterprise cash, position, trade, and custodian reconciliation are not the product category.**

## 10.3 Customization boundary

The Python and YAML architecture makes PPAR highly configurable. That is useful but commercially dangerous.

Customization should be accepted when it improves a reusable product capability, such as:

- a generally useful mapping option,
- a supported transaction-treatment policy,
- a broadly relevant rule,
- a standard report configuration.

Customization should be resisted when it creates:

- client-specific accounting logic with no broader value,
- unique report formats that become permanent maintenance obligations,
- unsupported asset-class claims,
- hidden code forks,
- implementation dependence on the founder.

The first-client pilot must be designed to learn without turning PPAR into consulting software.

---

# 11. Product-Claims Discipline

## 11.1 Compelling and currently defensible claims

Subject to exact version and documentation, the following claims are supportable:

- PPAR Audit runs locally so portfolio data remains within the client’s environment.
- It compares two configured snapshots of portfolio-accounting data.
- It identifies changed portfolio and security performance by period.
- It connects supported source-data changes to Modified Dietz performance differences.
- It distinguishes fully explained, partly explained, and unexplained differences.
- It preserves supporting source detail and evidence.
- It flags suspicious source-data relationships for human review.
- It produces Excel, HTML, and supporting CSV artifacts.
- It includes Axys/APX-oriented starter datasets and mappings.
- It uses explicit YAML mappings and treatment rather than silently guessing.
- It applies conservation and report-reconciliation checks to counted causes.
- Within supported comparison surfaces, it preserves reportable differences as counted causes or review evidence and does not treat suppression as deletion from the complete audit trail.
- It validates lineage, currency/unit treatment, period boundaries, cross-format report semantics, and deterministic repeat-run output under its documented safety contract.
- It has undergone substantial automated and financial-invariant testing, provided this claim can be substantiated with test records.

## 11.2 Claims that require qualification

### “Automatically explains performance changes”

Preferred:

> “Automatically explains supported causes of changed performance and identifies the residual that still requires review.”

Avoid implying universal explanation.

### “Works with Axys/APX”

Before client validation, preferred:

> “Includes an accepted Axys/APX-oriented packaged-demo and onboarding seed and is seeking validation partners with real Axys/APX exports.”

“Release-candidate quality for the packaged demo scope” is an internal engineering statement, not evidence of production compatibility across Axys/APX firms. After successful client validation, the claim can be strengthened only within the tested export patterns, transaction semantics, and site policies.

### “Highly customizable”

This is technically attractive but commercially ambiguous.

Preferred:

> “Uses configurable local mappings and transaction treatment.”

“Highly customizable” may cause prospects to expect bespoke engineering.

### “Audit”

Use with an explanation that it is an operational performance audit, not an independent assurance engagement.

## 11.3 Claims that should wait for pilot evidence

Do not make these claims until measured:

- reduces investigation time by a stated percentage,
- saves a stated number of staff hours,
- materially reduces reporting risk,
- improves client retention or satisfaction,
- detects errors before reporting,
- works across heterogeneous Axys/APX installations,
- is production-proven on real client data,
- scales to a stated portfolio count or file size in client conditions,
- requires minimal implementation,
- can be operated without specialist support,
- produces case-study-quality results consistently.

## 11.4 Claims to avoid

Avoid:

- “explains every performance change,”
- “guarantees accurate performance,”
- “detects all data errors,”
- “replaces reconciliation,”
- “replaces portfolio accounting,”
- “works with any accounting platform,”
- “production-ready for Axys/APX firms,” before real-client validation,
- “supports all Axys/APX transaction types,”
- “audit-ready,” unless defined and legally reviewed,
- “compliance certified,”
- “GIPS compliant” or “GIPS verification,”
- “AI-powered auditing,” under the current deterministic product identity,
- “enterprise proven,” before appropriate evidence.

## 11.5 Preferred language

Prefer:

- supported,
- configured,
- traceable,
- quantified,
- reconciled,
- reviewable,
- evidence-based,
- deterministic,
- locally executed,
- requires review,
- within scope.

Use sparingly or avoid:

- exact,
- complete,
- guaranteed,
- intelligent,
- automatic,
- universal,
- seamless,
- no-touch,
- error-proof.

---

# 12. Terminology and Definitions

## Accounting snapshot

A configured collection of portfolio-accounting exports representing the data available at a point in time or reporting state.

## Snapshot A

One neutrally labeled source-data collection in the comparison. It is often the earlier or previously reported snapshot, but that convention is not required.

## Snapshot B

The second neutrally labeled source-data collection. Numeric comparison deltas ordinarily use Snapshot B minus Snapshot A.

The labels do not establish which snapshot is newer, corrected, complete, or authoritative.

## Audit run

One execution of PPAR Audit against a defined configuration and pair of snapshots.

## Audit scope

The portfolios, securities, periods, report levels, datasets, and rules included in an audit run.

## Audit period

A from-date and through-date interval for which reported performance is compared.

## Reported performance difference

The difference between performance reported in Snapshot B and Snapshot A for the same configured entity and period.

## Source-data difference

A changed, added, or removed value identified between the normalized datasets in the two snapshots.

## Counted cause

A source-data difference or derived relationship that is explicitly permitted under configured policy to contribute quantitatively to the explained performance difference.

## Context evidence

A source-data change or relationship relevant to understanding the investigation but not counted independently as a performance cause.

## Explained difference

The portion of the reported performance difference accounted for by counted causes.

## Unexplained difference

The residual performance difference not accounted for by counted causes within the supplied data, configured policy, and supported methodology.

## Explanation completeness

A measure of how much of the material reported performance difference is accounted for by counted causes.

The exact formula and behavior near zero must be defined during detailed specification. It is not synonymous with confidence.

## Confidence

A claim about certainty or reliability. PPAR Audit does not currently have a defined probabilistic confidence model. The term should not be used as a product metric without formal specification and validation.

## Fully Explained

An analytical status indicating that counted causes reconcile to the explained difference and displayed causes reconcile after rounding, subject to scope and materiality.

## Partly Explained

An analytical status indicating that some but not all material difference is accounted for, or that a potential cause remains uncounted under current configuration.

## Unexplained

An analytical status indicating that the material difference is not defensibly accounted for by counted causes within current scope.

## Data-quality finding

A suspicious source-data relationship that deserves review. A finding is not necessarily a confirmed error or a performance cause.

## Review item

Any unresolved cause, context row, data-quality finding, or configuration issue requiring human attention.

## Rule

A transparent, versioned definition that evaluates data or a relationship and produces a classified finding or controlled calculation.

## Rule severity

The urgency or importance assigned to a rule finding. Severity is not the same as quantified performance impact.

## Tolerance

An absolute, relative, or combined threshold used to determine whether an observed relationship produces a finding.

## Materiality

The level at which a difference or finding matters for the intended review. Materiality may apply to performance, source values, or operational significance and must be explicit.

## Site configuration

The client-specific mappings, code treatment, assumptions, thresholds, scope, and output settings used by PPAR Audit.

## Lineage

The traceable connection from a reviewer-facing conclusion to the source values, normalized fields, policy, and calculation that produced it.

## Conservation

The requirement that counted cause amounts are neither lost nor duplicated as they are assembled and summarized.

## Reconciliation

The mathematical agreement between counted causes, explained totals, displayed totals, and common report-source tables. This term does not mean general custodian reconciliation.

## Reconstruction diagnostic

A secondary calculation intended to help diagnose reported returns. It is not the ordinary first review surface and does not convert PPAR Audit into the official performance engine.

## Evidence package

The workbook, HTML report, source-detail CSV, supporting files, configuration reference, and other artifacts needed to review or reproduce an audit result.

## Human disposition

An authorized reviewer’s decision concerning a finding, such as accepted, corrected, explained externally, immaterial, or closed.

## Local-first

A permanent product principle requiring PPAR Audit calculations and client portfolio data to remain within a client-controlled environment rather than a PPAR-operated hosted processing service.

## Reportable source difference

A Snapshot A versus Snapshot B change emitted after normalization, record matching, and the applicable comparison tolerance. It may be a row addition/removal, numeric change, or nonnumeric change.

## Permitted disposition

The required analytical treatment of a reportable difference: `counted_cause` or `review_evidence`. Suppression is metadata and is not a third disposition.

## Counted economic effect

One financial change that may be represented by several related source rows. The related rows may remain visible, but exactly one designated representation may own the explained amount.

## Complete audit trail

The lossless finding-level evidence from which summaries are derived, including suppressed and unsuppressed rows, stable logical locators, fingerprints, dispositions, and lineage.

## Source contract

The defined set of source files, normalized fields, meanings, mappings, transaction context, currency basis, timing conventions, and configuration assumptions that PPAR is allowed to interpret for one implementation.

## Internal logic error

A condition in which PPAR’s own arithmetic, lineage, evidence ownership, or artifact semantics are inconsistent. Generation must stop.

## Source contract error

A condition in which supplied data or configuration cannot be interpreted safely. The affected workflow must stop with actionable guidance.

## Visible review finding

A suspicious but interpretable relationship that requires human judgment. The report may proceed, but the finding must not be counted automatically without an explicit rule.

## Transaction match status

A transparent description of how transaction row identity was established or withheld, such as stable-ID match, exact singleton fallback, added row, missing row, or ambiguous fallback.

## Client validation

Testing and review using real client exports, client-approved mappings, and client-recognized investigation cases. Synthetic test coverage is necessary but not a substitute.

## Supported platform

A portfolio-accounting platform and export pattern that has a documented, tested, and commercially supportable implementation. Configurability alone does not establish support.

---

# 13. First-Client Validation Doctrine

## 13.1 Purpose of the first pilots

The first 2–5 clients should primarily validate:

- real-world data compatibility,
- mapping and configuration,
- analytical correctness,
- review usability,
- implementation effort,
- product trust,
- repeatability.

Revenue is desirable but secondary to learning that makes a standardized software business possible.

## 13.2 What a first pilot must not become

A first pilot should not become:

- an open-ended custom development engagement,
- a promise to support every export variation,
- a replacement implementation for an existing platform,
- a broad data-cleaning project,
- a guarantee that all historical performance can be explained,
- a case study written before the client has independently accepted the result.

## 13.3 Validation dimensions

### Data compatibility and source contract

- Can required exports be produced within the client-controlled environment?
- Are the source files actual operational exports rather than hand-curated demo shapes?
- Are identifiers and dates stable and case-preserving?
- Are portfolio and security performance periods comparable?
- Are FX and split files available where needed?
- Can ambiguous transaction families be resolved from source context or reviewed local policy?
- Can the data be normalized without client-specific code?
- Does the documented source contract match what the client believes each field means?

### Configuration and readiness

- Can transaction codes be classified without unsafe code-only inference?
- Can client policy be expressed transparently?
- Are missing treatments, missing context fields, and unknown changed fields detected before report generation?
- Can configuration be reviewed, versioned, and approved by an authorized client owner?
- Can a client operator understand and resolve readiness blockers without founder-only intervention?

### Analytical correctness and safety behavior

- Do known source changes appear in the complete audit trail?
- Are counted causes correct and owned only once?
- Are inherited effects and period boundaries handled correctly?
- Do totals conserve and reconcile internally, at displayed precision, and in written artifacts?
- Are source-backed causes traceable in both directions?
- Are unsafe currency/unit treatments blocked?
- Are unexplained residuals honest?
- Are false explanations and forced transaction matches avoided?

### Data-quality usefulness

- Do findings identify real review items?
- What is the false-positive rate?
- Are tolerances appropriate?
- Can the reviewer understand the business rationale?
- Are findings grouped and prioritized usefully?

### Operational usefulness and local deployment

- Can PPAR be installed, updated, licensed, and run within the client-controlled environment?
- Does PPAR reduce manual comparison work?
- Does the reviewer reach a conclusion faster?
- Is the evidence easier to preserve and hand off?
- Can another reviewer follow the result from the workbook/HTML into the evidence pack?
- Is the report usable without the founder explaining every row?
- Can support be provided without routine transfer of client portfolio data outside the environment?

### Repeatability

- Can the client rerun the process?
- Can a second reporting cycle be handled with less implementation effort?
- Can lessons be converted into reusable starter mappings, rules, and documentation?
- Can the next client be implemented materially faster?

## 13.4 Pilot evidence required before stronger claims

Before claiming a repeatable product, PPAR should possess:

- a documented and client-approved source contract,
- client-approved input mapping,
- client-approved transaction treatment and explicit handling of ambiguous/backlog families,
- successful runs on multiple real snapshot pairs,
- reconciliation of selected cases to client manual analysis,
- documented examples of fully, partly, and unexplained outcomes,
- evidence that material issues were not hidden, lost, or double-counted,
- evidence that report formats and supporting artifacts remained semantically aligned,
- measured review-time comparison,
- documented implementation effort,
- known limitations,
- client statement of continued value,
- permission for any public case-study claims.

## 13.5 Trust over automation

The pilot should not be judged by maximizing the fully explained rate.

A stronger result may be:

> “PPAR explained 92% of the difference, clearly isolated the remaining 8%, and prevented an unsupported transaction from being counted.”

That is more credible than a 100% explanation produced by an unapproved assumption.

---

# 14. Product Maturity Roadmap Summary

This is a directional maturity sequence, not a dated release commitment. Detailed prioritization will be written in Phase 5.

## Stage A — Real-Client Validation

### Objective

Prove that the current audit engine and review package work against real Axys/APX-oriented exports.

### Required evidence

- correct ingestion,
- approved mappings,
- accepted cause calculations,
- useful unresolved-item handling,
- measured implementation effort,
- repeated use by qualified reviewers.

### Exit gate

A validation partner trusts the results, can rerun the audit, and is willing to continue using the product.

## Stage B — Repeatable Axys/APX Product

### Objective

Convert first-client learning into a standardized software implementation.

### Likely capabilities

- stronger starter mappings,
- documented data contracts,
- improved readiness checks,
- a prioritized initial rules library,
- standard onboarding and troubleshooting,
- refined executive investigation summary,
- defined support boundaries.

### Exit gate

A second or third client can be implemented materially faster without redesigning the product.

## Stage C — Proactive Performance Quality Assurance

### Objective

Move from reactive investigation to routine quality review.

### Likely capabilities

- repeatable or scheduled runs,
- audit readiness,
- historical audit repository,
- stability and trend metrics,
- audit health dashboard,
- cause and finding summaries.

### Exit gate

Clients use PPAR before or during reporting cycles, not only after someone notices a changed return.

## Stage D — Organizational Knowledge and Review Workflow

### Objective

Turn accumulated audits into controlled organizational knowledge.

### Candidate capabilities

- finding disposition,
- reviewer ownership,
- comments and evidence references,
- closure and reopening,
- rule packs,
- site-specific thresholds,
- audit history and trend comparison.

### Constraint

Workflow must remain purpose-built for performance quality. PPAR should not become a generic ticketing system.

## Stage E — Broader Platform and Enterprise Support

### Objective

Expand only after the initial product is repeatable.

### Candidate capabilities

- additional accounting-platform starter packs,
- enterprise deployment and packaging,
- team and permission controls,
- cross-portfolio monitoring,
- standardized integrations,
- broader asset-class rule packs.

### Exit gate

Each supported platform or scope has a documented, tested, commercially supportable implementation.

---

# 15. Decisions Confirmed in Phase 1

The following decisions are treated as confirmed unless explicitly reopened:

1. **Focus on PPAR Audit.** Performance Analytics is excluded from the present product-design process.
2. **Separate product identity.** Audit should not be marketed as a co-equal half of an “Audit and Analytics” bundle.
3. **Validation before scale.** The first objective is 2–5 strong validation partners, leading to a repeatable software business.
4. **Changed performance is the wedge.** The core product problem is explaining why previously reported performance changed.
5. **Performance Quality Assurance is the long-term territory.** Expansion remains centered on performance-change explanation.
6. **Evidence and human review are central.** The product must not hide ambiguity or unsupported causes.
7. **Axys/APX is the initial implementation focus.** This is a wedge, not a permanent limit on product identity.
8. **The current product does not rebuild a full accounting ledger.**
9. **Local-first execution is a permanent product principle.** Client portfolio data and calculations remain within a client-controlled environment; a PPAR-operated hosted processing model is outside the product doctrine.
10. **The rules library is a strategic product direction.**
11. **The foundational design is internal and candid first, externally reusable second.**
12. **Prior expansion ideas are carried forward only with explicit status labels.**
13. **The safety-invariant contract is part of the current product doctrine.** No-lost-difference, no-double-counting, Fully Explained arithmetic, lineage, financial-input integrity, report parity, determinism, and fail-closed policy may not be silently weakened.
14. **Packaged-demo coverage and general production support are different claims.** Narrow context-gated examples do not establish universal Axys/APX transaction semantics.

---

# 16. Open Decisions for Founder Review

These questions materially affect later phases.

## 16.1 Product and brand

1. Is **PPAR Audit** intended to be the permanent commercial product name, or a working name under a future company/product brand?
2. Should the external category language lead with **Performance Audit**, **Performance Change Investigation**, or **Performance Quality Assurance**?
3. How prominently should the Modified Dietz methodology appear in product identity versus technical qualification?

## 16.2 Deployment and data control

4. What non-portfolio metadata, if any, may leave the client-controlled environment for licensing, update checks, telemetry, or support?
5. During implementation and support, under what exceptional, client-authorized conditions may redacted or anonymized evidence leave the environment?
6. What minimum offline capability and grace period are required for licensing and operation?

## 16.3 Methodology and scope

7. Is Modified Dietz intended to remain the primary supported explanation methodology, or should the long-term architecture anticipate other performance methods?
8. Which asset classes are intended to be in the initial commercially supported scope?
9. Should portfolio-level and security-level audits both be required in the first pilot, or should one be the primary commercial surface?
10. Who has authority to approve site transaction classifications and materiality thresholds?

## 16.4 Product behavior

11. Should **Partly Explained** and **Unexplained** be strictly mathematical statuses, with separate workflow statuses for review and closure?
12. Should the product ever allow a human reviewer to override or supplement a calculated cause, and if so, how must that be represented?
13. Is the long-term audit-history repository expected to remain file-based, use a local database, or support an optional centralized store?
14. How much user-defined rule logic should be allowed before supportability and auditability are compromised?

## 16.5 Commercial risk and liability

15. How conservative should the product be in using the word **audit** in contracts and marketing?
16. What level of professional liability or reliance does the founder intend to accept?
17. Will the first pilot be positioned as software validation, operational consulting, or a controlled combination of both?
18. What evidence must exist before the company permits a named client case study?

These decisions do not all need immediate answers. Phase 2 v0.3 uses explicit
recommended defaults where a decision is not required to draft safely and
isolates the five decisions that matter most for founder review in Section
17.10.2. The broader brand, methodology-expansion, and commercial questions
remain open for later phases.

---

# 17. Phase 2 — Users, Workflows, and Conceptual Product Architecture

## 17.1 Phase 2 outcome and design posture

Phase 2 defines how PPAR Audit should operate from the customer’s and reviewer’s
perspective. It does not specify implementation stories. Its later founder
approval, recorded in Section 17.9, authorized Phase 3 to begin.

The design keeps five kinds of authority separate:

1. **Source authority:** the client decides which exports and snapshots represent
   the requested reporting states.
2. **Business-policy authority:** qualified client personnel approve field
   meanings, transaction semantics, return basis, methodology, tolerances, and
   materiality.
3. **Product calculation authority:** PPAR applies the approved configuration,
   supported formulas, evidence dispositions, and safety invariants
   deterministically.
4. **Human review authority:** qualified client reviewers determine whether a
   source value is wrong, whether a valid restatement is accepted, what action
   to take, and when an investigation is operationally closed.
5. **Assurance authority:** an independent verifier, auditor, regulator, or other
   authorized party retains any assurance or compliance role. PPAR does not
   assume that role.

This separation reflects both product doctrine and external evidence. Current
GIPS guidance emphasizes documented and consistently applied firm policies,
supporting records, and firm responsibility for performance information,
including outsourced inputs. It also distinguishes firm-wide verification from
assurance on the accuracy of a specific performance report. Those principles
support the decision-rights model below; they do not make PPAR a GIPS product or
an assurance provider.

### 17.1.1 First-pilot operating hypothesis

The recommended first-pilot model is:

- **Primary daily user — APPROVED DIRECTION; CURRENT — REQUIRES CLIENT
  VALIDATION:** a performance analyst acting as the investigation operator. A
  portfolio-accounting or investment-operations analyst may fill the role when
  that person owns performance corrections.
- **Business-policy approver — APPROVED DIRECTION; CURRENT — REQUIRES CLIENT
  VALIDATION:** a performance or investment-operations manager with authority
  over return methodology and transaction treatment.
- **Source/extract owner — APPROVED DIRECTION; CURRENT — REQUIRES CLIENT
  VALIDATION:** the Axys/APX administrator, report writer, or operations data
  owner who can explain and reproduce the exports.
- **Primary commercial surface — APPROVED DIRECTION; CURRENT — REQUIRES CLIENT
  VALIDATION:** portfolio-level audit first, with security-level audit used as
  drilldown and selected validation evidence. Both report levels remain current
  capabilities; this recommendation narrows pilot workflow, not product
  capability.
- **Current operating form — CURRENT — DOCUMENTED; CURRENT — REQUIRES CLIENT
  VALIDATION:** local command-line/setup workflow plus generated XLSX, HTML,
  CSV, JSON, and ZIP artifacts.
- **Near-term product direction — APPROVED DIRECTION:** make readiness,
  executive handoff, recurring use, and local audit history coherent without
  moving portfolio data outside the client-controlled boundary.
- **Workflow software — CANDIDATE:** comments, assignment, disposition, closure,
  and reopening. Human review is mandatory doctrine, but a product-managed
  workflow layer is not current.

These hypotheses are sufficient to draft and test a controlled pilot. They must
be validated with the first client and approved by the founder before being
treated as the permanent commercial operating model.

## 17.2 Actors and role profiles

Job titles vary by firm. The design therefore defines functional roles first.
One person may hold several roles in a smaller firm, but the underlying decision
rights must remain explicit.

### 17.2.1 Daily investigation and source roles

| Actor | Primary jobs and pain | Frequency and inputs | Decisions allowed | Decisions not allowed | Owned evidence/configuration | Trust and success criteria | Product support status |
|---|---|---|---|---|---|---|---|
| Performance analyst | Identify changed returns, explain supported causes, isolate residuals, prepare a defensible answer, and rerun after correction. Pain is manual spreadsheet comparison, weak lineage, and key-person dependence. | Event-driven initially; potentially every reporting cycle. Consumes both snapshots, the approved configuration, `Performance Differences`, causes, Data Audit Issues, and source detail. | Select a preapproved scope; run the audit; triage findings; request corrections; propose a human disposition. | Must not silently redefine mappings, transaction semantics, return basis, or product-calculated causes. | Investigation notes, review selections, rerun request, and evidence references. | Can reach and communicate a supportable conclusion without founder explanation; unresolved items remain visible. | CURRENT — DEMONSTRATED for report review; CURRENT — REQUIRES CLIENT VALIDATION for real operating use; CANDIDATE for in-product disposition. |
| Portfolio-accounting analyst | Explain accounting changes, locate source postings, distinguish valid corrections from extraction problems, and coordinate source-system remediation. | Event-driven during investigation and rerun. Consumes source rows, transaction/holding detail, extract context, and Data Audit Issues. | Explain source-system behavior; propose or execute corrections under client controls; validate whether a restatement is expected. | Must not overwrite PPAR calculations or approve methodology outside delegated authority. | Correction evidence, accounting explanation, and source-system reference. | PPAR evidence maps cleanly to records the analyst recognizes; rerun reflects authorized corrections. | CURRENT — DOCUMENTED as a human reviewer role; CURRENT — REQUIRES CLIENT VALIDATION. |
| Investment-operations analyst | Coordinate extracts, investigate operational exceptions, and ensure the reporting process completes. | Event-driven or recurring. Consumes readiness output, audit artifacts, and action list. | Operate approved jobs; coordinate handoffs; propose severity/materiality based on client policy. | Must not approve accounting or performance methodology merely by operating the process. | Run log, scope record, handoff, and operational checklist. | Failures are actionable; artifacts can be retained and transferred internally without hidden steps. | CURRENT — DOCUMENTED for setup/run surfaces; CURRENT — REQUIRES CLIENT VALIDATION for workflow fit. |
| System/source-extract administrator | Produce reproducible Axys/APX exports, document field meanings, preserve native values/case, and resolve schema drift. | Initial implementation, source changes, and failed preflight. Consumes extract requirements, local source contract, column mappings, and validation errors. | Approve the factual extract definition and confirm which fields/reports are available. | Must not decide Modified Dietz treatment solely from a field name or transaction code unless also authorized as a business-policy owner. | Source/extract contract, export procedure, schema/version metadata, and sample files. | Another authorized operator can reproduce the same extract; ambiguous context fields are explicit. | CURRENT — DOCUMENTED technical contract; CURRENT — REQUIRES CLIENT VALIDATION for real Axys/APX sites. |
| Local product administrator | Install, configure paths, manage local versions/access, execute validation, retain evidence, and support internal scheduling. | Initial setup, upgrades, access changes, and recurring operation. Consumes package/configuration documentation and client infrastructure controls. | Manage local deployment and approved configuration releases; execute signed-off jobs. | Must not approve accounting semantics simply because they can edit YAML. | Installed version, configuration release, access record, local schedule, and retention location. | Reproducible operation with least-necessary access and no undisclosed data transfer. | CURRENT — DOCUMENTED; CURRENT — REQUIRES CLIENT VALIDATION. |

### 17.2.2 Approval, oversight, and commercial roles

| Actor | Primary jobs and pain | Frequency and outputs consumed | Decisions allowed | Decisions not allowed | Owned evidence/approval | Trust and success criteria | Product support status |
|---|---|---|---|---|---|---|---|
| Performance/operations manager | Own investigation quality, approve performance methodology, allocate review work, and decide whether conclusions are operationally acceptable. | Implementation, material investigations, periodic control review. Consumes summary, causes, residuals, Data Audit Issues, and policy exceptions. | Approve mappings, transaction treatment, return basis, tolerances/materiality, and final operational disposition when delegated by the firm. | Must not transform an unsupported product result into a calculated cause or external assurance claim. | Methodology approval, site-policy approval, materiality policy, and disposition approval. | Explanations are complete enough for the decision; important uncertainty is not hidden. | CURRENT — REQUIRES CLIENT VALIDATION for report use; CANDIDATE for managed approvals/work queues. |
| Head/Director of Investment Operations | Sponsor the control, set accountability, resolve cross-team issues, and judge implementation value. | Pilot approval, major exceptions, periodic management review. Consumes executive summary, unresolved exposure, recurring issues, and implementation measures. | Approve scope, staffing, risk acceptance, support model, and continued use. | Must not certify calculation correctness without qualified review evidence. | Pilot charter, accountability matrix, risk acceptance, and escalation decisions. | Material investigations become more reproducible and less dependent on individual analysts. | APPROVED DIRECTION for executive summary and health views; CURRENT — REQUIRES CLIENT VALIDATION for value evidence. |
| Compliance or GIPS reviewer | Assess whether investigation records, methodology changes, error treatment, and disclosures fit firm policy and applicable obligations. | Material restatements, periodic review, or request. Consumes retained evidence, approvals, methodology records, and human dispositions. | Determine compliance handling within the client’s governance; request additional evidence or disclosure. | Must not treat PPAR output as independent verification, legal advice, or a compliance certificate. | Compliance disposition, disclosure reference, and retention requirement. | Product boundaries are clear; records support review without overstating assurance. | CURRENT — DOCUMENTED as a potential reviewer; CURRENT — REQUIRES CLIENT VALIDATION; no dedicated compliance workflow is current. |
| Internal technology/information-security reviewer | Assess deployment, access, dependency, update, licensing, logging, and support-data boundaries. | Procurement/onboarding, upgrade, and incident review. Consumes architecture, data-flow, permissions, and network/support design. | Approve client infrastructure and permitted connectivity under client policy. | Must not approve business mappings or methodology. | Security assessment, deployment approval, outbound-data policy, and exception authorization. | Portfolio data remains inside the approved boundary; any outbound metadata is explicit and minimal. | CURRENT — DOCUMENTED local boundary; CURRENT — REQUIRES CLIENT VALIDATION for enterprise controls. |
| Executive/economic buyer | Decide whether the product merits budget and organizational adoption. | Pilot decision, continuation, renewal, and expansion. Consumes concise outcomes, measured effort/value, risks, and support obligations. | Approve commercial engagement and accountable sponsor. | Must not substitute commercial acceptance for analytical validation. | Budget, sponsor, success measures, and procurement decision. | Evidence of trusted use, repeatability, implementation effort, and bounded risk. | CANDIDATE customer hypothesis; requires discovery and pilot evidence. |
| Implementation partner or PPAR support | Help install, interpret validation errors, map source contracts, train operators, and convert client learning into reusable product guidance. | Initial implementation, upgrades, and controlled troubleshooting. Consumes redacted diagnostics or client-run outputs authorized for support. | Recommend mappings and rules; diagnose documented behavior; propose reusable product changes. | Must not approve client accounting policy, receive portfolio data routinely, alter findings, or provide independent assurance. | Support record, recommendation, disclosed product limitation, and reusable learning item. | Client can operate locally; support does not require routine portfolio-data transfer or founder-only knowledge. | CURRENT — DOCUMENTED for technical surfaces; CURRENT — REQUIRES CLIENT VALIDATION for supportability; redaction tooling is CANDIDATE. |

### 17.2.3 Role-design conclusions

1. A performance analyst is the strongest primary-user hypothesis because the
   current output begins with changed returns and quantified causes, and CFA
   Institute describes performance-measurement staff as producing routine and
   non-routine reports for internal and external consumers. This remains a
   hypothesis, not customer validation.
2. The source administrator and methodology approver are distinct roles even if
   one person fills both. Export facts and accounting interpretation are
   different approvals.
3. The performance/operations manager is the recommended first-pilot business-
   policy authority. PPAR support may advise but must not own client policy.
4. Compliance, technology/security, and the economic buyer are review and
   approval stakeholders, not ordinary row-level operators.
5. The product needs a concise management handoff, but not a separate management
   calculation. The same validated tables must support analyst and executive
   views.

## 17.3 Decision rights and approval model

### 17.3.1 Governing rules

- PPAR may calculate only from supplied source facts, normalized facts,
  supported methods, and approved configuration.
- A human may disagree with a result, but disagreement does not rewrite the
  product-calculated result. Correct source-data or configuration must be
  approved and the audit rerun.
- A human explanation may eventually supplement the record only if it is
  labeled as human-supplied, names its author and date, cites evidence, and
  remains outside `Explained Difference` unless a supported product rule is
  implemented and rerun. This is a **CANDIDATE** design, not current behavior.
- Analytical status and workflow status are orthogonal. Closing an investigation
  does not turn `Partly Explained` or `Unexplained` into `Fully Explained`.
- Approval must be attributable. The ability to edit YAML or files is not, by
  itself, business authority.
- Suppression is review-priority metadata. It is never deletion from the
  complete audit trail and never substitutes for field classification.

### 17.3.2 Approval matrix

| Decision or action | Proposes or performs | Required approver | Consulted/reviewed by | Product behavior and record | Capability status |
|---|---|---|---|---|---|
| Select audit portfolios, periods, report level, and snapshots | Performance/operations analyst | Performance/operations manager for pilot or material runs | Source administrator; compliance when material | Scope and snapshot identifiers must be retained with the evidence pack. Snapshot labels remain neutral. | CURRENT — DOCUMENTED; CURRENT — REQUIRES CLIENT VALIDATION |
| Define and change the source/extract contract | Source-extract administrator | Source owner plus performance/operations manager for business meaning | Local administrator; PPAR support | Revalidate before report generation; preserve contract version/path and extract context. | CURRENT — DOCUMENTED; CURRENT — REQUIRES CLIENT VALIDATION |
| Approve field mappings and accounting roles | Source administrator and analyst | Performance/operations manager or delegated methodology owner | Portfolio accounting; PPAR support | Unknown or ambiguous performance-relevant fields fail closed; approved configuration is versioned. | CURRENT — DOCUMENTED; CURRENT — REQUIRES CLIENT VALIDATION |
| Approve transaction classifications, signs, and flow semantics | Portfolio-accounting/performance specialist | Delegated performance methodology owner | Source administrator; compliance as required | Native code remains visible; ambiguous families require context or reviewed local policy; code-only guessing is prohibited. | CURRENT — DOCUMENTED; CURRENT — REQUIRES CLIENT VALIDATION |
| Approve return basis and Modified Dietz assumptions | Performance methodology owner | Performance/operations manager or the client’s formal methodology authority | Compliance/GIPS reviewer where applicable | Configuration identifies method, timing, day count, inclusion rule, flow/income categories, return basis, and sign convention. | CURRENT — DOCUMENTED; CURRENT — REQUIRES CLIENT VALIDATION |
| Set comparison tolerances and operational materiality | Analyst or methodology owner | Performance/operations manager | Compliance and executive sponsor for material policy | Comparison tolerance determines reportable differences; business materiality governs review/escalation and must not silently erase findings. | CURRENT — DOCUMENTED for tolerances; CANDIDATE for a coherent materiality workflow |
| Create or change a suppression | Analyst proposes; authorized configuration owner implements | Performance/operations manager | Compliance for material or recurring suppressions | Suppressed rows remain in `findings.csv`; reason/approval should be retained. Requiring formal reasons is CANDIDATE. | CURRENT — DOCUMENTED technical behavior; CANDIDATE approval workflow |
| Run the audit or rerun after correction | Authorized analyst/operator | Preapproved operating procedure; manager approval for scope/policy changes | Local administrator | New artifacts should identify inputs, configuration, and product version; prior artifacts remain retained under policy. Current bundles identify the comparison path and source context but not the installed PPAR version or a complete configuration fingerprint. | CURRENT — DEMONSTRATED for execution and partial provenance; APPROVED DIRECTION for complete run provenance; CURRENT — REQUIRES CLIENT VALIDATION |
| View complete versus prioritized evidence | Authorized reviewer | Client access-control owner | Technology/security | Summaries may prioritize; complete evidence remains available to authorized reviewers and validators. | CURRENT — DOCUMENTED artifact separation; CURRENT — REQUIRES CLIENT VALIDATION for access model |
| Add a human note or disposition | Qualified analyst or manager | Manager for material closure | Portfolio accounting/compliance as applicable | If later justified, store separately from calculated causes with author, time, evidence reference, and workflow status. | DEFERRED; requires a serious business case and client validation |
| Add a supplemental human explanation | Qualified specialist | Methodology owner | Compliance for external reliance | If later justified, it must be explicitly human-supplied and non-additive; it cannot change analytical status or Explained Difference. | APPROVED DIRECTION concept; DEFERRED implementation pending business case and client validation |
| Override a product-calculated cause or amount | No direct override permitted | Not applicable | Methodology owner may challenge result | Correct source/configuration or product logic, then rerun. Preserve the superseded run and reason. | OUT OF SCOPE for direct override |
| Mark accepted, correction required, closed, or reopened | Analyst proposes | Performance/operations manager under client policy | Compliance when relevant | If later justified, workflow status changes only; analytical status remains immutable for that run. | DEFERRED pending business case and client validation |
| Correct the accounting system or official reported performance | Authorized client accounting personnel | Client’s source-system/change-control authority | Performance/operations manager | PPAR never writes back. A correction creates a new source state and usually a rerun. | OUT OF SCOPE for PPAR writeback; client action is required |
| Export artifacts internally | Authorized operator | Client data owner under normal policy | Technology/security | Preserve bundle integrity and provenance. | CURRENT — DOCUMENTED; CURRENT — REQUIRES CLIENT VALIDATION |
| Share redacted/anonymized evidence for external support | Client operator prepares | Data owner plus technology/security and business owner | Legal/compliance as required | Explicit, case-specific authorization; disclose exactly what leaves; no routine transfer. Redaction tooling is not current. | OPEN DECISION; CANDIDATE tooling |
| Permit license/update metadata to leave | Local administrator | Technology/security and procurement | Business owner | Only approved non-portfolio metadata; no portfolio data or evidence. Exact fields and offline behavior remain undecided. | OPEN DECISION |
| Approve an external case-study claim | PPAR proposes | Authorized client executive/legal representative and PPAR leadership | Analyst, compliance, and product owner | Separate written permission after client validation; claim must match measured evidence. | OPEN DECISION |

### 17.3.3 Recommended defaults pending founder review

1. **No direct calculated-result override.** This is the strongest default and
   follows current deterministic/evidence doctrine.
2. **Allow future human annotations, not human-calculated causes.** A note can
   explain external evidence or a business decision without changing PPAR
   arithmetic.
3. **Manager approval for policy; analyst operation within policy.** This avoids
   making every routine run a governance project while protecting methodology.
4. **Versioned local files first for pilot history.** Preserve immutable run
   directories, manifest/configuration references, and an investigation index.
   A local database remains an approved direction only after the file-based
   workflow proves what entities and queries are actually needed.
5. **Portfolio audit first, security drilldown second.** This narrows the pilot
   conversation to the headline changed return while retaining security-level
   evidence where it materially helps.

## 17.4 End-to-end workflows

The workflows below distinguish implemented behavior from the desired product
experience. “Current” never means client validated.

### 17.4.1 Initial implementation and site setup

**Capability status:** CURRENT — DOCUMENTED technical setup; CURRENT — REQUIRES
CLIENT VALIDATION for real-site implementation; APPROVED DIRECTION for a
repeatable onboarding experience.

- **Trigger:** a validation partner approves a bounded pilot or an existing
  client adopts a new site/source contract.
- **Primary actors:** implementation lead, source-extract administrator, local
  product administrator, performance analyst, methodology owner, and
  technology/security reviewer.
- **Prerequisites:** pilot scope and responsibilities; client-controlled
  environment; package installation path; representative extract inventory;
  data-access approval; named source and methodology owners.
- **Inputs:** starter workspace; local extract samples and column lists; source-
  system/report context; candidate mappings; transaction-semantics matrix;
  extract-contract template; client return/methodology policy.
- **Current steps:** install locally; run `ppar setup`; preserve the demo; create
  a site workspace; replace CSVs; map fields; define comparison level and
  source paths; configure transaction rules, impact treatment, reconstruction,
  tolerances, and suppressions; run configuration validation; generate bounded
  validation reports.
- **Required approval points:** source owner confirms extract facts;
  methodology owner approves accounting roles, transaction treatment, return
  basis, and tolerances; technology/security approves local deployment and any
  network behavior.
- **Outputs and handoff:** approved source/extract contract; approved and
  versioned configuration; documented export procedure; validation bundle;
  issue/assumption log; first accepted baseline; named operator and escalation
  path.
- **Failure paths:** unavailable required exports, ambiguous codes without
  context, inconsistent period/currency basis, client-specific logic that
  cannot be expressed transparently, or inability to identify an accountable
  methodology owner. These block or narrow the pilot; they are not reasons to
  guess.
- **Pilot acceptance:** selected known restatements reconcile to client review;
  unresolved items are accepted as honest; another client operator can rerun
  the workflow; implementation effort and exceptions are measured.

### 17.4.2 Audit readiness and preflight

**Capability status:** CURRENT — DOCUMENTED for technical validation and hard
stops; APPROVED DIRECTION for a unified operator-facing readiness decision;
CURRENT — REQUIRES CLIENT VALIDATION for usefulness.

- **Trigger:** before the first run, after a schema/configuration/product change,
  or when a prior run failed.
- **Primary actors:** analyst/operator, source administrator, methodology owner,
  and local administrator.
- **Inputs:** requested scope; Snapshot A and B; resolved YAML; source/extract
  contract; installed product version; prior accepted baseline if available.
- **Checks:** file presence; required normalized columns; extract-context
  requirements; unique/valid identities; explicit field roles; complete impact
  policy; transaction semantics; currency/base-value integrity; period ordering
  and overlap; required reconstruction inputs; report-size risk; configuration
  approval/version.
- **Decision model — CANDIDATE:**
  - `Ready`: all required contracts pass for requested scope.
  - `Qualified Ready`: no unsafe condition exists, but missing optional evidence
    limits explanation depth and the limitation is explicit.
  - `Blocked`: a required input, meaning, policy, integrity rule, or safe output
    boundary is unresolved.
- **Current behavior:** validation already hard-stops many unsafe conditions and
  allows omitted optional evidence where the configured workflow remains safe.
  It does not yet present one durable readiness object or all three named
  statuses.
- **Outputs and handoff:** readiness result; blocking/qualifying items; affected
  scope; accountable owner; exact corrective action; approved exception if one
  exists; link to resolved configuration.
- **Failure rule:** a technical validation error is never converted into
  `Qualified Ready` for convenience. Qualification applies only when the source
  contract explicitly permits missing optional evidence.
- **Acceptance:** an operator can tell whether to run, what will be limited, and
  who must act without reading Python internals.

### 17.4.3 Performance-change investigation run

**Capability status:** CURRENT — DEMONSTRATED; CURRENT — DOCUMENTED; CURRENT —
REQUIRES CLIENT VALIDATION.

- **Trigger:** a previously reported return changes, a new accounting snapshot
  arrives, or a controlled validation case is selected.
- **Primary actors:** authorized operator; source and methodology owners on call
  for failures.
- **Prerequisites:** approved scope/configuration; a safe readiness result under
  current controls; write access to a client-controlled output location.
- **Deterministic processing sequence:**
  1. Load Snapshot A and Snapshot B without assuming either is correct.
  2. Normalize configured datasets and preserve source identity/provenance.
  3. Match records conservatively; retain additions, removals, and ambiguity.
  4. Identify reportable portfolio/security and supporting source differences.
  5. Link dated evidence to unambiguous performance periods and formula roles.
  6. Apply supported Modified Dietz reconstruction/attribution and explicit
     impact policy.
  7. Assign every reportable source difference to `counted_cause` or
     `review_evidence` and enforce single-owner economic effects.
  8. Run conservation, arithmetic, lineage, continuity, currency/unit,
     period-boundary, policy, parity, determinism, and output-size controls.
  9. Run configured Data Audit checks, separately from additive causes.
  10. Generate level-specific XLSX/HTML, promoted source detail, and the
      validated manifest v4 support bundle.
- **Outputs:** analytical statuses and amounts; causes; Data Audit Issues;
  complete finding trail; cause lineage; triage/context/diagnostic tables;
  manifest and review summary.
- **Failure paths:** source-contract errors stop the affected workflow with an
  actionable error; internal logic/parity/lineage failures stop generation;
  suspicious but interpretable relationships remain visible findings;
  report-size overflow stops before unusable artifacts are written.
- **Acceptance:** the run is reproducible from its inputs/configuration/version;
  no unsafe partial bundle is presented as a successful audit.

### 17.4.4 Analyst review and triage

**Capability status:** CURRENT — DEMONSTRATED report surfaces; CURRENT —
REQUIRES CLIENT VALIDATION for real reviewer sequence and usability; APPROVED
DIRECTION for a concise executive entry layer.

- **Trigger:** a validated bundle is generated.
- **Primary actor:** performance/operations analyst; portfolio accounting and
  manager join as needed.
- **Normal review sequence:**
  1. Open the level-specific workbook or HTML report.
  2. Start with `Performance Differences` to identify changed periods,
     analytical status, explained amount, and residual.
  3. Open `Performance Difference Causes` for counted inputs and supporting
     relationships.
  4. Review `Data Audit Issues` as independent source-quality signals; do not
     assume they caused the return difference.
  5. Use root-level `source_detail.csv` for active finding detail and native
     values.
  6. Use needs-review, context, matching, cross-check, reconstruction, complete
     findings, and cause-lineage artifacts only when the question requires
     deeper evidence.
- **Decision questions:** Is the explanation arithmetically complete? Is the
  configured treatment approved? Does the source evidence support the
  narrative? Is the restatement valid, suspicious, or still limited by source/
  methodology? What action and owner are required?
- **Outputs and handoff:** prioritized review list; accepted product explanation
  or challenged assumption; request for source/configuration correction;
  unresolved question; evidence reference; escalation owner.
- **Failure paths:** reviewer cannot interpret units or basis; evidence conflicts
  with client knowledge; source row is missing; configured rule produces noise;
  or report volume is unmanageable. Preserve the result and escalate—do not
  edit calculated artifacts.
- **Acceptance:** a qualified reviewer can navigate from changed performance to
  source evidence and understand why an item is counted, review-only, or
  unresolved.

### 17.4.5 Human disposition and investigation closure

**Capability status:** DEFERRED. Human judgment is current doctrine; product-
managed notes, assignments, approvals, and closure require a serious business
case and client validation before implementation.

- **Trigger:** analyst triage reaches a decision or identifies required action.
- **Primary actors:** analyst proposes; methodology/operations manager approves;
  portfolio accounting or compliance participates when relevant.
- **Candidate workflow states:** `Open`, `In Review`, `Accepted`, `Correction
  Required`, `Rerun Required`, `Closed`, and `Reopened`.
- **Permitted dispositions:** accept the product explanation; accept a valid
  restatement; request source correction; request configuration/methodology
  correction; document an externally supported explanation; accept an
  unresolved limitation under client policy; rerun; close; reopen.
- **Record requirements:** immutable run identifier; analytical status; workflow
  status; author/approver/time; reason; evidence references; action owner; link
  to superseding rerun; any externally supplied explanation labeled as human.
- **Non-negotiable boundary:** workflow closure does not alter product arithmetic
  or analytical status. Direct override of a product-calculated cause remains
  OUT OF SCOPE.
- **Outputs:** approved disposition record, action/rerun request, closure record,
  or escalation.
- **Failure paths:** no authorized approver; disagreement over methodology;
  missing evidence; material unresolved item; or attempted direct artifact edit.
  Keep the investigation open or record a governed risk acceptance outside
  PPAR until the workflow capability is specified.
- **Acceptance:** another reviewer can distinguish what PPAR calculated from
  what a human decided and why the investigation was closed.

### 17.4.6 Recurring audit and proactive quality review

**Capability status:** APPROVED DIRECTION as a long-term possibility; DEFERRED
implementation pending business case and client validation. Repeatable manual
runs are current; scheduled operation, durable history, recurrence detection,
and management trend views are not current product capabilities.

- **Trigger:** reporting-cycle schedule, approved source-state change, or
  periodic control procedure.
- **Primary actors:** local administrator/operator, performance analyst, and
  manager.
- **Prerequisites:** repeatable source/extract contract; approved schedule;
  stable configuration/version policy; local retention; accepted comparison
  key for “prior reported state”; change-management procedure.
- **Desired steps:** produce new snapshots; run readiness; run audit; compare
  findings/statuses to local history; identify new, recurring, resolved, and
  reopened issues; produce analyst and management summaries; retain the run.
- **Conditional history recommendation:** only if recurring use establishes the
  business case, begin with immutable versioned directories plus the smallest
  useful local index. Do not add a database or index merely to appear
  enterprise-ready.
- **Outputs:** current investigation bundle, recurrence classification, local
  history entry, manager summary, and exception backlog.
- **Failure paths:** incomparable scopes/configuration; missing prior state;
  retention collision; schema/version drift; or schedule runs without an
  accountable reviewer. Mark the run incomparable or block recurrence metrics;
  never create a misleading trend.
- **Acceptance:** repeated operation reduces setup effort and preserves
  comparability without sending portfolio data to PPAR.

### 17.4.7 Evidence-package retention and handoff

**Capability status:** CURRENT — DEMONSTRATED; CURRENT — DOCUMENTED for
bundle generation/validation; CURRENT — REQUIRES CLIENT VALIDATION for client
retention and internal handoff; cross-run history is DEFERRED pending business
case and client validation.

- **Trigger:** successful report generation or approved rerun.
- **Authoritative run artifacts:** level-specific workbook/HTML, manifest v4,
  review summary, complete `findings.csv`, cause lineage, source detail, primary
  CSV tables, and referenced configuration/source-contract/version context.
- **Primary actors:** operator creates; local administrator retains; analyst and
  manager review; compliance/technology set policy.
- **Retention rules:** keep artifacts immutable; retain or reproducibly reference
  inputs and approved configuration under client policy; preserve the manifest
  and bundle fingerprint; link superseded and superseding runs; do not retain
  less evidence merely because a summary is convenient.
- **Internal handoff:** use declared review entrypoints; identify scope, status,
  unresolved items, action owner, and required detail artifacts.
- **External handoff:** OUT OF SCOPE by default for portfolio data. Any redacted
  or anonymized support package requires explicit authorization and a defined
  generation/review procedure.
- **Failure paths:** missing required artifact, invalid fingerprint, corrupted
  archive, unknown configuration, or ambiguous run identity. Treat the evidence
  pack as invalid and regenerate or escalate; do not hand off a partial package
  as authoritative.
- **Acceptance:** a second authorized reviewer can reproduce the review path and
  establish which inputs, configuration, and version produced the result.

### 17.4.8 Support and troubleshooting under local-first doctrine

**Capability status:** CURRENT — DOCUMENTED for local operation, stable errors,
validators, and support artifacts; CURRENT — REQUIRES CLIENT VALIDATION for a
repeatable support model; CANDIDATE for automated redaction diagnostics.

- **Trigger:** installation/configuration failure, source-contract error,
  unexpected finding, artifact-validation failure, or upgrade issue.
- **Primary actors:** client operator/local administrator first; source or
  methodology owner for domain questions; PPAR support for product behavior.
- **Local-first sequence:** reproduce locally; run documented validation;
  inspect stable error code/context; inspect manifest and non-sensitive
  diagnostics; verify version/configuration; isolate source-contract versus
  product-logic question; prepare a minimal client-approved support record.
- **Data boundary:** raw snapshots, source detail, portfolio identifiers,
  evidence packs, and calculated audit results remain local by default. Screen
  sharing, file transfer, or copied diagnostics must follow client policy.
- **Potential external support:** only redacted/anonymized evidence explicitly
  authorized for the case. The client reviews the package before transfer;
  PPAR records purpose, contents, recipient, retention, and deletion.
- **Licensing/update boundary:** may eventually exchange approved entitlement,
  product version, machine/token, and update metadata; exact fields, offline
  grace, and telemetry remain OPEN DECISION. No mechanism may silently include
  portfolio data or audit evidence.
- **Failure paths:** issue cannot be reproduced without sensitive data;
  authorization denied; redaction cannot preserve needed semantics; or product
  defect suspected. Continue with client-run probes, provide a local patch/test
  path when appropriate, or record a blocked support case—do not broaden data
  access informally.
- **Acceptance:** most issues are diagnosable from stable local errors and
  client-run validation; exceptional data sharing is rare, explicit, minimal,
  and auditable.

### 17.4.9 Implementation and product change management

**Capability status:** CURRENT — DOCUMENTED for strict validation and regression
tests; CURRENT — REQUIRES CLIENT VALIDATION for client change control;
APPROVED DIRECTION for a standardized operating procedure.

- **Triggers:** source schema/report change; new transaction code or asset type;
  methodology/return-basis change; tolerance/suppression change; product
  upgrade; environment change; new report scope.
- **Primary actors:** change requester, source administrator, methodology owner,
  local administrator, analyst validator, and PPAR support when product changes
  are required.
- **Required steps:** classify the change; preserve prior configuration/version;
  update evidence and contract before treatment; validate mappings/semantics;
  run focused accepted cases and safety checks; compare artifacts; obtain
  business approval; deploy locally; retain change record; monitor first run.
- **High-risk rule:** a new transaction code or source field is not activated as
  a counted cause from name/code inference alone. Evidence, role, formula,
  policy, test, and approval must agree.
- **Upgrade rule:** a product version change requires regression against accepted
  client cases and reapproval when output semantics, policy, or a safety
  invariant changes.
- **Outputs:** change request, evidence/source-contract update, approved
  configuration, regression result, deployment record, known limitation, and
  rollback/supersession reference.
- **Failure paths:** changed semantics without evidence, regression mismatch,
  invariant change, schema ambiguity, or unsupported asset/transaction family.
  Block, remain review-only, or defer. Safety gates may change only through
  explicit product and engineering approval; no threshold is relaxed merely to
  make the change pass.
- **Acceptance:** every production-relevant change has an owner, evidence,
  approval, and reproducible validation result.

## 17.5 Conceptual product architecture

This architecture describes product responsibilities and handoffs, not Python
packages. Each component has an explicit capability status and human owner.

### 17.5.1 Client-Controlled Deployment Boundary

- **Purpose and status:** define where code executes, where portfolio data and
  evidence may reside, and which infrastructure is client controlled. CURRENT —
  DOCUMENTED; CURRENT — REQUIRES CLIENT VALIDATION. The permanent local-first
  principle is approved doctrine.
- **Inputs, outputs, dependencies:** client infrastructure/security policy,
  installation package, local identities/access, and storage locations produce
  an approved local runtime boundary. All other components depend on it.
- **Human owner:** client technology/security owner with the business sponsor;
  local product administrator operates within approval.
- **Safety/local-first/future:** deny undisclosed data transfer; minimize access;
  keep calculations and artifacts local. Client-controlled server/private-
  infrastructure deployment is compatible. PPAR-operated hosted processing of
  client portfolio data is OUT OF SCOPE.

### 17.5.2 Local Workspace, Configuration, and Version Context

- **Purpose and status:** hold source paths, mappings, policy, report options,
  product version, and output location. CURRENT — DEMONSTRATED; CURRENT —
  DOCUMENTED; CURRENT — REQUIRES CLIENT VALIDATION.
- **Inputs, outputs, dependencies:** setup template, local files, YAML, column
  mappings, extract contract, and version produce a resolved run context used by
  readiness and calculation.
- **Human owner:** local product administrator maintains; source and methodology
  owners approve their respective content.
- **Safety/local-first/future:** configuration is controlled business policy;
  unknown fields and incomplete impact treatment fail closed. APPROVED DIRECTION
  is attributable configuration approval and a concise resolved-policy export;
  hidden client code forks are OUT OF SCOPE.

### 17.5.3 Snapshot Intake and Source/Extract Contract

- **Purpose and status:** define exactly what Snapshot A and B contain, what
  fields mean, how the extracts were produced, and what PPAR may interpret.
  CURRENT — DOCUMENTED; CURRENT — REQUIRES CLIENT VALIDATION.
- **Inputs, outputs, dependencies:** local CSVs, export procedure, column
  mappings, source context, dataset requirements, and ambiguous-transaction
  context produce two neutral, contract-bound snapshot inputs.
- **Human owner:** source-extract administrator owns factual extract content;
  methodology owner approves meanings that affect performance treatment.
- **Safety/local-first/future:** required missing/ambiguous inputs block; optional
  evidence may qualify depth. Native values/case remain visible. APPROVED
  DIRECTION is a repeatable site-contract approval and change-detection
  experience; universal Axys/APX extraction is not established.

### 17.5.4 Audit Readiness and Preflight

- **Purpose and status:** decide whether the requested audit can run safely and
  explain limitations before expensive processing. CURRENT — DOCUMENTED
  technical controls; APPROVED DIRECTION unified user experience; CURRENT —
  REQUIRES CLIENT VALIDATION.
- **Inputs, outputs, dependencies:** resolved workspace, snapshots, source
  contract, policy, version, and scope produce actionable pass/block/limitation
  evidence. It is upstream of normalization/report generation.
- **Human owner:** operator resolves routine issues; source, methodology, or
  technology owner resolves domain-specific blockers.
- **Safety/local-first/future:** never downgrade an unsafe contract to a warning.
  `Ready`, `Qualified Ready`, and `Blocked` are CANDIDATE readiness states; exact
  persistence and UI are Phase 3 matters.

### 17.5.5 Normalization and Explicit Accounting Roles

- **Purpose and status:** translate site fields into a stable internal vocabulary
  while preserving source provenance and separating performance inputs,
  components, reported outputs, and context. CURRENT — DEMONSTRATED; CURRENT —
  DOCUMENTED; CURRENT — REQUIRES CLIENT VALIDATION.
- **Inputs, outputs, dependencies:** loaded source rows, schema aliases, column
  mappings, currency basis, and role registry produce normalized portfolio
  performance, security performance, holdings, transactions, FX rates, and
  optional split evidence.
- **Human owner:** source administrator approves factual mapping; methodology
  owner approves accounting role; PPAR owns deterministic normalization.
- **Safety/local-first/future:** preserve native identifiers/codes and stable
  locators; reject unknown changed fields and unsafe unit/currency treatment.
  Platform expansion is CANDIDATE only after a validated source contract.

### 17.5.6 Record Identity and Difference Detection

- **Purpose and status:** determine which records can safely be compared and
  emit additions, removals, and changed values without invented linkage.
  CURRENT — DEMONSTRATED; CURRENT — DOCUMENTED; CURRENT — REQUIRES CLIENT
  VALIDATION.
- **Inputs, outputs, dependencies:** normalized snapshots, stable IDs where
  available, exact keys, and tolerances produce reportable source differences,
  match statuses, and ambiguity evidence.
- **Human owner:** PPAR owns deterministic matching; source owner explains
  identity availability; reviewer evaluates unmatched/ambiguous groups.
- **Safety/local-first/future:** stable transaction ID is strongest; exact
  singleton fallback is weaker and labeled; fuzzy matching is OUT OF SCOPE under
  current doctrine. Duplicate/ambiguous identities remain visible or block where
  unsafe.

### 17.5.7 Performance Formula / Reconstruction Boundary

- **Purpose and status:** define the limited Modified Dietz calculation surface
  used to test and explain reported-return changes without rebuilding the
  accounting ledger. CURRENT — DEMONSTRATED; CURRENT — DOCUMENTED; CURRENT —
  REQUIRES CLIENT VALIDATION.
- **Inputs, outputs, dependencies:** performance periods, beginning/end values,
  approved external/security flows, income/fee treatment, timing/day count,
  return basis, currency basis, and denominators produce reconstruction evidence
  and formula-level effects.
- **Human owner:** methodology owner approves policy; PPAR owns supported
  calculation; client remains authority for official reported performance.
- **Safety/local-first/future:** reject missing method inputs, ambiguous periods,
  unsafe units, and unsupported methods. Other performance methods are OPEN
  DECISION; a universal performance engine and full-ledger reconstruction are
  OUT OF SCOPE.

### 17.5.8 Cause Attribution, Economic-Effect Ownership, and Conservation

- **Purpose and status:** connect supported source changes to reported-
  performance differences while ensuring every reportable difference remains
  represented and every explained economic effect is counted once. CURRENT —
  DEMONSTRATED; CURRENT — DOCUMENTED; CURRENT — REQUIRES CLIENT VALIDATION.
- **Inputs, outputs, dependencies:** reportable differences, formula rows,
  explicit impact policy, period links, and role classifications produce counted
  causes, review evidence, explained totals, residuals, and analytical status.
- **Human owner:** PPAR owns deterministic attribution inside the approved
  contract; methodology owner approves policy; analyst reviews unsupported
  residuals.
- **Safety/local-first/future:** enforce SN-01, SN-02, SN-03, SN-06, SN-07, and
  SN-12. Support rows—including FX and split factors—cannot double count an
  owned input effect. Unrestricted causal inference and direct human override
  are OUT OF SCOPE.

### 17.5.9 Evidence and Bidirectional Lineage

- **Purpose and status:** preserve the path from source fact through normalized
  difference, policy, cause, report row, and back again. CURRENT — DEMONSTRATED;
  CURRENT — DOCUMENTED; CURRENT — REQUIRES CLIENT VALIDATION.
- **Inputs, outputs, dependencies:** findings, stable locators/fingerprints,
  cause rows, lineage type, disposition, and economic-effect ID produce the
  complete finding trail and cause-lineage artifact.
- **Human owner:** PPAR owns lineage generation/validation; local administrator
  retains it; reviewer uses it when detail is disputed.
- **Safety/local-first/future:** enforce SN-05 and preserve suppressed rows.
  Summaries may filter but not become the only surviving evidence. External
  evidence transfer remains prohibited by default.

### 17.5.10 Data Quality Rules Engine

- **Purpose and status:** identify suspicious source relationships independently
  of additive performance causes. CURRENT — DEMONSTRATED; CURRENT — DOCUMENTED;
  CURRENT — REQUIRES CLIENT VALIDATION. Broader reusable rule families are
  APPROVED DIRECTION.
- **Inputs, outputs, dependencies:** normalized union of both snapshots,
  configured optional checks/tolerances/filters, and mandatory continuity
  checks produce `Data Audit Issues` with observed/reference values and
  explanations.
- **Human owner:** methodology/operations manager approves policy and tolerance;
  analyst reviews; PPAR evaluates transparent rules.
- **Safety/local-first/future:** a finding is not automatically an error or a
  cause. Current optional rules are the seven listed in Section 6.7; continuity
  is mandatory. Split plausibility, severity, prioritization, profiles, and a
  standalone Data Audit workflow are CANDIDATE until specified and validated.

### 17.5.11 Analytical Review Model

- **Purpose and status:** organize the investigation from headline difference to
  cause, independent data-quality issue, and supporting evidence. CURRENT —
  DEMONSTRATED; CURRENT — DOCUMENTED; CURRENT — REQUIRES CLIENT VALIDATION.
- **Inputs, outputs, dependencies:** validated shared tables produce the three
  normal workbook/HTML review surfaces plus promoted source detail and declared
  diagnostic entrypoints.
- **Human owner:** performance/operations analyst owns triage; manager owns
  escalation and acceptance under client policy.
- **Safety/local-first/future:** analyst and management views must share the same
  validated calculations. Review prioritization may improve, but complete
  evidence must remain available. Optional reconstruction stays secondary.

### 17.5.12 Human Disposition and Workflow Layer

- **Purpose and status:** potentially record who reviewed what, what action was
  chosen, and whether an investigation is open or closed. DEFERRED pending a
  serious business case and client validation.
- **Inputs, outputs, dependencies:** immutable run/analytical status, reviewer
  identity, evidence references, comments, actions, and approvals would produce
  attributable workflow status and links to reruns.
- **Human owner:** analyst proposes; performance/operations manager approves;
  client governance defines compliance involvement.
- **Safety/local-first/future:** must remain separate from product calculations,
  retain local history, and avoid becoming generic case management. Human-note
  and disposition infrastructure is DEFERRED until a serious business case and
  client validation justify it; calculated override is OUT OF SCOPE.

### 17.5.13 Executive Investigation Summary

- **Purpose and status:** give management a concise first view of scope,
  performance difference, explanation completeness, major supported causes,
  residuals, priority findings, and next actions. CURRENT — DOCUMENTED technical
  handoff foundation; APPROVED DIRECTION for the intended experience.
- **Inputs, outputs, dependencies:** validated differences, causes, residuals,
  Data Audit findings, and review entrypoints produce a derived summary—never a
  separate calculation.
- **Human owner:** product generates; analyst validates narrative context;
  manager consumes and approves external use.
- **Safety/local-first/future:** use explanation completeness, not an unsupported
  confidence score; retain limitations and evidence links. Exact content,
  materiality, and action language require Phase 3 and client validation.

### 17.5.14 Evidence-Pack Generation and Validation

- **Purpose and status:** serialize reviewer and machine artifacts into a
  reproducible, semantically aligned handoff. CURRENT — DEMONSTRATED; CURRENT —
  DOCUMENTED; CURRENT — REQUIRES CLIENT VALIDATION.
- **Inputs, outputs, dependencies:** validated tables and review views produce
  level-specific XLSX/HTML, promoted CSV, `audit_support.zip` or expanded files,
  manifest v4, review summary v1, typed fingerprints, and bundle fingerprint.
- **Human owner:** PPAR generates/validates; local administrator retains;
  reviewer chooses the entrypoint.
- **Safety/local-first/future:** enforce SN-10 and SN-11; parity or determinism
  drift is an internal logic error. Packaging may evolve, but required evidence
  and volatile exclusions are change-controlled contracts.

### 17.5.15 Local Audit History and Operational Intelligence

- **Purpose and status:** potentially compare investigations across time to
  identify new, recurring, resolved, and unstable patterns. APPROVED DIRECTION
  only as a long-term possibility; implementation is DEFERRED pending business
  case and client validation. No current persistent history product exists.
- **Inputs, outputs, dependencies:** comparable retained runs, stable
  configuration/rule versions, workflow dispositions, and local index/history
  produce recurrence and trend evidence.
- **Human owner:** local administrator retains; performance manager governs
  comparability/metrics; analyst investigates patterns.
- **Safety/local-first/future:** if the capability is later justified, start
  with versioned local files and the smallest useful index; do not produce
  trends across incomparable scopes. Storage selection is DEFERRED.
  Cross-client or PPAR-hosted portfolio history is OUT OF SCOPE.

### 17.5.16 Licensing, Updates, and Support Boundary

- **Purpose and status:** allow legitimate product operation and support without
  eroding local-first trust. Commercial licensing is APPROVED DIRECTION in the
  engineering roadmap; exact mechanism, telemetry, offline behavior, and
  redaction support are OPEN DECISION.
- **Inputs, outputs, dependencies:** approved entitlement and minimal product/
  environment metadata may eventually produce local activation/update state and
  a support record. This component depends on client security approval.
- **Human owner:** client technology/security and procurement approve; local
  administrator operates; PPAR support maintains the service boundary.
- **Safety/local-first/future:** portfolio data, identifiers, findings,
  calculations, and evidence do not leave silently. A reasonable offline path
  is required by doctrine but not yet specified. A PPAR-operated hosted
  portfolio-data processing service is OUT OF SCOPE.

## 17.6 Information, evidence, and accountability flow

### 17.6.1 Conceptual flow

```text
Client-controlled deployment boundary
  -> CURRENT — DOCUMENTED local workspace and version context
  -> CURRENT — REQUIRES CLIENT VALIDATION approved site source/extract contract
  -> CURRENT — DEMONSTRATED Snapshot A + Snapshot B
  -> CURRENT — DOCUMENTED technical validation and hard-stop preflight
  -> APPROVED DIRECTION unified Ready / Qualified Ready / Blocked experience
  -> CURRENT — DEMONSTRATED normalization + explicit accounting roles
  -> CURRENT — DEMONSTRATED conservative identity + difference detection
  -> CURRENT — DEMONSTRATED performance-period/formula assignment
  -> CURRENT — DEMONSTRATED counted_cause or review_evidence disposition
  -> CURRENT — DOCUMENTED single-owner economic effects + safety checks
  -> CURRENT — DEMONSTRATED Data Audit Issues, separate from additive causes
  -> CURRENT — DEMONSTRATED validated XLSX / HTML / CSV / JSON / ZIP pack
  -> CURRENT — REQUIRES CLIENT VALIDATION analyst review and client approval
  -> CANDIDATE human disposition / closure / reopening
  -> DEFERRED comparable local history + operational learning, pending client evidence
```

The arrows indicate accountability handoffs, not only data transformations.
Before calculation, the client supplies and approves facts and policy. During
calculation, PPAR owns deterministic treatment and safety. After calculation,
the client owns judgment, source correction, official reporting, and reliance.

### 17.6.2 Evidence progression

| Stage | Evidence object | Owner of factual meaning | Owner of transformation/decision | Required handoff |
|---|---|---|---|---|
| Export | Raw local snapshot files and export procedure | Source-extract administrator | Client source system/report process | Reproducible source/extract contract |
| Configuration | Mappings, roles, transaction semantics, method, tolerance, suppression | Client methodology/source owners | Authorized configuration owner | Attributable approved version |
| Normalization | Normalized row plus native source identity and locator | Client confirms mapping | PPAR applies deterministic mapping | Stable provenance and validation result |
| Difference | Added, removed, changed, or ambiguous record | Source facts remain client-owned | PPAR detects under explicit keys/tolerance | Finding fingerprint, match status, period context |
| Analytical treatment | `counted_cause` or `review_evidence` | Client policy defines allowed meaning | PPAR applies supported formula and ownership rules | Cause/evidence row, economic-effect ID, lineage |
| Integrity | Conservation, arithmetic, lineage, units, period, parity, determinism | Not a business opinion | PPAR owns enforcement | Valid bundle or stopped workflow |
| Review | Analytical status, Data Audit Issues, source detail | Evidence remains traceable | Qualified reviewer interprets | Review conclusion, challenge, or action request |
| Human disposition | Possible future accepted/correction/rerun/closure record | Authorized client reviewer | Client workflow authority | DEFERRED; attributable decision must remain distinct from calculation if later justified |
| History | Possible future comparable retained runs and dispositions | Client owns retention and comparability policy | No current product component | DEFERRED; provenance-preserving index only if later justified |

### 17.6.3 Handoff controls

- Every handoff identifies an artifact, owner, decision, and next actor.
- Missing optional evidence is a limitation, not evidence that no cause exists.
- A summary may point to detail; it may not replace the complete audit trail.
- A human decision may close workflow; it may not rewrite the immutable
  analytical result for that run.
- A correction produces a new source/configuration state and a new run. The old
  evidence pack remains retained under policy.
- External support or case-study use is a separate authorized export decision,
  never an implied consequence of normal operation.

## 17.7 Orthogonal status and failure model

### 17.7.1 Analytical status

**Capability status:** CURRENT — DEMONSTRATED; CURRENT — REQUIRES CLIENT
VALIDATION.

| Status | Meaning | What it does not mean |
|---|---|---|
| `Fully Explained` | Counted causes reconcile to the explained and reported performance difference within declared precision, including serialized display arithmetic. | It does not prove the source exports, mappings, vendor methodology, or official return are correct. |
| `Partly Explained` | Some material difference is explained, but a residual remains or potential cause cannot be counted under approved current policy. | It is not a failed run and does not authorize filling the residual with a human estimate. |
| `Unexplained` | No material difference is defensibly accounted for by counted causes within current scope. | It does not mean no real cause exists; it may indicate missing evidence, timing, unsupported methodology, or source-contract limits. |

The exact explanation-completeness metric and near-zero behavior remain for
Phase 3. No confidence score is introduced.

### 17.7.2 Readiness status

**Capability status:** CANDIDATE names over CURRENT — DOCUMENTED technical
controls; the unified experience is APPROVED DIRECTION.

| Status | Meaning | Allowed next step |
|---|---|---|
| `Ready` | Required source, policy, integrity, and output contracts pass for the requested scope. | Run under the approved context. |
| `Qualified Ready` | No unsafe condition exists, but explicitly optional missing evidence limits explanation depth. | Run only with visible qualification. |
| `Blocked` | Required data, meaning, policy, integrity, approval, or safe output boundary is unresolved. | Correct/narrow/approve, then revalidate. |

### 17.7.3 Workflow status

**Capability status:** DEFERRED pending business case and client validation.

| Status | Meaning |
|---|---|
| `Open` | Generated investigation awaits review. |
| `In Review` | A named reviewer is evaluating evidence. |
| `Accepted` | Authorized reviewer accepts the calculated explanation or valid restatement for the stated purpose. |
| `Correction Required` | Source-data or approved configuration must change. |
| `Rerun Required` | A new immutable run is needed after approved change. |
| `Closed` | Required action and approval are complete; analytical status remains unchanged. |
| `Reopened` | New evidence, disagreement, or later restatement requires renewed review. |

### 17.7.4 Data-quality finding status

The current report emits Data Audit issue rows but no durable lifecycle. A
future **CANDIDATE** lifecycle may distinguish `Open`, `Reviewed — Not
Confirmed`, `Confirmed Issue`, `Accepted Exception`, `Correction Required`, and
`Resolved in Rerun`. These states record human handling; they do not convert a
suspicious relationship into a source fact or an additive performance cause.

### 17.7.5 Source-contract/configuration error

**Capability status:** CURRENT — DOCUMENTED.

This is a fail-closed processing class, not an analytical or workflow status.
The affected workflow stops with actionable data/configuration guidance. A
manager cannot approve an unsafe run merely by marking the error accepted; the
source contract must be corrected, the scope safely narrowed, or an explicit
supported policy path used.

### 17.7.6 Internal logic failure

**Capability status:** CURRENT — DOCUMENTED.

Arithmetic, ownership, lineage, parity, determinism, or artifact inconsistency
is a product defect signal. Generation stops; the condition is never downgraded
to a reviewer warning. Any resulting partial artifacts are not an authoritative
evidence pack.

### 17.7.7 Status interaction rules

1. Readiness governs whether processing may begin.
2. Analytical status describes only calculated explanation for one immutable
   run.
3. Data-quality findings remain independent of additive cause status.
4. Workflow status describes human handling after a valid run.
5. Source-contract errors and internal logic failures interrupt processing and
   do not receive analytical status.
6. A rerun creates a new analytical result; it does not mutate the prior one.

## 17.8 Failure and exception paths

| Condition | Product classification | Required behavior | Accountable human next step | Capability status |
|---|---|---|---|---|
| Missing required performance file or configured required evidence file | Source-contract error | Stop before analysis; name file, dataset, snapshot, and requirement. | Source administrator supplies/corrects the extract or scope. | CURRENT — DOCUMENTED |
| Missing optional evidence | Qualification, not proof of absence | Continue only when the configured workflow is safe; state reduced explanation depth. | Analyst decides whether qualified scope is useful; source owner may add evidence. | CURRENT — DOCUMENTED behavior; CANDIDATE unified `Qualified Ready` status |
| Unknown field or accounting role | Source-contract error | Stop even if a suppression would match. | Source and methodology owners classify meaning/role. | CURRENT — DOCUMENTED |
| Incomplete impact policy | Source-contract error | Stop rather than choose additive, evidence-only, or suppressed treatment. | Methodology owner approves explicit policy. | CURRENT — DOCUMENTED |
| Ambiguous transaction semantics or missing required context | Source-contract error or review-only boundary | Block unsafe classification; use richer extract/REP semantics, explicit reviewed policy, or keep unsupported family outside counted treatment. | Source administrator and methodology owner resolve together. | CURRENT — DOCUMENTED; CURRENT — REQUIRES CLIENT VALIDATION |
| Duplicate or ambiguous identities | Source-contract error for unsafe comparison keys; visible unmatched evidence for allowed transaction ambiguity | Never fuzzy-match or collapse; label match status and retain rows. | Source owner supplies stable ID/context or reviewer handles unmatched evidence. | CURRENT — DOCUMENTED |
| Unsafe currency or unit treatment | Source-contract error | Stop affected workflow; require explicit base values and valid currency/quote direction. | Source/methodology owners correct mapping/data. | CURRENT — DOCUMENTED |
| Reversed, overlapping, or multiply assigned periods | Source-contract error | Stop; do not allow ambiguous dated evidence to own impact. | Source/methodology owner corrects scope/period contract. | CURRENT — DOCUMENTED |
| Report-size overflow | Source-contract/scope error | Stop before writing unusable reports; identify oversized table and contributors; never truncate silently. | Operator narrows scope or corrects upstream differences. | CURRENT — DOCUMENTED; CURRENT — REQUIRES CLIENT VALIDATION |
| Lost/broken bidirectional lineage | Internal logic error | Stop generation; invalidate partial bundle. | PPAR engineering diagnoses; client preserves inputs/configuration locally. | CURRENT — DOCUMENTED |
| Arithmetic, ownership, format-parity, or determinism failure | Internal logic error | Stop; never downgrade to a warning or adjust a tolerance to pass. | PPAR engineering fixes and reruns validation. | CURRENT — DOCUMENTED |
| Suspicious but interpretable source relationship | Visible review finding | Generate safely; show observed/reference values, tolerance, and explanation; do not count automatically. | Qualified reviewer determines whether it is an error or accepted condition. | CURRENT — DEMONSTRATED; CURRENT — REQUIRES CLIENT VALIDATION |
| Unsupported asset or transaction type | Source-contract error, review-only evidence, or deferred scope depending on safe observability | Do not infer treatment. Narrow, preserve as review evidence, or block. | Methodology/source owners provide evidence; product owner decides whether reusable support is justified. | CANDIDATE or DEFERRED unless current contract covers it |
| Source schema drift | Change-management event; often source-contract error | Detect missing/ambiguous columns or semantic change; revalidate and reapprove before trusted run. | Source administrator updates contract; methodology owner reapproves affected policy. | CURRENT — DOCUMENTED technical detection; APPROVED DIRECTION operating workflow |
| Client disagrees with a configured rule or cause | Human challenge, not direct override | Preserve result; identify whether disagreement concerns source fact, mapping, method, rule, or defect; approve correction and rerun if warranted. | Methodology owner decides business policy; PPAR support addresses product defect. | CURRENT — REQUIRES CLIENT VALIDATION; CANDIDATE managed challenge workflow |
| Evidence pack is missing, corrupted, or has invalid fingerprints | Internal logic/retention failure | Do not treat as authoritative; regenerate from preserved inputs or escalate. | Local administrator restores/reproduces; PPAR support investigates validator defects. | CURRENT — DOCUMENTED validation; CURRENT — REQUIRES CLIENT VALIDATION retention |
| External support needs sensitive evidence | Support-boundary exception | Keep local by default; use client-run probes or explicitly authorized minimal redaction. | Data owner and technology/security approve any transfer. | OPEN DECISION; CANDIDATE redaction tooling |

## 17.9 Phase 2 acceptance assessment

| Acceptance criterion | Assessment in v0.3 |
|---|---|
| Primary actors and decision rights are explicit. | Met as a first-pilot hypothesis. Founder and client validation remain required. |
| Current and future workflows are separated. | Met through exact capability-status taxonomy labels. |
| Conceptual architecture preserves local-first and safety invariants. | Met; no component weakens the twelve current invariant contracts. |
| Every important handoff has an owner and artifact. | Met conceptually; exact UI/storage fields remain Phase 3 work. |
| Human review and product calculation authority are distinct. | Met; direct calculated-result override is OUT OF SCOPE. |
| Failure paths are designed rather than implicit. | Met for the required Phase 2 exception set. |
| First-client workflow is realistic enough to inform a controlled pilot. | Met and founder-approved as a working model; client discovery remains required. |
| Open founder decisions are isolated and prioritized. | Met in Section 17.10. |
| Canonical document is updated and coherent. | Met and founder-approved on 2026-07-16; this remains the unversioned canonical working file. |

Phase 2 was approved by the founder on 2026-07-16. The founder also authorized
the recommended defaults in Section 17.10 as working assumptions for Phase 3.
This approval satisfies the Phase 3 gate; it does not convert client hypotheses
into validated market facts or settle later contractual/commercial policy.

## 17.10 Consequential assumptions and open founder decisions

### 17.10.1 Consequential assumptions used in this draft

1. A performance analyst is the primary daily user; portfolio accounting and
   operations specialists are essential collaborators.
2. A performance/operations manager is the client business-policy authority;
   PPAR support advises but does not approve site policy.
3. Portfolio-level audit is the first commercial conversation; security-level
   audit is drilldown and selected validation evidence.
4. Human notes are a possible long-term capability, not an assumed near-term
   requirement; infrastructure requires a serious business case and client
   validation. Human-calculated overrides remain OUT OF SCOPE.
5. Cross-run history is a possible long-term capability, not an assumed
   first-pilot requirement. If validated, it should begin with immutable local
   files and the smallest justified index before considering a database.
6. Licensing/support may exchange only explicitly approved non-portfolio
   metadata; routine portfolio-data or evidence transfer is prohibited.
7. Compliance/GIPS reviewers are secondary stakeholders; PPAR does not provide
   verification, compliance certification, or an independent audit opinion.

### 17.10.2 Founder-authorized Phase 3 working assumptions

1. **Primary pilot user and surface — APPROVED DIRECTION; CURRENT — REQUIRES
   CLIENT VALIDATION:** use the performance analyst as primary operator and
   portfolio-level audit as the first commercial surface, with security-level
   audit as drilldown and selected validation evidence.
2. **Human-supplied explanation — APPROVED DIRECTION concept; DEFERRED
   implementation:** attributable, separately labeled, non-additive notes are
   permissible only if client validation establishes a serious business case.
   Direct override of a calculated cause or analytical status remains OUT OF
   SCOPE.
3. **Initial local history — DEFERRED:** do not add a history index for the first
   pilot without a serious business reason and client validation. If recurring
   use later justifies the capability, begin with immutable versioned files and
   the smallest useful local index; database selection remains DEFERRED.
4. **Outbound metadata and exceptional support — OPEN DECISION:** Phase 3 may
   assume portfolio data remains local and that any transfer is explicit,
   minimal, and client authorized. Exact entitlement fields, offline grace, and
   redaction policy remain later commercial/security decisions.
5. **Reliance and liability posture — APPROVED DIRECTION for pilot design:**
   treat PPAR as operational decision support under explicit client authority,
   not independent assurance, certification, or a guarantee of official
   performance correctness. Final contractual language remains OPEN DECISION.

---

# 18. Phase 3A — Performance Change Investigation

## 18.1 Specification identity, purpose, and status

**Capability:** Performance Change Investigation  
**Primary user:** performance analyst  
**Primary first-pilot surface:** portfolio-level investigation, with
security-level drilldown  
**Overall status:** CURRENT — DEMONSTRATED; CURRENT — DOCUMENTED; CURRENT —
REQUIRES CLIENT VALIDATION; APPROVED DIRECTION for the history, disposition,
and stability extensions defined below

The capability answers one bounded question:

> Why did a reported portfolio or security return change between two declared
> source states, which source-data changes can PPAR quantify under approved
> policy, and what evidence remains for human review?

It does not decide whether Snapshot A or Snapshot B is correct, calculate the
client's official return, certify the source system, or provide independent
assurance. It compares two neutral snapshots and produces a deterministic,
evidence-preserving investigation.

This section is a product specification, not a Python package design. In the
requirements below:

- **MUST** identifies a safety, calculation, evidence, or product-boundary
  requirement that cannot be omitted without explicit product approval.
- **SHOULD** identifies the intended product behavior that may require a
  documented exception during client validation.
- **MAY** identifies permitted optional behavior that must not weaken a MUST.

Requirement identifiers use `PCI` for Performance Change Investigation. They
are intended to remain stable when implementation stories are created.

## 18.2 User problem, actors, and trigger

### 18.2.1 User problem

A performance analyst receives a newly calculated or newly reported return that
differs from a previously reported value. The analyst must establish:

1. which portfolio/security periods changed;
2. how large each reported change is;
3. which changed Modified Dietz inputs or supported source relationships explain
   it;
4. which rows are supporting evidence rather than additive causes;
5. which residuals, source limitations, or policy questions remain;
6. what another reviewer needs to reproduce and challenge the conclusion.

The product must reduce manual comparison and reconstruction without replacing
qualified judgment or hiding ambiguity.

### 18.2.2 Primary actors and authority

| Actor | Phase 3A authority |
|---|---|
| Performance analyst | Select an approved scope, operate the run, review evidence, challenge a result, and propose an action or disposition. |
| Portfolio-accounting or operations analyst | Explain source postings, reproduce extracts, and initiate authorized source correction. |
| Source/extract administrator | Approve factual extract definitions and resolve missing or changed source fields. |
| Performance/operations manager | Approve mappings, return basis, transaction semantics, materiality policy, and operational disposition. |
| Local product administrator | Maintain the local installation, approved configuration versions, access, output location, and retained runs. |
| Compliance/GIPS reviewer | Review methodology, error handling, evidence, and reliance under client policy; does not convert PPAR into an assurance provider. |
| PPAR support | Explain product behavior and diagnose defects from client-approved evidence; does not approve client accounting policy. |

### 18.2.3 Triggering events

The capability SHOULD be initiated when:

- a previously reported portfolio return changes;
- a security return or contribution changes and affects investigation;
- a corrected accounting snapshot is produced;
- a scheduled control compares the prior reported state with a new state;
- an analyst selects a known restatement for controlled validation;
- an approved source, mapping, methodology, or product change requires a rerun.

`PCI-001` — Every run MUST identify its trigger or investigation purpose in
local run metadata. CURRENT — DOCUMENTED metadata foundations; APPROVED
DIRECTION for a required human-readable trigger field.

## 18.3 Preconditions and investigation scope

### 18.3.1 Preconditions

Before an authoritative investigation bundle is generated:

1. Snapshot A and Snapshot B MUST be identified and accessible locally.
2. The requested portfolio/security scope and report level MUST be explicit.
3. Required datasets and normalized columns MUST satisfy the applicable source
   contract.
4. Field mappings and accounting roles MUST be complete for observed changed
   fields.
5. Transaction codes that can affect performance MUST have supported semantics
   from sufficient row context or approved local policy.
6. Return method, basis, flow timing, day count, inclusion rule, categories,
   sign convention, denominators, and impact treatments MUST be explicit where
   applicable.
7. Currency and unit relationships MUST be safe for every counted value.
8. Periods MUST be ordered, non-reversed, and unambiguously assign dated
   evidence.
9. The output scope MUST remain within report-size safety limits.

`PCI-002` — An unmet required precondition MUST produce a source-contract or
configuration error before authoritative artifacts are written. It MUST NOT be
represented as `Unexplained`. CURRENT — DOCUMENTED.

`PCI-003` — Missing explicitly optional evidence MAY permit a run only when the
remaining workflow is safe. The limitation MUST be visible and MUST NOT be
treated as evidence that no cause exists. CURRENT — DOCUMENTED; APPROVED
DIRECTION for a unified `Qualified Ready` handoff.

### 18.3.2 Scope object

The investigation scope MUST contain or resolve:

- comparison name and local run identifier;
- Snapshot A and Snapshot B labels and paths;
- requested portfolio IDs or approved all-portfolio scope;
- requested dates/periods;
- report level: `portfolio`, `security`, or both;
- approved configuration and product version;
- source/extract-contract identity;
- output location and selected artifact modes;
- when history is enabled, the prior run or reporting state to which this run is
  related.

`PCI-004` — PPAR MUST NOT infer that the newer file is correct merely because it
is Snapshot B. Snapshot labels express comparison direction, not authority.
CURRENT — DOCUMENTED.

## 18.4 Source contract, inputs, configuration, and permissions

### 18.4.1 Required and optional inputs

| Input | Requirement | Current product boundary |
|---|---|---|
| Portfolio performance | Required for portfolio-level investigation; requires portfolio, period, and reported return. | CURRENT — DEMONSTRATED; CURRENT — DOCUMENTED |
| Security performance | Required for security-level investigation; requires portfolio, security, period, and reported security return. | CURRENT — DEMONSTRATED; CURRENT — DOCUMENTED |
| Holdings | Required when beginning/ending value evidence or holding-based attribution is requested. | CURRENT — DEMONSTRATED; CURRENT — DOCUMENTED |
| Transactions | Required when dated-flow, income, fee, trade, or transaction attribution is requested. | CURRENT — DEMONSTRATED; CURRENT — DOCUMENTED |
| FX rates | Optional unless needed to establish an approved base-currency relationship or explain a configured FX treatment. | CURRENT — DEMONSTRATED evidence; CURRENT — DOCUMENTED boundary |
| Splits | Optional comparison/support dataset. | CURRENT — DEMONSTRATED evidence-only; not a current counted split cause |
| Source/extract context | Required when ambiguous transaction or field meaning cannot be established from the ordinary normalized row. | CURRENT — DOCUMENTED |
| Prior investigation/history record | Not required for a two-snapshot run; required only if repeated-restatement and stability features are later justified. | DEFERRED pending business case and client validation |

An absent optional dataset cannot create a finding about facts that were never
supplied. Product language MUST distinguish “not supplied,” “supplied with no
reportable difference,” and “supplied with a reportable difference.”

### 18.4.2 Required configuration

`PCI-005` — Configuration MUST explicitly define or resolve:

- snapshot paths, dataset filenames, and schema mappings;
- comparison level and return-field mapping;
- comparison tolerances by supported field family;
- accounting role for every observed comparable field;
- additive, evidence-only, cross-check-only, or suppressed treatment where
  policy is required;
- portfolio and security Modified Dietz policy where reconstruction or
  attribution uses it;
- transaction classification and sign/flow semantics;
- base-currency and quote-direction rules where applicable;
- output modes and report-size limit;
- suppressions with scope and rationale.

CURRENT — DOCUMENTED.

`PCI-006` — A suppression MUST NOT classify an unknown field, supply missing
transaction semantics, erase a reportable difference from `findings.csv`, or
turn review evidence into a counted cause. CURRENT — DOCUMENTED.

### 18.4.3 Configuration authority and permissions

- An authorized operator MAY run an approved configuration without obtaining a
  new approval for each routine execution.
- A source owner MUST approve factual mappings and extract changes.
- A methodology owner MUST approve return basis, transaction meaning, flow
  policy, denominators, timing, tolerances, and materiality.
- A local administrator MAY deploy approved configuration but MUST NOT gain
  methodology authority merely through file access.
- A reviewer MAY challenge a cause or add an attributable note in a future
  workflow, but MUST NOT directly overwrite calculated values.

Managed in-product permissions are CANDIDATE. Client filesystem and process
controls are the CURRENT — REQUIRES CLIENT VALIDATION operating boundary.

## 18.5 Core processing contract

`PCI-007` — A valid run MUST execute the following conceptual sequence:

1. resolve the approved scope, configuration, source contract, and versions;
2. load both snapshots without treating either as correct;
3. normalize configured datasets while preserving native identity and values;
4. establish conservative record identity and retain additions, removals, and
   ambiguity;
5. emit every reportable source difference after applicable field tolerance;
6. identify reportable portfolio/security return differences;
7. assign dated evidence to one unambiguous performance period and formula role;
8. calculate only supported, policy-approved return-impact estimates;
9. assign every reportable difference to `counted_cause` or `review_evidence`;
10. establish economic-effect ownership and prevent double counting;
11. calculate explained difference, residual, and analytical status;
12. run source-contract, currency/unit, period, continuity, conservation,
    lineage, arithmetic, parity, determinism, and report-size controls;
13. generate shared reviewer tables and serialize the selected XLSX, HTML, CSV,
    JSON, and ZIP surfaces;
14. validate the completed evidence pack before treating it as authoritative.

CURRENT — DEMONSTRATED; CURRENT — DOCUMENTED.

`PCI-008` — Processing MUST be deterministic for identical normalized inputs,
approved configuration, product/contract versions, and output options, excluding
only declared volatile metadata. CURRENT — DOCUMENTED.

## 18.6 Portfolio-level performance differences

### 18.6.1 Identity and calculation

The portfolio investigation key is:

```text
(portfolio_id, from_date, thru_date)
```

For a matched period:

```text
Performance Difference = Snapshot B Reported Return - Snapshot A Reported Return
```

`PCI-009` — A matched portfolio-period row MUST enter the primary review table
when its absolute reported-return difference exceeds the approved return
comparison tolerance. CURRENT — DEMONSTRATED; CURRENT — DOCUMENTED.

`PCI-010` — The portfolio primary row MUST include portfolio, from/thru dates,
performance difference, explained difference, unexplained difference when
nonzero at declared precision, analytical status, reviewer explanation, and a
stable review key. CURRENT — DEMONSTRATED.

`PCI-011` — Added, removed, duplicated, or ambiguously identified performance
periods MUST remain visible as source findings. PPAR MUST NOT invent a zero
return for the missing side to manufacture a numeric performance difference.
CURRENT — DOCUMENTED safety doctrine; edge behavior requires client validation.

### 18.6.2 Portfolio evidence boundary

Portfolio explanation MAY use supported changes in:

- beginning and ending portfolio value;
- dated investor external flows and weighted flows;
- configured income and fee treatment;
- holding market value and accrued value;
- transaction amount where approved as a performance input;
- security price/weight relationships where the supported method applies;
- other explicit formula inputs implemented under the current contract.

Reported portfolio components such as reported income, gain/loss, flow, or
market value MAY be compared and retained as diagnostics. They MUST NOT become
counted root causes merely because they correlate with the changed return.

`PCI-012` — Portfolio transaction wording MUST describe the portfolio-return
role. A buy, sell, income, fee, or external-flow row may support a changed cash
holding or formula input; the wording MUST NOT imply a separate additive effect
when that effect is already owned elsewhere. CURRENT — DEMONSTRATED.

## 18.7 Security-level performance differences

### 18.7.1 Identity and calculation

The security investigation key is:

```text
(portfolio_id, from_date, thru_date, security_id)
```

For a matched security period:

```text
Security Performance Difference = Snapshot B Security Return
                                  - Snapshot A Security Return
```

`PCI-013` — Security rows MUST remain scoped to the portfolio/security/period
container; a same-named security in another portfolio MUST NOT share causes or
status. CURRENT — DEMONSTRATED; CURRENT — DOCUMENTED.

`PCI-014` — The security primary row MUST include all portfolio-row fields plus
security ID. Review keys MUST include security ID. CURRENT — DEMONSTRATED.

`PCI-015` — The security report MAY include explicit no-security-difference rows
for changed portfolio periods to show that portfolio change exists without a
corresponding changed security return. Such rows MUST be labeled as no
difference and MUST NOT receive a fabricated analytical explanation. CURRENT —
DOCUMENTED.

### 18.7.2 Security evidence boundary

Security-level Modified Dietz treats buys and sells as capital flows into or out
of the security container; this does not make those trades investor external
flows at the portfolio level. Income and fee treatment depends on the approved
return basis. Cash securities may appear as security containers when present in
the reported security-performance source.

`PCI-016` — Security guidance MUST use the affected security-return role and
MUST remain distinct from portfolio-level flow language even when both reports
reuse one underlying finding. CURRENT — DEMONSTRATED.

`PCI-017` — Shared holdings, transaction, FX, and split findings MUST be compared
once when both report levels are requested, but each report level MUST apply its
own approved impact policy, review key, and economic-effect ownership. CURRENT —
DOCUMENTED.

## 18.8 Modified Dietz attribution boundary

### 18.8.1 Formula policy

The current onboarding method is Modified Dietz. At a conceptual level:

```text
Modified Dietz Return =
    (Ending Value - Beginning Value - Sum of External Flows)
    / (Beginning Value + Sum of Weighted External Flows)
```

Each external-flow weight depends on the approved timing field, calendar-day
count, and beginning-of-day or end-of-day inclusion rule. The client-approved
sign convention determines how source amounts enter the formula.

Primary standards material describes Modified Dietz as a daily-weighted
external-cash-flow estimate whose timing assumption must be defined and applied
consistently. It can be less accurate when large external flows coincide with
high volatility. PPAR therefore treats method, timing, flow classification, and
return basis as approved policy—not inferred facts—and does not claim that its
reconstruction is the client's official return.

`PCI-018` — The resolved policy MUST identify method, beginning/ending value
source, flow source, timing field, day count, inclusion rule, flow and income
categories, return basis, sign convention, and denominator source where
applicable. CURRENT — DOCUMENTED.

`PCI-019` — A source-row impact MAY be counted only when the row has a supported
method, sufficient numeric inputs, safe unit/currency basis, unambiguous period,
and approved policy. Otherwise it MUST remain review evidence or block when the
missing condition makes the workflow unsafe. CURRENT — DOCUMENTED.

### 18.8.2 Source-row attribution

Current supported attribution can promote:

- reconstructed beginning value;
- reconstructed ending value;
- net external flow;
- weighted external flow;
- configured income/fee amount;
- holding market-value or accrued-value change over an approved denominator;
- transaction amount change over an approved denominator;
- security transaction-flow Modified Dietz effect;
- price change relative to Snapshot A price and approved weight;
- other explicitly supported current methods named in the resolved contract.

`PCI-020` — Every numeric cause row MUST show the source change, impact method or
formula role, explained amount, report key, source lineage, safety disposition,
economic-effect ID, and counted owner. Some of these fields MAY remain in the
machine/detail artifact rather than the first-stop workbook, but they MUST be
available in the evidence pack. CURRENT — DEMONSTRATED; CURRENT — DOCUMENTED.

`PCI-021` — Formula rows derived from several source rows MUST use
`derived_formula` lineage and MUST retain links to the underlying source
findings. They MUST NOT masquerade as native source rows. CURRENT — DOCUMENTED.

### 18.8.3 Reconstruction versus explanation

Return reconstruction is a bounded diagnostic and formula-evidence surface.
The product MUST distinguish:

- changed official reported return;
- source-derived reconstructed return;
- change in the reconstructed return;
- source-row explanation of the reported difference;
- any residual between reported and explained change.

`PCI-022` — Agreement between reconstructed and reported changes strengthens
review evidence but does not prove source correctness. Disagreement MUST remain
visible and MUST NOT be forced into an additive cause. CURRENT — DOCUMENTED.

## 18.9 Cause families, roles, and grouping

### 18.9.1 Reviewer-facing row roles

| Row role | Meaning | Additive? | Status |
|---|---|---:|---|
| Formula Input | Derived Modified Dietz component retained to make formula coverage visible. | Yes, when selected as the counted representation. | CURRENT — DEMONSTRATED |
| Explained Cause | Source-row change with a defensible selected return-impact estimate. | Yes. | CURRENT — DEMONSTRATED |
| Supporting Evidence | Input component or linked fact that helps explain an owned effect. | No. | CURRENT — DEMONSTRATED |
| Review Context | Relevant changed fact without a supported additive interpretation. | No. | CURRENT — DEMONSTRATED |
| Possible Cause | Plausible row shown to direct review of an unresolved period. | No. | CURRENT — DEMONSTRATED |
| Diagnostic | Cross-check or implementation evidence. | No. | CURRENT — DOCUMENTED |

`PCI-023` — A row's display role MUST agree with its safety disposition. Only a
`counted_cause` owner may contribute to Explained Difference. Supporting,
possible, context, and diagnostic rows MUST have blank/non-additive explained
amounts. CURRENT — DOCUMENTED.

### 18.9.2 Coarse cause areas

Current helper tables group findings into coarse areas including:

- security return or contribution;
- market value or holding;
- transaction activity;
- FX rate;
- portfolio performance input;
- unexplained/other.

Classification/reference is present in the broader explanation vocabulary but
is not a distinct current dataset mapping in the coarse helper. Promoting it to
a separate reported cause area is a CANDIDATE and must remain non-additive
unless a supported method exists.

`PCI-024` — Cause-area grouping MAY summarize evidence and coverage, but the
group MUST NOT become an additional counted cause. Group totals MUST be derived
from the same selected row-level owners as the primary explanation. CURRENT —
DOCUMENTED.

`PCI-025` — Cause labels and stable finding codes SHOULD remain stable across
releases. New detail SHOULD be added through versioned fields or subcategories,
not by silently changing the meaning of an existing code. APPROVED DIRECTION
over current stable-code foundations.

## 18.10 Explanation completeness and analytical status

### 18.10.1 Required arithmetic

For one portfolio/security investigation key:

```text
Explained Difference   = sum(selected counted-cause impacts)
Unexplained Difference = Performance Difference - Explained Difference
```

`PCI-026` — Calculations MUST retain sufficient internal precision to avoid
classification changes caused by early rounding. Displayed six-decimal values
and serialized workbook values MUST independently reconcile. CURRENT —
DOCUMENTED.

The current workbook uses `0.0000005` in decimal-return units as the half-unit
six-decimal presentation threshold for treating the residual as zero. This is
an arithmetic/serialization precision, not business materiality and not the
field comparison tolerance.

### 18.10.2 Analytical statuses

| Status | Normative rule | Reviewer meaning |
|---|---|---|
| `Fully Explained` | A numeric performance difference and numeric explained total exist; the residual is within declared arithmetic precision; all displayed/serialized causes reconcile. | Current supported causes account for the reported change arithmetically. It does not prove source or policy correctness. |
| `Partly Explained` | At least one nonzero counted cause exists and a residual remains outside declared arithmetic precision. | Some of the change is quantified; review the residual and evidence limitations. |
| `Unexplained` | No nonzero counted cause defensibly accounts for the changed reported return, or no estimate is available under supported policy. | A real cause may exist, but PPAR cannot count one from current evidence and policy. |

`PCI-027` — These three statuses MUST describe only explanation arithmetic for a
valid immutable run. Workflow closure, management acceptance, or a human note
MUST NOT change them. CURRENT — DEMONSTRATED; APPROVED DIRECTION for managed
workflow separation.

`PCI-028` — Strict user-facing bundle generation MUST treat incomplete required
YAML or unsafe source policy as a configuration/source-contract error rather
than a fourth analytical status. The current internal compatibility label
`Missing YAML Specifications` MAY remain in diagnostic/test surfaces but MUST
NOT represent a successful authoritative investigation. CURRENT — DOCUMENTED
strict path; APPROVED DIRECTION for vocabulary consolidation.

### 18.10.3 Explanation-completeness measure

A percentage-like completeness metric is CANDIDATE and MUST NOT be introduced
until zero, near-zero, sign-reversing, and over-explained cases are specified.
If later adopted:

- it MUST be derived from immutable performance, explained, and residual values;
- it MUST preserve direction/sign information or explicitly explain its use of
  absolute values;
- it MUST not be labeled confidence;
- it MUST not replace the three analytical statuses;
- it MUST be suppressed or specially labeled when the denominator is too close
  to zero to be meaningful.

## 18.11 Comparison tolerance, arithmetic precision, and materiality

These controls are deliberately separate:

| Control | Purpose | Current status | Prohibited use |
|---|---|---|---|
| Field comparison tolerance | Determines whether a normalized A-versus-B field change becomes a reportable source difference. | CURRENT — DOCUMENTED | Must not be changed merely to make a test or investigation pass. |
| Return comparison tolerance | Determines whether a reported return change enters primary review. | CURRENT — DOCUMENTED | Must not stand in for business materiality. |
| Arithmetic/serialization precision | Determines whether calculated/displayed residual is zero for reconciliation. | CURRENT — DOCUMENTED | Must not be used to suppress source findings. |
| Operational materiality | Prioritizes review, escalation, approval, and management attention. | CANDIDATE | Must not erase findings or manufacture `Fully Explained`. |

`PCI-029` — Every reportable source difference MUST be defined independently of
finding severity and financial-statement materiality, consistent with SN-01.
CURRENT — DOCUMENTED.

`PCI-030` — A future materiality policy SHOULD support absolute return,
basis-point, currency-impact, relative-portfolio, and policy-specific thresholds
only when their basis and precedence are explicit. It MUST retain below-
materiality findings in the complete trail and SHOULD label them as lower
review priority. CANDIDATE.

`PCI-031` — Tolerance and materiality changes MUST be versioned, attributable,
approved, and applied prospectively to new runs. Historical runs MUST remain
interpretable under the values used when generated. APPROVED DIRECTION.

## 18.12 Supporting evidence, context, and possible causes

`PCI-032` — Every reportable source difference MUST have exactly one visible
safety disposition: `counted_cause` or `review_evidence`. Suppression metadata
is not a disposition. CURRENT — DOCUMENTED.

Supporting evidence includes, when linked and applicable:

- transaction quantity, price, or commission supporting transaction amount;
- holding quantity or price supporting holding value;
- FX rate supporting an explicit base-currency amount;
- split factor supporting holding quantity/value movement;
- reported components that confirm the location of a change;
- transaction match and semantics context;
- cross-check-only flow or reconstruction diagnostics.

`PCI-033` — A support row that describes the same economic effect as a counted
input MUST share or reference that economic-effect relationship and MUST NOT own
a second additive amount. CURRENT — DOCUMENTED.

`PCI-034` — Possible-cause guidance MUST use conditional language, show the
actual observed change, state what policy/input is missing, and direct a
reviewer action. It MUST NOT imply that correlation proves cause. CURRENT —
DEMONSTRATED; CURRENT — REQUIRES CLIENT VALIDATION for usefulness.

`PCI-035` — If a changed period has no visible cause or promoted evidence row,
the report MUST provide a diagnostic such as `no_underlying_causes_found` and
retain the period as `Unexplained`. CURRENT — DOCUMENTED.

## 18.13 Transaction identity and semantic boundaries

### 18.13.1 Identity

`PCI-036` — Transaction matching MUST prefer stable transaction ID. Exact
singleton fallback MAY be used only when the configured comparison fields
produce one unambiguous row on each side. Fuzzy matching is OUT OF SCOPE.
CURRENT — DOCUMENTED.

The complete evidence MUST distinguish at least:

- `matched_by_id`;
- `matched_by_singleton_fallback`;
- `added_in_snapshot_b`;
- `missing_from_snapshot_b`;
- `ambiguous_fallback_match`;
- `transaction_id_unmatched`;
- `strict_fallback_unmatched`.

`PCI-037` — Added, removed, and ambiguous transactions MUST remain visible and
MUST NOT be collapsed merely to improve match rates. CURRENT — DOCUMENTED.

### 18.13.2 Semantics

`PCI-038` — Transaction code alone MAY determine treatment only when the current
contract identifies that code/family as safe without additional context. An
ambiguous family MUST use required source/destination, security-type,
special-security, report, or approved local-policy evidence. CURRENT —
DOCUMENTED.

`PCI-039` — Packaged-demo coverage is evidence of one controlled story, not a
claim of universal site semantics. `by`, `sl`, `dv`, and ordinary `in` are the
strongest packaged starting families; other families remain context-gated,
narrow, test-only, or backlog according to the machine-readable matrix.
CURRENT — DOCUMENTED; CURRENT — REQUIRES CLIENT VALIDATION.

`PCI-040` — Portfolio investor external flows, security-level capital flows,
performance income/fees, neutral transfers, and corporate-action context MUST
remain distinct semantic categories. A security buy/sell flow MUST NOT be
treated as a portfolio investor external flow without separate evidence.
CURRENT — DOCUMENTED.

## 18.14 Reviewer-facing outputs and guidance

### 18.14.1 Required review path

The normal review sequence MUST be:

1. `Performance Differences` — identify changed periods, status, explained
   amount, and residual;
2. `Performance Difference Causes` — review counted causes and promoted support;
3. `Data Audit Issues` — review independent source-quality findings without
   assuming they caused performance;
4. `source_detail.csv` — inspect active finding-level source evidence;
5. declared support artifacts — inspect lineage, needs-review, context,
   transaction, cross-check, or reconstruction detail when needed.

`PCI-041` — XLSX, HTML, and promoted CSV surfaces MUST be derived from shared
validated tables. Presentation differences MAY improve ergonomics but MUST NOT
change financial meaning. CURRENT — DOCUMENTED.

`PCI-042` — The first-stop surfaces MUST use the same review keys and MUST allow
a reviewer to navigate from a changed performance row to cause/evidence rows.
CURRENT — DEMONSTRATED; APPROVED DIRECTION for stronger in-product navigation.

### 18.14.2 Guidance requirements

Reviewer guidance SHOULD answer:

- what changed;
- whether the row is counted, supporting, possible, context, or diagnostic;
- how a counted row enters the return explanation;
- why a row is not counted;
- what evidence or policy is missing;
- what the reviewer should do next;
- which detailed artifact supports the conclusion.

`PCI-043` — Guidance MUST preserve uncertainty. It MUST NOT use “caused” for a
review-only relationship unless the text clearly names the owned input effect
and does not imply a second additive impact. CURRENT — REQUIRES CLIENT
VALIDATION for language comprehension.

`PCI-044` — The report SHOULD prioritize unresolved or policy-limited periods
before fully explained periods, without hiding the latter. Current primary
review tables and the derived needs-review summary provide the technical
foundation; a separate first-screen executive entry layer, management
materiality, and exact ordering remain CANDIDATE.

## 18.15 Evidence, lineage, and immutable audit trail

`PCI-045` — Every native source finding MUST retain source dataset, snapshot,
source file, stable record locator, source column, native A/B values, normalized
delta where applicable, finding code, sequence, and fingerprint. CURRENT —
DOCUMENTED.

`PCI-046` — Every report cause MUST trace backward to one or more source finding
fingerprints or be explicitly typed as a derived formula row. Every counted
source finding MUST trace forward to its cause/report representation. CURRENT —
DOCUMENTED.

`PCI-047` — The evidence pack MUST retain complete `findings.csv`, cause lineage,
primary review tables, source detail, manifest v4, review summary v1, resolved
source context, and declared reviewer entrypoints. CURRENT — DEMONSTRATED;
CURRENT — DOCUMENTED.

`PCI-048` — A failed arithmetic, lineage, ownership, parity, or determinism check
MUST invalidate the authoritative pack. Partial files MUST NOT be presented as a
successful investigation. CURRENT — DOCUMENTED.

`PCI-049` — A rerun after correction MUST create a new immutable result. The
prior run MUST remain retained under client policy and MUST NOT be edited to
look as though it used the corrected source/configuration. APPROVED DIRECTION
over current deterministic bundle foundations.

## 18.16 Analytical and workflow transitions

**Implementation status:** DEFERRED. The model below preserves a possible
long-term boundary; it is not a first-pilot infrastructure requirement. A
serious business case and client validation must precede implementation.

Analytical status is calculated during a valid run and is immutable for that
run. A future human workflow MAY then move through:

```text
Open -> In Review -> Accepted
                  -> Correction Required -> Rerun Required -> Closed
                  -> Closed
Closed -> Reopened
```

`PCI-050` — If workflow infrastructure is later justified, transitions MUST
record actor, timestamp, reason, evidence reference, and approver where
required. They MUST NOT change calculated causes, residual, or analytical
status. DEFERRED pending business case and client validation.

`PCI-051` — If client evidence later justifies human-note infrastructure, a
human-supplied explanation MAY be stored as an attributable, separately
labeled, non-additive note. It MUST identify its author, date, evidence, and
approval; it MUST NOT enter Explained Difference. The conceptual boundary is
APPROVED DIRECTION; implementation is DEFERRED.

`PCI-052` — Direct override of a calculated cause, amount, or analytical status
is OUT OF SCOPE. Correct source-data, configuration, or product logic and rerun.

## 18.17 Failure behavior and edge cases

| Condition | Required Phase 3A behavior | Status |
|---|---|---|
| Missing required performance dataset/column | Stop requested report level with actionable source-contract error. | CURRENT — DOCUMENTED |
| Optional holdings/transaction/FX/split evidence absent | Continue only if safe; state reduced evidence depth; never infer no cause. | CURRENT — DOCUMENTED |
| One-sided performance period | Preserve addition/removal finding; do not invent missing return or numeric difference. | CURRENT — DOCUMENTED boundary; CURRENT — REQUIRES CLIENT VALIDATION |
| Duplicate/ambiguous performance identity | Stop unsafe primary comparison; name conflicting key and source rows. | CURRENT — DOCUMENTED |
| Unknown changed field/role | Stop before bundle; suppression cannot bypass. | CURRENT — DOCUMENTED |
| Missing impact policy | Stop strict authoritative bundle; diagnostic mode may expose setup gap. | CURRENT — DOCUMENTED |
| Ambiguous transaction semantics | Require context or approved local policy; otherwise block or keep outside counted treatment where safe. | CURRENT — DOCUMENTED |
| Foreign nonzero value without explicit base counterpart | Do not count; stop affected workflow when required for explanation. | CURRENT — DOCUMENTED |
| Invalid/reversed FX quote direction | Stop affected workflow. | CURRENT — DOCUMENTED |
| Zero, near-zero, or invalid denominator | Withhold estimate, expose limitation, and never divide through it. | CURRENT — DOCUMENTED |
| Flow exactly on period boundary | Apply approved timing/inclusion policy once; prevent multiple-period ownership. | CURRENT — DOCUMENTED |
| Reversed/overlapping/multiple period assignment | Stop with source-contract error. | CURRENT — DOCUMENTED |
| Large flow during high volatility | Preserve Modified Dietz limitation and recommend source/methodology review; do not claim exact official cause from approximation alone. | APPROVED DIRECTION; CURRENT — REQUIRES CLIENT VALIDATION |
| Gross/net fee-basis mismatch | Block or require corrected return-basis policy; never assume fees are additive. | CURRENT — DOCUMENTED |
| Several rows describe one cash/value effect | Select one counted owner and retain others as support. | CURRENT — DOCUMENTED |
| Explained total overshoots or reverses the reported difference | Preserve arithmetic and classify by residual; direct reviewer attention; do not cap the amount to 100%. | CURRENT — DOCUMENTED arithmetic; reviewer design CANDIDATE |
| Display rounding creates a residual | Allocate presentation residual only under deterministic reconciliation rules; preserve full-precision evidence. | CURRENT — DOCUMENTED |
| No underlying cause rows | Emit visible diagnostic and `Unexplained` primary row. | CURRENT — DOCUMENTED |
| Report-size limit exceeded | Stop before unusable artifacts are written and identify the oversized table/scope. | CURRENT — DOCUMENTED |
| Bundle fingerprint/parity failure | Invalidate pack and treat as internal logic error. | CURRENT — DOCUMENTED |
| Reviewer disputes policy or cause | Preserve run; classify dispute; approve correction and rerun if warranted. | APPROVED DIRECTION |

## 18.18 Scale and performance considerations

`PCI-053` — Scale optimization MUST preserve identical financial meaning,
finding disposition, ownership, lineage, status, and reviewer-visible ordering.
No safety invariant may be disabled for performance. CURRENT — DOCUMENTED.

`PCI-054` — Shared source findings and reconstruction inputs SHOULD be computed
once when portfolio and security reports are generated together. Report-level
policy MUST still be applied separately. CURRENT — DOCUMENTED.

`PCI-055` — The product MUST fail before producing an unusable report when row
volume exceeds the supported output boundary. Future pagination, partitioning,
or portfolio selection MAY extend the boundary, but silent truncation is OUT OF
SCOPE. CURRENT — DOCUMENTED; CANDIDATE for richer scoping.

`PCI-056` — Production runs SHOULD enforce inexpensive conservation, lineage,
financial-input, and explanation-reconciliation checks. Redundant full-artifact
reparsing MAY remain in validators and release gates when production cost would
materially degrade operation. CURRENT — DOCUMENTED project doctrine.

## 18.19 Local-first, security, and privacy

`PCI-057` — Snapshot files, normalized rows, portfolio/security identifiers,
calculated findings, reports, and evidence packs MUST remain within the
client-controlled environment during ordinary operation. APPROVED DIRECTION;
CURRENT — DOCUMENTED local execution; CURRENT — REQUIRES CLIENT VALIDATION.

`PCI-058` — Output locations and access MUST be client controlled. PPAR MUST NOT
silently transmit investigation content, usage evidence, or portfolio metadata.
APPROVED DIRECTION.

`PCI-059` — External support SHOULD begin with stable local error messages,
client-run validators, version/configuration metadata, and non-sensitive
diagnostics. Any redacted evidence transfer requires explicit case-specific
authorization. Exact tooling and support retention remain OPEN DECISION.

`PCI-060` — Licensing or update mechanisms MAY exchange only approved
non-portfolio metadata. Exact fields and offline grace remain OPEN DECISION and
are not prerequisites for the Phase 3A calculation design.

## 18.20 Repeated-restatement history

**Capability status:** APPROVED DIRECTION only as a possible long-term product;
implementation is DEFERRED until a serious business reason and client
validation justify the added infrastructure. No current persistent
investigation-history product exists.

### 18.20.1 Purpose and initial storage

Repeated-restatement history connects immutable two-snapshot investigations so
the client can determine whether a period changed once, changed repeatedly, was
corrected, or was reopened by later evidence.

No history index is required for the first pilot. If validated recurring use
later establishes the business case, the least-complex starting design is
versioned local run directories plus the smallest useful local index. A
database remains DEFERRED.

`PCI-061` — Each history entry MUST reference, without duplicating or mutating:

- local run ID and creation time;
- scope and comparison level;
- Snapshot A/B labels and capture/extract provenance;
- product, manifest, normalization, configuration, source-contract, and rule
  versions/fingerprints;
- primary investigation keys and analytical statuses;
- performance, explained, and residual values;
- link to the immutable bundle;
- relationship to prior/superseded run;
- human workflow disposition when available.

DEFERRED conditional specification.

`PCI-062` — The product MUST NOT infer chronological authority solely from
filename, directory order, or Snapshot B. If history is implemented, the
operator/source contract must declare capture time and relationship to the
prior reported state. DEFERRED conditional specification.

### 18.20.2 Comparability and recurrence

A run is comparable for recurrence only when portfolio/security identity,
period basis, report level, return basis/method, currency basis, and material
configuration semantics are compatible. Product-version differences MAY be
allowed only when the relevant contracts are declared comparable.

`PCI-063` — If history is implemented, incomparable runs MUST be labeled and
excluded from recurrence or trend metrics; they MUST NOT be coerced into a
continuous series. DEFERRED conditional specification.

Candidate recurrence labels are:

- `New Restatement` — first comparable changed result for the key;
- `Repeated Restatement` — a later source state changes the same key again;
- `Unchanged Since Prior Audit` — same reported value/status under comparable
  scope;
- `Resolved in Rerun` — correction produces the accepted expected result;
- `Reopened` — later evidence or restatement returns a closed key to review;
- `Incomparable` — method, scope, identity, or policy prevents a valid trend.

These labels are CANDIDATE until client workflow validates their meaning.

## 18.21 Portfolio stability

**Capability status:** APPROVED DIRECTION only as a possible long-term
capability; implementation is DEFERRED until cross-run history has a validated
business case. Metrics and presentation remain CANDIDATE after that gate.

Portfolio stability is a descriptive view of comparable audit history. It is
not return volatility, investment risk, source-system quality certification, or
an employee score.

Potential measures include:

- number and percentage of previously reported periods restated;
- count of repeated restatements for the same period;
- absolute and signed magnitude of return changes;
- frequency of `Partly Explained` and `Unexplained` outcomes;
- age and recurrence of unresolved residuals;
- repeated supported cause areas and Data Audit finding families;
- time/runs from detected issue to accepted rerun where workflow data exists.

`PCI-064` — Every stability measure MUST disclose its population, period,
comparison basis, materiality/tolerance version, and excluded incomparable runs.
CANDIDATE.

`PCI-065` — No composite “stability score,” ranking, traffic-light threshold, or
confidence score may be introduced until validated with real client history and
approved for interpretation. CANDIDATE; otherwise DEFERRED.

`PCI-066` — Stability views MUST link to underlying immutable investigations and
MUST preserve honest residuals. They MUST NOT imply blame or predict source
correctness from frequency alone. APPROVED DIRECTION.

## 18.22 Validation and safety invariants

Phase 3A inherits all twelve current safety invariants. The most direct mappings
are:

| Invariant | Phase 3A acceptance implication |
|---|---|
| SN-01 No lost differences | Every reportable source difference has a visible disposition and remains in the complete trail. |
| SN-02 No double counting | Each economic effect has at most one counted owner per report key. |
| SN-03 Fully Explained arithmetic | Full-precision, displayed, and serialized cause totals reconcile. |
| SN-04 Beginning/end continuity | Discontinuity is a mandatory visible Data Audit finding, not silently repaired. |
| SN-05 Bidirectional lineage | Source-to-report and report-to-source traceability both pass. |
| SN-06 Currency/unit consistency | Unsafe monetary inputs fail closed and cannot enter attribution. |
| SN-07 Period-boundary safety | Dated evidence owns at most one unambiguous period/formula role. |
| SN-08/09 Demo preservation/isolation | Packaged scenario intent remains controlled and test fixtures do not become product claims. |
| SN-10 Report parity | XLSX, HTML, and CSV meanings agree. |
| SN-11 Deterministic output | Stable content/fingerprints repeat outside declared volatility. |
| SN-12 Fail-closed policy | Unknown roles, semantics, or required impact policies block authoritative output. |

`PCI-067` — A Phase 3A implementation change is unacceptable if it weakens any
invariant, converts an internal logic error to a warning, raises a threshold to
hide a regression, or removes evidence to improve presentation. CURRENT —
DOCUMENTED project gate.

## 18.23 Acceptance criteria

Phase 3A is functionally acceptable only when all applicable criteria pass:

1. **Scope and identity:** portfolio and security keys are explicit; additions,
   removals, duplicates, and ambiguity are handled without invented matches.
2. **Arithmetic:** performance, explained, and residual values reconcile at
   internal, display, and serialized precision.
3. **Status:** valid runs use the three analytical statuses consistently;
   configuration failure is not mislabeled as unexplained performance.
4. **Attribution:** every counted cause has supported method, inputs, policy,
   units, period, lineage, and one economic-effect owner.
5. **Evidence:** every reportable difference remains available as counted cause
   or review evidence, including suppressions in the complete trail.
6. **Level separation:** portfolio and security reports reuse shared findings
   without sharing incompatible policy or double counting.
7. **Transactions:** identity and semantic boundaries remain conservative and
   context gated.
8. **Outputs:** primary XLSX/HTML/CSV tables agree and the manifest/review-summary
   contract validates.
9. **Reviewer guidance:** a qualified reviewer can move from changed return to
   counted cause, support, residual, and next action without founder coaching.
10. **Failure behavior:** source-contract and internal logic failures stop
    safely with actionable classification.
11. **Local-first:** ordinary operation requires no portfolio-data transfer.
12. **Client validation:** real-client evidence supports mappings, semantics,
    method interpretation, usefulness, and supportability before stronger
    commercial claims.

The first nine are CURRENT — DEMONSTRATED or CURRENT — DOCUMENTED within the
packaged scope but remain CURRENT — REQUIRES CLIENT VALIDATION. History,
workflow, and stability criteria are APPROVED DIRECTION or CANDIDATE and are not
release claims.

## 18.24 Dependencies and approved working assumptions

### Dependencies

- approved Phase 2 actors, decision rights, and workflow;
- client-controlled deployment boundary;
- approved source/extract contract and mappings;
- field-role, transaction-semantics, and impact-policy contracts;
- Modified Dietz reconstruction and attribution components;
- safety-invariant and bundle-validation contracts;
- current review workbook/HTML/CSV surfaces;
- Phase 3C readiness design for a unified preflight experience;
- Phase 3F human disposition design for managed review workflow;
- local history capability before stability measures.

### Founder-approved Phase 3A working assumptions

The founder approved Phase 3A on 2026-07-16 with these working assumptions:

1. **Explanation precision — APPROVED DIRECTION:** retain a fixed product
   contract rather than exposing routine client configuration. Any future
   change remains change controlled and must preserve internal, display, and
   serialization reconciliation.
2. **Operational materiality — APPROVED DIRECTION; CURRENT — REQUIRES CLIENT
   VALIDATION:** use materiality only to prioritize visible work, never to
   suppress a reportable difference or change explanation arithmetic. Initial
   pilots may use client-approved basis-point thresholds; unresolved,
   recurring, continuity, integrity, and policy-ambiguity conditions take
   precedence over magnitude alone. Exact thresholds require client approval
   and validation.
3. **Explanation-completeness percentage — DEFERRED:** retain analytical status,
   full-precision explained amount, and residual without an initial percentage.
4. **Pilot report scope — APPROVED DIRECTION; CURRENT — REQUIRES CLIENT
   VALIDATION:** generate the portfolio report in every pilot. Generate the
   security report when the selected case or reviewer workflow requires
   security-level drilldown; do not require it mechanically for every run.

Human-note/workflow infrastructure and local-history infrastructure remain
DEFERRED until a serious business case and client validation justify their
complexity. These assumptions do not authorize direct calculated-result
override, hidden evidence, or weaker safety behavior.

## 18.25 Real-client validation plan

### Stage 1 — Source and policy discovery

- obtain client-authorized representative portfolio and security extracts;
- identify export/report provenance and reproducibility;
- map required/optional fields and transaction context;
- document official return method, basis, timing, day count, flow policy,
  currency basis, tolerances, and known restatements;
- identify source, methodology, and approval owners.

**Exit evidence:** approved source/extract contract and configuration draft with
no guessed semantics.

### Stage 2 — Known-case shadow validation

- run known valid restatements and known source errors;
- compare PPAR differences and causes with client analyst conclusions;
- trace every material cause to recognized source evidence;
- review false positives, missed expected evidence, residuals, and language;
- confirm portfolio-first and security-drilldown usability.

**Exit evidence:** case-by-case validation record showing correct, limited, and
incorrect outcomes without hiding failures.

### Stage 3 — Independent analyst operation

- have a client analyst execute setup/run/review with bounded support;
- measure time, interventions, configuration changes, and unresolved questions;
- have a second reviewer reproduce the evidence path;
- validate local retention and support boundaries.

**Exit evidence:** another authorized operator can produce and explain a trusted
bundle without founder-led interpretation.

### Stage 4 — Recurring-use validation

- repeat across reporting cycles or multiple source states;
- test configuration/version change management;
- determine whether recurrence creates enough client value to justify any
  history infrastructure; only then validate comparability concepts or
  stability metrics;
- measure review value, noise, support cost, and repeatability.

**Exit evidence:** justified decision on repeatable product scope, local history,
and the next commercial claim boundary.

## 18.26 Claims supported and not supported

### Claims the capability can support now, with qualification

- PPAR compares two configured portfolio-accounting snapshots and identifies
  changed reported portfolio/security returns.
- Within the current supported contract, PPAR quantifies selected Modified Dietz
  input changes and preserves supporting evidence and residuals.
- PPAR separates counted causes from review evidence and enforces arithmetic,
  ownership, lineage, parity, determinism, and fail-closed controls.
- PPAR produces reproducible local review and evidence-pack artifacts for the
  packaged Axys/APX-style scope.

Each claim remains CURRENT — REQUIRES CLIENT VALIDATION for a real site.

### Claims the capability does not support

- that Snapshot B, the client's official return, or every source value is
  correct;
- that all performance changes are automatically explained;
- that PPAR reconstructs a complete accounting ledger;
- that one packaged transaction story establishes universal Axys/APX semantics;
- that `Fully Explained` means independently verified, compliant, or free of
  source/configuration error;
- that Modified Dietz is exact under every flow/volatility pattern or matches
  every vendor methodology;
- that portfolio stability or repeated-restatement history is a current
  implemented product;
- that PPAR provides an audit opinion, certification, or assurance.

## 18.27 Phase 3A approval and next gate

Phase 3A was founder-approved on 2026-07-16 with the four working assumptions
in Section 18.24. That approval authorized Phase 3B drafting; it did not
authorize application-code changes or begin Phase 3C.

---

# 19. Phase 3B — Performance Data Quality Audit

## 19.1 Specification identity, purpose, and status

**Capability:** Performance Data Quality Audit

**Primary user:** performance or investment-operations analyst

**Primary first-pilot surface:** the Data Audit Issues section inside the
portfolio audit bundle

**Overall status:** CURRENT — DEMONSTRATED; CURRENT — DOCUMENTED; CURRENT —
REQUIRES CLIENT VALIDATION; CANDIDATE for explicit rule-execution coverage and
summary behavior pending founder review

This capability answers a different question from Phase 3A:

> Which relationships inside each declared source state look internally
> inconsistent, incomplete, duplicated, or discontinuous, and what evidence
> should a qualified reviewer examine?

It does not decide that a value is wrong, determine which snapshot is correct,
or automatically turn a suspicious relationship into a performance cause.
The current implementation evaluates transparent checks against normalized
Snapshot A and Snapshot B inputs and publishes evidence rows for review.

This section follows the same `MUST`, `SHOULD`, and `MAY` meanings defined in
Section 18.1. Requirement identifiers use `PDQ` for Performance Data Quality.

`PDQ-001` — A Data Audit result MUST be described as a rule-based review
finding, not as proof of error, source-system failure, control deficiency,
financial misstatement, or audit conclusion. CURRENT — DOCUMENTED.

`PDQ-002` — Data Audit MUST remain analytically separate from the additive
Performance Change Investigation calculation. CURRENT — DEMONSTRATED; CURRENT
— DOCUMENTED.

`PDQ-003` — Initial commercial claims MUST remain limited to the configured
Axys/APX-style source contract and MUST retain CURRENT — REQUIRES CLIENT
VALIDATION until real client cases establish rule utility, tolerances, and
false-positive behavior.

## 19.2 User problem, actors, trigger, and decision authority

### 19.2.1 User problem

A reviewer may have a valid performance-change explanation while the underlying
extract still contains a suspicious price relationship, duplicate transaction,
apparently missing dividend, inconsistent rate, or period discontinuity.
Conversely, a Data Audit issue may be operationally harmless and unrelated to
the changed return. The product must expose both truths without conflating them.

`PDQ-004` — The capability MUST help a reviewer answer:

1. which rule observed the condition;
2. in which snapshot, portfolio, date, dataset field, and security it occurred;
3. what values or absence triggered it;
4. what tolerance or comparison basis was applied;
5. whether the rule was mandatory, enabled, disabled, inapplicable, or unable
   to evaluate; and
6. what the reviewer should validate before concluding that the condition is
   an error.

Items 1–4 are CURRENT — DEMONSTRATED in finding detail. Item 5 is only partly
visible today and is CANDIDATE for rule-execution coverage. Item 6 is
CURRENT — DOCUMENTED in rule explanations and setup comments but requires
client validation.

### 19.2.2 Actors and authority

| Actor | Phase 3B responsibility | Authority boundary |
|---|---|---|
| Performance or operations analyst | Reviews issue rows, validates source context, and escalates likely errors | May not rewrite PPAR calculations or source records through PPAR |
| Methodology owner | Approves rule meaning, tolerance, and any performance-method interpretation | Owns client policy, not PPAR implementation |
| Source/extract administrator | Explains export construction, identifiers, fields, and known valuation overrides | Owns extract facts, not performance conclusions |
| Performance or operations manager | Approves pilot scope, review priority, and operational response | Does not convert a rule finding into independent assurance |
| Local technology administrator | Controls installation, configuration-file access, execution, and retention | Does not approve financial semantics unless separately authorized |
| PPAR | Executes declared checks deterministically and preserves evidence | Does not judge official correctness or alter the book of record |

`PDQ-005` — Rule enablement, filters, and tolerances that affect a client run
MUST be attributable to an authorized client policy owner and retained with the
run configuration. APPROVED DIRECTION; the current local YAML is retained, but
the product has no role-based approval workflow.

### 19.2.3 Trigger

The current trigger is a normal `ppar audit` execution after a valid local
workspace and two snapshots have been declared. The same Data Audit table is
reused in portfolio and security bundles generated by that execution.

`PDQ-006` — Phase 3B MUST NOT require a changed reported return before a rule
can find an issue. Checks operate on the declared source states, not only on
rows that differ between Snapshot A and Snapshot B. CURRENT — DEMONSTRATED.

A standalone Data Audit command is CANDIDATE. It is not required for the first
pilot and must not be described as current.

## 19.3 Boundary from Performance Change Investigation

Data Audit and Performance Change Investigation share normalized inputs and
bundle infrastructure but create different evidence:

| Dimension | Performance Change Investigation | Performance Data Quality Audit |
|---|---|---|
| Primary question | Why did reported performance change? | What source relationship looks suspicious? |
| Population | Reportable differences between A and B | Each snapshot's eligible rows, across the union of A and B |
| Quantification | Selected supported effects on the return difference | Difference, range, rate, count, absence, or continuity evidence |
| Counted in explained amount | Yes, only for supported owned causes | Never merely because a Data Audit finding exists |
| Normal output | Performance Differences and Performance Difference Causes | Data Audit Issues |
| Correctness conclusion | Not established | Not established |

`PDQ-007` — A Data Audit finding MUST NOT add to, subtract from, or otherwise
change `Performance Difference Explained`, the residual, or the Phase 3A
analytical status. CURRENT — DEMONSTRATED; protected by report-table separation.

`PDQ-008` — When the same underlying source fact is relevant to both
capabilities, each representation MUST retain its own role and lineage. The
Phase 3A cause may count a supported economic effect once; the Phase 3B row
remains independent review evidence. APPROVED DIRECTION consistent with SN-02
and the current table model.

`PDQ-009` — A `Fully Explained` performance difference MUST NOT suppress,
downgrade, or close an independent Data Audit finding. Likewise, a Data Audit
finding MUST NOT force a valid performance-change residual. APPROVED DIRECTION.

## 19.4 Preconditions, source contract, inputs, and configuration

### 19.4.1 Preconditions

The current capability depends on:

- a resolved local audit configuration;
- declared Snapshot A and Snapshot B directories;
- normalized portfolio-performance and, when present, security-performance,
  holdings, and transaction datasets;
- valid identifiers, dates, numeric types, and currency/unit treatment required
  by the applicable source contract;
- client-reviewed transaction-code context for rules that use transaction
  families; and
- enough eligible peer rows or adjacent periods for the selected rule.

`PDQ-010` — Required source-contract failures MUST be resolved before PPAR
publishes an authoritative Data Audit result. A malformed required input is not
equivalent to “no issue detected.” CURRENT — DOCUMENTED.

`PDQ-011` — Optional dataset absence or insufficient comparison population MUST
not be represented as a clean test. The target rule-execution summary MUST
distinguish `No Issue Detected` from `Not Evaluated — Input Unavailable or
Insufficient`. CANDIDATE; the current detail table does not expose
this distinction.

### 19.4.2 Required and optional inputs

| Input | Use in Phase 3B | Status |
|---|---|---|
| Portfolio performance | Mandatory portfolio continuity | CURRENT — DEMONSTRATED |
| Security performance | Security continuity when the dataset is available | CURRENT — DEMONSTRATED |
| Holdings | Holdings price, missing-dividend qualification, accrued-rate, and rate-quantity fallback | CURRENT — DEMONSTRATED |
| Transactions | Duplicate, transaction-price, dividend-rate, missing-dividend, and PA/SA-rate checks | CURRENT — DEMONSTRATED |
| Split evidence | No current Phase 3B rule | CANDIDATE only |
| Snapshot labels and configuration | Scope, enablement, filters, and tolerances | CURRENT — DEMONSTRATED |

`PDQ-012` — Optional inputs MUST be described as optional evidence, not silently
treated as comprehensive source coverage. A rule that needs an absent dataset
cannot establish that the relevant condition does not exist. APPROVED
DIRECTION.

### 19.4.3 Current configuration surface

The current `data_audit_checks` YAML supports:

- top-level `enabled` for the seven optional checks;
- per-check `enabled`;
- per-check `only` and `exclude` exact-match row filters;
- `absolute_tolerance` and `percent_tolerance` for numeric comparison checks;
  and
- separate continuity tolerance blocks keyed by
  `portfolio_market_value_continuity` and
  `security_market_value_continuity`.

Every optional check defaults to enabled unless an exact Boolean `false` is
provided. Top-level disablement does not disable continuity.

`PDQ-013` — The generated evidence pack SHOULD make the effective rule set,
filters, tolerances, source-contract version, and configuration fingerprint
easy to identify without asking the operator to reconstruct defaults.
CANDIDATE; current bundles identify the comparison path and source context and
preserve the result table, but they do not embed an effective rule summary or
complete configuration snapshot.

`PDQ-014` — Malformed rule configuration MUST eventually fail validation with
an actionable location rather than silently falling back to a default.
CANDIDATE and current gap: some non-Boolean enablement values,
non-numeric tolerances, or malformed filter shapes currently fall back or are
ignored by the rule evaluator. No application-code change is authorized by
this specification.

## 19.5 Current rule catalog

The current implemented catalog contains seven optional checks and two
mandatory continuity issue types. All are CURRENT — DEMONSTRATED; CURRENT —
DOCUMENTED; CURRENT — REQUIRES CLIENT VALIDATION.

| Rule / issue type | Current logic | Current output meaning |
|---|---|---|
| `holdings_price_range` | Within one snapshot, compare same-date, same-security `holdings.price` values across eligible portfolios | Emits the minimum, maximum, range, and tolerance for every participating row when the range is over tolerance |
| `transactions_price_range` | Within one snapshot, compare same-date, same-security `transactions.price` values across eligible portfolios | Emits the minimum, maximum, range, and tolerance for every participating row |
| `duplicate_transactions` | Within one snapshot and portfolio, group exact same date, security, code, amount, quantity, and price | Emits every row in a group of two or more and reports the group count |
| `dividend_rate` | Within one snapshot, compare same-date, same-security `dv` amount-per-unit rates across portfolios | Emits min/max rates and tolerance; uses transaction quantity or a holdings fallback |
| `missing_dividend` | Find portfolios conservatively appearing to hold a security across a date when another portfolio has a same-day `dv` | Emits the apparently missing portfolio and the portfolio(s) where the dividend exists |
| `pa_sa_rate` | Within one snapshot, compare same-date, same-security amount-per-unit rates across portfolios, separately for `pa` and `sa` | Emits min/max rate and tolerance for participating rows |
| `holdings_accrued_rate` | Within one snapshot, compare same-date, same-security `holdings.accrued / abs(quantity)` across portfolios | Emits min/max rate and tolerance for participating rows |
| `portfolio_market_value_continuity` | Within one snapshot and portfolio, compare an adjacent prior period's ending market value with the next period's beginning market value | Emits prior ending value, next beginning value, difference, tolerance, and SN-04 explanation |
| `security_market_value_continuity` | Same continuity test at portfolio/security grain | Emits the security-grain SN-04 evidence |

`PDQ-015` — The rule identifiers above MUST remain machine-readable and
semantically stable or be versioned with an explicit migration. APPROVED
DIRECTION.

`PDQ-016` — The current catalog MUST NOT be described as comprehensive data
validation. It samples specific relationships relevant to portfolio-
performance review. CURRENT — DOCUMENTED.

`PDQ-017` — Split plausibility, FX anomalies, stale prices, identifier changes,
broader corporate actions, return outliers, and other rule families remain
CANDIDATE for Phase 4 unless separately implemented, tested, and approved.

## 19.6 Rule execution model

The current execution sequence is:

1. resolve the comparison specification and normalized loaders;
2. load Snapshot A and Snapshot B frames for performance, holdings, and
   transactions when available;
3. evaluate mandatory portfolio and security continuity;
4. if top-level optional checks are enabled, materialize eligible holding and
   transaction rows once;
5. run each enabled optional rule with its filters and tolerances;
6. normalize issue rows to a stable table schema;
7. sort by snapshot, portfolio, date, issue type, and security; and
8. reuse the resulting table in the portfolio and, when generated, security
   evidence packs.

`PDQ-018` — Rule evaluation MUST be deterministic for the same normalized
inputs, configuration, product version, and declared nonvolatile context.
CURRENT — DOCUMENTED under SN-11.

`PDQ-019` — Each rule MUST declare its eligible datasets, fields, population,
grouping key, comparison formula, threshold, output meaning, and known
limitations before it can become part of the supported catalog. APPROVED
DIRECTION for the Phase 4 rule schema.

`PDQ-020` — A rule MUST NOT infer correctness from the frequency or majority of
a value. The current range and rate rules report peer inconsistency; they do not
label the minimum, maximum, or most common value as correct. CURRENT —
DOCUMENTED.

`PDQ-021` — A rule that cannot safely interpret a row SHOULD exclude that row
from the calculation and MUST make material coverage limitations visible in
the target execution summary. Current numeric helpers skip absent, nonfinite,
or unusable values; coverage disclosure is APPROVED DIRECTION.

## 19.7 Union-of-snapshots and comparison behavior

“Across the union of Snapshot A and Snapshot B” means that eligible rows from
both declared source states are examined. It does **not** mean that Snapshot A
and Snapshot B values are combined into one peer group.

`PDQ-022` — Every current rule MUST preserve the snapshot label in its grouping
or period scope. A same-day Snapshot A value and Snapshot B value MUST NOT be
treated as two portfolio peers merely because their date and security match.
CURRENT — DEMONSTRATED.

`PDQ-023` — Optional rules MUST evaluate unchanged rows as well as rows that
differ between snapshots. This capability is source-state consistency review,
not delta-only review. CURRENT — DEMONSTRATED.

`PDQ-024` — Duplicate groups MUST remain snapshot-specific. An otherwise
identical transaction appearing once in A and once in B is not a duplicate
under the current rule. CURRENT — DEMONSTRATED.

`PDQ-025` — Continuity MUST compare periods only within the same snapshot and
grain. It MUST NOT compare Snapshot A ending value directly with Snapshot B
beginning value. CURRENT — DEMONSTRATED.

`PDQ-026` — Current continuity candidates require exact calendar adjacency:
the next `from_date` equals the prior `thru_date + 1 day`. Nonadjacent periods
do not create a continuity finding. CURRENT — DEMONSTRATED; client calendars
and intentionally gapped extracts require validation.

## 19.8 Tolerances and filters

### 19.8.1 Numeric tolerances

For current numeric range and rate rules, the trigger threshold is:

> greater of configured absolute tolerance or
>
> absolute reference value × configured percent tolerance / 100

The reference is the maximum observed peer value for range/rate rules and the
prior ending market value for continuity. A finding is emitted only when the
absolute continuity difference or positive peer range is strictly greater than
that threshold.

`PDQ-027` — Both the observed difference and human-readable effective
tolerance MUST remain in the finding detail. CURRENT — DEMONSTRATED.

`PDQ-028` — Optional numeric rules default to zero absolute and percent
tolerance when no valid value is configured. Continuity uses an effective
absolute tolerance of `0.01` when both configured tolerances resolve to zero.
CURRENT — DEMONSTRATED.

`PDQ-029` — Pilot tolerances MUST be approved in the units and economic context
of the rule. A percent tolerance is not an operational-materiality threshold,
and neither type may change Phase 3A explanation arithmetic. APPROVED DIRECTION.

`PDQ-030` — Tolerance changes MUST be versioned and regression-tested against
known true-positive and known acceptable cases. They MUST NOT be widened merely
to hide noise or a failing gate. APPROVED DIRECTION consistent with project
test-gate doctrine.

### 19.8.2 Exact-match filters

Current filters are case-insensitive string matches:

- every configured `only` field must match;
- a match on any configured `exclude` field removes the row;
- common aliases include snapshot, portfolio/portfolio_id,
  security/security_id, security_type, asset_class, and transaction_code; and
- a dataset-qualified field such as `transactions.transaction_code` resolves to
  the normalized field after the final dot.

They do not provide ranges, regular expressions, wildcards, or semantic
security-master predicates.

`PDQ-031` — Filters MUST be understood as population definition, not
post-finding cosmetic filtering. Removing rows before grouping can change the
peer set and whether an issue exists. CURRENT — DOCUMENTED.

`PDQ-032` — A client pilot SHOULD begin with deliberately narrow,
client-approved populations where the relationship is expected to be
comparable, then expand only after reviewing noise and missed cases. CURRENT —
DOCUMENTED setup guidance; CURRENT — REQUIRES CLIENT VALIDATION.

`PDQ-033` — Global rule disablement SHOULD be the last response to false
positives. Prefer correcting source interpretation, narrowing eligible
populations, or calibrating a documented tolerance. APPROVED DIRECTION.

## 19.9 Mandatory and optional findings

“Mandatory” describes rule execution, not automatic run failure.

`PDQ-034` — Portfolio and security beginning/end market-value continuity are
mandatory safety checks when the applicable performance data is present. They
MUST remain active even when `data_audit_checks.enabled` is false. CURRENT —
DEMONSTRATED under SN-04.

`PDQ-035` — The seven other checks are optional and individually configurable.
Their absence from an issue table does not by itself prove they ran, had
eligible inputs, or found no issue. CURRENT — DOCUMENTED.

`PDQ-036` — The target rule-execution summary MUST use these non-workflow
execution outcomes:

- `Issue Detected`;
- `No Issue Detected`;
- `Not Run — Disabled`;
- `Not Evaluated — Input Unavailable or Insufficient`; and
- `Blocked — Invalid Contract or Configuration`.

This summary is CANDIDATE. These labels do not create human case
management and do not supersede the Phase 2 run or analytical statuses.

`PDQ-037` — A mandatory continuity finding MUST remain visible and prioritized
for review, but its existence alone MUST NOT be described as proof that the
audit run or reported return is invalid. APPROVED DIRECTION.

## 19.10 Evidence, disposition, and separation from counted causes

The current issue schema contains:

- snapshot;
- portfolio identifier;
- as-of date;
- dataset field;
- security identifier when applicable;
- issue type;
- reference/minimum value;
- observed/maximum value;
- difference;
- tolerance;
- explanation; and
- deterministic review key.

`PDQ-038` — The reviewer-facing row MUST contain enough detail to reproduce the
rule comparison from retained local evidence or identify why reproduction is
not possible. APPROVED DIRECTION; current rows provide the comparison detail
but broader source locators vary by rule.

`PDQ-039` — The current `review_key` MUST be treated as a deterministic review
locator, not as a guaranteed unique source-row identity. Multiple rows in the
same issue group, including exact duplicates, may legitimately share it.
CURRENT — DOCUMENTED.

`PDQ-040` — Data Audit rows MUST NOT receive an explained amount, economic-
effect ownership, or cause disposition merely because they share a portfolio,
period, security, or field with Phase 3A evidence. APPROVED DIRECTION.

`PDQ-041` — Human disposition, comments, assignment, acceptance, and closure
remain DEFERRED with the broader workflow infrastructure. Until a serious
business case is validated, the immutable generated issue table is the product
record and any external client workflow remains outside PPAR.

## 19.11 Summary, detail, and reviewer workflow

### 19.11.1 Current detail outputs

`Data Audit Issues` is one of exactly three normal review surfaces in both the
XLSX workbook and HTML report. Its canonical CSV counterpart,
`x_ref_issues.csv`, is preserved in the supporting bundle and promoted in
CSV-only output. The bundle manifest records the artifact and table contract.

`PDQ-042` — XLSX, HTML, CSV, manifest counts, and fingerprints MUST remain
semantically aligned. Any parity failure is an internal logic error, not a
review warning. CURRENT — DEMONSTRATED under SN-10 and SN-11.

`PDQ-043` — The detail table MUST remain available even when a future summary
or prioritization view is added. Summary presentation MUST NOT become the only
surviving evidence. APPROVED DIRECTION.

### 19.11.2 Minimum useful summary

The current report shows the detail table but no dedicated rule-execution
coverage summary. The minimum useful target summary is:

- total issue-row count;
- count by issue type and snapshot;
- affected portfolio and security counts;
- mandatory versus optional rule identity;
- effective execution outcome for every configured/current rule;
- configured tolerance/filter indicator;
- input-coverage limitation; and
- direct link to detail.

`PDQ-044` — A summary MUST distinguish “zero findings” from “rule not
evaluated.” It MUST NOT use an unsupported severity, confidence, quality score,
or pass/fail certification. CANDIDATE.

`PDQ-045` — Initial prioritization MAY use the founder-approved operational-
materiality policy from Section 18.24, but continuity, integrity,
insufficient-coverage, recurring, and unresolved conditions take precedence
over magnitude alone. APPROVED DIRECTION; exact presentation requires client
validation.

### 19.11.3 Reviewer path

The intended first-pilot path is:

1. start with the portfolio audit's changed-performance view;
2. review independent Data Audit summary/issue rows even if the performance
   difference is fully explained;
3. filter by snapshot, rule, portfolio, date, field, and security;
4. inspect the observed/reference values, tolerance, and explanation;
5. return to the local extract and configuration for factual validation;
6. escalate source correction or policy clarification outside PPAR; and
7. rerun with a new immutable evidence pack when inputs or approved policy
   change.

`PDQ-046` — The product SHOULD explain that a rerun supersedes the reviewed
evidence operationally but does not mutate the prior bundle. APPROVED
DIRECTION; cross-run history remains DEFERRED.

## 19.12 Blocking and nonblocking policy

### 19.12.1 Nonblocking review findings

All current Data Audit issue rows, including mandatory continuity findings, are
nonblocking review findings. The audit continues and preserves them visibly.

`PDQ-047` — A plausible but interpretable inconsistency SHOULD produce a
finding rather than stop report generation. CURRENT — DEMONSTRATED.

### 19.12.2 Blocking conditions

Data Audit must remain subject to the broader fail-closed product contracts.
Blocking conditions include:

- invalid required source schema, type, identity, currency/unit, or period
  contract;
- overlapping or otherwise unsafe performance periods;
- required transaction or formula semantics that cannot be resolved for an
  authoritative output;
- malformed configuration once the target validation in `PDQ-014` is
  implemented;
- internal arithmetic, conservation, lineage, output-parity, determinism, or
  bundle-validation failure; and
- a review table exceeding the current 100,000-row output safety limit.

`PDQ-048` — Blocking failures MUST identify the failure class and corrective
owner. They MUST NOT be converted into a clean result, hidden row, generic
“unexplained” status, or weaker warning. CURRENT — DOCUMENTED.

`PDQ-049` — When a primary review table would exceed 100,000 rows, current
behavior MUST remain fail-closed: write no files for that oversized report,
identify major row contributors, and require narrower scope or corrected
upstream conditions. CURRENT — DEMONSTRATED.

## 19.13 False-positive and exception management

A finding can be technically correct under its rule and still be operationally
acceptable—for example, portfolio-specific valuation overrides or legitimate
intraday transaction price variation. Phase 3B therefore requires controlled
rule calibration, not stronger correctness language.

`PDQ-050` — Each pilot rule MUST be validated against three populations where
available:

1. known error or expected finding;
2. known acceptable variation; and
3. ambiguous case requiring human review.

CURRENT — REQUIRES CLIENT VALIDATION.

`PDQ-051` — Validation SHOULD record, per rule, eligible population, issue
groups, issue rows, confirmed errors, acceptable variations, indeterminate
cases, missed expected cases, review time, and configuration changes.
APPROVED DIRECTION.

`PDQ-052` — PPAR MUST NOT introduce a global “false positive” percentage
without a defined denominator and client-reviewed ground truth. Rule-level
precision or yield measures MAY be evaluated during validation but are not
current product claims.

`PDQ-053` — Per-finding suppression, comments, exception expiry, assignment,
and approval would require the deferred workflow infrastructure. They MUST NOT
be improvised as hidden report filtering. DEFERRED.

`PDQ-054` — An approved filter or tolerance exception MUST be visible in
effective configuration and must not alter retained historical bundles.
APPROVED DIRECTION.

## 19.14 Rule-specific limitations and edge cases

### 19.14.1 Peer range and rate checks

- Fewer than two usable peer values produce no issue under the current rules.
- Range and rate groups are same-snapshot, same-date, and same-security.
- `pa` and `sa` rates are grouped separately.
- Transaction rates use `abs(amount) / abs(quantity)`.
- If transaction quantity is absent or zero, dividend and PA/SA rules use the
  first same-snapshot, same-portfolio, same-security holdings quantity on or
  after the transaction date.
- Holdings accrued rate uses `accrued / abs(quantity)`.
- Filtering a participant out can change the min/max and the result.

`PDQ-055` — The quantity fallback MUST be disclosed because it is an inferred
calculation basis, not transaction-row evidence. CURRENT — DOCUMENTED in setup
guidance; CURRENT — REQUIRES CLIENT VALIDATION.

`PDQ-056` — A peer range finding MUST not identify which portfolio is wrong.
Every participating row may be emitted so the reviewer can assess context.
CURRENT — DEMONSTRATED.

### 19.14.2 Duplicate transactions

The current duplicate rule has no durable transaction identifier in its group
key. It flags exact repeated rows at the configured fields and emits each member
of the group.

`PDQ-057` — The rule MUST be described as duplicate-row detection, not proof of
duplicate economic activity. Legitimate separate events can share the current
fields if the extract omits a durable identifier. CURRENT — DOCUMENTED.

### 19.14.3 Missing dividend

The current rule begins with a same-snapshot dividend event in at least one
eligible portfolio. Another portfolio qualifies for review only when
consecutive holdings dates bracket the dividend date and either:

- the earlier holding quantity is positive; or
- positive buy activity occurs before the dividend date.

Any other transaction activity between the earlier holding date and dividend
date prevents qualification. A same-date dividend in the candidate portfolio
prevents a finding.

`PDQ-058` — The rule MUST be described as conservative apparent-absence
detection. It does not establish entitlement, ex-date/pay-date correctness,
tax treatment, corporate-action completeness, or a missing official posting.
CURRENT — DOCUMENTED.

### 19.14.4 Continuity

- Only consecutive periods at the same grain are compared.
- Nonadjacent periods, missing begin/end values, or unavailable security
  performance do not produce a continuity row.
- Difference is next beginning market value minus prior ending market value.
- Continuity is evaluated independently inside A and B.

`PDQ-059` — Continuity MUST remain a visible safety finding and MUST never be
silently repaired, filled, or netted against another value. CURRENT —
DEMONSTRATED under SN-04.

## 19.15 Lineage, audit trail, and invariants

Each issue row is generated from normalized local data and carries a
deterministic review locator. The canonical CSV is included in the validated
supporting evidence, and the report bundle records table metadata and content
fingerprints.

`PDQ-060` — The bundle MUST preserve the exact issue detail used by every
presentation surface and MUST validate before delivery. CURRENT —
DEMONSTRATED.

`PDQ-061` — Data Audit implementation changes MUST preserve all applicable
safety invariants, especially:

| Invariant | Phase 3B implication |
|---|---|
| SN-02 No double counting | Data Audit evidence cannot become an additive cause without separate supported ownership |
| SN-04 Continuity | Applicable continuity always executes and remains visible |
| SN-05 Bidirectional lineage | Any cross-link to Phase 3A must be traceable in both directions |
| SN-06 Currency/unit consistency | Unsafe units cannot enter a meaningful comparison |
| SN-07 Period-boundary safety | Continuity and dated evidence use unambiguous periods |
| SN-10 Report parity | XLSX, HTML, and CSV convey the same issue semantics |
| SN-11 Deterministic output | Rule results and bundle fingerprints repeat |
| SN-12 Fail-closed policy | Unknown required meaning cannot become a clean result |

CURRENT — DOCUMENTED project gate.

`PDQ-062` — No implementation or configuration change may weaken a test,
tolerance, invariant, or release gate merely because it produces an
uncomfortable result. Any intentional product-policy change requires explicit
approval, evidence, and tradeoff documentation. CURRENT — DOCUMENTED.

## 19.16 Scale and performance

The current evaluator loads each applicable snapshot dataset once, reuses
materialized holding/transaction rows across enabled rules, precompiles filters,
and uses grouped comparisons. This is CURRENT — DEMONSTRATED in the packaged
scope and engineering scale checks, but CURRENT — REQUIRES CLIENT VALIDATION
for real extract shapes.

`PDQ-063` — Rule design SHOULD avoid repeated full-dataset parsing and
unbounded pairwise comparisons when a grouped deterministic calculation is
available. APPROVED DIRECTION.

`PDQ-064` — Scale validation MUST measure source row counts, eligible row
counts, group cardinality, issue-row amplification, runtime, memory, report
size, and reviewer usability by rule. APPROVED DIRECTION.

`PDQ-065` — A noisy rule that creates an unusable report is not commercially
validated merely because it is computationally fast. Review effort and
actionable yield are required validation evidence.

`PDQ-066` — The 500x scale check remains part of the release-candidate workflow
for major cross-cutting, reporting, audit, safety-net, or performance changes.
CURRENT — DOCUMENTED project gate.

## 19.17 Local-first, security, privacy, and support

`PDQ-067` — Source extracts, normalized rows, issue calculations, configuration,
and generated evidence MUST remain in the client-controlled environment during
ordinary operation. CURRENT — DOCUMENTED permanent doctrine.

`PDQ-068` — Phase 3B MUST NOT require PPAR-operated upload, telemetry, hosted
portfolio-data processing, or remote rule execution. OUT OF SCOPE under current
doctrine.

`PDQ-069` — Support SHOULD begin with locally generated validation results,
product/configuration versions, rule-execution metadata, error classification,
and non-sensitive diagnostics. Any evidence transfer requires explicit,
case-specific client authorization. APPROVED DIRECTION.

`PDQ-070` — Current access control is the client's operating-system and file
permission boundary. Built-in role-based authorization, identity, comments,
and approval workflow are not current capabilities and remain DEFERRED pending
a validated business case.

## 19.18 Acceptance criteria

Phase 3B is functionally acceptable only when all applicable criteria pass:

1. **Boundary:** issue rows never alter Phase 3A counted causes, explained
   amount, residual, or analytical status.
2. **Rule truth:** every supported rule has explicit population, grouping,
   formula, threshold, output meaning, and limitations.
3. **Snapshot separation:** A and B are both examined but never mixed into one
   current peer group or continuity sequence.
4. **Mandatory behavior:** applicable portfolio/security continuity cannot be
   disabled with optional checks.
5. **Configuration:** effective enablement, filters, tolerances, and versions
   are reproducible; malformed target configuration fails clearly.
6. **Coverage:** reviewers can distinguish detected, clean, disabled,
   insufficient, and blocked execution outcomes.
7. **Evidence:** detail identifies scope, observed/reference values or absence,
   difference, tolerance, explanation, and review locator.
8. **Outputs:** XLSX, HTML, CSV, manifest, and fingerprints agree.
9. **Failure behavior:** unsafe contracts and internal logic failures block;
   interpretable suspicious relationships remain visible nonblocking findings.
10. **False positives:** each enabled pilot rule is calibrated against known
    errors, acceptable variation, and ambiguous cases where available.
11. **Scale:** issue amplification remains computationally and operationally
    usable, with the 100,000-row report guard intact.
12. **Local-first:** ordinary execution and evidence retention require no
    portfolio-data transfer.
13. **Claims:** outputs are described as transparent review findings, not
    assurance, certification, or comprehensive validation.

Current implementation demonstrates much of criteria 1–4, 7–9, 11–12 within
the packaged scope. Explicit execution coverage, stricter malformed-
configuration validation, calibrated false-positive evidence, and real-client
rule utility remain APPROVED DIRECTION or CURRENT — REQUIRES CLIENT VALIDATION.

## 19.19 Dependencies and approved working assumptions

### Dependencies

- founder-approved Phase 2 actors, decision rights, and local-first workflow;
- founder-approved Phase 3A evidence/cause boundary and pilot report scope;
- current source/extract, normalization, safety-invariant, and bundle contracts;
- Phase 3C design for a unified readiness and preflight experience;
- Phase 4 rule schema, prioritization, and catalog-governance design;
- real-client source and policy owners for tolerance and filter approval; and
- deferred workflow/history capabilities only if later business validation
  justifies them.

### Founder-approved Phase 3B working assumptions

The founder approved Phase 3B on 2026-07-16 with these working assumptions:

1. **Rule-execution summary — APPROVED DIRECTION:** adopt the five execution
   outcomes in `PDQ-036` and the minimum summary in `PDQ-044`, without adding
   human workflow infrastructure.
2. **Pilot rule-set policy — APPROVED DIRECTION; CURRENT — REQUIRES CLIENT
   VALIDATION:** configure optional rules only for understood, comparable
   source populations; always retain mandatory continuity.
3. **Malformed configuration — APPROVED DIRECTION:** invalid enablement,
   tolerance, and filter shapes should fail validation rather than silently
   fall back. This is not yet current application behavior.
4. **Data Audit placement — APPROVED DIRECTION:** retain Data Audit primarily
   inside the integrated audit bundle for initial pilots. A standalone entry
   point remains CANDIDATE only after independent client demand is shown.
5. **Rule-result priority — APPROVED DIRECTION; CURRENT — REQUIRES CLIENT
   VALIDATION:** reuse the Phase 3A operational-materiality dimensions initially.
   Do not add a separate rule-priority model, confidence score, or composite
   quality score without client evidence.

These assumptions do not authorize per-finding workflow, hidden suppression,
hosted processing, or expansion beyond performance-quality review.

## 19.20 Real-client validation plan

### Stage 1 — Rule and source discovery

- inventory client extract availability and field meaning;
- identify which portfolios, security types, transaction codes, and valuation
  practices create legitimately comparable populations;
- select mandatory and optional pilot rules;
- approve initial exact filters and tolerances;
- identify known accepted overrides and data limitations.

**Exit evidence:** an attributable pilot rule matrix with population,
configuration, rationale, owner, and expected limitation.

### Stage 2 — Known-case calibration

- run known data errors and known acceptable variations;
- verify expected issue types, snapshot scope, values, dates, and portfolios;
- look for missed expected cases and unexpected row amplification;
- confirm that no finding alters Phase 3A explanation arithmetic;
- adjust filters or tolerances only with recorded evidence.

**Exit evidence:** rule-by-rule labeled cases and an approved configuration
revision, including errors the rule did not detect.

### Stage 3 — Blind operational review

- have an analyst review findings without being told the planted/known outcome;
- record likely error, acceptable variation, and indeterminate judgments;
- reconcile judgments with source and methodology owners;
- measure review time, escalation burden, and actionable yield;
- test whether the summary and explanations support independent operation.

**Exit evidence:** client-reviewed usefulness and false-positive record with no
unsupported correctness claim.

### Stage 4 — Recurring-use decision

- repeat on later reporting cycles or source states;
- measure configuration stability and rule-specific noise;
- identify repeated conditions outside PPAR if the client already has a
  workflow;
- decide which rule families merit Phase 4 investment;
- assess whether standalone Data Audit or workflow/history infrastructure has
  a serious business case.

**Exit evidence:** evidence-supported catalog priorities and an explicit
continue, revise, or retire decision for each pilot rule.

## 19.21 Claims supported and not supported

### Claims supported now, with qualification

- PPAR applies configured transparent consistency checks to eligible normalized
  rows in both Snapshot A and Snapshot B.
- PPAR currently reports seven optional issue families plus mandatory
  beginning/end market-value continuity when applicable.
- PPAR preserves snapshot, portfolio, date, dataset field, security, values,
  difference, tolerance, explanation, and review-key context in a validated
  local evidence pack.
- Data Audit findings remain separate from additive performance causes.

Each claim is CURRENT — REQUIRES CLIENT VALIDATION for a real site.

### Claims not supported

- that PPAR validates all portfolio-accounting data;
- that every issue row is an error;
- that no issue rows means the data is correct or every rule had coverage;
- that the reference/minimum/maximum/majority value is authoritative;
- that a detected duplicate row proves duplicated economic activity;
- that an apparent missing dividend establishes legal or accounting
  entitlement;
- that Data Audit provides an audit opinion, control certification, regulatory
  assurance, GIPS verification, or official-performance approval;
- that current rules have validated severity, confidence, or quality scores;
- that PPAR currently provides durable issue workflow, cross-run history, or a
  standalone data-quality product; or
- that packaged-demo calibration establishes universal Axys/APX behavior.

## 19.22 Phase 3B approval and next gate

Phase 3B was founder-approved on 2026-07-16 with the five working assumptions
in Section 19.19. That approval authorized Phase 3C drafting; it did not
authorize application-code changes or begin Phase 3D.

---

# 20. Phase 3C — Audit Readiness

## 20.1 Specification identity, purpose, and status

**Capability:** Audit Readiness and Preflight

**Primary user:** local operator or performance/operations analyst

**Primary first-pilot surface:** a pre-run decision for the requested portfolio
audit scope, with explicit limitations and remediation

**Overall status:** CURRENT — DEMONSTRATED and CURRENT — DOCUMENTED for
distributed technical validators and fail-closed run gates; CURRENT — REQUIRES
CLIENT VALIDATION for real client inputs and operating usefulness; APPROVED
DIRECTION for a unified operator-facing readiness experience

Audit Readiness answers:

> Are these declared snapshots, contracts, configuration, product context, and
> requested outputs safe and sufficiently evidenced for this specific audit
> purpose—and, if not, exactly what must happen next?

It is a precondition decision, not an opinion that the extracts are complete,
the configured policy is correct, the performance is accurate, or the eventual
investigation will be fully explained.

This section follows the `MUST`, `SHOULD`, and `MAY` meanings in Section 18.1.
Requirement identifiers use `ARD` for Audit Readiness.

`ARD-001` — Readiness MUST be bound to a declared run purpose, requested report
scope, Snapshot A and B, source/extract contract, configuration, and product
context. It MUST NOT be a generic certification of the client environment.
APPROVED DIRECTION.

`ARD-002` — A readiness decision MUST NOT weaken or replace any calculation,
lineage, output-integrity, or bundle-validation gate that occurs during or
after the run. APPROVED DIRECTION.

`ARD-003` — Current technical validation MUST be described candidly: it can
demonstrate that the supplied case passes implemented controls, but it has not
established compatibility with a real client's Axys/APX exports. CURRENT —
REQUIRES CLIENT VALIDATION.

## 20.2 User problem, actors, trigger, and authority

### 20.2.1 User problem

Today, a knowledgeable operator can run the configuration validator or start an
audit and receive useful failures and a detailed successful summary. However,
the controls are distributed across specification parsing, data loading,
comparison, financial-integrity checks, report generation, and bundle
validation. The operator does not receive one durable readiness object that
answers:

- what was evaluated;
- which requested outputs are safe to attempt;
- what required evidence passed;
- what optional evidence is absent and how that limits the result;
- what remains unapproved or ambiguous;
- who must act; and
- which changes invalidate the decision.

`ARD-004` — The unified experience MUST turn technical controls into an
operator decision and corrective path. It SHOULD not merely reproduce a long
exception stack or relabel every warning as “readiness.” APPROVED DIRECTION.

### 20.2.2 Actors and authority

| Actor | Readiness responsibility | Authority boundary |
|---|---|---|
| Performance/operations analyst | Declares intended use and scope; reviews qualifications | Cannot approve extract facts or methodology merely by operating the run |
| Source/extract administrator | Confirms export procedure, fields, provenance, and reproducibility | Cannot approve performance treatment unless separately authorized |
| Methodology owner | Approves return basis, formula policy, transaction semantics, tolerances, and material configuration | Does not certify file delivery or local infrastructure |
| Performance/operations manager | Approves pilot use, material exceptions, and whether a qualified result is acceptable | Cannot waive a safety blocker |
| Local product administrator | Installs the approved version, controls local files, runs validation, and retains evidence | Cannot approve accounting meaning solely because the administrator can edit YAML |
| Technology/security owner | Approves execution boundary, access, output location, updates, and permitted support data | Does not approve financial calculations |
| PPAR | Evaluates declared contracts and reports deterministic pass, qualification, or block evidence | Does not infer missing approval or official correctness |

`ARD-005` — Readiness MUST distinguish factual source approval, business-policy
approval, technical execution authority, and product-calculated status.
APPROVED DIRECTION.

`ARD-006` — No human role may convert an unresolved safety requirement into
`Qualified Ready` by waiver. The source/configuration must be corrected, the
requested scope safely narrowed, or the workflow must remain `Blocked`.
APPROVED DIRECTION.

### 20.2.3 Triggering events

Readiness should be evaluated:

- before the first pilot run;
- after either snapshot or extract procedure changes;
- after a schema, mapping, transaction rule, tolerance, suppression, formula,
  source contract, or report-scope change;
- after a product or dependency upgrade;
- after an earlier readiness or audit failure;
- before an operationally relied-upon rerun; and
- when an accepted client regression case no longer matches.

`ARD-007` — A prior readiness result MUST be invalidated when any material
bound input, contract, configuration, requested scope, or product context
changes. APPROVED DIRECTION.

## 20.3 Readiness decision model

### 20.3.1 Readiness statuses

The proposed three-status model is:

- **`Ready`** — all required contracts evaluated for the requested purpose and
  scope pass; no declared optional-evidence limitation materially qualifies
  the intended interpretation.
- **`Qualified Ready`** — no safety blocker exists, but explicitly optional
  evidence is absent or another permitted limitation narrows what the audit can
  explain or review. The limitation and affected scope are visible.
- **`Blocked`** — a required input, meaning, approval, integrity condition,
  configuration, regression requirement, or safe-output boundary is unresolved.

These statuses are APPROVED DIRECTION under the founder-approved Phase 3C
working assumptions. They are not the Phase 3A analytical statuses or future
human workflow statuses.

`ARD-008` — Overall status MUST be calculated deterministically:

1. any blocking item produces `Blocked`;
2. otherwise, any applicable qualification produces `Qualified Ready`; and
3. otherwise, the result is `Ready`.

APPROVED DIRECTION.

`ARD-009` — Readiness items MUST use a separate disposition:

- `Blocker`;
- `Qualification`; or
- `Information`.

An informational item does not change overall status. APPROVED DIRECTION.

`ARD-010` — `Qualified Ready` MUST be allowed only for an explicit optional or
permitted limitation whose safety effect is understood. Unknown required
meaning, missing required data, incomplete required policy, or an internal
logic failure MUST produce `Blocked`. APPROVED DIRECTION.

`ARD-011` — `Ready` means safe to proceed past the evaluated preflight for the
declared purpose. It does not guarantee that no later data-dependent,
report-size, arithmetic, lineage, parity, determinism, filesystem, or bundle
gate will fail. APPROVED DIRECTION.

### 20.3.2 Diagnostic-only execution

The current `validate_config --allow-incomplete-yaml` option intentionally lets
maintainers and implementers inspect a case whose YAML setup is incomplete.
That is useful diagnosis, not readiness.

`ARD-012` — A run that bypasses complete-YAML validation MUST be labeled
`Diagnostic Only` and MUST NOT receive `Ready` or `Qualified Ready`. CURRENT —
DOCUMENTED boundary; unified presentation is APPROVED DIRECTION.

`ARD-013` — Diagnostic bypasses MUST remain explicit in the command, output,
and any generated artifact. They MUST NOT become the default or a way to
produce authoritative pilot evidence. CURRENT — DOCUMENTED.

### 20.3.3 Intended use and reliance

The first clients are validation partners. A case can be safe for controlled
validation without being approved for recurring operational reliance.

`ARD-014` — The readiness request SHOULD declare intended use, at minimum
`validation` or `operational review`. Both uses retain identical safety
requirements; operational review additionally requires the client's applicable
approval and regression evidence. APPROVED DIRECTION.

`ARD-015` — A validation-purpose `Ready` result MUST retain CURRENT — REQUIRES
CLIENT VALIDATION and MUST NOT be represented as production certification.

## 20.4 Current technical baseline and gaps

### 20.4.1 Current demonstrated foundation

The current `validate_config` path:

1. parses the YAML with duplicate-key rejection;
2. resolves comparison level, snapshots, schemas, files, and reconstruction
   settings;
3. checks required file existence;
4. loads and compares the configured datasets;
5. validates normalized columns and source mappings;
6. applies current identity, transaction, extract-contract, currency/unit, and
   period controls reached by the comparison;
7. validates complete YAML treatment for observed changed fields by default;
8. loads transaction data from both snapshots;
9. reports observed transaction codes, rule coverage, and semantic sources;
10. reports configured and minimum required datasets/columns;
11. reports missing configured optional files; and
12. returns nonzero with an actionable error when validation fails.

The `ppar setup` workflow also invokes the same validator when starter snapshot
files are present. Normal audit generation repeats applicable source and policy
checks and adds conservation, lineage, report-size, output-parity,
determinism, and bundle-validation gates.

`ARD-016` — These controls are CURRENT — DEMONSTRATED; CURRENT — DOCUMENTED;
CURRENT — REQUIRES CLIENT VALIDATION. They are distributed technical gates, not
the completed unified readiness product.

### 20.4.2 Current gaps

| Gap | Current truth | Target status |
|---|---|---|
| Unified readiness status | Validator prints pass/fail and summary, not Ready/Qualified Ready/Blocked | APPROVED DIRECTION |
| Multiple issue collection | Most failures stop at the first raised product error | CANDIDATE bounded issue collection where safe |
| Optional-evidence interpretation | Missing configured optional files are listed, but their effect is not calculated into a readiness qualification | CANDIDATE |
| Durable readiness object | No canonical readiness JSON/table exists | CANDIDATE |
| Human approval record | No built-in contract/configuration approval workflow exists | DEFERRED workflow; CANDIDATE external approval reference |
| Product-version provenance | `ppar.__version__` exists, but current bundles do not embed it | APPROVED DIRECTION |
| Complete configuration provenance | Current bundles identify the comparison path, not an embedded configuration snapshot/fingerprint | APPROVED DIRECTION |
| Client regression baseline | Internal demo/release regression exists; no client accepted-case registry exists | CURRENT — REQUIRES CLIENT VALIDATION; CANDIDATE minimal local case files |
| Data Audit configuration shape | Some malformed settings can currently fall back or be ignored | APPROVED DIRECTION from Phase 3B |
| Public preflight surface | Maintainer module exists; no unified top-level operator experience is established | CANDIDATE |

`ARD-017` — Phase 3C MUST NOT describe any target item in this gap table as a
current implemented capability.

## 20.5 Preflight contract layers

### 20.5.1 Layer 1 — Local workspace and execution context

This layer establishes:

- requested site directory and audit configuration path;
- installed PPAR version and dependency/runtime identity;
- requested report level(s) and output mode;
- output destination;
- client-controlled execution boundary; and
- intended use and operator identity or local job identity where available.

Current commands validate key paths and required package dependencies during
normal use. They do not create a complete readiness context record.

`ARD-018` — The target preflight MUST identify the exact installed PPAR
version, requested outputs, and configuration path before evaluating financial
meaning. APPROVED DIRECTION.

`ARD-019` — Output-directory existence, write permission, available capacity,
and dependency availability SHOULD be checked before expensive processing when
the selected output requires them. CANDIDATE.

`ARD-020` — Infrastructure checks MUST remain inside the client-controlled
environment and MUST NOT require portfolio-data telemetry. APPROVED DIRECTION.

### 20.5.2 Layer 2 — Specification and configuration shape

Current specification validation includes:

- YAML must parse to a mapping;
- duplicate mapping keys fail;
- Snapshot A and B definitions and paths must have valid shape;
- comparison level must be portfolio or security;
- configured dataset names must be supported;
- file definitions, `required` flags, schemas, and paths must have valid shape;
- selected performance dataset must be configured;
- active reconstruction settings must contain supported keys and enum values;
- required reconstruction datasets cannot opt out of required status; and
- legacy/unsupported policy sections fail.

`ARD-021` — Duplicate, unknown, malformed, or internally contradictory policy
configuration MUST be a blocker. CURRENT — DEMONSTRATED for current validated
sections; APPROVED DIRECTION for complete schema coverage.

`ARD-022` — Phase 3B's approved fail-closed Data Audit configuration behavior
MUST become part of this layer before malformed Data Audit settings can be
considered ready. APPROVED DIRECTION; not current application behavior.

### 20.5.3 Layer 3 — Source/extract contract and file structure

This layer evaluates:

- Snapshot A/B directories and configured file presence;
- required versus optional file status;
- required normalized columns and unambiguous alias resolution;
- extract-contract path and shape;
- required transaction context fields;
- supported contract column names and Boolean flags;
- source/report provenance and reproducible export procedure where declared;
- native identifier/code visibility; and
- selected report scope against available performance datasets.

`ARD-023` — A required configured file missing from either snapshot MUST
produce `Blocked` with the dataset, snapshot, resolved path, owner, and
corrective action. CURRENT — DEMONSTRATED failure; unified item format is
CANDIDATE.

`ARD-024` — A configured optional file that exists MUST still pass its
applicable schema and integrity contract before it contributes evidence.
CURRENT — DOCUMENTED.

`ARD-025` — A required normalized column that is absent or resolves
ambiguously MUST produce `Blocked`. CURRENT — DEMONSTRATED.

`ARD-026` — Readiness MUST NOT infer that a vendor export is complete merely
because the CSV header passes. Source provenance, export procedure, applicable
fields, and known omissions remain client-owned evidence. APPROVED DIRECTION.

### 20.5.4 Layer 4 — Transaction and accounting semantics

This layer evaluates:

- observed transaction codes in both snapshots;
- configured conditional rules and their source;
- extract-contract enforcement for ambiguous Axys/APX-style flows;
- required source/destination, special-security, or report-semantic context;
- normalized transaction category, flow sign, performance sign, and formula
  role when applicable;
- field-role registry coverage; and
- explicit additive, evidence-only, or suppression treatment for changed fields
  that require policy.

`ARD-027` — Ambiguous `dp`, `li`, `lo`, or `wd` rows without the required
contract context MUST be `Blocked` when enforcement is on. CURRENT —
DEMONSTRATED.

`ARD-028` — Setting
`extract_contract.enforce_ambiguous_axys_flows: false` MUST NOT by itself prove
readiness. The target readiness decision must require an attributable reviewed
local opt-out and identify the risk owner; otherwise the case is `Blocked`.
APPROVED DIRECTION; the current technical validator checks the Boolean setting
but does not record human approval.

`ARD-029` — An unknown or unsupported transaction family MAY remain
non-additive review evidence only when safe quarantine is explicit and the
requested calculation does not depend on guessed semantics. Otherwise it is a
blocker. APPROVED DIRECTION.

`ARD-030` — No code name, field name, majority behavior, or packaged-demo
example may substitute for client-approved transaction or accounting policy.
CURRENT — DOCUMENTED.

### 20.5.5 Layer 5 — Financial and period integrity

Applicable current controls include:

- three-letter normalized currency codes;
- explicit base-currency counterparts for foreign countable values;
- agreement between local/base values when currencies are the same;
- valid, positive, uniquely scoped FX rates and quote direction;
- portfolio-specific FX target currency agreement;
- required reconstruction inputs and supported formula configuration;
- valid ordered performance periods;
- rejection of reversed or overlapping periods;
- safe dated-evidence ownership; and
- beginning/end continuity as a mandatory visible Data Audit finding.

`ARD-031` — Unsafe currency, unit, FX, period, or required reconstruction input
conditions MUST produce `Blocked` before authoritative output. CURRENT —
DEMONSTRATED for implemented contracts.

`ARD-032` — A continuity mismatch is not a preflight blocker merely because it
exists. It remains a mandatory visible review finding unless the underlying
period or value contract is itself invalid. CURRENT — DOCUMENTED distinction.

`ARD-033` — A legitimate unresolved performance difference is not a readiness
failure. Missing required setup or unsafe evidence is a readiness blocker;
honest unexplained residual after a safe run is an analytical outcome.
APPROVED DIRECTION.

### 20.5.6 Layer 6 — Requested output and downstream gates

The readiness request must evaluate each selected output scope separately:

- portfolio report, mandatory for the initial pilot;
- security report when the selected case requires drilldown;
- XLSX, HTML, or CSV-only presentation;
- optional reconstruction diagnostics; and
- compact or expanded supporting files.

`ARD-034` — Security-level readiness MUST require security performance and any
active security reconstruction inputs only when security output is requested.
APPROVED DIRECTION consistent with the founder-approved Phase 3A pilot scope.

`ARD-035` — If `both` is requested but security inputs are unavailable, the
target preflight MUST explicitly narrow the request to portfolio with
authorization or return `Blocked` for the unchanged request. It MUST NOT
silently call the full request ready. APPROVED DIRECTION; current `both`
execution may skip unavailable security output with a status message.

`ARD-036` — Readiness SHOULD estimate review-table row amplification and output
risk where practical. The exact 100,000-row limit remains a hard generation
gate even if the estimate passes. CANDIDATE estimate; CURRENT — DEMONSTRATED
generation gate.

`ARD-037` — Post-generation arithmetic, lineage, parity, determinism, and bundle
validation are downstream completion gates. A preflight `Ready` result MUST
not pre-authorize a failed evidence pack. CURRENT — DOCUMENTED.

## 20.6 Required versus optional evidence

### 20.6.1 Four evidence classes

Readiness needs a more precise model than “file exists”:

1. **Structurally required** — required for the selected comparison/output to
   load safely.
2. **Conditionally required** — required because a selected method, currency,
   transaction family, asset condition, or report level is present.
3. **Optional but limiting** — safe to omit, but absence reduces explanation,
   rule coverage, drilldown, or reviewer evidence.
4. **Optional and nonlimiting for declared scope** — absent evidence that does
   not affect the requested purpose.

`ARD-038` — Structurally or conditionally required evidence missing for the
declared case MUST produce `Blocked`. APPROVED DIRECTION.

`ARD-039` — Optional-but-limiting evidence missing MUST produce a visible
`Qualification` that names the unavailable capability, affected population,
and claim limitation. CANDIDATE.

`ARD-040` — Optional evidence that is irrelevant to the declared scope MAY be
an informational item and need not prevent `Ready`. CANDIDATE.

### 20.6.2 Relationship to extraction labels

The current extraction guidance uses:

- `Required` for ordinary evidence needed to make Fully Explained possible;
- `Required only when applicable` for named features or data conditions; and
- `Optional` for safe omissions.

The runtime structural minimum is intentionally narrower. A field can be
structurally optional to a loader but necessary to explain a specific changed
value.

`ARD-041` — Readiness MUST disclose which meaning of “required” is being used:
runtime safety, selected feature/method, or explanation depth. It MUST NOT
collapse all three into one misleading checklist. APPROVED DIRECTION.

`ARD-042` — A missing field that prevents `Fully Explained` but does not make
the run unsafe SHOULD normally produce `Qualified Ready`, not `Blocked`,
provided the limitation is explicit and no required policy is missing.
CANDIDATE; CURRENT — REQUIRES CLIENT VALIDATION.

`ARD-043` — Optional evidence absence MUST NOT be described as proof that no
relevant event occurred. APPROVED DIRECTION.

## 20.7 Processing and evaluation sequence

The target unified preflight should use this sequence:

1. **Declare request:** intended use, portfolio/security scope, snapshots,
   outputs, and local configuration.
2. **Static validation:** paths, YAML shape, duplicate keys, supported settings,
   contract references, and required declarations.
3. **Source loading:** required files, headers, mappings, types, identifiers,
   dates, and optional-file inventory.
4. **Data-dependent validation:** observed transaction semantics, field roles,
   changed-field policy coverage, currency/unit, period, reconstruction, and
   selected Data Audit configuration.
5. **Output-risk assessment:** requested surfaces, optional limitations,
   estimated row amplification, output destination, and downstream gates.
6. **Version/regression assessment:** product/configuration/contract identity
   and applicable accepted-case evidence.
7. **Decision:** item list, Ready/Qualified Ready/Blocked, remediation, and
   validity boundary.

`ARD-044` — Static checks SHOULD collect multiple independent issues when doing
so is safe and useful. Data-dependent evaluation MUST stop when continuing
would require interpreting invalid structure or unsafe semantics. APPROVED
DIRECTION.

`ARD-045` — The unified preflight SHOULD reuse loaded/normalized results or a
shared deterministic plan with the immediately following audit. It SHOULD not
double full-run cost merely to create a readiness label. CANDIDATE.

`ARD-046` — The current validator's full comparison is a meaningful
data-dependent check, but a pass applies only to changed fields and conditions
observed in those supplied snapshots. It MUST NOT be represented as proof that
configuration covers every future field or transaction. CURRENT — DOCUMENTED
boundary.

`ARD-047` — If inputs change after preflight and before bundle completion, the
readiness result MUST be invalidated or the run MUST prove it used the same
bound content. APPROVED DIRECTION; content binding is not a current complete
capability.

## 20.8 Source/extract contract approval

### 20.8.1 Required contract content

A pilot source/extract contract should identify:

- source system and relevant version when known;
- producing report, IMEX profile, REP/custom report, or export procedure;
- dataset and field inventory;
- snapshot capture/as-of provenance;
- normalized mappings and preserved native fields;
- conditional transaction context;
- required/conditional/optional evidence;
- known omissions and client-specific enrichments;
- currency, unit, period, and identifier conventions;
- responsible source owner and methodology owner;
- contract version or stable content fingerprint; and
- approval reference and effective date.

`ARD-048` — PPAR MUST NOT invent absent source provenance or approval metadata.
Missing approval required for the declared intended use is a blocker; missing
optional descriptive context may be a qualification only when safety is not
affected. APPROVED DIRECTION.

`ARD-049` — A packaged contract is a versioned product default for its declared
scope, not proof that the client's extract conforms to it. CURRENT —
DOCUMENTED.

### 20.8.2 Approval without workflow infrastructure

No PPAR approval database or human-signature workflow is required for the first
pilot. Approval may remain in the client's change-control system or a small
local controlled record referenced by the configuration.

`ARD-050` — The initial product SHOULD record an approval reference, approver
role, approval date, and contract/configuration fingerprint without building
generic workflow infrastructure. APPROVED DIRECTION.

`ARD-051` — PPAR MUST treat an approval reference as provenance, not as proof
that the approver had authority or that the policy is correct. APPROVED
DIRECTION.

`ARD-052` — Any change to contract-relevant fields, export procedure, guarded
transaction context, or semantic mappings MUST invalidate the applicable
approval and require revalidation. APPROVED DIRECTION.

## 20.9 Configuration completeness

### 20.9.1 Current completeness boundary

Current complete-YAML validation checks observed reportable changed fields. It
requires explicit accounting role and, where applicable, additive,
evidence-only, or suppression policy before normal bundle generation.

`ARD-053` — Readiness MUST distinguish:

- syntactic validity;
- structural/schema validity;
- observed-case policy completeness;
- selected-method completeness; and
- client approval completeness.

One passing dimension MUST NOT imply the others. APPROVED DIRECTION.

`ARD-054` — A configuration that passes because the supplied snapshots contain
no instance of a future field or transaction family is complete only for the
observed case. CURRENT — DOCUMENTED boundary.

`ARD-055` — Missing required impact policy, unknown field role, unsupported
method key, or ambiguous required transaction semantics MUST produce
`Blocked`. CURRENT — DEMONSTRATED for implemented contracts.

### 20.9.2 Supported attribution versus full explanation

The strict supported-attribution option checks whether known supported
attribution methods have the required setup. It does not require every
performance difference to be fully explained.

`ARD-056` — Readiness for a requested strict-attribution run MUST include that
strict check. Readiness for an ordinary review MAY permit a safe unexplained
result, but MUST NOT permit missing setup for a field the product claims to
treat authoritatively. APPROVED DIRECTION.

`ARD-057` — `Fully Explained` is never a readiness prerequisite by itself.
Configuration safety and explanation outcome are different dimensions.
APPROVED DIRECTION.

## 20.10 Product version, configuration identity, and regression

### 20.10.1 Current version and artifact truth

Current project evidence shows:

- `pyproject.toml` is the package-version authority;
- the installed package exposes `ppar.__version__`;
- the current package version is `0.1.5` at this intake;
- the report-bundle manifest contract is version 4;
- output-integrity normalization is version 1;
- review-summary contract is version 1;
- extract-availability and transaction-semantics machine contracts are
  independently versioned; and
- generated bundles contain deterministic table, review-surface, and bundle
  fingerprints.

Current generated bundles do **not** embed the installed PPAR version or a
complete configuration snapshot/fingerprint.

`ARD-058` — Any future readiness artifact MUST record the installed PPAR
version and relevant contract/schema versions. APPROVED DIRECTION.

`ARD-059` — The authoritative run bundle SHOULD record a deterministic
configuration fingerprint and either retain the approved resolved
configuration or identify a controlled immutable copy. APPROVED DIRECTION.

`ARD-060` — A path alone MUST NOT be treated as immutable configuration
identity because the file can change in place. APPROVED DIRECTION.

### 20.10.2 Regression model

Current internal regression infrastructure includes:

- unit and integration tests;
- safety-invariant tests;
- packaged scenario matrix;
- operational demo rebuild/freeze checks;
- portfolio/security bundle generation and validation;
- cross-format parity and determinism checks;
- wheel/install smoke tests; and
- the maintained 500x scale workflow.

This is strong internal engineering evidence, not client accepted-case
regression.

`ARD-061` — Before an operationally relied-upon client upgrade, the new product
and configuration SHOULD be run against a small approved set of client cases
covering known explanations, unresolved results, Data Audit findings, and
source-contract boundaries. APPROVED DIRECTION; CURRENT — REQUIRES CLIENT
VALIDATION.

`ARD-062` — Client regression cases MAY begin as immutable local input/config
directories plus expected semantic outcomes and validated bundles. A database
or cross-run history system is not required. CANDIDATE.

`ARD-063` — Regression comparison MUST focus on semantic outcomes, calculations,
statuses, findings, lineage, and contract versions. Volatile timestamps and
permitted presentation-only differences MUST be excluded explicitly. APPROVED
DIRECTION consistent with SN-11.

`ARD-064` — A product, configuration, source-contract, or mapping change that
alters an accepted semantic outcome MUST produce `Blocked` for operational use
until the change is explained, approved, and the baseline is intentionally
superseded. APPROVED DIRECTION.

`ARD-065` — The absence of client regression cases during the first validation
pilot MUST be disclosed. It may qualify a validation-purpose run; it cannot be
silently treated as established operational regression coverage. APPROVED
DIRECTION.

## 20.11 Actionable remediation

Each target readiness item should include:

- stable check identifier and category;
- disposition: Blocker, Qualification, or Information;
- affected requested scope and snapshot;
- dataset, file/path, field, transaction code, configuration key, or contract
  reference as applicable;
- concise observed condition;
- why it matters;
- required evidence or correction;
- responsible owner role;
- exact next validation action; and
- status after correction only when deterministic.

`ARD-066` — Remediation MUST tell the operator what to change or obtain and who
owns the decision. “Contact support” alone is insufficient when the repository
can identify a source, policy, configuration, or infrastructure owner.
APPROVED DIRECTION.

`ARD-067` — Remediation MUST NOT recommend weakening a safety invariant,
disabling ambiguous-flow enforcement without reviewed evidence, suppressing an
unknown field, or widening a tolerance merely to obtain a pass. CURRENT —
DOCUMENTED doctrine.

`ARD-068` — When multiple remedies are safe, the product SHOULD present the
least expansive correction first—for example, add the required context, narrow
the requested scope, or keep an unsupported item review-only. CANDIDATE.

`ARD-069` — Error output SHOULD avoid unnecessarily echoing sensitive row
values. Stable identifiers, paths, fields, counts, and locally inspectable
details should normally be enough for first-line support. APPROVED DIRECTION.

## 20.12 Reviewer-facing and machine-readable outputs

### 20.12.1 Current output

Current `validate_config` provides a console pass/fail result and, on success,
prints snapshot paths, configured/minimum datasets, required columns, missing
optional files, configured impact methods, transaction-rule count, contract
context, observed transaction codes, codes without YAML rules, and semantic
source counts.

`ARD-070` — This current output is CURRENT — DEMONSTRATED and useful for
maintainers, but it is not a durable readiness decision or complete client
handoff.

### 20.12.2 Target output

The minimum target output is:

1. overall status and declared intended use;
2. requested and evaluated scope;
3. bound snapshot/configuration/product/contract identity;
4. required-check summary;
5. qualifications and their exact interpretation limits;
6. blockers ordered by responsible owner and dependency;
7. informational coverage summary;
8. remediation and rerun instruction;
9. downstream gates that remain; and
10. creation time plus invalidation conditions.

`ARD-071` — Human-readable and machine-readable readiness outputs MUST derive
from one validated result model. APPROVED DIRECTION under the same parity
doctrine as report outputs.

`ARD-072` — The initial machine-readable artifact SHOULD be a small local
versioned JSON object or equivalent embedded in the eventual evidence pack. It
MUST NOT require a database, hosted service, or human-workflow platform.
APPROVED DIRECTION.

`ARD-073` — A `Blocked` preflight MAY be retained locally for troubleshooting,
but it MUST be clearly separated from a successful audit evidence pack and
must not imply that reports were generated. APPROVED DIRECTION.

## 20.13 Status transitions and re-evaluation

Readiness is recalculated, not manually closed:

```text
Blocked
  -> correct required input/policy/environment
  -> rerun preflight
  -> Ready or Qualified Ready

Qualified Ready
  -> add optional evidence or safely narrow interpretation
  -> rerun preflight
  -> Ready or remain Qualified Ready

Ready / Qualified Ready
  -> bound input, contract, configuration, version, or scope changes
  -> Invalidated
  -> rerun preflight
```

`ARD-074` — `Invalidated` is a lifecycle condition, not a fourth readiness
status. An invalidated result cannot authorize a run until reevaluated.
CANDIDATE.

`ARD-075` — Readiness transitions MUST NOT change Phase 3A analytical status,
Phase 3B issue results, or future human workflow status. APPROVED DIRECTION.

`ARD-076` — Manual comments, assignments, approvals, and closure remain outside
the initial readiness object except for a minimal external approval reference.
The deferred workflow decision remains unchanged.

## 20.14 Failure behavior and edge cases

| Condition | Readiness treatment | Current/target status |
|---|---|---|
| Missing required performance file | Blocked | CURRENT — DEMONSTRATED |
| Missing reconstruction-required holdings or transactions | Blocked | CURRENT — DEMONSTRATED |
| Missing configured optional file | Information today; Qualification only when the target can establish a limitation | CURRENT — DEMONSTRATED listing; CANDIDATE status |
| Optional file exists but has invalid required columns | Blocked for its configured contribution; narrow scope only if explicitly safe | CURRENT — DOCUMENTED |
| Duplicate YAML key | Blocked | CURRENT — DEMONSTRATED |
| Unsupported/malformed reconstruction policy | Blocked | CURRENT — DEMONSTRATED |
| Unknown changed field or missing required impact policy | Blocked | CURRENT — DEMONSTRATED |
| Diagnostic incomplete-YAML option | Diagnostic Only; never Ready | CURRENT — DOCUMENTED |
| Ambiguous flow codes without contract context | Blocked | CURRENT — DEMONSTRATED |
| Reviewed ambiguous-flow opt-out | Requires approval reference; ready only within declared risk boundary | APPROVED DIRECTION |
| Safe unmatched/ambiguous transaction identity | Qualification or later review evidence when formula ownership remains safe | CURRENT — DOCUMENTED behavior; CANDIDATE readiness wording |
| Unsafe currency/unit/FX basis | Blocked | CURRENT — DEMONSTRATED |
| Reversed/overlapping periods | Blocked | CURRENT — DEMONSTRATED |
| Continuity mismatch | Nonblocking mandatory review finding | CURRENT — DEMONSTRATED |
| No reported performance changes | May still be Ready; report/data-quality result may simply be clean | CURRENT — DOCUMENTED |
| A and B resolve to the same files or content | Block unless an explicit validation purpose makes this intentional | CANDIDATE |
| Requested security report lacks security data | Block requested scope or obtain authorization to narrow to portfolio | APPROVED DIRECTION |
| Estimated output risk | Qualification or blocker based on declared hard boundary; generation limit remains final | CANDIDATE |
| Review table exceeds 100,000 rows | Generation blocks and writes no files for that report | CURRENT — DEMONSTRATED |
| Output destination unwritable or insufficient | Block before writing when detectable | CANDIDATE preflight; current write failure remains possible |
| Internal arithmetic/lineage/parity/determinism failure | Not a readiness qualification; run/bundle invalid | CURRENT — DOCUMENTED |
| Product/config changed since readiness | Invalidate and rerun | APPROVED DIRECTION |
| Client accepted-case regression differs | Block operational use pending review | APPROVED DIRECTION |

`ARD-077` — Narrowing scope is a new request, not a waiver. The readiness
output MUST show the original blocker and the newly authorized scope.
APPROVED DIRECTION.

`ARD-078` — If identical or apparently stale snapshots are intentionally used
for a clean baseline test, the purpose and expected zero-change behavior SHOULD
be explicit. CANDIDATE.

## 20.15 Lineage, audit trail, and safety invariants

`ARD-079` — Every readiness item SHOULD trace to the evaluated configuration
key, contract clause, source file/header, normalized validation, regression
case, or runtime condition that produced it. CANDIDATE.

`ARD-080` — The readiness result MUST preserve the distinction among:

- source-contract error;
- configuration/policy error;
- optional evidence limitation;
- visible review finding;
- internal logic failure; and
- downstream output failure.

APPROVED DIRECTION.

`ARD-081` — Phase 3C directly inherits:

| Invariant | Readiness implication |
|---|---|
| SN-01 No lost differences | Diagnostic bypass cannot authorize omission of an observed unclassified change |
| SN-02 No double counting | Readiness cannot approve ambiguous effect ownership |
| SN-03 Fully Explained arithmetic | Readiness does not predeclare explanation success; downstream reconciliation remains mandatory |
| SN-04 Continuity | Continuity remains mandatory and visible, not a disabling option |
| SN-05 Bidirectional lineage | A successful pack still requires complete lineage |
| SN-06 Currency/unit consistency | Unsafe monetary basis blocks |
| SN-07 Period-boundary safety | Reversed/overlapping or ambiguous periods block |
| SN-08/09 Demo contracts | Internal fixture readiness cannot become client compatibility evidence |
| SN-10 Report parity | Readiness cannot authorize inconsistent output meanings |
| SN-11 Deterministic output | Bound content and semantic regression exclude declared volatility only |
| SN-12 Fail-closed policy | Unknown required role, semantics, or impact policy blocks |

CURRENT — DOCUMENTED product contract.

`ARD-082` — A Phase 3C implementation change is unacceptable if it converts a
current blocker to qualification without explicit product approval and evidence
that safety is preserved. APPROVED DIRECTION.

## 20.16 Scale and performance

Current `validate_config` executes a real comparison, so it can detect
data-dependent policy gaps but may approach audit processing cost. The report
then repeats some work and adds output-specific gates.

`ARD-083` — Preflight performance MUST be measured separately for static
validation, source loading, data-dependent checks, and output-risk assessment.
CANDIDATE.

`ARD-084` — Cheap required source, currency, period, policy, and integrity
checks SHOULD remain enabled in ordinary production operation. Redundant
full-artifact reparsing and similarly expensive independent checks MAY remain
in test/release gates when production cost is material. CURRENT — DOCUMENTED
project doctrine.

`ARD-085` — Readiness MUST fail safely rather than sample away a required
condition merely to meet a runtime target. APPROVED DIRECTION.

`ARD-086` — Large-site validation SHOULD report row counts by dataset,
comparison scope, missing optional files, issue counts by readiness category,
elapsed time, and estimated review-output amplification. CANDIDATE.

`ARD-087` — Major readiness, reporting, financial-integrity, or cross-cutting
changes remain subject to the maintained 500x release-candidate scale check.
CURRENT — DOCUMENTED.

## 20.17 Local-first, security, privacy, and support

`ARD-088` — All snapshot inspection, configuration resolution, contract
evaluation, fingerprinting, regression comparison, and readiness output MUST
execute and remain inside the client-controlled environment during ordinary
operation. APPROVED DIRECTION.

`ARD-089` — Audit Readiness MUST NOT require hosted portfolio-data processing,
remote rule execution, or routine evidence upload. OUT OF SCOPE.

`ARD-090` — Product/version update checks, if later added, MAY exchange only
explicitly approved non-portfolio metadata. Exact entitlement, update,
connectivity, and offline-grace fields remain OPEN DECISION.

`ARD-091` — A readiness output intended for support SHOULD separate
non-sensitive environment/version/check metadata from portfolio identifiers,
paths, values, and business policy. Any transfer remains explicit and
client-authorized. APPROVED DIRECTION.

`ARD-092` — Readiness does not grant file or report access. Client operating-
system permissions and governance remain the current access boundary; built-in
role-based authorization is not current. CURRENT — DOCUMENTED.

## 20.18 Acceptance criteria

Phase 3C is functionally acceptable only when all applicable criteria pass:

1. **Bound request:** purpose, scope, snapshots, outputs, configuration,
   contract, and product context are explicit.
2. **Status:** Ready, Qualified Ready, and Blocked follow one deterministic
   precedence; Diagnostic Only cannot masquerade as ready.
3. **Required evidence:** structural and conditional requirements block when
   absent.
4. **Optional evidence:** safe limitations are distinguished from irrelevant
   omissions and unsafe gaps.
5. **Source contract:** extract shape, provenance, context, approval reference,
   and limitations are visible without claiming universal vendor behavior.
6. **Configuration:** syntax, structure, observed-case policy, selected method,
   and approval completeness are distinct.
7. **Semantics:** unknown fields and unsafe transaction meaning fail closed;
   review-only quarantine remains available when safe.
8. **Financial integrity:** currency/unit, FX, period, and reconstruction
   preconditions remain enforced.
9. **Requested output:** portfolio/security scope and output formats are
   evaluated separately; later report/bundle gates remain active.
10. **Remediation:** each blocker/qualification identifies why, owner, evidence,
    correction, and recheck path.
11. **Provenance:** product and contract versions plus configuration/content
    identity make the decision reproducible.
12. **Regression:** operational upgrades compare accepted semantic outcomes;
    first-pilot absence is disclosed.
13. **Outputs:** human and machine readiness views agree and remain local.
14. **Invalidation:** material changes force reevaluation.
15. **Claims:** readiness is safe-to-proceed evidence, not correctness,
    completeness, assurance, or production certification.

Current technical controls demonstrate substantial parts of criteria 3, 6–9,
and 13 within the packaged scope. Unified statuses, optional-evidence impact,
approval references, complete provenance, and client regression remain
APPROVED DIRECTION, CANDIDATE, or CURRENT — REQUIRES CLIENT VALIDATION.

## 20.19 Dependencies and approved working assumptions

### Dependencies

- founder-approved Phase 2 actors, decision rights, and exception paths;
- founder-approved Phase 3A input, policy, attribution, and failure boundaries;
- founder-approved Phase 3B rule configuration and integrated placement;
- current source-data contract, field-role registry, transaction-semantics
  matrix, extract-contract templates, and site-readiness checklist;
- current configuration validator, source loaders, financial-integrity checks,
  output guardrails, safety invariants, and bundle contract;
- client source, methodology, technology, and approval owners; and
- accepted client regression cases only after validation creates them.

### Founder-approved working assumptions

The founder approved these six working assumptions on 2026-07-16:

1. **Status model:** use `Ready`, `Qualified Ready`, and `Blocked` with the
   precedence in `ARD-008`, while keeping `Diagnostic Only` outside the
   readiness statuses.
2. **Operator surface:** expose one readiness engine both as an explicit
   no-report preflight and automatically at the start of `ppar audit`. Exact
   command naming remains implementation design.
3. **Readiness artifact:** create a small local versioned machine-readable
   summary and include the successful result in the eventual evidence pack;
   do not add a database.
4. **Approval provenance:** record a reference to the client's existing
   approval/change-control record plus role/date/fingerprint rather than
   building PPAR workflow.
5. **Version and configuration binding:** add installed product version,
   relevant contract versions, and resolved configuration/content fingerprints
   to readiness and run provenance.
6. **Intended-use distinction:** distinguish controlled validation from
   operational review so absence of client regression/approval evidence can
   qualify a validation run but block operational reliance.

These decisions add a bounded preflight record, not the deferred comments,
assignment, case-management, or cross-run history infrastructure.

## 20.20 Real-client validation plan

### Stage 1 — Source and approval discovery

- identify Axys/APX version, report/export tool, extract procedure, dataset
  inventory, and accountable owners;
- classify each field/file as structurally required, conditionally required,
  optional limiting, or optional nonlimiting for the selected case;
- document source, methodology, technology, and approval responsibilities;
- identify the client's existing change-control reference mechanism;
- run current validation and compare its output with the client's expectation.

**Exit evidence:** approved pilot request and source/extract contract with no
guessed semantics.

### Stage 2 — Known readiness cases

- run known-good inputs;
- remove or corrupt required files/columns;
- remove optional evidence;
- introduce unknown fields, ambiguous flow context, currency/unit conflicts,
  period overlap, incomplete policy, and output-scope mismatches;
- verify status precedence and remediation ownership;
- confirm Diagnostic Only never becomes ready.

**Exit evidence:** labeled Ready, Qualified Ready, and Blocked cases that source
and methodology owners agree are correctly classified.

### Stage 3 — Independent operator remediation

- have a client operator run preflight without founder guidance;
- measure time to identify and route each issue;
- have the proper owner correct or approve the condition;
- rerun and verify deterministic status transition;
- confirm sensitive data remains local during support.

**Exit evidence:** another authorized operator can reach a correct readiness
decision and remediation path using bounded support.

### Stage 4 — Change and regression validation

- change extract schema, transaction family, configuration, product version,
  and requested scope in controlled cases;
- confirm prior readiness invalidates;
- rerun accepted client cases and compare semantic outcomes;
- test approval-reference and version/configuration provenance;
- measure validation cost and false readiness classifications.

**Exit evidence:** an evidence-supported client change-control procedure and
decision on which provenance/regression fields are required for repeatability.

## 20.21 Claims supported and not supported

### Claims supported now, with qualification

- PPAR currently validates substantial configuration, required-file, normalized
  schema, transaction-context, accounting-role, policy, currency/unit, period,
  and reconstruction conditions before or during audit processing.
- The current validator reports resolved datasets, minimum required columns,
  missing optional files, extract-contract context, and transaction semantic
  coverage for the supplied snapshots.
- Current audit generation retains fail-closed arithmetic, lineage,
  report-size, parity, determinism, and bundle-validation gates.
- The packaged configuration and demo matrix pass their current validators.

Each claim remains CURRENT — REQUIRES CLIENT VALIDATION for a real site.

### Claims not supported

- that current validation is a completed unified readiness product;
- that a passing config proves the client's extract is complete or policy is
  correct;
- that a pass covers fields or transaction families absent from the supplied
  snapshots;
- that missing optional evidence never limits explanation;
- that current bundles embed the installed product version or complete
  configuration snapshot/fingerprint;
- that PPAR currently stores attributable contract/configuration approvals;
- that internal demo/release regression equals client accepted-case regression;
- that `Ready` guarantees a successful later report or fully explained result;
- that `Qualified Ready` can waive a safety blocker;
- that Audit Readiness provides independent assurance, certification,
  production approval, regulatory compliance, or official-performance
  correctness; or
- that packaged Axys/APX readiness establishes universal Axys/APX
  compatibility.

## 20.22 Phase 3C approval and next gate

Phase 3C was founder-approved on 2026-07-16 with the six working assumptions
in Section 20.19. It defines the current technical baseline, approved readiness
statuses, preflight layers, required/optional evidence model, remediation,
source-contract approval boundary, configuration completeness,
version/regression behavior, outputs, invalidation, acceptance criteria, and
client-validation plan.

That approval authorized Phase 3D drafting; it did not authorize application
code changes or begin Phase 3E.

---

# 21. Phase 3D — Executive Investigation Summary

## 21.1 Specification identity, purpose, and status

**Capability:** Executive Investigation Summary

**Primary user:** performance/operations manager or senior reviewer

**Primary first-pilot surface:** the first view of every authoritative
portfolio investigation report, with direct paths to analyst evidence

**Overall status:** CURRENT — DEMONSTRATED and CURRENT — DOCUMENTED for the
shared analytical tables, three reviewer surfaces, derived triage/supporting
tables, handoff metadata, and evidence-pack validation; CURRENT — REQUIRES
CLIENT VALIDATION for comprehension and operating usefulness; APPROVED
DIRECTION for the intended executive entry layer

The Executive Investigation Summary answers:

> What changed, how completely is it explained, what requires attention, what
> should happen next, and where is the evidence—without asking management to
> reconstruct the answer from technical tables?

It is a communication and navigation layer over an authoritative investigation.
It is not a second calculation, a workflow decision, a certification, a
confidence score, or a replacement for detailed evidence.

This section follows the `MUST`, `SHOULD`, and `MAY` meanings in Section 18.1.
Requirement identifiers use `EIS` for Executive Investigation Summary.

`EIS-001` — Every summary value, status, cause, finding, and action cue MUST be
derived from the same validated result model as the detailed report. The
summary MUST NOT recalculate performance independently. APPROVED DIRECTION.

`EIS-002` — The summary MUST preserve the independent meanings of Audit
Readiness, Performance Change Investigation, Performance Data Quality Audit,
and any future human disposition. It MUST NOT collapse them into one opaque
health or confidence score. APPROVED DIRECTION.

`EIS-003` — The initial product MUST remain deterministic and local-first. A
hosted service, database, generative narrative engine, reviewer-comment system,
or cross-run history index is not required. APPROVED DIRECTION consistent with
the founder-approved infrastructure boundary.

## 21.2 User problem, actors, trigger, and authority

### 21.2.1 User problem

The current portfolio and security reports begin with `Performance
Differences`, followed by `Performance Difference Causes` and `Data Audit
Issues`. The bundle also contains useful derived tables such as
`needs_review_summary.csv`, `portfolio_period_summary.csv`,
`cause_summary.csv`, `impact_coverage.csv`, `context_evidence_summary.csv`, and
`residual_status.csv`, plus compact `review_summary.json` handoff metadata.

These are strong analytical and machine-handoff foundations, but they do not
yet provide one concise management answer. A manager must currently infer:

- whether any performance result remains partly explained or unexplained;
- which individual changed periods are largest or highest priority;
- which supported causes matter most within those periods;
- whether independent Data Audit issues require attention;
- which limitations prevent stronger interpretation;
- what action belongs to which owner; and
- which report row or supporting artifact proves each statement.

`EIS-004` — The summary MUST reduce navigation effort without reducing evidence
depth. Complete detail MUST remain available to authorized reviewers.
APPROVED DIRECTION.

`EIS-005` — The summary MUST not describe the current `review_summary.json` as
an executive narrative. That file is compact review-handoff metadata containing
entrypoints, source context, counts, transaction semantics, and artifact
references. CURRENT — DEMONSTRATED.

`EIS-006` — The summary MUST not describe the current HTML as already containing
a separate first-screen `Problems` grid. Current implementation and generated
artifacts begin with the shared `Performance Differences` table. A lower-level
implementation-design statement to the contrary is superseded by executable
and generated evidence. CURRENT — DEMONSTRATED correction.

### 21.2.2 Actors and authority

| Actor | Summary responsibility | Authority boundary |
|---|---|---|
| Performance/operations analyst | Validates scope, explanation status, cause interpretation, findings, and evidence navigation before reliance | Cannot rewrite calculated values or mark unresolved evidence as explained |
| Performance/operations manager | Uses the first view to prioritize review and decide whether external use needs further approval | Cannot waive readiness, arithmetic, lineage, or source-contract blockers |
| Methodology owner | Approves return method, impact treatment, materiality policy, and action language tied to methodology | Does not establish source completeness or official correctness merely by approving wording |
| Source/extract owner | Resolves source, export, mapping, and missing-evidence actions named by the summary | Does not approve performance interpretation unless separately authorized |
| Compliance/GIPS reviewer | Reviews claims, material unresolved items, and permitted external use where applicable | Does not turn PPAR output into independent assurance |
| Local administrator | Generates, validates, retains, and securely distributes the local evidence pack | Does not approve analytical conclusions solely through technical access |
| PPAR | Derives concise facts, statuses, priorities, action cues, and evidence references deterministically | Does not close investigations, approve official returns, or infer missing human judgment |

`EIS-007` — Summary language MUST make product-calculated facts, client-approved
policy, and human decisions distinguishable. APPROVED DIRECTION.

`EIS-008` — A manager's use of the summary MUST NOT alter an analytical status
or create a disposition record. Human comments, assignment, approval, closure,
and reopening remain DEFERRED pending a serious business case and client
validation. APPROVED DIRECTION.

### 21.2.3 Trigger and preconditions

The summary is generated only after:

1. the declared audit request passes applicable readiness and runtime gates;
2. source differences and Data Audit findings are calculated;
3. canonical review tables reconcile;
4. explanation, conservation, ownership, and lineage invariants pass; and
5. the requested report/bundle can be validated.

`EIS-009` — An authoritative summary MUST be generated as part of the same
successful report-bundle transaction as its detailed evidence. It MUST NOT be
published from partial or unvalidated intermediate files. APPROVED DIRECTION.

`EIS-010` — A `Blocked` readiness result MUST not produce an authoritative
investigation summary. A separately labeled diagnostic preflight record may
exist, but it is not an investigation result. APPROVED DIRECTION.

`EIS-011` — If summary derivation, cross-reference validation, or output parity
fails, the evidence pack MUST fail rather than silently omit or stale the
summary. APPROVED DIRECTION consistent with SN-10 and SN-11.

## 21.3 Capability boundary and shared truth model

### 21.3.1 One model, several presentations

The target model is one versioned in-memory summary result derived from current
validated tables. That result may render as:

- the first section of the HTML report;
- the first sheet of the XLSX workbook;
- a small machine-readable summary artifact for CSV-only and integration use;
  and
- entrypoint and evidence-reference metadata in the bundle manifest/handoff.

The three detailed analytical surfaces remain:

1. `Performance Differences`;
2. `Performance Difference Causes`; and
3. `Data Audit Issues`.

The Executive Investigation Summary is an entry layer over those surfaces, not
a fourth source of analytical truth.

`EIS-012` — HTML, XLSX, and machine-readable summary outputs MUST share the same
canonical fields, values, statuses, ordering keys, and evidence references.
Presentation-only differences MAY improve usability but MUST NOT change
meaning. APPROVED DIRECTION.

`EIS-013` — The summary SHOULD be assembled from already computed table caches
or a shared result object. It SHOULD NOT fully reparse artifacts or repeat an
expensive comparison merely to create presentation. CANDIDATE implementation
design.

`EIS-014` — Full-artifact reparsing and independent format comparison belong in
tests and release-candidate validation when production cost would be material.
Inexpensive reconciliation invariants MUST remain enabled in production.
APPROVED DIRECTION consistent with project test-gate doctrine.

### 21.3.2 Independent truth rails

The summary must show at least these independent dimensions:

| Dimension | Source truth | Permitted summary statement |
|---|---|---|
| Readiness/run validity | Phase 3C result and successful bundle gates | Evaluated scope and limitations; never a correctness certification |
| Performance change | `Performance Differences` | Changed review units and exact per-unit explanation status |
| Cause evidence | counted and supporting rows in `Performance Difference Causes` plus lineage | Largest eligible causes and their evidence role |
| Data quality | `Data Audit Issues` and rule-execution coverage when available | Priority findings and coverage limitations; never automatic causation |
| Human disposition | No current product layer | Not shown as calculated closure; future only if separately implemented |

`EIS-015` — A `Fully Explained` performance result MUST NOT suppress or downgrade
an independent Data Audit issue. A clean Data Audit result MUST NOT change a
`Partly Explained` or `Unexplained` performance result. APPROVED DIRECTION.

`EIS-016` — The initial summary MUST NOT create a single overall “passed,”
“healthy,” or “complete” status. It SHOULD present separate performance and
Data Audit attention statements, with readiness/run provenance separately.
APPROVED DIRECTION.

## 21.4 Content hierarchy

### 21.4.1 Required first-view order

The initial summary should use the following hierarchy:

1. **Investigation identity and scope** — portfolio/security level, entities,
   comparison snapshots, reporting span, intended use, and generated artifact.
2. **Attention statements** — separate performance-explanation and Data Audit
   outcomes, without a composite score.
3. **Key counts** — changed review units by analytical status and Data Audit
   items by supported priority/coverage category.
4. **Priority performance changes** — the most important changed review units,
   with performance difference, explained amount, residual, and status.
5. **Largest supported causes** — cause families within the displayed review
   unit, with basis and evidence role.
6. **Unresolved and limited items** — residuals, withheld residuals, missing
   inputs, evidence-only areas, ambiguous or unsupported interpretation, and
   safe limitations.
7. **Data Audit highlights** — independent findings and rule-coverage caveats.
8. **Recommended actions** — deterministic, owner-routed next steps.
9. **Evidence navigation** — stable links/references to detailed rows and
   supporting artifacts.
10. **Method, provenance, and limitations** — calculation basis, configuration
    identity, product/contract versions, and claims boundary.

`EIS-017` — The first screen/page SHOULD answer scope, attention, most important
change, unresolved state, highest-priority Data Audit issue, and next action
without requiring detailed-table reading. APPROVED DIRECTION; exact layout is
CANDIDATE pending client comprehension testing.

`EIS-018` — The first view MUST stay bounded. It MUST show complete counts and
limitations even when it displays only a prioritized subset of rows. Full
detail MUST remain one navigation step away. APPROVED DIRECTION.

`EIS-019` — “No reportable performance differences” is permitted only when the
canonical `Performance Differences` population is empty for the declared
scope. It MUST NOT imply that Data Audit found no issue or that source-data is
correct. APPROVED DIRECTION.

`EIS-020` — “No Data Audit issues detected” is permitted only for rules that
were actually evaluated over their declared eligible populations. Until
rule-execution coverage is implemented, the summary MUST use narrower wording
and disclose unavailable or insufficient rule evidence. APPROVED DIRECTION.

### 21.4.2 Investigation identity and scope

Required identity fields should include:

- report title and stable bundle/run identity;
- portfolio or security comparison level;
- requested entities or an explicit all-configured-scope statement;
- earliest and latest applicable period dates;
- Snapshot A and Snapshot B labels/source context;
- intended use: controlled validation or operational review;
- output generation time;
- installed PPAR and relevant contract versions when Phase 3C provenance is
  implemented; and
- resolved configuration/content fingerprint when implemented.

`EIS-021` — Scope labels MUST distinguish the compared reporting span from
individual affected periods. A minimum/maximum date range MUST NOT imply that
every intervening period changed or was evaluated by every optional rule.
APPROVED DIRECTION.

`EIS-022` — Large entity populations MAY be summarized by count on the first
view, but the exact included population MUST remain available in the validated
evidence pack. APPROVED DIRECTION.

## 21.5 Explanation completeness

### 21.5.1 Definition

Explanation completeness is evaluated separately for each canonical review
unit: portfolio-period for portfolio reporting and security-period (with its
declared return container) for security reporting.

For a review unit:

```text
reported performance difference
  = explained performance difference + unexplained difference
```

The corresponding Phase 3A statuses are:

- **`Fully Explained`** — the explained difference reconciles to the reported
  difference within fixed product precision and no unresolved owned amount
  remains;
- **`Partly Explained`** — some supported amount is explained and a reportable
  unexplained amount remains; and
- **`Unexplained`** — no supported owned cause explains the reportable change.

`EIS-023` — The summary MUST reuse the exact Phase 3A arithmetic, precision,
and status from the canonical performance table. It MUST NOT recompute the
status from rounded display values. APPROVED DIRECTION.

`EIS-024` — The initial summary MUST show status plus the full-precision
explained and unexplained amounts available through the underlying contract.
It MUST NOT add an explanation-completeness percentage. Founder-approved
working assumption.

`EIS-025` — A report-level explanation summary MUST use counts of review units
by status. It MUST NOT sum or average return differences, explained amounts,
or residuals across unrelated portfolios, securities, periods, currencies, or
return containers. APPROVED DIRECTION.

`EIS-026` — A rounded visual zero MUST NOT be labeled Fully Explained unless the
unrounded canonical arithmetic satisfies the Phase 3A invariant. APPROVED
DIRECTION.

### 21.5.2 Withheld residuals and incomplete coverage

Current supporting evidence can label a residual
`withheld_partial_estimates` when coverage is incomplete or estimates may
overlap. That status protects reviewers from treating an unsafe subtraction as
an authoritative residual.

`EIS-027` — When the authoritative review table has a valid unexplained amount,
the summary MUST show it. When the applicable residual is withheld, the summary
MUST say that it is withheld and why; it MUST NOT derive a replacement by
subtracting a partial estimate total. APPROVED DIRECTION.

`EIS-028` — `impact_coverage.csv`, `residual_status.csv`, and
`needs_review_summary.csv` are current supporting inputs for limitations and
triage. They MUST NOT override the canonical `Performance Differences` status
or become a second explanation calculation. CURRENT — DEMONSTRATED boundary.

## 21.6 Priority performance changes

Priority ordering inherits the founder-approved Phase 3A model:

1. unresolved data-integrity or safety concern;
2. `Unexplained` before `Partly Explained` before `Fully Explained`;
3. client-approved operational materiality when configured;
4. recurring or policy-ambiguous conditions when supported by evidence;
5. larger absolute performance difference within a comparable basis; and
6. stable portfolio/security/date tie-breakers.

`EIS-029` — Operational materiality MAY prioritize a reportable item but MUST
NOT suppress it or change explanation arithmetic. APPROVED DIRECTION.

`EIS-030` — Until client-approved materiality exists, the summary MUST avoid
calling an item “material.” It MAY use “largest” for a transparently defined
absolute magnitude within comparable review units. APPROVED DIRECTION.

`EIS-031` — A bounded first view SHOULD display the highest-priority review
units and the number not shown. The default row limit is CANDIDATE and must be
tested with real reviewers; it is not a financial threshold.

`EIS-032` — Stable ordering MUST be deterministic across HTML, XLSX, and
machine-readable output. APPROVED DIRECTION.

## 21.7 Largest supported causes

### 21.7.1 Eligibility

A cause is eligible for an executive “largest supported causes” statement only
when:

- it belongs to the displayed review unit;
- its economic effect is owned once under the Phase 3A model;
- it is counted in the canonical explained amount or explicitly labeled as
  non-additive support;
- its impact basis is compatible with the review unit;
- its lineage resolves to source or derived formula evidence; and
- any confidence/coverage limitation remains visible.

`EIS-033` — Cause ranking MUST use authoritative counted cause rows or a
validated aggregation over them. It MUST NOT rank raw input-value differences,
context rows, cross-checks, Data Audit findings, or overlapping candidate
estimates as though they were additive return causes. APPROVED DIRECTION.

`EIS-034` — Causes MAY be aggregated into approved cause families within one
review unit only when all included effects share compatible units, sign
convention, impact basis, and ownership. APPROVED DIRECTION.

`EIS-035` — The initial product MUST NOT total a cause family across unrelated
periods or portfolios for the executive summary. Cross-run and trend analysis
belongs to Phase 3E and remains gated. APPROVED DIRECTION.

`EIS-036` — A “largest cause” row SHOULD show cause family, signed explained
impact, plain-language effect, basis/method, and evidence reference. If the
effect is review-only or evidence-only, the wording MUST say so and omit it
from the additive ranking. APPROVED DIRECTION.

### 21.7.2 Current table boundary

Current `cause_summary.csv` groups period-level cause areas and exposes
estimated return impact, impact basis, confidence, representative codes, and a
message. Current `Performance Difference Causes` is the reviewer-facing cause
surface whose counted rows reconcile to the displayed explanation.

`EIS-037` — `cause_summary.csv` is a useful current foundation, but it MUST NOT
be promoted unchanged as the executive largest-cause model until eligibility,
aggregation, and parity with the canonical explained amount are explicitly
validated. CANDIDATE.

`EIS-038` — If no eligible quantified cause exists, the summary MUST say
“No supported quantified cause” or equivalent. It MUST NOT promote a possible
cause merely to avoid an empty section. APPROVED DIRECTION.

## 21.8 Unresolved items and limitations

The summary should distinguish:

- valid unexplained amount;
- partly explained amount;
- withheld residual because estimates are incomplete or overlapping;
- missing impact input;
- evidence-only cause area;
- review-only context;
- ambiguous or unsupported transaction meaning;
- Data Audit issue requiring independent review;
- unavailable or insufficient optional rule evidence;
- readiness or provenance qualification retained with an otherwise successful
  validation-purpose run; and
- downstream limitation on claims or external use.

`EIS-039` — Each unresolved item MUST state the affected review unit or scope,
the limitation type, why it matters, the responsible owner role, the next safe
action, and an evidence reference. APPROVED DIRECTION.

`EIS-040` — The summary MUST preserve “unknown,” “not evaluated,” “withheld,”
and “unexplained” as distinct conditions. It MUST NOT render any of them as
zero, clean, or not applicable without evidence. APPROVED DIRECTION.

`EIS-041` — Suppressed findings MUST remain in the complete audit trail and
summary counts/provenance where contractually required. A suppression MUST NOT
make a material unresolved condition disappear from management visibility.
APPROVED DIRECTION; precise suppression-summary policy is CURRENT — REQUIRES
CLIENT VALIDATION.

## 21.9 Data Audit highlights

Data Audit highlights are an independent section sourced from `Data Audit
Issues` and, when implemented, Phase 3B rule-execution coverage. The first-pilot
priority model reuses Phase 3A's approved operational prioritization rather
than creating a separate score.

`EIS-042` — The summary MUST label Data Audit rows as findings or review issues,
not proven errors and not performance causes. APPROVED DIRECTION.

`EIS-043` — A mandatory continuity finding and any client-prioritized integrity
issue MUST remain visible even when every performance difference is Fully
Explained. APPROVED DIRECTION.

`EIS-044` — The summary SHOULD show check/rule identity, snapshot, affected
portfolio/security/date scope, observed relationship, priority basis, and
evidence link for each highlighted issue. APPROVED DIRECTION.

`EIS-045` — Current `x_ref_issues.csv` may contain multiple affected rows for a
shared condition. Until a validated stable issue-group identifier exists, the
summary MUST describe row counts and affected entities; it MUST NOT claim a
precise number of independent incidents. APPROVED DIRECTION.

`EIS-046` — The absence of an optional dataset, insufficient peer population,
or disabled rule MUST qualify coverage rather than appear as “no issue.”
APPROVED DIRECTION inherited from Phase 3B.

## 21.10 Recommended actions

Initial actions should be deterministic templates selected from calculated
conditions and routed to an owner role. Examples include:

- review a named unexplained period and linked source detail;
- complete or approve a named attribution/configuration input;
- obtain missing source or transaction context from the extract owner;
- review a Data Audit finding in the source system;
- validate a low-confidence or evidence-only area before relying on totals;
- narrow requested scope safely and rerun;
- retain the current result with an explicit limitation; or
- escalate a methodology or external-use decision to the authorized owner.

`EIS-047` — Recommended actions MUST be traceable to a calculated status,
finding, limitation, or approved policy. The product MUST NOT invent a business
decision or recommend altering source records without human validation.
APPROVED DIRECTION.

`EIS-048` — Action ordering SHOULD follow dependency and owner: safety/source
contract first, then methodology/configuration, then analytical review, then
external-use decision. APPROVED DIRECTION.

`EIS-049` — Current deterministic `suggested_next_step` cues in
`needs_review_summary.csv` are a useful foundation. They require client
language testing and broader condition coverage before they can serve as the
complete executive action model. CURRENT — DEMONSTRATED; CURRENT — REQUIRES
CLIENT VALIDATION.

`EIS-050` — The initial summary MUST NOT contain editable comments, assignment,
approval, due-date, or closure fields. Those features remain DEFERRED and would
require a separately justified workflow design. APPROVED DIRECTION.

## 21.11 Evidence links and bidirectional lineage

Every displayed review item should carry a stable evidence target appropriate
to its output format:

- **HTML:** a local relative anchor to the exact detailed row or the smallest
  applicable detailed section;
- **XLSX:** an internal hyperlink to the exact sheet/row when stable, otherwise
  to the filtered detail sheet with the review key visible;
- **machine-readable output:** artifact name, stable review key, row/finding
  fingerprint where applicable, and relationship type; and
- **supporting bundle:** relative artifact paths only, never a hosted link.

`EIS-051` — A summary fact MUST trace backward to its canonical review row and,
for causes or findings, to source/derived lineage. Detailed evidence SHOULD
trace forward to the summary item when it was promoted. APPROVED DIRECTION.

`EIS-052` — Evidence references MUST use stable keys or fingerprints, not row
position alone. Sorting, pagination, or presentation changes MUST NOT silently
retarget a link. APPROVED DIRECTION.

`EIS-053` — Links MUST remain local and portable within the validated bundle.
They MUST NOT expose absolute workstation paths, require network access, or
send portfolio data outside the client-controlled environment. APPROVED
DIRECTION.

`EIS-054` — A broken required summary-to-detail reference MUST invalidate the
authoritative bundle. CANDIDATE implementation; required by the existing
lineage and bundle-integrity doctrine.

## 21.12 Management and analyst views

### 21.12.1 One progressive summary

The initial product should provide one progressive summary rather than separate
management and analyst report packages:

- management sees concise identity, attention statements, key counts, priority
  items, actions, and limitations first;
- analysts use the same summary to open exact calculation and evidence detail;
  and
- both audiences see the same numbers, statuses, and ordering.

`EIS-055` — The initial distinction between management and analyst use MUST be
progressive disclosure over one validated result, not separate calculations or
independently editable narratives. APPROVED DIRECTION.

`EIS-056` — The management first view MUST not hide uncertainty, Data Audit
issues, coverage gaps, or claim limitations merely to appear concise.
APPROVED DIRECTION.

`EIS-057` — Separate role-based summary artifacts or permission-specific views
are DEFERRED until client access needs and a serious business case justify the
additional parity, distribution, and support burden.

### 21.12.2 Analyst detail

Analyst drilldown should expose:

- canonical full-precision values and review keys;
- calculation and impact basis;
- counted versus supporting/evidence-only role;
- residual and coverage status;
- source and formula lineage;
- Data Audit trigger/tolerance evidence;
- configuration or contract reference; and
- complete artifact references.

`EIS-058` — Analyst detail MAY be more technical, but it MUST not introduce a
different conclusion from the management first view. APPROVED DIRECTION.

## 21.13 Portfolio and security behavior

The founder-approved Phase 3A pilot scope requires a portfolio report for every
pilot and uses the security report only when the investigation needs
security-level drilldown.

`EIS-059` — The portfolio summary MUST be the primary management view for the
initial pilot. It MUST show portfolio-period outcomes even when a security
report is also generated. APPROVED DIRECTION.

`EIS-060` — A security summary MUST preserve the security report's declared
return container and status semantics. It MUST NOT aggregate security returns
or impacts into a portfolio conclusion unless an explicit validated method
owns that aggregation. APPROVED DIRECTION.

`EIS-061` — When both reports exist, each summary SHOULD link to the other at
the bundle level while retaining its own scope and lineage. Shared Data Audit
evidence MUST not be presented as two independent incidents merely because it
appears in both report packages. APPROVED DIRECTION.

`EIS-062` — Absence of a security report MUST be disclosed as scope, not treated
as evidence that no security-level issue exists. APPROVED DIRECTION.

## 21.14 Configuration and permissions

### 21.14.1 Product-controlled behavior

The following should be fixed or version-controlled product behavior initially:

- content hierarchy and section meaning;
- analytical status labels;
- explanation arithmetic and precision;
- cause eligibility and ownership rules;
- independent Data Audit treatment;
- deterministic tie-breakers;
- required limitation language; and
- summary/detail parity and lineage requirements.

`EIS-063` — Client configuration MUST NOT redefine Fully Explained, cause
ownership, residual arithmetic, or the distinction between cause and Data Audit
evidence. APPROVED DIRECTION.

### 21.14.2 Client-controlled behavior

The client may eventually configure, under attributable approval:

- operational-materiality thresholds and dimensions;
- entity/scope selection;
- permitted output formats;
- approved rule enablement, filters, and tolerances;
- report title and controlled business labels; and
- local retention/distribution policy.

`EIS-064` — A configured threshold MAY change priority or first-view inclusion,
but every reportable item and complete audit-trail row MUST remain available.
APPROVED DIRECTION.

`EIS-065` — A presentation preference MUST NOT change machine-readable facts or
remove a mandatory limitation. APPROVED DIRECTION.

`EIS-066` — Summary configuration that affects interpretation MUST be retained
with the run and bound to the Phase 3C configuration identity. APPROVED
DIRECTION.

## 21.15 Processing model

The target derivation sequence is:

1. receive the already validated canonical performance, cause, Data Audit,
   coverage, triage, lineage, and provenance results;
2. establish exact report scope and comparable review units;
3. calculate per-unit attention keys without changing analytical values;
4. select eligible causes and verify their sum/ownership against canonical
   explanation data;
5. select unresolved/limited items and Data Audit highlights independently;
6. derive deterministic action cues and owner roles;
7. create stable evidence references;
8. render all requested formats from one summary result;
9. validate semantic parity, links, counts, ordering, and required limitations;
   and
10. include the summary in bundle fingerprints and final validation.

`EIS-067` — Summary derivation MUST be a pure, deterministic transformation of
validated inputs plus approved configuration. Identical normalized inputs and
configuration MUST produce identical semantic output aside from declared
volatile metadata. APPROVED DIRECTION.

`EIS-068` — The product MUST reconcile summary counts to canonical table
populations and summary financial values to canonical rows before publication.
APPROVED DIRECTION.

`EIS-069` — A displayed subset MUST carry total population, displayed count,
selection basis, and evidence path to the complete set. APPROVED DIRECTION.

`EIS-070` — Recommended-action templates and attention statements MUST have
stable identifiers or versioned vocabulary so client validation and regression
can detect semantic changes. CANDIDATE.

## 21.16 Machine-readable contract and outputs

### 21.16.1 Minimum result shape

The initial machine-readable summary should contain:

- schema/version and bundle identity;
- scope and provenance reference;
- readiness/run-validity reference;
- separate performance and Data Audit attention statements;
- exact counts by analytical status and applicable Data Audit category;
- ordered priority review-unit items;
- eligible cause items and evidence roles;
- unresolved/limitation items;
- Data Audit highlight items and coverage limitations;
- recommended actions with owner role and source condition;
- evidence targets; and
- claim/interpretation limitations.

`EIS-071` — The machine-readable summary SHOULD be a small versioned JSON
artifact or equivalent structured object included in the evidence pack. It
MUST NOT require a database or persistent service. APPROVED DIRECTION.

`EIS-072` — The exact schema is CANDIDATE implementation work. It MUST use typed
numbers/dates, stable enums, explicit null/withheld semantics, and relative
artifact references. CSV-only mode MAY additionally promote flat tables, but
those tables MUST derive from the same object.

`EIS-073` — Current `review_summary.json` SHOULD remain a compatible handoff
foundation or be versioned deliberately. The executive summary MUST not
silently repurpose version 1 fields with new meanings. APPROVED DIRECTION.

### 21.16.2 Human-readable surfaces

The target presentation is:

- **HTML:** Executive Investigation Summary first, followed by the three
  detailed analytical surfaces and any declared appendix detail;
- **XLSX:** Executive Summary first, followed by the same three detailed
  analytical sheets and any declared diagnostics; and
- **CSV-only/integration:** structured executive summary plus the existing
  canonical primary and supporting tables.

`EIS-074` — Adding the entry layer MUST NOT remove or rename the three current
primary analytical tables without a separately versioned contract change.
APPROVED DIRECTION.

`EIS-075` — The summary itself SHOULD be mandatory in every authoritative
portfolio pilot bundle once implemented. Security summary generation follows
the selected security-output scope. CANDIDATE pending founder approval and
client validation.

## 21.17 Failure behavior and edge cases

| Condition | Required summary behavior |
|---|---|
| No reportable performance differences; no evaluated Data Audit rows | State the performance result narrowly; disclose Data Audit coverage rather than claim universal cleanliness |
| No performance differences; Data Audit findings exist | Lead with no reportable performance difference and a separate Data Audit review-required statement |
| Fully explained differences; Data Audit findings exist | Show Fully Explained counts and retain Data Audit highlights independently |
| Partly explained or unexplained difference | Prioritize the affected review unit and show exact residual/status/action |
| Residual withheld | State withheld and reason; do not calculate a replacement |
| Causes exist but none is eligible for additive ranking | State no supported quantified cause; link review-only evidence separately |
| Optional input absent | Show affected capability and coverage limitation; never convert absence to clean result |
| Data Audit rule not evaluated | State not evaluated/unavailable/insufficient according to coverage evidence |
| Very large review population | Show bounded priority subset, exact totals, and link to complete detail |
| Equal priority/magnitude | Use stable entity/date/key tie-breakers |
| Security detail unavailable | Disclose scope limitation; do not imply no security issue |
| Broken link, count mismatch, parity drift, or stale summary | Fail authoritative bundle validation |
| Diagnostic/incomplete-YAML run | Do not publish an authoritative executive investigation summary |

`EIS-076` — Empty-state language MUST name what was evaluated and what was not.
“No issues” without scope and coverage is prohibited. APPROVED DIRECTION.

`EIS-077` — A summary MUST NOT present a report-generation failure as an
analytical outcome. Failure diagnostics remain separate from validated result
content. APPROVED DIRECTION.

`EIS-078` — Unsupported character, long label, extreme magnitude, negative
amount, missing optional description, or empty detail table MUST not break
typed machine output or cause silent value truncation. CANDIDATE test contract.

## 21.18 Scale, performance, local-first, and support

`EIS-079` — Summary work SHOULD scale with canonical summary/review tables and
selected highlights, not with repeated full-artifact parsing. CANDIDATE.

`EIS-080` — First-view row limits MUST bound rendering cost but MUST NOT limit
calculation, counts, validation, or retained evidence. APPROVED DIRECTION.

`EIS-081` — The 500x scale check MUST remain in the release-candidate workflow
after implementing this cross-cutting reporting layer. Any summary-specific
latency/memory budget must be measured before a gate is established; existing
gates MUST NOT be relaxed merely because the new layer fails. CURRENT project
test-gate doctrine.

`EIS-082` — Summary generation, viewing, evidence navigation, and validation
MUST work without internet access inside the client-controlled environment.
APPROVED DIRECTION.

`EIS-083` — Support guidance SHOULD identify the local artifact, review key,
condition ID, and version needed to diagnose a summary issue without requiring
the client to transmit unrestricted portfolio data. APPROVED DIRECTION.

`EIS-084` — Any future remote support transfer remains client-authorized and
data-minimized. The summary feature MUST NOT add telemetry. APPROVED DIRECTION.

## 21.19 Acceptance criteria

Phase 3D is functionally acceptable only when all applicable criteria pass:

1. **Shared truth:** all summary formats derive from the same validated model
   as the detailed analytical surfaces.
2. **Scope:** report level, entities, periods, snapshots, intended use, and
   provenance are explicit.
3. **Independent dimensions:** readiness/run validity, performance explanation,
   Data Audit, and human disposition remain separate.
4. **Explanation completeness:** per-unit status and exact amounts reuse Phase
   3A arithmetic; no percentage is introduced.
5. **Aggregation safety:** returns/residuals are not totaled or averaged across
   incompatible review units.
6. **Cause eligibility:** only owned, compatible, validated causes enter largest
   supported-cause ranking; evidence-only and cross-check rows remain separate.
7. **Unresolved truth:** unexplained, withheld, unknown, not evaluated, and
   optional-limitation states remain distinct.
8. **Data Audit:** findings remain independent of performance causation and
   include honest rule-coverage boundaries.
9. **Actions:** every recommended action is deterministic, owner-routed, and
   traceable to a condition.
10. **Navigation:** required evidence references resolve through stable keys or
    fingerprints.
11. **Views:** management and analyst presentations share facts and differ only
    through progressive disclosure.
12. **Formats:** HTML, XLSX, and machine output agree semantically.
13. **Scale:** first-view rendering is bounded while complete evidence and
    counts remain retained.
14. **Local-first:** no hosted service, database, telemetry, or network is
    required.
15. **Failure:** summary mismatch or broken required lineage invalidates the
    bundle.
16. **Claims:** the summary communicates a validated investigation; it does not
    certify official correctness, assurance, compliance, or closure.

`EIS-085` — Client acceptance MUST include both management comprehension and
analyst evidence-navigation testing. Internal semantic correctness alone is
not sufficient to claim the signature experience works. CURRENT — REQUIRES
CLIENT VALIDATION.

## 21.20 Dependencies and open decisions

### Dependencies

- founder-approved Phase 2 actors, decision rights, workflows, and local-first
  architecture;
- founder-approved Phase 3A explanation arithmetic, status, materiality,
  evidence, and portfolio/security scope;
- founder-approved Phase 3B independent Data Audit model, rule configuration,
  priority reuse, and coverage direction;
- founder-approved Phase 3C readiness, provenance, intended-use, and versioned
  local artifact direction;
- current canonical review tables, supporting summaries, lineage, manifest,
  review handoff, bundle validation, parity, and determinism controls; and
- real-client examples of management questions, terminology comprehension,
  materiality policy, evidence navigation, and acceptable action language.

### Founder-approved Phase 3D working assumptions

The founder approved these six working assumptions on 2026-07-16:

1. **Entry-layer placement:** add the Executive Investigation Summary as the
   first HTML section and first XLSX sheet, while preserving the three existing
   analytical surfaces.
2. **Audience model:** use one progressive summary for managers and analysts
   rather than separate report packages or permissioned views.
3. **Attention model:** show separate performance-explanation and Data Audit
   attention statements; do not introduce a composite status, score, or
   “passed” badge.
4. **Cause ranking:** rank only eligible supported causes within a single
   comparable review unit; prohibit cross-period or cross-portfolio cause
   totals in Phase 3D.
5. **Data Audit counting:** show affected row/entity counts and coverage until
   a stable validated issue-group contract exists; do not call them incident
   counts.
6. **Narrative and actions:** use deterministic statements and action templates
   only; defer editable notes, workflow, and generative narrative.

These decisions add a bounded report-entry layer and a small structured result,
not the deferred comments, assignment, case-management, or cross-run history
infrastructure.

## 21.21 Real-client validation plan

### Stage 1 — Question and language discovery

- observe how managers and analysts currently review restatements;
- collect the first questions they ask and the evidence they trust;
- test `Fully Explained`, `Partly Explained`, `Unexplained`, `withheld`, and
  Data Audit language without coaching;
- identify client-approved materiality and escalation owners; and
- confirm which summary statements are safe for internal and external use.

**Exit evidence:** approved first-view vocabulary, priority policy, and
responsibility map.

### Stage 2 — Labeled summary cases

- generate clean/no-difference, Fully Explained, Partly Explained,
  Unexplained, withheld-residual, evidence-only, and missing-input cases;
- pair each with no Data Audit finding, independent finding, mandatory
  continuity finding, and unavailable-rule evidence as applicable;
- verify that management states the correct conclusion and limitation; and
- verify that analysts reach the exact supporting row without founder help.

**Exit evidence:** labeled semantic cases and comprehension results with no
dimension conflation.

### Stage 3 — Action and handoff testing

- have a manager prioritize cases using the summary;
- have the named owner follow each recommended action;
- measure time to evidence, wrong-path navigation, and language ambiguity;
- confirm deterministic action templates do not imply unauthorized correction;
  and
- test a rerun as a new result without adding workflow infrastructure.

**Exit evidence:** another authorized operator can interpret, route, and
evidence the investigation using bounded support.

### Stage 4 — Scale, parity, and claims validation

- test large entity/period populations and bounded first views;
- validate HTML/XLSX/machine parity and all summary/detail links;
- test version/configuration changes and accepted semantic regressions;
- review claim language with the client's appropriate governance owner; and
- run the maintained release-candidate and 500x workflows.

**Exit evidence:** approved pilot summary contract, measured usability, stable
performance, and defensible claims boundary.

## 21.22 Claims supported, claims not supported, and next gate

### Claims supported now, with qualification

- PPAR currently generates shared HTML/XLSX reviewer surfaces for Performance
  Differences, Performance Difference Causes, and Data Audit Issues.
- Current bundles contain deterministic period-level triage, cause, coverage,
  residual, context, lineage, manifest, and handoff artifacts.
- Current primary performance tables show per-review-unit performance
  difference, explained difference, unexplained difference, status, and
  comments, subject to current method/configuration boundaries.
- Current bundle validation protects required artifacts, typed content,
  canonical review content, parity, and deterministic semantic output.

These are CURRENT — DEMONSTRATED and CURRENT — DOCUMENTED foundations. Their
usefulness and correctness on real client exports remain CURRENT — REQUIRES
CLIENT VALIDATION.

### Claims not supported

- that the intended executive first-view experience is currently implemented;
- that current `review_summary.json` is an executive narrative;
- that current HTML begins with a separate `Problems` grid;
- that a summary establishes official-performance correctness, independent
  assurance, compliance, approval, or investigation closure;
- that Fully Explained means source-data contains no independent quality issue;
- that no displayed Data Audit row means every applicable rule was evaluated;
- that cause estimates can be totaled safely across periods or portfolios;
- that current artifacts support an explanation-completeness percentage,
  composite health score, or confidence score;
- that PPAR currently provides management/analyst access control, comments,
  assignment, approval, or case workflow; or
- that Phase 3D requires or authorizes cross-run history, dashboard, or
  Operational Intelligence infrastructure.

### Completion and next gate

Phase 3D is complete as a **founder-approved functional specification**.
It defines content hierarchy, explanation completeness, priority performance
changes, largest supported causes, unresolved items, independent Data Audit
highlights, deterministic actions, evidence links, progressive management and
analyst use, shared calculation/table behavior, outputs, failures, scale,
acceptance, and client validation.

The founder approved Phase 3D on 2026-07-16 with the six working assumptions in
Section 21.20. That approval authorized Phase 3E drafting; it did not authorize
application-code changes or begin Phase 3F.

---

# 22. Phase 3E — Audit Health Dashboard and Operational Intelligence

## 22.1 Phase outcome and design posture

**Capability status:** APPROVED DIRECTION; implementation remains DEFERRED
until recurring client use establishes a serious business case and validates
the comparability model. This section is a draft functional specification for
founder review, not authorization to implement history infrastructure.

Phase 3E defines how PPAR Audit could use multiple retained, validated
investigations to answer a management question:

> Across comparable reporting cycles, where is reported performance changing,
> what remains unresolved, and which operational patterns deserve attention?

The dashboard is not a visualization over one workbook. It depends on a
trustworthy local history of immutable investigations, declared source-state
relationships, version/configuration provenance, and metric-specific
comparability. Operational Intelligence is the evidence-backed pattern layer
over that same history; it is not a separate causal-inference or BI product.

`AHD-001` — Phase 3E MUST NOT be described as current product behavior. Current
bundles provide strong per-run inputs, but PPAR has no persistent audit-history
repository, source-state chain, cross-run comparability engine, or health
dashboard. CURRENT — DOCUMENTED limitation.

`AHD-002` — Phase 3E implementation MUST remain deferred until a validation
partner demonstrates recurring use, retained comparable runs, a management
decision improved by history, and willingness to govern the required local
provenance. APPROVED DIRECTION.

`AHD-003` — The dashboard MUST preserve separate dimensions for readiness/run
validity, performance-change frequency, explanation status, unresolved work,
Data Audit findings, and human disposition. It MUST NOT collapse them into one
health score, confidence score, grade, traffic light, or pass/fail badge.
APPROVED DIRECTION.

## 22.2 User problem, value, and non-goals

### 22.2.1 User problem

A manager can review one investigation using Phase 3D, but repeated operation
creates different questions:

- Is the same portfolio or reporting period changing again?
- Are unexplained outcomes becoming more or less common?
- Which supported cause families recur across comparable investigations?
- Which Data Audit checks repeatedly affect the evaluated population?
- Is a perceived improvement real, or did scope, rules, configuration, or
  optional evidence change?
- Which current investigation should the team open first?

Without a governed history model, teams may answer these questions with manual
spreadsheets that double-count regenerated reports, mix incomparable methods,
ignore rule-coverage changes, or mistake absence of evidence for improvement.

### 22.2.2 Intended value

Phase 3E should:

- reduce manual compilation of recurring investigation results;
- expose repeated change and unresolved patterns without hiding individual
  evidence;
- give managers a bounded view of current and historical review demand;
- help analysts navigate from a pattern to the exact immutable investigation;
- distinguish real observed change from coverage or comparability change; and
- identify client-validated opportunities for process improvement or reusable
  rules.

### 22.2.3 Explicit non-goals

Phase 3E is not:

- proof that official performance is correct;
- an independent assurance, compliance, GIPS, or control certification;
- investment-performance, risk, or return-volatility analytics;
- an employee, team, custodian, vendor, or portfolio “scorecard”;
- a cross-client benchmark or PPAR-hosted data network;
- a generic BI platform, data warehouse, or ticketing system;
- a prediction of future errors;
- a substitute for Phase 3D investigation detail;
- a license to aggregate unlike returns, residuals, or cause amounts; or
- evidence that a recurring association is the organizational root cause.

`AHD-004` — “Health” MUST mean disclosed descriptive evidence about the
performance-review process within an explicit client-controlled population. It
MUST NOT imply medicalized diagnosis, certification, or a universal quality
standard. APPROVED DIRECTION.

## 22.3 Current foundation and prerequisite gaps

| Foundation or gap | Current truth | Phase 3E implication |
|---|---|---|
| Immutable per-run evidence packs | Manifest v4, canonical tables, review keys, lineage, and normalized bundle fingerprints are current | Reuse as the authoritative evidence source; do not copy or mutate their facts |
| Per-run analytical results | Performance Differences, causes, residual/status, Data Audit Issues, coverage, and triage are current | History facts must derive from these canonical results |
| Bundle determinism and parity | Current safety controls validate semantic output across formats | History/dashboard output must inherit equivalent parity and determinism controls |
| Product version in bundle | Not current | Required before authoritative cross-version history |
| Resolved configuration fingerprint | Not current | Required for comparability and interpretation |
| Source-state identity/relationship | Snapshot labels and extract context exist, but an authoritative A-to-B-to-C chain is not current | Required to distinguish repeated restatement from report regeneration |
| Unified readiness artifact | Phase 3C approved direction, not current | Operational history should bind to the readiness result once implemented |
| Executive summary object | Phase 3D approved direction, not current | Preferred compact ingestion surface when implemented |
| Durable history/index | Not current | Conditional prerequisite; implementation remains deferred |
| Human workflow/disposition | Not current | Resolution time, aging, closure, and owner-backlog metrics remain unavailable |
| Real-client recurring-use evidence | Not available | Blocks product and claims validation |

`AHD-005` — A normalized bundle fingerprint proves the recorded bundle
semantics and supports deduplication; it is not a source-state identifier,
configuration approval, or comparability decision. APPROVED DIRECTION.

`AHD-006` — Existing bundles that lack required product, configuration, source-
state, or intended-use provenance MAY be registered for discovery, but MUST be
excluded from authoritative metrics whose interpretation requires the missing
facts. CANDIDATE migration behavior.

## 22.4 Actors and decision rights

| Actor | Responsibility | May decide | Must not decide merely through access |
|---|---|---|---|
| Performance/operations analyst | Review patterns and open underlying investigations | Which evidence to investigate and what source/methodology question to raise | That frequency proves source error, blame, or official correction |
| Performance/operations manager | Own dashboard purpose, review cadence, population, and escalation policy | Approved scope, history window, operational materiality, and action priority | To waive missing comparability or rewrite product facts |
| Methodology owner | Govern return basis, material configuration semantics, and permitted cross-version comparison | Whether a method/configuration change preserves metric comparability | That administrative continuity proves methodological equivalence |
| Source/extract administrator | Govern source-state identity, capture provenance, and export procedure | Whether state B succeeds state A and whether source identity is stable | Performance methodology or explanation status |
| Local product administrator | Retain bundles, generate/index local history, validate artifacts, and manage access | Local paths, retention operations, approved software release, and job execution | Accounting meaning or business acceptance |
| Compliance/GIPS reviewer | Review reliance, definitions, limitations, and retention under client policy | Applicable governance challenge and permitted use | Convert the dashboard into independent assurance |
| Technology/security owner | Approve local storage, access, backup, and support boundary | Client-controlled deployment and distribution policy | Financial interpretation |
| PPAR | Validate, deduplicate, classify metric eligibility, calculate descriptive measures, and link evidence | Deterministic product results under approved policy | Human closure, official correctness, blame, or source correction |

`AHD-007` — Any manual inclusion, exclusion, identity mapping, or comparability
override MUST be an attributable, versioned policy decision with a reason and
approval reference. It MUST NOT modify an underlying bundle. APPROVED
DIRECTION.

`AHD-008` — PPAR-calculated historical facts MUST remain distinguishable from
client-supplied classifications and future Phase 3F dispositions. APPROVED
DIRECTION.

## 22.5 History and metric terminology

Phase 3E uses exact units to prevent denominator and double-counting errors:

- **Evidence bundle:** one validated portfolio- or security-level report bundle
  produced by one audit execution.
- **Audit execution:** the local operation that may produce one portfolio
  bundle, one security bundle, or both for the same source-state comparison.
- **Source state:** a client-declared extract/capture state with stable identity,
  capture/as-of provenance, and relationship to another state.
- **State transition:** one declared comparison from source state A to source
  state B. Snapshot B is not presumed correct or later unless provenance says
  so.
- **Review unit:** one compatible analytical grain, such as portfolio-period or
  portfolio-security-period, identified by report level and stable entity/
  period keys.
- **Changed review unit:** a review unit with a reportable performance change
  under the run's declared comparison contract.
- **Observation:** one review unit's result for one validated state transition.
- **Comparable series:** ordered observations eligible for one named metric
  under an explicit comparability policy.
- **Repeated change:** the same stable review unit changes again in a later
  comparable state transition.
- **Affected row/entity count:** the Phase 3D-approved Data Audit population
  count; it is not an incident count.
- **Dashboard generation:** a deterministic local result derived at an “as of”
  time from a declared history/index version and filter policy.

`AHD-009` — A regenerated bundle with identical normalized semantics MUST NOT
be counted as a new source-state observation merely because its file path or
creation timestamp differs. APPROVED DIRECTION.

`AHD-010` — Portfolio and security bundles from the same audit execution MUST
remain separate analytical levels. Shared source evidence or Data Audit rows
MUST NOT be counted twice as independent operational events. APPROVED
DIRECTION.

`AHD-011` — “Restatement” SHOULD be used only when the client has established
that a later source state changes a previously reported value. Otherwise the
product MUST use the narrower term “observed performance change.” APPROVED
DIRECTION.

## 22.6 Conditional local history model

### 22.6.1 Storage posture

If the implementation gate is later satisfied, the initial history should use:

1. immutable, client-retained evidence bundles as the authoritative facts;
2. a small versioned local registry/index containing references and derived
   facts needed for comparability and dashboard generation; and
3. generated dashboard artifacts that can always identify their index version
   and underlying bundle fingerprints.

The registry may be JSON plus typed flat tables or an equivalent file-based
contract. SQLite or another local database remains CANDIDATE only if measured
volume, concurrency, query, locking, or recovery needs justify it. A hosted
database is OUT OF SCOPE.

`AHD-012` — The index MUST reference rather than replace authoritative bundles.
Removing a dashboard or rebuilding an index MUST NOT remove or rewrite retained
investigation evidence. APPROVED DIRECTION.

`AHD-013` — Index creation and update MUST be atomic, versioned, recoverable,
and deterministic. A partial index update MUST NOT publish a dashboard.
CANDIDATE implementation contract.

### 22.6.2 Minimum history record

Each registered audit execution should carry or reference:

- history schema version and local execution ID;
- creation/import time and accountable local job/operator identity where
  available;
- portfolio and/or security bundle paths and normalized fingerprints;
- manifest, review-summary, executive-summary, readiness, and history-contract
  versions;
- installed PPAR version and relevant rule/normalization/contract versions;
- resolved configuration fingerprint and approved immutable reference;
- readiness status, intended use, qualifications, and approval reference;
- comparison level, requested/evaluated scope, return basis, currency/unit, and
  tolerance/materiality policy identity;
- source-state A/B IDs, capture/as-of provenance, and declared relationship;
- stable portfolio/security/period identities and any approved identity-map
  version;
- per-review-unit performance difference, explained difference, residual or
  withheld state, and analytical status;
- eligible supported cause families and economic-effect ownership references;
- Data Audit affected rows/entities, rule identity/version, and evaluation
  coverage;
- links to canonical evidence; and
- future Phase 3F disposition references only when available.

`AHD-014` — The history record MUST preserve typed values and stable enums. It
MUST NOT scrape displayed HTML/XLSX text when a canonical machine-readable
artifact is available. APPROVED DIRECTION.

`AHD-015` — The index MUST retain the provenance and policy used when a fact was
generated. A later configuration change MUST NOT retroactively reinterpret an
old bundle as though it used the new policy. APPROVED DIRECTION.

### 22.6.3 Ingestion and validation

The target ingestion sequence is:

1. locate a client-selected evidence bundle;
2. validate required artifacts, manifest, fingerprints, lineage, and formats;
3. bind readiness/intended-use and configuration/product provenance;
4. identify the audit execution and source-state transition;
5. reject or flag duplicate semantic observations;
6. extract typed review-unit, cause, Data Audit, and coverage facts;
7. classify eligibility independently for each metric family;
8. write a new index version atomically; and
9. validate indexed facts back to the immutable bundle.

`AHD-016` — A bundle that fails current authoritative validation MUST NOT enter
authoritative history. The failure is not a historical health observation.
APPROVED DIRECTION.

`AHD-017` — An imported legacy bundle MAY remain visible in an excluded/
limited-history register with exact reasons. The dashboard MUST NOT silently
discard it or silently treat missing provenance as comparable. CANDIDATE.

## 22.7 Identity, sequence, and deduplication

History requires three distinct identities:

1. **bundle identity** — normalized semantic fingerprint and artifact reference;
2. **audit-execution identity** — groups portfolio/security outputs produced for
   one declared comparison; and
3. **source-state identity** — identifies the client extract/reporting state
   being compared.

These identities must not be substituted for one another.

`AHD-018` — The product MUST NOT infer source-state order from directory names,
file modification time, import time, lexical Snapshot A/B labels, or manifest
creation time. APPROVED DIRECTION inherited from `PCI-062`.

`AHD-019` — One source-state transition may have multiple generated formats and
report levels. Format copies are never new observations; report levels remain
separate observations linked to the same transition. APPROVED DIRECTION.

`AHD-020` — If the same bundle fingerprint is registered twice under different
paths, the index MUST retain path/provenance information as needed but count the
semantic observation once. CANDIDATE.

`AHD-021` — If two different bundle fingerprints claim the same audit execution,
level, scope, and source-state transition, ingestion MUST stop or retain an
explicit versioned supersession relationship. It MUST NOT choose by timestamp.
CANDIDATE.

`AHD-022` — Entity renames, identifier changes, mergers, composite changes, or
security identifier reuse require an approved identity mapping or break the
series. Fuzzy name matching is prohibited. APPROVED DIRECTION.

## 22.8 Metric-specific comparability contract

A single global “comparable/not comparable” flag is too coarse. For example, a
change in Data Audit rule coverage may invalidate a rule-frequency trend while
leaving performance-change observations comparable. Phase 3E therefore uses
metric-specific eligibility.

### 22.8.1 Eligibility statuses

- **`Eligible`** — all required dimensions are compatible for the named metric.
- **`Limited`** — the metric remains interpretable only within an explicitly
  narrowed population or coverage boundary.
- **`Ineligible`** — a required identity, method, unit, provenance, or coverage
  condition is incompatible or unknown.

These are historical-metric eligibility statuses, not readiness, analytical,
Data Audit, or workflow statuses.

`AHD-023` — Each displayed metric MUST identify its eligibility rule, included
population, excluded/limited population, window, denominator, and relevant
policy versions. APPROVED DIRECTION.

### 22.8.2 Comparability dimensions

| Dimension | Performance-change metrics | Explanation metrics | Cause metrics | Data Audit metrics |
|---|---|---|---|---|
| Stable entity and period identity | Required | Required | Required | Required at the rule's grain |
| Declared source-state sequence | Required for recurrence | Required for trend | Required for recurrence | Required for trend |
| Portfolio/security report level | Must remain separate | Must remain separate | Must remain separate | Shared rows require deduplication |
| Return basis, currency, and unit | Required | Required | Required for amount interpretation | Required when rule meaning depends on them |
| Comparison and arithmetic policy | Required | Required | Required | Rule-specific |
| Configuration semantics | Material changes may break the series | Required | Required | Required where rule input/logic changes |
| Product/contract compatibility | Declared compatibility required | Declared compatibility required | Declared compatibility required | Rule-version compatibility required |
| Optional input coverage | Disclose if scope changes | May make status trend limited | May make cause absence uninterpretable | Required per rule/population |
| Materiality policy | Does not change reportable population; required for priority views | Required for priority views | Required for priority views | Required for priority views |
| Workflow status | Not required | Not required | Not required | Not required; required for closure/aging metrics |

`AHD-024` — Incompatible observations MUST remain available in the history
register and exclusion counts but MUST NOT be coerced into a continuous trend.
APPROVED DIRECTION.

`AHD-025` — A product, rule, or configuration version change MUST default to
`Ineligible` for affected metrics unless an explicit compatibility contract or
client-approved regression demonstrates semantic equivalence. APPROVED
DIRECTION.

`AHD-026` — `Limited` MUST name the exact unaffected metric and narrowed
population. It MUST NOT become a general waiver for unknown semantics.
APPROVED DIRECTION.

## 22.9 Dashboard information architecture

The initial dashboard should use progressive disclosure and this order:

1. **Scope, as-of state, and comparability coverage** — client/site, history
   window, latest included transition, included/excluded runs, evaluated levels,
   and material limitations.
2. **Current review demand** — latest comparable changed review units, Partly
   Explained/Unexplained units, independent Data Audit attention, and links to
   Phase 3D summaries.
3. **Repeated change** — review units observed changing again, first-observed
   changes, reappearing changes, and incomparable sequences.
4. **Explanation outcomes** — counts and proportions by Fully Explained,
   Partly Explained, Unexplained, and applicable withheld states.
5. **Recurring supported cause areas** — frequency by affected comparable
   review unit, never cross-unit summed return impact.
6. **Data Audit findings and coverage** — affected rows/entities by rule plus
   evaluated/not-evaluated populations.
7. **Descriptive entity patterns** — portfolios/securities with repeated change
   shown with opportunity counts, denominators, and minimum-history disclosure.
8. **Limitations and evidence** — excluded observations, coverage changes,
   policy versions, and direct bundle references.

`AHD-027` — The first view MUST show whether the apparent pattern is based on
one observation, a pairwise change, or a multi-observation series. APPROVED
DIRECTION.

`AHD-028` — One validated run MAY support a current-state inventory, but it MUST
NOT be labeled a trend. Two comparable observations MAY support a stated
change. “Trend” language SHOULD require at least three comparable observations
and MUST always disclose the observation count. CANDIDATE threshold.

`AHD-029` — Management and analyst views SHOULD use one shared result with
progressive disclosure, consistent with Phase 3D. Separate facts or permissioned
calculation models are not part of the initial design. APPROVED DIRECTION.

## 22.10 Metric catalog and aggregation rules

### 22.10.1 Coverage and population metrics

| Metric | Definition | Required disclosure | Status |
|---|---|---|---|
| Registered audit executions | Unique executions in the selected local history | Window and registry version | CANDIDATE |
| Eligible/limited/ineligible executions | Executions by metric-family eligibility | Reason counts and affected metric family | CANDIDATE |
| Evaluated review units | Distinct review units actually evaluated in eligible observations | Level, scope, and observation opportunities | CANDIDATE |
| Rule evaluation coverage | Rows/entities or declared populations evaluated by each Data Audit rule | Rule version, filters, exclusions, and unavailable inputs | APPROVED DIRECTION from Phase 3B |
| Current-state coverage | Latest valid observation available for each requested entity/scope | Missing, stale, and incomparable entities | CANDIDATE |

`AHD-030` — Every rate MUST expose its numerator and denominator. A denominator
MUST represent actual comparable observation opportunities, not all calendar
periods, all portfolios known to the client, or all registered bundles unless
those populations were truly evaluated. APPROVED DIRECTION.

### 22.10.2 Performance-change metrics

Candidate metrics are:

- changed review-unit count and share of comparable evaluated opportunities;
- distinct portfolio-periods or security-periods affected;
- first-observed versus repeated-change counts;
- number of comparable transitions in which a stable review unit changed;
- largest individual absolute reported-return changes; and
- counts within client-approved descriptive magnitude bands.

`AHD-031` — Portfolio and security metrics MUST remain separate. A security
change and its portfolio-level effect MUST NOT be treated as two independent
portfolio restatements. APPROVED DIRECTION.

`AHD-032` — Reported-return changes, explained amounts, and residuals MUST NOT
be summed or averaged across different portfolios, periods, report levels,
currencies, return containers, or incompatible methods. APPROVED DIRECTION.

`AHD-033` — “Largest observed changes” MAY order individual compatible review
units by absolute return difference. The display MUST retain sign, scope,
period, unit, method, status, and evidence link and MUST NOT imply that Snapshot
B is a correction or that magnitude alone defines operational priority.
CANDIDATE.

`AHD-034` — Aggregate currency impact is not a current metric because the
product does not have a validated, universally comparable monetary-loss model.
DEFERRED.

### 22.10.3 Explanation metrics

The initial explanation view should use:

- counts and proportions of comparable changed review units by Fully Explained,
  Partly Explained, and Unexplained;
- separate counts for residual-withheld or method-limited states;
- current Partly Explained/Unexplained review units with exact per-unit values;
- consecutive comparable observations in which a unit remains unresolved; and
- first-observed and latest-observed dates/states for the unresolved condition.

`AHD-035` — Phase 3E MUST NOT introduce an explanation-completeness percentage.
It inherits the Phase 3A decision to retain analytical status plus exact
per-review-unit explained and residual values. APPROVED DIRECTION.

`AHD-036` — A status proportion is the share of comparable changed review units
in a named analytical status. It is not the percentage of performance change
explained and MUST be labeled accordingly. APPROVED DIRECTION.

`AHD-037` — Without Phase 3F workflow evidence, the product MAY describe how
many consecutive comparable observations remain Partly Explained or
Unexplained. It MUST NOT call elapsed time “open age,” “time to resolution,” or
“backlog age.” APPROVED DIRECTION.

### 22.10.4 Cause metrics

The initial cause view should show:

- number and share of comparable changed review units containing each eligible
  supported cause family;
- number of distinct entities/periods affected by the family;
- first/latest observed transition; and
- links to representative and complete underlying investigations.

`AHD-038` — A cause family is counted at most once per review unit per source-
state observation for frequency metrics, even when multiple source rows support
the same family. Physical row count MAY be shown separately when useful.
CANDIDATE.

`AHD-039` — Only Phase 3A-eligible, owned, validated supported causes may enter
cause-frequency metrics. Review evidence, supporting components, related
outputs, Data Audit issues, and cross-check diagnostics MUST remain separate.
APPROVED DIRECTION.

`AHD-040` — Cause return impacts MUST NOT be totaled across review units. A
frequency pattern may justify investigation; it does not prove one common root
cause. APPROVED DIRECTION.

### 22.10.5 Data Audit metrics

The initial Data Audit view should show, by stable rule ID/version:

- affected row count;
- affected entity count at the rule's declared grain;
- evaluated row/entity population;
- not-evaluated/unavailable/insufficient population;
- number/share of comparable observations with at least one affected entity;
- first/latest observed transition; and
- current high-priority affected entities with evidence links.

`AHD-041` — Phase 3E inherits Phase 3D's affected-row/entity counting decision.
It MUST NOT convert repeated rows into “incidents” until a stable, validated
issue-group identity contract exists. APPROVED DIRECTION.

`AHD-042` — Rule-frequency trends require stable rule meaning, input coverage,
filters, tolerances, and evaluated population. A rule not evaluated MUST NOT be
counted as zero findings. APPROVED DIRECTION.

`AHD-043` — Mandatory continuity findings remain visible and independent from
optional-rule trends. Their presence MUST NOT alter performance explanation
arithmetic. CURRENT doctrine; APPROVED DIRECTION for history views.

### 22.10.6 Entity stability and operational pattern views

Descriptive entity views may show:

- comparable observation opportunities by portfolio/security;
- changed-review-unit count and rate;
- repeated-change count;
- Partly Explained/Unexplained count;
- recurring supported cause families; and
- Data Audit affected-entity counts with coverage.

`AHD-044` — An entity table MAY be sorted by one disclosed descriptive metric,
but MUST NOT be labeled a quality ranking or league table. It MUST show the
denominator and minimum comparable history next to the numerator. CANDIDATE.

`AHD-045` — Cross-portfolio comparison requires equivalent scope and observation
opportunities. A portfolio audited more frequently or across more periods MUST
not appear worse merely because it had more chances to produce findings.
APPROVED DIRECTION.

`AHD-046` — Custodian, source, desk, administrator, or other organizational
patterns may be shown only when the source contract provides stable approved
identity and adequate comparable coverage. The product MUST avoid blame or
causal language. CANDIDATE; otherwise DEFERRED.

## 22.11 Recurrence and observed-state model

Until Phase 3F supplies governed disposition, Phase 3E should use observation
language rather than closure language:

- **`First Observed Change`** — first eligible changed observation for a stable
  review unit in retained history;
- **`Changed Again`** — a later eligible state transition changes that unit
  again;
- **`Not Changed in Next Comparable Observation`** — the next comparable
  transition contains no reportable performance change for the unit;
- **`Reappeared`** — a changed observation follows at least one eligible
  not-changed observation;
- **`No Comparable Follow-up`** — no later eligible observation exists; and
- **`Incomparable`** — a required identity or policy dimension prevents the
  recurrence conclusion.

`Resolved`, `Accepted`, `Closed`, and `Reopened` require governed human
disposition or an approved external workflow reference and belong to Phase 3F.

`AHD-047` — Absence of a changed review unit in a later bundle means “not
observed as changed” only when the later run evaluated the same unit under a
comparable contract. Missing scope or evidence MUST NOT be treated as
resolution. APPROVED DIRECTION.

`AHD-048` — A repeated change MUST follow the declared source-state chain. Two
independent A-to-B comparisons with no established relationship MUST not be
ordered into a sequence. APPROVED DIRECTION.

`AHD-049` — Branching source-state histories, such as two alternate corrections
from the same prior state, MUST remain explicit branches. The product MUST NOT
choose a canonical branch without client authority. CANDIDATE.

`AHD-050` — Operational Intelligence MAY identify recurrence and co-occurrence.
It MUST NOT use “root cause” unless the underlying Phase 3A evidence supports
that exact cause and the historical grouping preserves its meaning. APPROVED
DIRECTION.

## 22.12 Materiality, filters, and presentation subsets

Operational materiality may prioritize the dashboard, but it must not change
historical truth.

`AHD-051` — The history index MUST retain every reportable changed review unit
and every applicable Data Audit row required by the source bundle, including
suppressed findings in the complete audit trail. A dashboard threshold may
change ordering or first-view inclusion only. APPROVED DIRECTION.

`AHD-052` — Each filtered or bounded view MUST show the complete eligible count,
displayed count, filter/threshold identity, and route to the complete local
detail. APPROVED DIRECTION.

`AHD-053` — Materiality-policy changes MUST be versioned and disclosed. They may
invalidate a priority trend even when the underlying reportable-difference
series remains comparable. APPROVED DIRECTION.

`AHD-054` — A local browser filter MAY hide rows interactively, but MUST NOT
recalculate canonical metrics or change the stored dashboard result.
CANDIDATE.

## 22.13 Recommended actions and management interpretation

Phase 3E actions should be deterministic and evidence-routed. Candidate cues
include:

| Condition | Candidate action | Default owner role |
|---|---|---|
| Changed Again | Open latest and prior Phase 3D summaries; confirm source-state sequence and repeated mechanism | Performance analyst |
| Reappeared | Review intervening observation, configuration, and source correction evidence | Performance/operations manager |
| Consecutive Partly Explained/Unexplained observations | Review missing evidence, mapping, methodology, and escalation policy | Performance analyst plus methodology owner |
| Cause family recurring across eligible units | Validate whether one reusable rule/process response exists; do not assume common cause | Operations manager |
| Data Audit affected-entity rate increases under stable coverage | Review exact rule population, tolerance, examples, and false positives | Rule owner/source owner |
| Rule coverage decreases | Restore evidence/coverage or label the trend unavailable | Local/source administrator |
| Comparability exclusions rise | Review configuration, version, identity, or source-contract drift | Methodology owner and local administrator |

`AHD-055` — Action text MUST identify the observed condition, owner role,
evidence path, and limitation. It MUST NOT instruct PPAR to write corrections
to the source system or declare a human disposition. APPROVED DIRECTION.

`AHD-056` — Operational Intelligence recommendations MUST be phrased as
investigation or process-learning hypotheses until real client evidence and an
authorized owner support a stronger conclusion. APPROVED DIRECTION.

## 22.14 Output and machine-readable contract

### 22.14.1 Canonical result

The target output should be one small, versioned, machine-readable dashboard
result containing:

- schema, generation, registry, and policy versions;
- as-of time, history window, and client-controlled scope;
- included/limited/excluded executions and observations with reason codes;
- metric-specific eligibility and coverage;
- current-state review-demand facts;
- recurrence classifications;
- explanation-status distributions;
- supported cause-family frequencies;
- Data Audit affected populations and rule coverage;
- entity descriptive facts;
- deterministic actions;
- bundle/evidence references; and
- required interpretation and claims limitations.

`AHD-057` — All human-readable dashboard formats MUST derive from one validated
machine-readable result. APPROVED DIRECTION.

`AHD-058` — The exact schema and filenames are CANDIDATE implementation work.
The likely initial package is local static HTML for management/reviewer use,
XLSX for controlled operational handoff, and JSON plus typed flat tables for
validation/integration.

### 22.14.2 Relationship to per-run bundles

The dashboard is a separate derived history artifact. It should reference
immutable run bundles rather than be inserted retrospectively into them.

`AHD-059` — A dashboard generation MUST identify every underlying bundle by a
stable relative reference or registry key plus fingerprint. A broken or changed
required reference invalidates authoritative publication. APPROVED DIRECTION.

`AHD-060` — Rebuilding a dashboard with a new history window or presentation
policy creates a new dashboard result. It MUST NOT alter prior dashboard
generations or source bundles. APPROVED DIRECTION.

`AHD-061` — HTML and XLSX should share the same section order, facts, counts,
labels, and evidence targets. Presentation-only differences must remain
declared and deterministic. APPROVED DIRECTION.

## 22.15 Processing model

The target Phase 3E sequence is:

1. load the selected versioned local history registry;
2. validate registry shape, policy, and referenced bundle identities;
3. deduplicate formats, regenerated bundles, and paired report-level facts;
4. establish declared source-state chains and identity mappings;
5. derive per-metric eligibility with stable reason codes;
6. construct typed review-unit observations from canonical run facts;
7. calculate recurrence, coverage, status, cause, and rule-frequency facts;
8. apply approved materiality only to priority and display selection;
9. generate deterministic actions and evidence targets;
10. render machine, HTML, XLSX, and flat-table outputs from one result; and
11. validate counts, financial values, eligibility, links, parity, and
    deterministic output before publication.

`AHD-062` — Dashboard derivation MUST be a pure deterministic transformation of
validated history facts plus approved versioned policy. APPROVED DIRECTION.

`AHD-063` — The engine MUST reconcile every aggregate count to its underlying
distinct observation keys and every displayed financial value to one canonical
review unit. APPROVED DIRECTION.

`AHD-064` — Incremental indexing MAY cache typed per-bundle facts, but cache
entries MUST be content-bound and invalidated when required source fingerprints
or schema versions change. CANDIDATE.

## 22.16 Failure behavior and edge cases

| Condition | Required behavior |
|---|---|
| Current/legacy bundle lacks product or configuration identity | Register as limited/ineligible where useful; exclude affected authoritative metrics |
| Bundle validation or fingerprint fails | Stop ingestion; do not publish using the invalid bundle |
| Same semantic bundle appears at multiple paths | Count one observation; retain controlled path provenance |
| Conflicting bundles claim the same execution/level/transition | Stop or require explicit supersession; never choose by timestamp |
| Source-state order is missing | Permit per-run inventory only; block recurrence/trend metrics |
| Source-state history branches | Preserve branches; do not merge or select authority automatically |
| Configuration/method changes | Evaluate each metric independently; exclude affected metrics absent compatibility evidence |
| Data Audit rule coverage changes | Preserve performance metrics when safe; limit/exclude affected rule trends |
| Optional input disappears | Show the coverage loss; do not interpret absent causes/findings as improvement |
| Entity identifier changes | Require approved identity mapping or break the series |
| Portfolio and security bundles share source evidence | Keep report-level facts separate and deduplicate shared operational rows |
| No performance changes in an eligible observation | Count as evaluated/not changed; do not claim universal correctness |
| No Data Audit rows displayed | Show evaluated rule coverage before any clean-language statement |
| One observation only | Show current state; prohibit trend language |
| Two observations only | Permit pairwise change language; avoid trend claim |
| Missing expected scheduled run | Show a history/coverage gap; do not interpolate |
| Negative or sign-reversing return difference | Preserve sign and individual value; do not use an unsafe aggregate ratio |
| Residual withheld | Preserve withheld state; do not substitute zero or estimate |
| Large history population | Bound first views; retain exact totals and full evidence navigation |
| Dashboard link breaks after bundle move | Invalidate publication until registry/reference is repaired and revalidated |
| A user filter hides all issues | Keep scope/coverage/filter banner; do not display an unqualified clean result |

`AHD-065` — Missing or incompatible history is a visible product limitation or
source/configuration-contract error, not an Unexplained analytical result.
APPROVED DIRECTION.

`AHD-066` — An aggregation, deduplication, lineage, parity, or deterministic-
output mismatch is an internal logic error and MUST stop authoritative
dashboard publication. APPROVED DIRECTION.

## 22.17 Safety-invariant mapping

Phase 3E inherits all twelve current safety invariants:

| Invariant | Phase 3E acceptance implication |
|---|---|
| SN-01 No lost differences | Priority filters cannot erase registered reportable differences; complete counts and evidence remain available |
| SN-02 No double counting | Duplicate bundles, formats, report levels, source rows, and economic effects cannot inflate metrics |
| SN-03 Fully Explained arithmetic | Historical status and exact values must match the source bundle at each review unit |
| SN-04 Beginning/ending continuity | Continuity findings remain visible and distinct; state chaining cannot fabricate continuity |
| SN-05 Bidirectional lineage | Every dashboard fact links to source observations, and every included observation identifies applicable metrics |
| SN-06 Currency/unit consistency | Incompatible units block amount aggregation and may block metric eligibility |
| SN-07 Period-boundary safety | Reversed, overlapping, or incompatible periods cannot form a valid series |
| SN-08 Demo scenario preservation | Any future dashboard fixture must preserve labeled recurrence/comparability stories |
| SN-09 Demo fixture isolation | Synthetic history must remain isolated from client history and product claims |
| SN-10 Report-format parity | HTML/XLSX/machine dashboard facts and counts agree semantically |
| SN-11 Deterministic output | Identical registry, bundles, and policy produce identical nonvolatile output |
| SN-12 Fail-closed policy coverage | Unknown identity, comparability, grouping, or required policy cannot be hidden by a filter or override |

`AHD-067` — Phase 3E implementation is unacceptable if it weakens any current
safety invariant or changes the underlying Phase 3A/3B calculations merely to
make a historical metric easier to compute. APPROVED DIRECTION.

## 22.18 Configuration and governance

The initial versioned dashboard policy may control:

- included client/site scope and report levels;
- history window and as-of state;
- approved source-state and entity identity references;
- minimum observation counts for pairwise/trend displays;
- operational-materiality priority thresholds;
- cause-family and rule taxonomy versions;
- first-view row limits;
- local output formats and retention path; and
- attributable inclusion/exclusion decisions.

It must not control:

- the meaning of Fully/Partly/Unexplained;
- source bundle arithmetic or evidence roles;
- whether an incompatible observation is silently treated as eligible;
- cause ownership or Data Audit independence;
- reportable-difference retention;
- a composite score or confidence claim; or
- official correctness, closure, or blame.

`AHD-068` — Dashboard policy MUST be fingerprinted and retained with each
generation. A policy change MUST produce a new result and disclose which
historical comparisons changed. APPROVED DIRECTION.

`AHD-069` — Initial access control SHOULD rely on the client-controlled file and
environment permissions approved in Phase 2. Building application-level roles
solely for Phase 3E is DEFERRED.

## 22.19 Scale, performance, local-first, security, and support

`AHD-070` — History ingestion, indexing, dashboard generation, and viewing MUST
operate within the client-controlled environment without internet access.
APPROVED DIRECTION.

`AHD-071` — No portfolio identifier, return, cause, Data Audit result, history
fact, or dashboard usage may be transmitted for licensing, updates, telemetry,
benchmarking, or support without explicit case-specific client authorization.
APPROVED DIRECTION.

`AHD-072` — The dashboard SHOULD index each validated bundle once and derive
routine views from compact typed facts rather than reparsing every XLSX/HTML or
full evidence artifact on every use. CANDIDATE.

`AHD-073` — First-view row limits MUST bound rendering cost, not calculation,
retention, counts, comparability assessment, or evidence availability.
APPROVED DIRECTION.

`AHD-074` — A Phase 3E implementation is a major cross-cutting reporting,
history, and safety-net change. It MUST run the complete relevant release-
candidate workflow and the maintained 500x scale check. Existing gates MUST not
be weakened because history processing fails. CURRENT project doctrine.

`AHD-075` — Backup, restore, retention, path migration, and access procedures
must be validated in the client environment before recurring operational
reliance. CURRENT — REQUIRES CLIENT VALIDATION.

## 22.20 Acceptance criteria

Phase 3E is functionally acceptable only when all applicable criteria pass:

1. **Business gate:** recurring use demonstrates a management decision and
   value that justify history infrastructure.
2. **Authority:** immutable validated bundles remain authoritative; the index
   and dashboard are derived and reproducible.
3. **Identity:** bundle, execution, source-state, entity, period, and report-
   level identities are explicit and not inferred from filenames/timestamps.
4. **Deduplication:** regenerated formats, duplicate paths, paired report levels,
   source rows, and economic effects cannot inflate metrics.
5. **Comparability:** eligibility is determined per metric family with stable
   reasons, versions, populations, and exclusions.
6. **Population:** every metric exposes its numerator, denominator, window,
   observation count, and scope.
7. **Aggregation:** incompatible returns, residuals, causes, currencies, units,
   report levels, or periods are not totaled or averaged.
8. **Explanation:** analytical statuses and exact per-unit values are preserved;
   no explanation-completeness percentage is introduced.
9. **Data Audit:** affected rows/entities and rule coverage remain explicit;
   not evaluated is never treated as zero findings.
10. **Recurrence:** observed-state language is distinct from human resolution,
    closure, acceptance, and reopening.
11. **Operational Intelligence:** patterns remain traceable hypotheses and do
    not imply blame or unrestricted root cause.
12. **Priority:** materiality and filters change attention only, never retained
    truth or calculation.
13. **Evidence:** every displayed fact resolves to exact local investigations;
    broken required references invalidate publication.
14. **Formats:** machine, HTML, XLSX, and flat-table outputs share one result and
    agree semantically.
15. **Determinism:** identical history, policy, and bundles produce identical
    nonvolatile output.
16. **Local-first:** no hosted store, server, telemetry, or network is required.
17. **Scale:** indexing and views remain bounded without losing evidence, and
    established release gates pass.
18. **Claims:** the dashboard describes observed comparable history; it does
    not certify quality, correctness, compliance, closure, or future outcomes.

`AHD-076` — Client acceptance MUST test management interpretation, analyst
evidence navigation, source/methodology owner agreement on comparability, and
the ability to detect misleading denominator or coverage changes. CURRENT —
REQUIRES CLIENT VALIDATION.

## 22.21 Dependencies

Phase 3E depends on:

- founder-approved Phase 2 actors, decision rights, recurring workflow, and
  local-first architecture;
- founder-approved Phase 3A review units, analytical status, recurrence,
  materiality, evidence, and stability boundaries;
- founder-approved Phase 3B rule identity, independent finding model, coverage,
  and false-positive controls;
- founder-approved Phase 3C readiness, intended use, product/configuration
  identity, source provenance, and local versioned artifact direction;
- founder-approved Phase 3D progressive summary, separate attention dimensions,
  cause eligibility, Data Audit counting, deterministic actions, and machine-
  readable summary direction;
- current immutable bundle, manifest v4, review-key, lineage, fingerprint,
  parity, and determinism foundations;
- future product/configuration/source-state provenance required for safe
  history;
- real-client recurring observations with stable approved semantics; and
- Phase 3F only for workflow status, closure, assignment, and true resolution-
  time metrics.

## 22.22 Founder-approved Phase 3E working assumptions

The founder approved these seven working assumptions on 2026-07-17:

1. **Implementation gate:** keep history/dashboard
   implementation DEFERRED until a validation partner demonstrates recurring
   use and a management decision that the capability improves.
2. **History foundation:** if the gate is met, begin with
   immutable local bundles plus the smallest versioned file-based registry/
   index; require evidence before adopting a database.
3. **Health model:** present separate coverage, performance-
   change, explanation, unresolved, cause, and Data Audit dimensions; prohibit
   a composite score, grade, traffic light, or confidence score.
4. **Comparability:** decide eligibility per metric family so a
   coverage or rule change invalidates only the trends it actually affects;
   retain excluded observations and reasons.
5. **Aggregation:** use counts, proportions with explicit
   denominators, individual magnitudes, and descriptive bands; prohibit cross-
   unit return/residual/cause totals and retain the Phase 3A decision against an
   explanation-completeness percentage.
6. **Recurrence language:** use First Observed Change, Changed
   Again, Not Changed in Next Comparable Observation, Reappeared, No Comparable
   Follow-up, and Incomparable; reserve Resolved/Closed/Reopened for Phase 3F
   disposition evidence.
7. **Initial output posture:** use one local machine-readable
   result with progressive static HTML/XLSX views and evidence links; do not
   require a server, hosted service, or application-level permission system.
   Exact formats remain subject to pilot workflow validation.

These decisions define a safe conditional design. Approval does not itself
satisfy the business gate or authorize implementation.

## 22.23 Real-client validation plan

### Stage 1 — Business-case and recurring-use gate

- operate the current two-snapshot workflow across multiple real reporting
  cycles without adding a history product;
- retain immutable bundles and manually identify repeated management questions;
- measure manual compilation time, decision value, frequency, and support cost;
- identify the specific manager and analyst actions history would improve; and
- confirm that the client will govern source-state, configuration, identity,
  and retention provenance.

**Exit evidence:** a serious recurring business case and founder decision to
permit history implementation. Without this evidence, Phase 3E remains
DEFERRED.

### Stage 2 — Provenance and labeled history cases

- define stable source-state IDs and A-to-B-to-C chains with source owners;
- bind product, configuration, contract, readiness, and rule identity;
- label duplicate generation, paired report levels, branches, gaps, identifier
  changes, and superseding bundles;
- test immutable bundle retention, index rebuild, and path migration; and
- confirm every indexed fact reconciles to its source bundle.

**Exit evidence:** approved history contract and labeled identity/deduplication
cases with no silent ordering assumptions.

### Stage 3 — Metric comparability and semantic validation

- vary return basis, configuration, tolerance/materiality, rule version, input
  coverage, product version, currency, report level, and entity identity;
- have methodology/rule owners label which metric families remain eligible;
- test numerator, denominator, exclusion, and limited-population behavior;
- test first-observed, changed-again, not-changed, reappeared, branch, and no-
  follow-up sequences; and
- verify no status proportion is mistaken for percent explained.

**Exit evidence:** versioned comparability rules and a semantic regression set
agreed by client owners.

### Stage 4 — Management and analyst workflow validation

- compare the dashboard with the client's existing recurring review process;
- test whether managers identify the right current priorities and limitations;
- test whether analysts navigate each pattern to exact bundle evidence;
- measure wrong-path navigation, false trend interpretation, and action utility;
- validate cause/rule frequency language for false blame or causal inference;
  and
- confirm bounded views do not hide the complete population.

**Exit evidence:** another authorized operator can interpret and act on the
dashboard without founder coaching or dimension conflation.

### Stage 5 — Scale, resilience, security, and claims

- test large histories, incremental updates, rebuilds, backup/restore, path
  migration, and interrupted writes;
- validate machine/HTML/XLSX parity, evidence links, and deterministic repeat;
- run release-candidate and 500x workflows;
- test client access, retention, offline operation, and support boundaries; and
- approve any external claim with the appropriate client governance owner.

**Exit evidence:** measured local operational reliability, stable interpretation,
and a defensible claims boundary.

## 22.24 Claims supported, claims not supported, and next gate

### Claims supported now, with qualification

- Current validated bundles contain per-run performance, explanation, cause,
  Data Audit, coverage, lineage, review-key, and fingerprint foundations that
  could support a future history model.
- The product doctrine and approved specifications define conservative repeated-
  restatement, stability, local-first, provenance, and evidence boundaries.
- Current deterministic bundle output can support client-retained manual
  comparisons, subject to real-client validation.

These claims do not establish a current dashboard or cross-run product.

### Claims not supported

- that PPAR currently stores or indexes investigation history;
- that it currently identifies recurrence, trends, resolution time, or backlog
  age;
- that existing bundles contain all provenance required for authoritative
  comparability;
- that the dashboard or Operational Intelligence is implemented or client-
  validated;
- that one or two observations establish a trend;
- that missing optional data or unevaluated rules represent improvement;
- that return changes, residuals, or cause impacts can be totaled across
  portfolios or periods;
- that PPAR has a validated explanation-completeness percentage, health score,
  confidence score, quality rank, or cross-client benchmark;
- that a recurring cause area proves organizational root cause or blame;
- that not observing a later change means a human accepted, resolved, or closed
  an investigation; or
- that Phase 3E provides independent assurance, certification, official
  correctness, or prediction.

### Completion and next gate

Phase 3E is complete as a **founder-approved functional specification**.
It defines the conditional business gate, local history foundation, identity
and deduplication, metric-specific comparability, descriptive metrics,
recurrence language, Operational Intelligence boundary, management views,
machine and human outputs, failure behavior, safety mapping, acceptance, and
client validation.

The founder approved Phase 3E on 2026-07-17 with the seven working assumptions
in Section 22.22. That approval authorized a Phase 3F boundary brief; it did not
authorize history/dashboard implementation or detailed speculative workflow
design.

---

# 23. Phase 3F — Human Review and Disposition Boundary

## 23.1 Design posture

**Capability status:** DEFERRED pending a serious business case and real-client
workflow evidence.

Human review is central to PPAR Audit, but product-managed comments,
assignments, approvals, dispositions, closure, and reopening are not current
capabilities. This section intentionally does **not** define a detailed workflow
system. It records the minimum product boundary that any later design must
inherit and the evidence required before that design begins.

This is a deliberate application of the evidence-horizon rule in Section 0.5.
The product should first observe how validation partners actually review,
approve, correct, rerun, retain, and close performance investigations. It should
not preselect a state machine or case-management model from theory.

## 23.2 Current truth

PPAR currently calculates analytical results and generates immutable local
evidence packs. Qualified client personnel review those results and make
business decisions through processes outside PPAR.

PPAR does not currently provide:

- user accounts or application-level permissions;
- comments, assignments, notifications, or work queues;
- approval or electronic-signature workflow;
- authoritative workflow status or closure;
- a human-disposition record linked across reruns; or
- a persistent investigation-history store.

That absence is not automatically a product gap. Existing client controls may
already handle some or all of this work more appropriately.

## 23.3 Non-negotiable boundary

Any future Human Review and Disposition capability must preserve these
contracts:

1. **Analytical status remains product-calculated.** A workflow decision cannot
   change Fully Explained, Partly Explained, Unexplained, explained amount,
   residual, evidence role, or cause ownership.
2. **Human content remains visibly human.** Any note, supplemental explanation,
   acceptance, or risk decision must identify its author, time, basis, and
   evidence reference and remain separate from PPAR-calculated content.
3. **No direct calculated-result override.** Correct source-data or approved
   configuration and rerun PPAR; do not edit a cause or residual into agreement.
4. **Reruns remain new evidence.** A later run may supersede an earlier one for
   operational use, but it must not mutate the earlier bundle or erase its
   provenance.
5. **Closure is not correctness.** A client may decide that no further action is
   warranted, but closure cannot certify official performance, source quality,
   assurance, compliance, or product correctness.
6. **Uncertainty remains visible.** A client decision to accept a limitation
   does not convert an unexplained result into a product explanation.
7. **Local-first remains permanent.** Routine workflow data, notes, evidence,
   identities, and history remain inside the client-controlled environment.
8. **PPAR does not become generic case management.** Workflow belongs in the
   product only to the extent that it materially improves performance-change
   investigation, evidence accountability, or safe rerun/closure handoff.

These are product contracts, not a proposed UI or schema.

## 23.4 Questions to validate before designing workflow

A validation partner should answer through observed use:

- Where are investigation decisions recorded today, and what actually fails in
  that process?
- Who reviews, who approves methodology, who corrects source-data, and who may
  accept an unresolved limitation?
- Which decisions recur often enough to justify product support?
- Is a reference to the client's existing control/ticket system sufficient?
- Do users need a human note, a disposition, assignment, approval, closure, or
  only a reproducible rerun link?
- What evidence is required to distinguish accepted restatement, requested
  correction, methodology limitation, and no-further-action decisions?
- Which retention, access, and audit requirements already exist?
- What measurable delay, error, or accountability problem would PPAR workflow
  remove?
- Would workflow create more implementation, support, security, and liability
  burden than client value?

The validation method should use real historical cases and fresh investigations,
not opinions about an imagined future interface.

## 23.5 Details intentionally deferred

The following must remain unspecified until the validation evidence requires a
decision:

- workflow state names and transition rules;
- comment, assignment, approval, and disposition schemas;
- user, role, and permission models;
- storage technology and history architecture;
- UI layout, inboxes, queues, dashboards, and notifications;
- integration with ticketing, identity, email, or collaboration systems;
- closure, reopening, aging, SLA, and escalation calculations;
- electronic signature or regulated-record behavior;
- API and export contracts; and
- packaging, pricing, and support obligations.

The candidate states and dispositions recorded earlier in Sections 17.4.5 and
17.5.12 are discovery hypotheses only. They are not a committed state model.

## 23.6 Evidence gate and possible product outcomes

Detailed design should begin only if repeated client evidence establishes:

- a material workflow problem tied directly to PPAR investigations;
- a named daily user and accountable decision owner;
- a recurring decision that existing client systems do not handle adequately;
- a smallest useful record or handoff that can be stated from observed cases;
- value sufficient to justify security, retention, migration, and support cost;
  and
- a design that preserves every boundary in Section 23.3.

Validation may legitimately conclude that PPAR should provide no managed
workflow. Other acceptable outcomes could be a minimal attributable local
disposition record or a reference to an external client-controlled system. A
larger assignment/approval layer should be considered only after those smaller
options prove insufficient.

## 23.7 Acceptance of the Phase 3F boundary

This boundary brief is acceptable when:

- current capability and absence are stated accurately;
- human authority and product calculation authority remain distinct;
- direct overrides and retroactive bundle mutation remain prohibited;
- closure, correctness, explanation, and assurance remain separate concepts;
- the local-first and non-generic-workflow boundaries are explicit;
- hypothetical implementation details remain deferred; and
- client evidence, rather than document completeness, controls later design.

No application-code change, workflow prototype, schema, or companion document is
authorized by this section.

## 23.8 Founder-approved Phase 3F and document posture

The founder approved these decisions on 2026-07-17:

1. **Phase 3F posture:** accept this evidence-gated boundary and do not produce
   a detailed Human Review and Disposition specification until distant-future
   validation evidence shows that PPAR should own part of the workflow.
2. **Document structure:** split the material in principle into a concise
   governing product constitution/roadmap and a separate approved-
   specifications reference, but review the outline before moving content.
3. **Information-first product focus:** maximize valuable and accurate
   information, including how it is presented. Do not assume user workflow is
   uniform or that PPAR should define it.

These decisions close Phase 3F at the appropriate evidence horizon. They do not
authorize workflow implementation or document restructuring.

## 23.9 Proposed document split outline — review only

No restructuring has occurred. The proposed future structure is:

### A. Concise governing document

**Proposed path:** retain
`docs/audit/PPAR_Audit_Foundational_Product_Design.md` as the canonical path,
but replace its contents through a controlled migration with a concise product
constitution and roadmap.

**Purpose:** let a founder, product leader, engineer, implementation lead, or
future commercial leader understand the product and its direction end to end
without reading implementation-level specifications.

**Proposed outline:**

1. Executive product doctrine and enduring promise
2. Problem, customer, user, and jobs-to-be-done hypotheses
3. Current product truth and capability-status taxonomy
4. Product principles, safety contracts, and trust model
5. Product boundaries, non-goals, and local-first doctrine
6. Information and presentation strategy
7. Capability map: current, approved, evidence-gated, deferred, and out of scope
8. First-client validation and claims discipline
9. Product roadmap, evidence gates, and near-term priorities
10. Confirmed decisions, open decisions, and links to governing references

The governing document should state **what**, **why**, **status**, **boundary**,
and **next evidence gate**. It should not repeat detailed processing rules,
schemas, edge cases, or acceptance tests already owned by specifications and
machine-readable contracts.

### B. Approved product-specifications reference

**Proposed path:**
`docs/audit/PPAR_Audit_Product_Specifications_Reference.md`.

**Purpose:** retain detailed, founder-approved product requirements and their
implementation/safety context without making them the entrypoint for product
strategy.

**Proposed contents:**

1. Source authority and implementation-evidence register
2. Phase 2 conceptual architecture, actors, and decision-rights reference
3. Phase 3A Performance Change Investigation specification
4. Phase 3B Performance Data Quality Audit specification
5. Phase 3C Audit Readiness specification
6. Phase 3D Executive Investigation Summary specification
7. Phase 3E Audit Health Dashboard and Operational Intelligence specification,
   clearly marked implementation-DEFERRED
8. Phase 3F Human Review and Disposition boundary, not a detailed specification
9. Cross-cutting acceptance, safety, validation, and claims references
10. Historical implementation-intake and external-evidence appendices only
    where they remain necessary for traceability

This reference should be updated when an approved product requirement changes
or evidence justifies greater specificity. It should not grow merely because a
hypothetical feature can be described.

### C. Source-of-truth and migration rules

If the outline is approved, restructuring should follow these rules:

- preserve the current v0.10 document before moving content;
- make the governing document canonical for product identity, status, scope,
  principles, roadmap, and founder decisions;
- make the specifications reference authoritative for approved detailed product
  requirements;
- keep executable behavior, tests, safety catalog, and machine-readable
  contracts authoritative for current implementation truth;
- link rather than duplicate detailed material;
- preserve decision dates, capability-status labels, requirement identifiers,
  and source traceability;
- remove repetition only after confirming that no unique decision or limitation
  is lost; and
- make no application-code or product-behavior changes during the document
  migration.

**Outline gate — OPEN DECISION:** approve, revise, or reject this outline before
any file is created, renamed, split, or moved. Phase 4 remains gated until the
outline and migration posture are approved.

---

# Appendix A — Source Register

## A.1 `README.md`

Relevant sections:

- product definition,
- Performance Auditing,
- setup,
- audit inputs,
- YAML normalization,
- audit outputs,
- large-report safety behavior.

## A.2 `docs/architecture.md`

Relevant sections:

- product surface,
- package map,
- audit data flow,
- shared-source findings,
- caching and common report tables,
- explicit boundary against full-ledger reconstruction,
- configuration boundary,
- calculation and validation contracts,
- report review order,
- supporting evidence.

## A.3 `performance_comparison_safety_invariants.md`

Classification:

- current maintainer safety contract,
- authoritative for the meaning and enforcement status of twelve safety invariants,
- authoritative for failure classifications and change-control expectations.

Relevant sections:

- no lost differences,
- no double counting,
- Fully Explained arithmetic,
- continuity, currency/unit, and period-boundary safety,
- bidirectional lineage,
- fail-closed policy coverage,
- report-format parity,
- deterministic output,
- complete audit trail and permitted dispositions.

## A.4 `docs/roadmap.md`

Classification:

- central current engineering roadmap plus historical implementation journal.

Use primarily:

- `Current Status`,
- `Current Open Items`,
- explicit current completion/status notes,
- Data Auditing direction,
- onboarding and source-contract boundaries.

Do not interpret every historical numbered phase as a current product commitment.

## A.5 `docs/audit/performance_comparison_design.md`

Classification:

- deep current/historical design reference.

Relevant content:

- current checkpoint and package behavior,
- normalized data and field roles,
- explicit YAML policy,
- conservative transaction matching,
- report vocabulary,
- evidence layers,
- Data Audit Issues design,
- source-contract and reconstruction boundaries.

Explicit current status notes supersede earlier design-only descriptions within the same file.

## A.6 `docs/audit/performance_comparison_transaction_boundary_snapshot.md`

Classification:

- compact reviewer snapshot; the referenced machine-readable matrix remains the implementation contract.

Relevant content:

- covered formula-input families,
- context-required ambiguous flows,
- review-only/context-only rows,
- fixed-income, capital-return, short-side, and standalone backlog gates.

## A.7 `docs/audit/archive/performance_comparison_evidence_pack_review.md`

Classification:

- reviewer aid for the evidence-pack checkpoint.

Relevant content:

- manifest and handoff metadata,
- site extract readiness,
- transaction boundaries,
- test-only fixtures,
- validator coverage,
- public package-surface boundaries.

## A.8 `PPAR.pdf`

Relevant content:

- product overview,
- audit positioning,
- representative audit images,
- analytics content excluded from this product-design phase,
- setup and input/output summary.

## A.9 Current generated audit artifacts

Inspected under:

- `_demo_output/performance_comparison_portfolio/`
- `_demo_output/performance_comparison_security/`

Both current workbooks contain:

- `Performance Differences`
- `Performance Difference Causes`
- `Data Audit Issues`

Current HTML reports use the same three normal review surfaces. Root-level
`source_detail.csv` and `audit_support.zip` accompany each level. The support
archives contain 21 required artifacts, including manifest v4 and review
summary v1. See Appendix B.

## A.10 Founder-provided product context

Material current facts:

- extensive automated testing,
- financial-invariant testing,
- report-reconciliation testing,
- large-dataset performance testing,
- no validation yet using real client Axys/APX exports,
- commercial goal of validation partners first and repeatable software second,
- decision to focus on `ppar audit`,
- support for the “quality assurance layer for portfolio performance” vision.

Current working-tree verification adds:

- project HEAD `e035174` on 2026-07-16;
- 823 tests run at intake with one documentation-style failure caused by nine
  legacy terminology occurrences in the then-supplied handoff/v0.2 documents;
- the v0.2 content is now consolidated into the canonical v0.3 document, while
  the governing handoff retains one legacy occurrence;
- no evidence that the single failure represents a calculation or generated-
  artifact regression.

## A.11 Machine-readable transaction and extract contracts

Primary files:

- `docs/axys_apx/contracts/transaction_semantics_matrix.yaml`
- `ppar/setup_templates/axys_apx_performance_comparison/demo_extract_availability.yaml`
- `docs/axys_apx/contracts/templates/site_extract_contract*.yaml`
- `ppar/performance_comparison/extract_contract.py`

The YAML transaction matrix outranks the compact boundary snapshot and rendered
Markdown when coverage labels drift. The packaged extract-availability contract
and any approved site contract define runtime context requirements.

## A.12 Current setup and policy surfaces

Primary files:

- `ppar/setup_templates/axys_apx_performance_comparison/README.md`
- `ppar/setup_templates/axys_apx_performance_comparison/axys_apx_performance_comparison.yaml`
- `docs/audit/demo_source_contract.md`
- `docs/audit/site_extract_readiness_checklist.md`
- `docs/axys_apx/axys_apx_common_core_export.md`

These establish the first-user path, exact current optional Data Audit rules,
split evidence, source-contract boundary, and site-readiness responsibilities.

## A.13 Executable report-bundle contract

`ppar.performance_comparison.bundle.report_bundle_contract()` is authoritative
for current bundle modes, artifact names, required manifest/review-summary keys,
review entrypoints, and versions. At this intake it reports manifest version 4,
normalization version 1, and review-summary version 1.

## A.14 Prior strategy drafts

The earlier product-expansion specification and roadmap supplied:

- five product areas,
- executive investigation summary,
- rules-library direction,
- multi-stage product vision.

Those ideas are incorporated here as approved directions or candidates rather than current product claims.

---

# Appendix B — Representative Workbook Observations

## B.1 Performance Differences

The current generated portfolio sheet contains 25 data rows and demonstrates:

- multiple portfolios,
- multiple date intervals,
- decimal performance differences,
- explained and unexplained columns,
- status,
- comments.

Observed analytical statuses:

- Fully Explained
- Partly Explained
- Unexplained

Notable examples:

- A fully explained period where the explained difference equals the reported performance difference.
- A partly explained `BALANCED` period with a residual of `0.000200` and a comment indicating that a JPM transaction may be a cause but is not counted under current YAML configuration.
- An `INCOME` period with an unexplained difference of `0.000350` and a comment directing review toward missing source-data, timing differences, or vendor methodology mismatch.

### Product-design observations

- The status model is a strong foundation.
- The comments demonstrate honest boundary behavior.
- The report is analyst-oriented.
- The difference units are displayed as decimals; future presentation should make decimal, percentage, and basis-point units unambiguous.
- The sheet does not yet provide a management-level root-cause summary.

## B.2 Performance Difference Causes

The current generated portfolio sheet contains 140 data rows and demonstrates:

- source dataset and field,
- source values from each snapshot,
- arithmetic source difference,
- quantified performance effect when counted,
- narrative relationship,
- inherited beginning-value effects,
- holdings, cash, transaction, price, quantity, commission, and accrued-interest examples.

### Product-design observations

- Counted effects and contextual relationships appear in one table; this is analytically valuable but may require stronger visual classification.
- Highlighting draws attention to quantified effects, but future outputs should not rely on formatting alone.
- Cause-category summaries would make large investigations easier to navigate.
- The narrative is evidence-based and specific, which should be preserved.
- The product should expose a stable cause ID or lineage reference in future detailed specifications.

## B.3 Data Audit Issues

The current generated portfolio sheet contains 26 data rows and demonstrates:

- snapshot-specific findings,
- portfolio, date, dataset/field, and security,
- issue type,
- reference and observed values,
- difference,
- tolerance,
- explanation.

Observed issue types include:

- `missing_dividend`
- `duplicate_transactions`
- `holdings_accrued_rate`
- `pa_sa_rate`
- `transactions_price_range`
- `holdings_price_range`
- `dividend_rate`

### Product-design observations

- The tolerance column is important and should remain visible.
- Cross-portfolio relationships can provide useful reference evidence.
- Rule output must avoid implying that the reference portfolio is necessarily correct.
- Duplicate transactions appear as multiple rows; detailed specification should determine whether users need row-level evidence, grouped exceptions, or both.
- Severity and recommended reviewer action are not present and are natural additions to the future rules framework.

---

# Appendix C — Incorporated Product-Expansion Inventory

This appendix preserves the content of the earlier expansion specification without implying implementation.

## C.1 Performance Change Investigation enhancements

- explanation completeness,
- root-cause summary,
- repeated-restatement timeline,
- portfolio stability ranking,
- cross-period investigation history,
- cause trends.

## C.2 Data-quality rule directions

### Pricing

- zero or negative prices,
- missing prices,
- stale prices,
- extreme price changes,
- price-source inconsistencies.

### Transactions

- duplicates,
- impossible amount/price/quantity relationships,
- settlement or trade-date anomalies when fields support them,
- unknown transaction codes,
- holiday or back-dated activity,
- income after position disposal.

### Holdings

- negative positions where unexpected,
- position appearance/disappearance,
- zero quantity with value,
- quantity inconsistencies,
- holdings unsupported by available activity.

### Income

- missing or duplicate dividends,
- dividend currency mismatch,
- coupon or accrued-interest inconsistencies,
- withholding-tax anomalies.

### Corporate actions

- split and reverse-split mismatches,
- missing spin-offs,
- merger conversion inconsistencies,
- identifier continuity issues.

### Foreign exchange

- missing rates,
- extreme changes,
- base or security currency mismatch,
- prior-period FX restatement.

## C.3 Audit Health Dashboard

Candidate metrics:

- restatements by month,
- most corrected portfolios,
- largest unresolved differences,
- explanation completeness,
- common cause categories,
- common rule failures,
- trend analysis.

## C.4 Audit Readiness

Candidate checks:

- missing files,
- missing columns or mappings,
- unknown transaction codes,
- missing prices,
- missing FX,
- missing dividends,
- configuration inconsistencies,
- unsupported requested scope.

## C.5 Operational Intelligence

Candidate analysis:

- most corrected portfolios,
- most corrected securities,
- recurring transaction types,
- recurring data sources or custodians when available,
- common configuration gaps,
- long-term quality trends.

## C.6 Executive Investigation Summary

Candidate content:

- entity and period,
- reported return change,
- explained and unexplained amounts,
- explanation completeness,
- largest causes,
- priority findings,
- recommended next actions,
- status,
- evidence references.

## C.7 Rules Library

Candidate rule-definition fields:

- stable ID,
- name,
- category,
- severity,
- rationale,
- required data,
- logic,
- tolerance,
- evidence,
- performance relationship,
- reviewer action,
- configuration,
- false positives,
- tests,
- status.

---

# Appendix D — Planned Foundational Design Sections

The canonical document will be extended through the following drafting phases.

## Phase 2 — Users, Workflows, and Conceptual Architecture

**Document status: founder-approved on 2026-07-16. Recommended defaults are
authorized as Phase 3 working assumptions.**

- actor models,
- decision rights,
- normal workflow,
- exception workflow,
- conceptual system components,
- evidence lifecycle.

## Phase 3 — Detailed Functional Specifications

- Performance Change Investigation — Phase 3A founder-approved on 2026-07-16
  with four authorized working assumptions,
- Performance Data Quality Audit — Phase 3B founder-approved on 2026-07-16
  with five authorized working assumptions,
- Audit Readiness — Phase 3C founder-approved on 2026-07-16 with six authorized
  working assumptions,
- Executive Investigation Summary — Phase 3D founder-approved on 2026-07-16
  with six authorized working assumptions,
- Audit Health Dashboard and Operational Intelligence — Phase 3E founder-
  approved on 2026-07-17 with seven authorized working assumptions; its
  implementation remains evidence-gated and DEFERRED,
- Human Review and Disposition — Phase 3F boundary founder-approved on
  2026-07-17; detailed specification remains deliberately DEFERRED pending
  distant-future client evidence.

Before Phase 4, the founder will review the v0.10 proposed split between a
concise canonical product constitution/roadmap and a separate approved-
specifications reference. No restructuring has yet occurred.

## Phase 4 — Audit Rules Catalog Framework

- rule schema,
- finding taxonomy,
- severity,
- tolerance,
- materiality,
- initial high-value rule priorities.

## Phase 5 — Roadmap and Release Gates

- current baseline,
- real-client validation,
- repeatable Axys/APX product,
- proactive quality assurance,
- workflow and organizational knowledge,
- broader platform support.

## Phase 6 — Commercial and Implementation Design

- ideal customer profile,
- buyer and champion,
- pilot design,
- implementation responsibilities,
- evidence required for case studies,
- packaging,
- trust and liability controls,
- renewal and expansion criteria.

---

# Closing Product Doctrine

PPAR Audit should become the quality assurance layer for portfolio performance, but it must earn that position through narrow, defensible excellence.

The product’s most important behavior is not producing a large report or maximizing automatic explanations. It is maintaining a trustworthy boundary between:

- what changed,
- what can be proven,
- what can be quantified,
- what appears suspicious,
- and what still requires human judgment.

The enduring promise is:

> **When reported performance changes, PPAR Audit tells the reviewer why—or clearly identifies what it cannot explain—and preserves the evidence needed to trust that distinction.**

---

# Appendix E — Additional Implementation-Document Intake

## E.1 Intake conclusion

The additional documents materially strengthen Phase 1. They do not change the central product identity, but they reveal that the current implementation has a more mature safety, evidence, readiness, and handoff foundation than the initial product overview alone showed.

The strongest additions are:

- an explicit twelve-invariant safety contract,
- lossless finding dispositions and economic-effect ownership,
- bidirectional source lineage,
- fail-closed currency, period, field-role, and policy handling,
- deterministic report bundles with cross-format semantic parity,
- conservative transaction row identity,
- extract-contract and ambiguous-transaction readiness controls,
- a machine-oriented evidence pack with manifests and review summaries.

These capabilities strengthen the product’s trust proposition. They do not resolve the absence of real-client validation.

## E.2 Phase 1 changes caused by the intake

This revision:

1. Records local-first execution as a permanent product principle.
2. Elevates the safety-invariant catalog into the product doctrine.
3. Reclassifies Audit Readiness from purely future to a current technical foundation plus a future user-facing experience.
4. Reclassifies the Executive Investigation Summary as having a current technical handoff foundation, while preserving the management experience as future work.
5. Recognizes current rule-system foundations without claiming a complete product-facing rules library.
6. Adds complete-audit-trail, disposition, economic-effect ownership, failure-class, transaction-match, source-contract, and local-first terminology.
7. Strengthens claims that are defensible from the current implementation while preserving the real-client validation boundary.
8. Adds transaction identity and semantic boundaries to the Phase 2 design scope.

## E.3 Discrepancies and uncertainty requiring controlled interpretation

### Cumulative documents mix current and historical statements

The deep design note and central roadmap preserve substantial implementation history. Earlier passages can describe a capability as future or diagnostic-only even when later status notes say it is implemented. Product truth should therefore be taken from current checkpoint/status sections and explicit implementation contracts, not by treating every paragraph as equally current.

### Packaged-demo release quality is not client production validation

The central roadmap describes release-candidate quality for the packaged Axys/APX demo scope. That is compatible with the founder’s statement that PPAR has not been validated using a real client’s Axys/APX exports. The foundational design must never convert demo/release engineering quality into a claim of broad production readiness.

### Narrow packaged transaction stories do not establish universal support

The central roadmap records narrow context-gated packaged examples for transaction families that the compact boundary snapshot still treats as backlog or context dependent in broader production use. The correct interpretation is:

- a realistic packaged example may exist,
- the code family may still be unsafe by code alone,
- production treatment still requires source context or reviewed local policy,
- general Axys/APX support must not be claimed from one fixture story.

The current machine-readable matrix was inspected in v0.3 and confirms that
the compact boundary snapshot and rendered Markdown lag later packaged-demo
coverage. The YAML matrix remains authoritative. In particular, `rc`, `pd`,
`ss`, and `cs` have narrow packaged-demo fixtures while code-only production
treatment remains unsafe; `;` has test-only site-variant coverage. None of this
establishes universal Axys/APX behavior.

### Manifest-version control descriptions contain stale v3 references

Current generated manifests, the executable bundle contract, and the current
safety-invariant document use manifest version 4. A `Current Open Items`
paragraph in the roadmap and the executable `SN-11` existing-control description
still say v3. Treat those two descriptions as stale control-inventory text; do
not downgrade the current artifact contract. The application code is unchanged
by this product-design revision.

### Data Audit language contains historical status drift

One design passage refers to `Data Audit Issues` as future workbook vocabulary,
while the current generated workbooks, README, setup YAML, and current roadmap
show it as implemented. The current artifacts and current-status documentation
take precedence. The exact current optional rule list contains seven checks;
mandatory beginning/end continuity is separate.

### Confidence and severity remain product-design questions

Internal finding structures and historical design ideas mention confidence and severity vocabulary, but the current Data Audit worksheet intentionally avoids a severity column and the product has no validated confidence metric. Phase 1 therefore retains explanation completeness rather than a confidence score, and treats severity/prioritization as future rules-library design.

### Split and broader corporate-action scope must stay qualified

`splits` is a current optional normalized dataset. The packaged CVNA split-
factor row is review evidence that supports a related holdings quantity/value
correction; the split row does not own an explained amount and is not a current
Data Audit rule. Mergers, spin-offs, ticker changes, and broader corporate-
action interpretation remain evidence blocked or future. “Split support” must
not be generalized into comprehensive corporate-action support.

## E.4 Additional documents incorporated for Phase 2

The v0.3 intake incorporated the previously requested next batch:

1. `docs/audit/demo_source_contract.md`
2. `docs/audit/site_extract_readiness_checklist.md`
3. `docs/axys_apx/contracts/transaction_semantics_matrix.yaml` and its
   rendered Markdown counterpart
4. the setup-installed audit README and current starter `ppar.yaml`

These documents informed the operator onboarding path, source-contract approval
workflow, readiness failure experience, transaction-policy decision rights, and
first-client responsibilities in Phase 2.

# Appendix F — External Evidence

External sources are contextual evidence for actor, governance, and market
framing. They do not establish a current PPAR capability and do not outrank the
repository for product truth.

## F.1 GIPS Standards Handbook for Firms

Source:
[GIPS Standards Handbook for Firms](https://www.gipsstandards.org/standards/gips-standards-for-firms/gips-standards-handbook-for-firms/)

Phase 2 relevance:

- firms document and consistently apply performance policies and procedures;
- performance responsibilities can require coordination among operations,
  performance, compliance, and marketing;
- records must support presented performance information;
- firms retain responsibility for policy and relied-upon outsourced inputs;
- verification has a defined independence/firm-wide policy boundary and does
  not imply assurance on every specific report.

Product interpretation: client policy authority, evidence retention, and the
assurance boundary must remain explicit. PPAR does not claim GIPS compliance or
verification.

## F.2 CFA Institute performance-measurement role evidence

Sources:

- [Performance Measurement Analyst career-path overview](https://www.cfainstitute.org/insights/events/2026/career-path-performance-measurement-analyst-part-4)
- [CIPM career prospects](https://www.cfainstitute.org/programs/cipm/career-prospects)

Phase 2 relevance: performance-measurement professionals produce routine and
non-routine reports and serve portfolio managers, investment leadership,
marketing, risk, and clients. This supports—but does not prove—the performance-
analyst primary-user hypothesis.

## F.3 SS&C product context

Source:
[SS&C Technologies 2025 Form 10-K](https://investor.ssctech.com/static-files/8d784cd6-7a12-4eaf-a8a6-8f7672e2fb2b)

Phase 2 relevance: SS&C describes APX as spanning portfolio accounting,
performance measurement, and reporting, and Axys as serving small-to-mid-sized
investment-management organizations. This supports the need to involve source-
system/report owners and reinforces Axys/APX as a plausible initial client
context. It does not prove PPAR compatibility with any site or export pattern.

## F.4 Modified Dietz methodology context for Phase 3A

Source:
[GIPS Standards Handbook for Firms](https://www.gipsstandards.org/standards/gips-standards-for-firms/gips-standards-handbook-for-firms/)

Phase 3A relevance: the handbook describes Modified Dietz as an estimate that
weights external cash flows according to the time they are held in the
portfolio. It requires a defined and consistently applied flow-treatment policy
and distinguishes beginning-of-day from end-of-day timing assumptions. It also
notes that the estimate is less accurate when large external flows coincide
with high volatility than a method that values at flow time.

Product interpretation: PPAR must treat flow classification, timing, day count,
inclusion rule, and sign convention as explicit client-approved policy; surface
limitations honestly; and avoid claiming that source-derived reconstruction is
the client's official return or an exact result under every flow pattern.
