# PPAR Audit

## Foundational Product Design

### Phases 1 through 3B — Doctrine, Conceptual Architecture, and Detailed Functional Specifications

| Document field | Value |
|---|---|
| Document status | Phases 2 and 3A founder-approved; Phase 3B draft for founder review |
| Version | 0.5 |
| Date | 2026-07-16 |
| Primary audience | Founder, product leadership, engineering, implementation, and future commercial leadership |
| Writing posture | Internal and candid first; externally reusable second |
| Canonical file | `PPAR_Audit_Foundational_Product_Design.md` |
| Phase covered | Phase 1 — Product Doctrine and Boundaries; Phase 2 — Users, Workflows, and Conceptual Product Architecture; Phase 3A — Performance Change Investigation; Phase 3B — Performance Data Quality Audit |
| Supersedes | Version 0.2; its reviewed Phase 1 content is incorporated into this canonical document |
| Next phase gate | Founder review and approval of Phase 3B; Phase 3C must not begin before approval |
| Confirmed deployment doctrine | Local-first execution within the client-controlled environment is a permanent product principle |

## Change Log

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
21. [Appendix A — Source Register](#appendix-a--source-register)
22. [Appendix B — Representative Workbook Observations](#appendix-b--representative-workbook-observations)
23. [Appendix C — Incorporated Product-Expansion Inventory](#appendix-c--incorporated-product-expansion-inventory)
24. [Appendix D — Planned Foundational Design Sections](#appendix-d--planned-foundational-design-sections)
25. [Appendix E — Additional Implementation-Document Intake](#appendix-e--additional-implementation-document-intake)
26. [Appendix F — External Evidence](#appendix-f--external-evidence)

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
- `docs/audit/performance_comparison_demo_source_contract.md`,
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

### 0.5 Capability-status language

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

The 2026-07-16 intake used project HEAD `e035174` and the current working tree.
The full suite ran 823 tests and completed with one documentation-style failure:
the then-supplied handoff and v0.2 snapshot contained nine instances of the
legacy unhyphenated term. The v0.2 content has since been consolidated into this
canonical document and the redundant snapshot removed; the governing handoff
retains one legacy occurrence. This failure is not evidence of an application
calculation or artifact regression, but the release gate is not described as
fully passing. Current generated bundles and the executable bundle contract
independently confirm manifest version 4 and the expected report surfaces.

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
| Run the audit or rerun after correction | Authorized analyst/operator | Preapproved operating procedure; manager approval for scope/policy changes | Local administrator | New artifacts must identify inputs, configuration, and product version; prior artifacts remain retained under policy. | CURRENT — DEMONSTRATED; CURRENT — REQUIRES CLIENT VALIDATION |
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
before fully explained periods, without hiding the latter. Current Problems and
needs-review summaries provide the technical foundation; management
materiality and exact ordering remain CANDIDATE.

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

## 19.19 Dependencies and open decisions

### Dependencies

- founder-approved Phase 2 actors, decision rights, and local-first workflow;
- founder-approved Phase 3A evidence/cause boundary and pilot report scope;
- current source/extract, normalization, safety-invariant, and bundle contracts;
- Phase 3C design for a unified readiness and preflight experience;
- Phase 4 rule schema, prioritization, and catalog-governance design;
- real-client source and policy owners for tolerance and filter approval; and
- deferred workflow/history capabilities only if later business validation
  justifies them.

### Open decisions for founder review

1. **Rule-execution summary — OPEN DECISION:** recommended direction is to
   adopt the five execution outcomes in `PDQ-036` and the minimum summary in
   `PDQ-044`, without adding human workflow infrastructure.
2. **Pilot rule-set policy — OPEN DECISION:** whether every pilot begins with
   all seven optional rules deliberately configured, or only the rules relevant
   to validated source populations. Recommended default: configure only
   understood populations while always retaining mandatory continuity.
3. **Malformed configuration — OPEN DECISION:** recommended direction is to
   make invalid enablement, tolerance, and filter shapes fail validation
   instead of silently falling back.
4. **Data Audit placement — OPEN DECISION:** retain Data Audit primarily inside
   the integrated audit bundle for initial pilots; consider a standalone entry
   point only after client workflow shows independent demand.
5. **Rule-result priority — OPEN DECISION:** validate whether the Phase 3A
   operational-materiality dimensions are sufficient or Phase 4 needs a
   separate rule-specific priority model. Do not introduce confidence or a
   composite quality score.

None of these decisions authorizes per-finding workflow, hidden suppression,
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

## 19.22 Phase 3B completion and next gate

Phase 3B is complete as a **draft functional specification for founder review**.
It defines the current rule surface, exact snapshot execution boundary,
configuration and tolerance behavior, mandatory/optional policy, evidence and
cause separation, outputs, failure policy, false-positive management,
acceptance criteria, and client-validation plan.

Phase 3C MUST NOT begin until the founder reviews and approves Phase 3B and
resolves or authorizes working assumptions for the open decisions in Section
19.19.

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
- `ppar/setup_templates/axysapx_performance_comparison/demo_extract_availability.yaml`
- `docs/axys_apx/contracts/templates/site_extract_contract*.yaml`
- `ppar/performance_comparison/extract_contract.py`

The YAML transaction matrix outranks the compact boundary snapshot and rendered
Markdown when coverage labels drift. The packaged extract-availability contract
and any approved site contract define runtime context requirements.

## A.12 Current setup and policy surfaces

Primary files:

- `ppar/setup_templates/axysapx_performance_comparison/README.md`
- `ppar/setup_templates/axysapx_performance_comparison/axysapx_performance_comparison.yaml`
- `docs/audit/performance_comparison_demo_source_contract.md`
- `docs/audit/site_extract_readiness_checklist.md`
- `docs/axys_apx/axysapx_common_core_export.md`

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
- Performance Data Quality Audit — Phase 3B draft completed in v0.5; awaiting
  founder review and approval,
- Audit Readiness — Phase 3C not started and gated on Phase 3B approval,
- Executive Investigation Summary,
- Audit Health Dashboard,
- Operational Intelligence.

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

1. `docs/audit/performance_comparison_demo_source_contract.md`
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
