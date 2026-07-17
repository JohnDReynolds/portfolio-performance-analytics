# PPAR Audit

## Product Constitution and Roadmap

| Document field | Value |
|---|---|
| Document status | Visionary product design complete — good enough for now; MVP completion is the active phase |
| Version | 1.4 |
| Date | 2026-07-17 |
| Primary audience | Founder, product leadership, engineering, implementation, and future commercial leadership |
| Writing posture | Internal and candid first; externally reusable second |
| Canonical scope | Product identity, current truth, principles, boundaries, capability status, validation doctrine, and roadmap |
| Specifications index | [`product_specifications_index.md`](product_specifications_index.md) |
| Preserved pre-split snapshot | [`archive/PPAR_Audit_Foundational_Product_Design_v0.10.md`](archive/PPAR_Audit_Foundational_Product_Design_v0.10.md) |
| Active MVP plan | [`mvp_plan.md`](mvp_plan.md) |
| Next gate | Slice 2 Executive Summary usability acceptance |

## Change Log

### Version 1.4 — 2026-07-17

- Recorded completion of the bounded Slice 2 Executive Summary shared model.
- Recorded founder rejection of its first jargon-heavy presentation and kept
  Slice 2 open for a genuine 30-second first-view refinement.
- Preserved the constitution's capability boundary and deferred product choices;
  executable detail and acceptance remain owned by the active MVP plan.
- The first additional Data Issues issue type remains sequenced after Slice 2
  usability acceptance.

### Version 1.3 — 2026-07-17

- Recorded the founder decision that bounded Axys/APX transaction-semantics,
  exact-case, source-contract, and demo coverage is the fourth MVP capability
  and is blocking for MVP completion.
- Approved the Executive Summary scope and a fixed, version-controlled initial
  limit of ten priority review units without YAML configuration.
- Clarified that this constitution owns the MVP capability boundary while the
  active MVP plan owns implementation sequence, detailed acceptance, and status.
- Made no application-code or executable-contract change.

### Version 1.2 — 2026-07-17

- Recorded that MVP Slice 1 is complete and narrowed the next founder gate to
  the Executive Summary scope and first additional Data Issues issue type.
- Replaced the near-duplicate detailed specifications file with a compact index
  to the complete archived v0.10 corpus and current owning contracts.
- Added an Audit documentation landing page and clarified the lifecycle of
  current, indexed, and archived material.
- Made no additional product-design commitment and did not expand workflow,
  dashboard, history, or speculative rule scope.

### Version 1.1 — 2026-07-17

- Closed the visionary product-design exercise as **good enough for now**.
- Replaced the planned next visionary phase with concrete MVP completion work.
- Recorded three founder-defined MVP gaps: an Executive Summary worksheet before
  Performance Differences; additional enumerated Data Issues issue types; and
  stronger codification of causes/issues with appropriate YAML control.
- Linked a separate active MVP implementation plan grounded in the current code,
  workbook contract, issue checks, cause areas, and YAML behavior.
- Kept history, dashboards, Operational Intelligence, broad rules catalogs, and
  managed workflow outside the active MVP phase.

### Version 1.0 — 2026-07-17

- Consolidated the approved v0.10 foundational design into a governing product
  constitution and roadmap that can be read end to end.
- Moved detailed approved specifications to a separate reference without
  changing their product status or authorizing implementation.
- Preserved the complete v0.10 document as a read-only migration snapshot.
- Elevated the information-first product principle: maximize the value,
  accuracy, structure, prioritization, explanation, and presentation of
  supported information.
- Kept managed workflow as a distant, evidence-gated possibility rather than an
  assumed product layer.
- Made no application-code or product-behavior changes.

## Contents

1. [Executive Product Doctrine](#1-executive-product-doctrine)
2. [Problem, Customer, Users, and Jobs](#2-problem-customer-users-and-jobs)
3. [Current Product Truth](#3-current-product-truth)
4. [Product Principles, Safety, and Trust](#4-product-principles-safety-and-trust)
5. [Product Boundaries and Local-First Doctrine](#5-product-boundaries-and-local-first-doctrine)
6. [Information and Presentation Strategy](#6-information-and-presentation-strategy)
7. [Capability Map](#7-capability-map)
8. [First-Client Validation and Claims](#8-first-client-validation-and-claims)
9. [Product Roadmap and Evidence Gates](#9-product-roadmap-and-evidence-gates)
10. [Decisions, Open Questions, and Governing References](#10-decisions-open-questions-and-governing-references)

---

# 1. Executive Product Doctrine

## 1.1 Product identity

PPAR stands for **Portfolio Performance Auditing & Analytics Reporting**. This
governing design concerns **PPAR Audit** and the `ppar audit` workflow only.
Performance Analytics has a separate product identity, buyer, use case,
message, and roadmap.

The internal category thesis is:

> **PPAR Audit is the quality assurance layer for portfolio performance.**

“Performance Quality Assurance” is an internal product territory, not a claim
that PPAR provides independent assurance or defines an established market
category.

## 1.2 Core problem and commercial wedge

PPAR Audit addresses one operational question:

> **Why did previously reported portfolio or security performance change
> between two portfolio-accounting snapshots?**

The commercial wedge is:

> **Structured investigation of changed reported portfolio performance.**

The plain-language lead message is:

> **Software that explains why reported portfolio performance changed.**

The enduring product promise is:

> **When reported performance changes, PPAR Audit tells the reviewer why—or
> clearly identifies what it cannot explain—and preserves the evidence needed
> to trust that distinction.**

The second half is essential. The product must preserve uncertainty rather than
manufacture certainty.

## 1.3 Mission and intended outcome

The mission is to help investment organizations continuously improve the
quality of reported portfolio performance.

For each supported investigation, PPAR should:

- identify changed reported performance;
- quantify defensible causes under approved policy;
- distinguish explained from unresolved differences;
- surface suspicious source-data relationships independently;
- preserve complete, navigable evidence; and
- make the result faster and easier for qualified people to review.

## 1.4 Commercial reality

The current product has substantial internal engineering validation, including
automated tests, financial invariants, report reconciliation, output integrity,
determinism, and scale checks within the supported packaged scope.

It has **not** been validated using a real client's production-style Axys/APX
exports and approved local accounting policy. This is the most important
current product and commercial limitation.

The near-term objective is to obtain approximately **2–5 strong validation
partners**, learn what implementation actually requires, and convert that
evidence into a repeatable Axys/APX-oriented software product before attempting
broad scale.

---

# 2. Problem, Customer, Users, and Jobs

## 2.1 Problem context

Historical returns can change for legitimate or erroneous reasons, including
revised holdings, back-dated transactions, price changes, FX changes, split
treatment, accrued interest, extract timing, mapping, or methodology.

Portfolio-accounting systems can show the changed result without providing a
structured, quantified, evidence-preserving explanation of why it changed.
Teams often investigate through repeated exports and spreadsheets, with
inconsistent evidence retention and heavy dependence on individual expertise.

Generic comparison tools can find differences. PPAR's differentiated value is
connecting supported source-data changes to changed performance while keeping
unsupported or ambiguous evidence visible for review.

## 2.2 Initial validation-partner hypothesis

The strongest first-partner hypothesis is a medium-sized RIA or institutional
investment manager that:

- uses Axys first, or APX where exports fit the current contract;
- has a dedicated operations or performance function;
- experiences recurring historical corrections or restatements;
- investigates through spreadsheets or other manual comparisons;
- can provide local CSV exports and approve mappings promptly;
- has analysts and managers willing to challenge PPAR results candidly;
- accepts a controlled local implementation; and
- may become a reference client only if measured evidence supports it.

Poor first-partner profiles include firms seeking universal platform support,
unwilling to approve accounting policy, unable to provide reproducible exports,
requiring a hosted PPAR data service, or treating an early pilot as independent
assurance.

## 2.3 User and authority hypotheses

| Role | Primary relationship to PPAR Audit |
|---|---|
| Performance, portfolio-accounting, or operations analyst | Daily investigator and evidence reviewer |
| Performance/operations manager | Product champion, policy approver, and escalation owner |
| Head/Director of Investment Operations, Director of Performance, COO | Economic buyer or executive sponsor hypothesis |
| Source/extract administrator | Owner of export facts, source-state provenance, and schema changes |
| Methodology owner | Approver of return basis, transaction semantics, tolerances, and material policy |
| Compliance/GIPS reviewer | Reviewer of methodology, evidence, error handling, and permitted reliance |
| Local product administrator and technology/security | Owner of client-controlled deployment, access, retention, and versions |
| PPAR support/implementation | Product guidance within the local-first and client-authority boundary |

These are hypotheses until validated. Technical access does not grant accounting
authority, and operating the software does not grant methodology authority.

## 2.4 Primary jobs to be done

### Investigation job

When a reported return changes, determine what changed, quantify supported
causes, identify the unresolved portion, and preserve evidence so the conclusion
can be reviewed and defended.

### Data-quality job

Identify suspicious source-data relationships that may affect performance
quality, without automatically declaring them errors or counted causes.

### Readiness job

Before running an investigation, determine whether files, fields, mappings,
policy, currency/unit treatment, periods, and requested scope are safe and
sufficient for the intended use.

### Management-information job

Present the most decision-useful, accurate result and limitation first, with a
clear route to complete evidence.

### Human authority job

Qualified client personnel approve policy, judge source correctness, make
corrections, and determine official reported performance. PPAR informs those
decisions; it does not assume ownership of each client's workflow.

---

# 3. Current Product Truth

## 3.1 Capability-status taxonomy

Use these labels consistently:

| Status | Meaning |
|---|---|
| **CURRENT — DEMONSTRATED** | Present in current artifacts or directly observed behavior |
| **CURRENT — DOCUMENTED** | Described as current in authoritative documentation or executable contracts |
| **CURRENT — REQUIRES CLIENT VALIDATION** | Implemented and internally tested, but not validated on real client exports and approved policy |
| **APPROVED DIRECTION** | Accepted product direction, not necessarily implemented |
| **CANDIDATE** | Aligned idea requiring prioritization, evidence, and design |
| **DEFERRED** | Potentially useful but intentionally postponed |
| **OUT OF SCOPE** | Inconsistent with the product identity or unacceptable dilution |
| **OPEN DECISION** | Requires explicit founder direction |

A packaged demo or passing synthetic test does not establish real-client
validation. Configurability does not establish vendor compatibility.

## 3.2 Public local workflow

The documented path is:

```text
ppar setup <site_directory>
ppar audit <site_directory>/audit
```

Setup creates a local starter workspace with commented configuration and
Axys/APX-oriented starter files. Routine data processing and calculations occur
inside the client-controlled environment.

## 3.3 Inputs and interpretation

- Snapshot A and Snapshot B are neutral labels; neither is presumed correct.
- Portfolio performance is the minimum top-level comparison surface.
- Depending on scope and configuration, supporting data can include security
  performance, holdings, transactions, FX rates, and split evidence.
- Split factors are an optional evidence-only normalized surface, not a current
  Data Issues rule or counted split cause.
- YAML maps files, fields, accounting roles, transaction semantics, impact
  treatment, tolerances, suppressions, and report assumptions.
- Required treatment is fail-closed. Unknown changed fields and incomplete
  policy must not be hidden through suppression or guessed silently.
- Transaction matching is conservative. Ambiguous no-ID groups remain unpaired,
  and transaction code alone may be insufficient to establish meaning.

PPAR does not rebuild the complete accounting ledger. Its cause attribution is
bounded by supplied data, configured Modified Dietz roles, explicit policy,
evidence lineage, and safety rules.

## 3.4 Current analytical behavior

Within the supported contract, PPAR can:

- normalize and compare two snapshots;
- identify portfolio- and security-level performance differences;
- identify additions, removals, and changed fields;
- conservatively match source records;
- attribute supported Modified Dietz effects to recognizable source rows;
- separate counted causes, input/support components, related outputs, context,
  and review-only evidence;
- classify review units as Fully Explained, Partly Explained, or Unexplained;
- preserve a complete finding-level audit trail and cause lineage; and
- stop on internal inconsistency or unsafe required interpretation.

This is supported formula-bound attribution, not unrestricted causal inference
or a universal return-calculation engine.

## 3.5 Current Data Issues surface

Current optional checks cover:

- conservatively identified missing dividends;
- exact duplicate transactions;
- holdings accrued-rate consistency;
- purchase/sale accrued-interest rate consistency;
- transaction price ranges;
- holdings price ranges; and
- dividend-rate consistency.

Beginning/end continuity remains a mandatory visible integrity finding.

Data Issues findings are independent of additive performance causes. A
suspicious relationship must not automatically change the explained amount.
Utility, tolerance calibration, false-positive behavior, and reviewer action
remain subject to client validation.

## 3.6 Current outputs and evidence

Current report bundles can include:

- level-specific XLSX and HTML review reports;
- the bounded first-view `Executive Summary` plus the unchanged primary
  `Performance Differences`, `Performance Difference Causes`, and `Data Issues`
  analytical surfaces;
- complete `source_detail.csv` and `findings.csv` evidence;
- primary and supporting CSV summaries and diagnostics;
- `review_summary.json` handoff metadata;
- `manifest.json` version 6;
- cause lineage and typed semantic/display fingerprints; and
- compact or expanded audit-support artifacts.

The normal review path begins with the Executive Summary, then exact performance
differences and their supported causes, independent Data Issues findings, source
detail, and supporting diagnostics. Reconstruction and transaction-matching
diagnostics remain secondary.

## 3.7 Current maturity summary

| Capability | Governing status and limitation |
|---|---|
| Local Python execution and setup | CURRENT — DOCUMENTED; client deployment/support validation still required |
| Axys/APX starter mappings and demo seed | CURRENT — DOCUMENTED; not proof of compatibility with every site |
| Two-snapshot portfolio/security comparison | CURRENT — DEMONSTRATED; real client variability untested |
| Modified Dietz source-row explanation | CURRENT — DEMONSTRATED and DOCUMENTED; coverage depends on evidence and policy |
| Fully/Partly/Unexplained classification | CURRENT — DEMONSTRATED; operational interpretation requires validation |
| Evidence, lineage, conservation, parity, determinism | CURRENT — DOCUMENTED safety foundation |
| Data Issues checks | CURRENT — DEMONSTRATED; value and noise require validation |
| Technical preflight/readiness controls | CURRENT — DOCUMENTED; unified product experience remains APPROVED DIRECTION |
| XLSX, HTML, CSV, manifest, and bundle output | CURRENT — DEMONSTRATED; management usability requires validation |
| Executive Summary | CURRENT — UNDER USABILITY REFINEMENT; shared model implemented, first presentation rejected in founder review |
| Persistent history, health dashboard, Operational Intelligence | APPROVED DIRECTION but implementation DEFERRED behind recurring-use evidence |
| Managed human workflow | Distant evidence-gated possibility; not an assumed product layer |
| Broad non-Axys/APX compatibility | CANDIDATE only after a documented, tested, supportable source contract |

---

# 4. Product Principles, Safety, and Trust

## 4.1 Governing principles

1. **Changed-performance explanation remains the center.** Expansion must
   strengthen performance quality or review.
2. **Information quality comes before workflow ownership.** Maximize useful,
   accurate information and how it is presented; do not assume PPAR should own
   each client's processing workflow.
3. **Evidence comes before assertion.** Every material conclusion should trace
   to inputs, normalized facts, policy, calculation, and evidence.
4. **Quantitative conservation is non-negotiable.** No reportable difference is
   silently lost, and one economic effect is counted once.
5. **Do not guess silently.** Block, qualify, or surface ambiguity.
6. **Keep cause, context, anomaly, configuration, and residual distinct.** They
   answer different questions and must not be blended for convenience.
7. **Human review is a designed outcome.** Unexplained can be correct and
   trustworthy.
8. **Determinism and reproducibility are default product behavior.** Formats
   present shared validated semantics rather than separate calculations.
9. **Configuration is controlled business policy.** It should be versioned,
   reviewable, attributable, and approved.
10. **Tolerance and materiality are explicit and separate.** Neither may erase
    the complete audit trail.
11. **Local-first is permanent.** Client portfolio data and routine processing
    remain client-controlled.
12. **PPAR complements the system of record.** It does not write corrections or
    become the official book of record.
13. **Reviewability matters more than report volume.** More rows, tabs, scores,
    and features do not necessarily create more value.
14. **Repeatability precedes breadth.** Earn a repeatable Axys/APX product before
    claiming other platforms.
15. **Claims follow evidence.** Marketing may not outrun behavior, validation,
    coverage, or reliance limits.
16. **Client learning should accumulate through reusable rules, mappings,
    validation, and presentation—not one-off forks.**
17. **Audit and Analytics retain separate product identities.**
18. **Safety guarantees are change-controlled contracts.**
19. **Design at the evidence horizon.** Deferred capabilities receive
    boundaries and validation gates, not speculative schemas or state machines.

## 4.2 Safety contracts

| ID | Guarantee |
|---|---|
| `SN-01` | No lost differences: every reportable source difference remains in the complete trail unless safe processing stops |
| `SN-02` | No double counting: one representation owns each explained economic effect |
| `SN-03` | Fully Explained arithmetic reconciles internally, at display precision, and in serialized output |
| `SN-04` | Beginning/end continuity anomalies remain visible at portfolio and security grain |
| `SN-05` | Source-backed causes and findings retain validated bidirectional lineage |
| `SN-06` | Unsafe currency or unit interpretation stops rather than being silently coerced |
| `SN-07` | Reversed, overlapping, ambiguous, or out-of-boundary periods cannot silently own explained performance |
| `SN-08` | Packaged demo scenario meaning is preserved against fixture drift |
| `SN-09` | Demo changes and carry-forward stories remain isolated and inspectable |
| `SN-10` | HTML, XLSX, CSV, and internal review semantics retain parity |
| `SN-11` | Identical inputs and configuration produce deterministic normalized output apart from declared volatility |
| `SN-12` | Unknown fields and incomplete required policy fail closed; suppression cannot replace classification |

No feature, presentation change, pilot accommodation, or roadmap item may
silently weaken these guarantees. The maintained 500x scale check remains part
of release-candidate work after major cross-cutting, reporting, audit, or
safety-net changes.

## 4.3 Failure classes

- **Internal logic error:** stop generation when arithmetic, lineage, ownership,
  parity, or determinism is inconsistent.
- **Source-contract error:** stop the affected workflow with actionable guidance
  when data or required policy cannot be interpreted safely.
- **Visible review finding:** generate the report and show suspicious but
  interpretable evidence without automatically counting it.
- **Demo maintenance error:** fail internal fixture maintenance without
  converting that problem into a client finding.

## 4.4 Authority and accountability

PPAR owns deterministic evaluation under supplied data and approved policy.
Clients own source facts, mappings, accounting meaning, methodology approval,
source corrections, human disposition, and official reported performance.

A human decision may accept a limitation or end further work, but it cannot
rewrite PPAR's analytical status, cause ownership, arithmetic, or immutable
evidence. A corrected result requires corrected data or approved configuration
and a new run.

---

# 5. Product Boundaries and Local-First Doctrine

## 5.1 What PPAR Audit is

PPAR Audit is intended to be:

- a Performance Change Investigation system;
- a Performance Data Quality system;
- a portfolio-performance quality-assurance layer;
- a purpose-built review-information product for investment operations and
  performance teams; and
- local-first, configurable, deterministic, and evidence-preserving software.

## 5.2 What PPAR Audit is not

PPAR Audit is not intended to become:

- a portfolio-accounting system or book of record;
- a full accounting-ledger reconstruction system;
- a universal performance engine;
- a general cash, position, trade, custodian, or accounting reconciliation
  platform;
- an automated source-system correction tool;
- a generic report builder, BI platform, or ticketing system;
- a PPAR-operated hosted portfolio-data processing service;
- an independent audit, assurance, GIPS verification, compliance, or
  certification provider;
- a black-box causal-inference or “AI-powered audit” product; or
- a broad investment-analytics product under the PPAR Audit identity.

## 5.3 Local-first boundary

Portfolio data, calculations, evidence, and routine operation must remain in a
client-controlled environment. This may be a client desktop, server, private
infrastructure, or internal scheduler; local-first does not mean one physical
workstation.

Licensing, updates, and support must not silently transmit portfolio data,
identifiers, results, or evidence. Any exceptional transfer of minimized,
redacted evidence requires explicit client authorization and a defined policy.

Cross-client portfolio-data aggregation and PPAR-hosted history are out of
scope.

## 5.4 Product design filter

Every proposed capability should materially strengthen at least one question:

1. Why did reported performance change?
2. Is the explanation complete?
3. What source-data appears suspicious?
4. What information should the reviewer examine next?
5. What recurring weakness does validated audit history reveal?

A feature that does not strengthen one of these questions should normally be
excluded, separated, or deferred.

---

# 6. Information and Presentation Strategy

## 6.1 Product focus

PPAR should provide as much valuable and accurate information as the available
evidence and supported methodology permit. Information design is part of the
product: structure, prioritization, terminology, limitations, navigation, and
format parity materially affect whether the result is useful and trusted.

PPAR should not maximize report volume. It should maximize decision usefulness
without hiding the complete evidence trail.

## 6.2 Information hierarchy

The intended review hierarchy is:

1. **Investigation scope and validity** — entities, periods, snapshots,
   intended use, readiness/limitations, and provenance.
2. **Performance result** — reported difference, explained amount, residual or
   withheld state, and Fully/Partly/Unexplained status.
3. **Supported causes** — only compatible, owned, traceable causes; never unsafe
   cross-period or cross-portfolio totals.
4. **Independent Data Issues attention** — affected rows/entities and rule
   coverage, kept separate from performance causation.
5. **Recommended next evidence** — deterministic, condition-based navigation
   and action cues.
6. **Complete evidence** — findings, source detail, lineage, configuration/
   contract context, and supporting diagnostics.

The Executive Summary is the first HTML section and first XLSX sheet while
preserving existing analytical surfaces. It derives from the same validated
tables, not a separate calculation.

## 6.3 Presentation rules

- Management and analyst views should use one shared truth with progressive
  disclosure.
- Fully/Partly/Unexplained, readiness, Data Issues, and any future human status
  remain separate dimensions.
- No composite health, confidence, quality, or “passed” score is approved.
- No explanation-completeness percentage is approved; retain analytical status
  and exact per-review-unit values.
- Data Issues should use affected-row/entity counts and evaluation coverage, not
  unvalidated incident counts.
- Narrative and action text should be deterministic and traceable.
- HTML, XLSX, and machine output must agree semantically.
- Bounded first views may prioritize, but they may not delete facts or become
  the only surviving evidence.
- Empty-state language must state what was and was not evaluated.

## 6.4 Workflow stance

Client workflows for reviewing, routing, approving, correcting, retaining, and
closing information will differ. PPAR should make evidence easy to use inside
those workflows without assuming it should own them.

Managed comments, assignments, approvals, closure, notification, and case
management are not current capabilities and are not an expected near-term
product layer. Detailed workflow design is deliberately deferred until distant-
future evidence establishes a common material problem that client systems and
clear PPAR evidence cannot handle adequately.

Validation may conclude that PPAR should provide no managed workflow.

---

# 7. Capability Map

| Capability | Product posture | Governing boundary |
|---|---|---|
| Performance Change Investigation | Current foundation and product center | Formula-bound, evidence-preserving, conservative, client validation required |
| Performance Data Quality Audit | Current selected checks; approved strengthening | Findings remain independent of counted performance causes |
| Audit Readiness | Current technical controls; APPROVED DIRECTION for unified experience | Ready/Qualified Ready/Blocked cannot waive safety; Diagnostic Only remains separate |
| Executive Summary worksheet | CURRENT — UNDER USABILITY REFINEMENT | First XLSX sheet and first HTML section; shared validated truth, no new financial calculation |
| Data Issues issue-type expansion | APPROVED DIRECTION and required MVP gap | Add only defensible, enumerated checks with evidence, YAML policy, and tests |
| Cause/issue codification and YAML control | APPROVED DIRECTION and required MVP gap | Stable vocabulary and fail-closed configuration; user policy cannot redefine product arithmetic |
| Axys/APX transaction semantics and demo coverage | FOUNDER-APPROVED and required MVP gap | Evidence-scoped scenarios, exact-case capability, source contracts, and staging controls must preserve posted-transaction boundaries |
| Broad Audit Rules Library | Long-term APPROVED DIRECTION | Defer a broad catalog; MVP work is limited to the codification and checks above |
| Repeated-restatement history | APPROVED DIRECTION but implementation DEFERRED | Requires recurring use, source-state identity, provenance, comparability, and local retention |
| Audit Health Dashboard and Operational Intelligence | Founder-approved conditional design; implementation DEFERRED | No superficial one-run dashboard, composite score, blame, or cross-client aggregation |
| Additional vendor support | CANDIDATE | Each platform needs a documented, tested, commercially supportable source contract |
| Human Review and Disposition | Distant evidence-gated possibility | Product calculation remains immutable; do not assume PPAR owns workflow |

## 7.1 Rules-library direction

The rules library is the strategic way for product learning to accumulate.
Each mature rule should eventually have a stable ID, rationale, applicable
scope, required inputs, transparent logic, evidence, disposition type,
configuration, false-positive conditions, validation examples, tests, and
status.

Do not begin by inventing a large catalog. Prioritize a small set based on
observed client value, frequency, materiality, defensibility, and cross-client
reuse. Do not invent checks that require reference data the source contract does
not provide.

## 7.2 Evidence-gated capabilities

History, dashboards, Operational Intelligence, additional vendor packs, and any
workflow layer remain behind evidence gates. Founder approval of their product
boundaries does not authorize implementation.

---

# 8. First-Client Validation and Claims

## 8.1 Validation doctrine

The first client is a validation partner, not merely a logo or revenue event.
The pilot should test the product as software rather than disguise bespoke
consulting.

A controlled pilot should likely include:

- one client-controlled environment;
- bounded portfolios, periods, and source files;
- an agreed source/extract contract;
- approved field mappings and transaction/methodology policy;
- known historical changes plus a fresh comparison;
- portfolio audit as the primary surface and selected security review where the
  case requires it;
- local installation, data retention, and evidence handoff;
- client-SME validation of labeled results and limitations;
- measured implementation and investigation effort; and
- explicit exclusions, reliance limits, and separate case-study permission.

## 8.2 Evidence to collect

The first 2–5 partners should establish:

- real Axys/APX export variability and reproducibility;
- actual mapping and transaction-policy effort;
- unsupported asset, transaction, currency, and methodology cases;
- false-positive and false-negative patterns for current checks;
- management and analyst comprehension of presentation and terminology;
- evidence-navigation success and wrong-path behavior;
- investigation time and repeatability compared with current practice;
- installation, update, security, retention, and support burden;
- whether readiness and executive-summary directions improve outcomes;
- which rules create reusable value; and
- whether history or workflow has any serious business case.

## 8.3 Claims supported before client validation

Use narrow language such as:

- compares two configured snapshots;
- identifies changed reported portfolio or security performance;
- explains supported causes under configured Modified Dietz treatment;
- identifies unresolved differences and review evidence;
- preserves source evidence and deterministic report artifacts;
- includes Axys/APX-oriented starter mappings and an accepted packaged-demo
  seed; and
- runs locally within the client-controlled environment.

## 8.4 Claims not supported

Do not claim:

- that PPAR explains every performance change or detects every error;
- guaranteed, certified, or officially correct performance;
- compatibility with every Axys/APX site or any unvalidated platform;
- production proof or quantified time savings without measurement;
- replacement of accounting, reconciliation, compliance, verification, or
  human judgment;
- independent audit, assurance, GIPS verification, or regulatory certification;
- that Fully Explained proves the source-data is correct;
- a validated confidence, health, quality, or completeness score;
- that dashboards, persistent history, Operational Intelligence, or managed
  workflow are current; or
- black-box AI causal inference.

Case-study and external claims require separate client permission and must match
measured evidence.

---

# 9. Product Roadmap and Evidence Gates

## 9.1 Roadmap doctrine

Validation comes before scale. Product specificity should stop at the evidence
horizon. A phase is not complete because every imaginable field is documented;
it is complete when current decisions, boundaries, evidence needs, and the next
safe gate are clear.

## 9.2 Stage 1 — Current foundation

**Status:** substantial internal foundation; commercial validation absent.

Current work has established the two-snapshot investigation engine, selected
Data Issues checks, evidence packs, safety invariants, deterministic outputs,
Axys/APX-oriented setup, and a packaged demonstration scope.

The product-design foundation through Phase 3F is founder-approved. Detailed
specifications are retained in the separate reference, but approval does not
mean each approved direction should be implemented.

## 9.3 Stage 2 — Validation partners

**Priority:** highest.

Exit evidence:

- at least one real client export can be mapped and run safely;
- client experts agree with labeled supported, unresolved, and Data Issues cases;
- implementation effort and source-contract gaps are measured;
- presentation supports correct management and analyst interpretation;
- safety and evidence behavior survive real-site variability; and
- the next product changes are supported by observed value rather than theory.

## 9.4 Stage 3 — Repeatable Axys/APX product

Begin only after pilot evidence.

Potential work includes:

- repeatable onboarding and approved starter contracts;
- better operator-facing readiness and remediation;
- validation-driven refinement of the current Executive Summary;
- prioritized, reusable high-value rules;
- accepted client regression cases;
- packaging, deployment, update, and support hardening; and
- measured claims and pilot-to-product conversion.

Exit evidence: a second and subsequent client can be implemented without a
client-specific code fork or founder-dependent interpretation.

## 9.5 Stage 4 — Recurring product use

Only after repeatability, evaluate whether recurring operation justifies local
history, repeated-change analysis, a health dashboard, or Operational
Intelligence.

The implementation gate requires retained comparable runs, stable provenance,
a management decision improved by history, and value sufficient to justify
storage and support complexity.

## 9.6 Distant possibilities

- additional vendor source contracts;
- broader asset/transaction rule packs;
- enterprise deployment and controlled integrations;
- cross-portfolio monitoring within one client-controlled environment; and
- a narrowly tailored human-disposition capability if distant-future evidence
  shows a common problem PPAR should own.

These are not near-term commitments.

## 9.7 Active phase — MVP completion

The visionary product-design phase is complete and **good enough for now**. The
active work is to define, implement, test, and demonstrate four bounded MVP
capabilities:

1. **Executive Summary:** retain the implemented concise first worksheet and
   equivalent first HTML section derived from existing validated review tables.
2. **Additional Data Issues issue types:** expand the current enumerated checks
   only where existing source-data supports transparent, high-signal logic and
   manageable false-positive controls.
3. **Cause/issue codification and YAML control:** create stable product
   vocabulary and fail-closed configuration for supported user choices without
   allowing YAML to redefine analytical truth or safety invariants.
4. **Axys/APX transaction semantics and demo coverage:** complete the
   founder-approved, evidence-scoped scenarios, exact-case capability, normalized
   source contracts, and staging controls defined by the active MVP plan without
   turning site- or workflow-specific evidence into universal semantics.

The active implementation plan is maintained in
[`mvp_plan.md`](mvp_plan.md).

History, dashboards, Operational Intelligence, additional platforms, managed
workflow, and a broad speculative rules catalog are not part of MVP completion.

## 9.8 Immediate priorities

1. Complete founder usability acceptance of the 30-second Executive Summary.
2. Review and select the first approved high-value additional Data Issues issue
   type, then implement it with fixtures, YAML policy,
   reviewer output, and false-positive tests.
3. Implement the current-capability Axys/APX scenarios defined by the active MVP
   plan.
4. Implement the exact-case, normalized source-contract, and staging-control
   work required by the active MVP plan.
5. Complete remaining approved issue types and re-run report parity,
   determinism, financial invariants, demo coverage, and
   the 500x release-candidate scale check after the cross-cutting MVP changes.
6. Only then decide whether the product is ready for validation-partner planning.

---

# 10. Decisions, Open Questions, and Governing References

## 10.1 Confirmed decisions

1. PPAR Audit is separate from Performance Analytics.
2. Changed-performance investigation is the commercial wedge.
3. Performance Quality Assurance is the long-term internal territory.
4. Validation precedes scale; target approximately 2–5 strong partners.
5. Local-first execution is permanent; PPAR-operated hosted portfolio-data
   processing is out of scope.
6. Evidence, determinism, conservation, and human review are central.
7. Axys/APX is the initial implementation focus, not a permanent brand limit.
8. PPAR does not rebuild the full ledger, replace accounting, write corrections,
   or provide independent assurance.
9. Unknown required meaning fails closed; ambiguity is not hidden.
10. Rules should accumulate reusable knowledge without client-specific forks.
11. Demo coverage and production support are different claims.
12. No unvalidated confidence, health, quality, or explanation-completeness
    score is approved.
13. Information value, accuracy, and presentation are core product concerns.
14. Client workflow varies; managed workflow is a distant possibility, not an
    assumed layer.
15. Product detail should stop at the evidence horizon.
16. Product identity/roadmap and detailed specifications have separate document
    lifecycles.
17. Visionary product design is good enough for now; MVP completion requires
    the Executive Summary, additional Data Issues issue types, stronger
    cause/issue codification with bounded YAML control, and the evidence-scoped
    Axys/APX transaction-semantics and demo work defined by the active MVP plan.
18. Workstream D is blocking for MVP completion but remains sequenced after the
    Executive Summary and first additional issue-type slices.
19. The Executive Summary uses the approved Section 5 boundary and a fixed,
    version-controlled initial limit of ten priority review units; YAML display
    control is deferred.

## 10.2 Material open questions

- Which firm and role will become the first strong validation partner and daily
  user?
- What exact source/extract and transaction-policy variation appears in real
  Axys/APX sites?
- Which information hierarchy, terminology, and actions work without coaching?
- Which current Data Issues checks create value, noise, or missing coverage?
- What pilot scope, commercial structure, responsibilities, and reliance limits
  best support product validation?
- What minimal non-portfolio metadata, if any, may leave the client environment
  for licensing or updates?
- Under what exceptional authorization may minimized/redacted evidence be used
  for support?
- Which small rules-library framework should be prioritized after client
  evidence?

Workflow schemas, assignments, comments, approval states, storage, and
notifications are intentionally **not** current open design questions. They are
distant discovery topics unless evidence changes the roadmap.

## 10.3 Authority hierarchy

When sources conflict, use this order:

1. Current executable behavior, tests, generated artifacts, and machine-
   readable contracts
2. Current safety-invariant catalog and maintainer contract
3. Current README, setup documentation, and architecture document
4. This document's current product scope, status, boundaries, and founder
   decisions
5. The active MVP plan's implementation sequence, detailed acceptance, and
   status
6. Explicit current-checkpoint notes in the deep design reference
7. Machine-readable transaction-semantics and extract contracts
8. Approved detailed product specifications
9. Historical journals, snapshots, and brainstorming material

Current implementation truth outranks a stale product description. Founder
product direction does not convert a future capability into current behavior.

## 10.4 Governing references

| Reference | Authority and use |
|---|---|
| [`product_specifications_index.md`](product_specifications_index.md) | Compact index to approved historical Phase 2/3 detail and its current owners |
| [`archive/PPAR_Audit_Foundational_Product_Design_v0.10.md`](archive/PPAR_Audit_Foundational_Product_Design_v0.10.md) | Preserved pre-split migration snapshot; historical, not maintained |
| [`mvp_plan.md`](mvp_plan.md) | Active implementation sequence, current-state evidence, detailed acceptance, and slice status within this constitution's MVP capability boundary |
| [`safety_invariants.md`](safety_invariants.md) | Current safety guarantees and failure classes |
| [`performance_comparison_design.md`](performance_comparison_design.md) | Deep current/historical implementation design reference |
| [`../roadmap.md`](../roadmap.md) | Portfolio-level priorities, shared-platform work, and document ownership |
| [`../archive/roadmap_through_v0.1.5.md`](../archive/roadmap_through_v0.1.5.md) | Frozen pre-restructure engineering journal; historical context only |
| [`../axys_apx/contracts/transaction_semantics_matrix.yaml`](../axys_apx/contracts/transaction_semantics_matrix.yaml) | Machine-readable transaction-semantics authority |
| [`../axys_apx/contracts/transaction_semantics_matrix.md`](../axys_apx/contracts/transaction_semantics_matrix.md) | Rendered transaction-semantics reference |
| [`demo_source_contract.md`](demo_source_contract.md) | Packaged-demo source boundary |
| [`site_extract_readiness_checklist.md`](site_extract_readiness_checklist.md) | Site-extract readiness guidance |

## 10.5 Document maintenance rule

Update this governing document only when product identity, current status,
principles, boundaries, roadmap, claims, or founder decisions change.

Update the specifications index only when the location, current interpretation,
or owner of approved historical detail changes. Record current product decisions
in this document or the active plan rather than expanding the index.

Update executable contracts, tests, and implementation documentation with the
code they govern. Link across documents instead of copying large sections.

Any change to the number or identity of MVP capabilities must update this
constitution and the active MVP plan in the same commit. The constitution owns
the capability boundary; the plan owns sequence, detailed acceptance, and
implementation status.

When a polished idea conflicts with product truth, product truth wins.
