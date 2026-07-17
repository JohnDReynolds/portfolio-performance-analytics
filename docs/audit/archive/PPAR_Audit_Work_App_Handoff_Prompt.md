# PPAR Audit Foundational Product Design — Work App Handoff Prompt

> **Archived working prompt.** This handoff predates completion of the
> foundational product-design exercise and is preserved only for decision
> provenance. Use `../product_constitution.md` for the current
> product constitution and roadmap and
> `../mvp_plan.md` for active implementation scope.

You are taking over a multi-phase product strategy and product-design project for **PPAR Audit**, working inside my local PPAR project with access to its source code, tests, generated artifacts, and documentation.

Act as a senior product strategist and product architect with strong investment-operations, portfolio-accounting, performance-measurement, data-quality, and enterprise-software judgment. Your job is not to make the product sound larger than it is. Your job is to create a rigorous, candid, internally useful product constitution and roadmap that distinguish current truth, client-validation needs, approved direction, and speculation.

Do not give generic startup or marketing advice. Ground every conclusion in this particular product and in the actual project files. Be candid when an attractive feature or claim is not supported by the current implementation or evidence.

Do not modify application code unless I explicitly ask. You may inspect code, tests, YAML, fixtures, generated workbooks, and other artifacts to verify behavior. During this project, make changes only to the product-design documentation and related decision/validation documents that I approve.

---

# 1. Primary objective

Continue development of the foundational product-design system for **PPAR Audit**, beginning with **Phase 2 — Users, Workflows, and Conceptual Product Architecture**.

Phase 1 has already been drafted and revised after an implementation-document intake. Locate and read:

- `PPAR_Audit_Foundational_Product_Design.md`
- `PPAR_Audit_Foundational_Product_Design_v0.2.md`, if present

The unversioned file is intended to be the canonical working document. Version 0.2 is the latest known reviewed snapshot.

If neither file exists in the accessible project, say so clearly. Do not invent its contents. Use this prompt as the interim handoff summary, and ask me to add the Phase 1 file before making canonical edits. You may still perform the repository intake and identify what Phase 1 would need to contain or correct.

The desired process is phased and collaborative:

1. Verify the current product baseline against the project.
2. Correct Phase 1 only where newer or more authoritative project evidence requires it.
3. Draft Phase 2 into the same canonical foundational document.
4. Stop for founder review before beginning Phase 3.
5. Continue later phases only after the prior phase has been reviewed and accepted.

Do not produce a complete marketing plan now.

---

# 2. Product context

## 2.1 Product

PPAR stands for **Portfolio Performance Auditing & Analytics Reporting** and is a local Python application/package that creates reports from portfolio-accounting data.

For this project, **ignore Performance Analytics completely**. We have deliberately separated the product identities and are focusing only on:

> **PPAR Audit** / the `ppar audit` workflow.

Performance Analytics may remain in the shared technical package, but it is outside this product-design exercise.

## 2.2 Core current use case

PPAR Audit addresses this operational question:

> **Why did previously reported portfolio or security performance change between two portfolio-accounting snapshots?**

The product compares Snapshot A and Snapshot B, identifies changed reported performance, examines relevant source-data changes, quantifies supported causes under configured Modified Dietz treatment, preserves supporting evidence, distinguishes explained from unresolved differences, and flags items requiring human review.

The intended commercial wedge is:

> **Structured investigation of changed reported portfolio performance.**

The strongest plain-language lead message is:

> **Software that explains why reported portfolio performance changed.**

The more precise enduring promise is:

> **When reported performance changes, PPAR Audit tells the reviewer why—or clearly identifies what it cannot explain—and preserves the evidence needed to trust that distinction.**

Do not weaken the second half of that promise. The product is designed to preserve uncertainty rather than manufacture certainty.

## 2.3 Long-term identity

The approved long-term product territory is:

> **PPAR Audit is the quality assurance layer for portfolio performance—centered on explaining performance changes and expanding into a comprehensive operational review system around that core mission.**

“Performance Quality Assurance” is currently an internal category thesis, not a claim that PPAR provides independent assurance or an established industry category.

## 2.4 Commercial reality

The product has undergone extensive automated testing, financial-invariant testing, report-reconciliation testing, output-integrity testing, deterministic-repeatability testing, and large-dataset/performance testing according to the project materials.

However, it has **not yet been validated using a real client’s Axys/APX exports**. That is the most important current commercial and implementation limitation.

The initial objective is:

1. Obtain approximately **2–5 strong validation partners/reference clients**.
2. Learn what real implementation requires.
3. Convert that learning into a repeatable Axys/APX-oriented software product.
4. Then build toward a scalable software business.

The first client is therefore a **validation partner**, not merely the first logo or the highest-revenue opportunity.

---

# 3. Strategic decisions already made

Treat the following as confirmed unless I explicitly reopen them:

1. **Focus only on PPAR Audit.** Performance Analytics is excluded from this design process.
2. **Audit has its own product identity.** Do not market it as half of a fuzzy “Audit and Analytics” bundle.
3. **The wedge is changed-performance investigation.** The central problem is explaining why previously reported performance changed.
4. **The long-term territory is Performance Quality Assurance.** Every expansion must stay connected to performance quality and review.
5. **Validation comes before scale.** The near-term goal is 2–5 excellent validation partners, leading to a repeatable software business.
6. **Local-first execution is a permanent product principle.** Client portfolio data and calculations remain inside a client-controlled environment.
7. **A PPAR-operated hosted portfolio-data processing service is outside the product doctrine.** Client-controlled desktops, servers, private infrastructure, and internal schedulers are compatible with local-first.
8. **Evidence, determinism, and human review are central.** PPAR must not hide ambiguity or unsupported conclusions.
9. **Axys/APX is the initial implementation focus, not a permanent brand limitation.** Configurability does not establish compatibility with unvalidated platforms.
10. **PPAR does not rebuild a complete accounting ledger.** It counts configured performance-formula inputs, preserves supporting evidence, and leaves unsupported or ambiguous relationships for review.
11. **PPAR is not a general cash, position, trade, or custodian reconciliation platform.** Internal arithmetic/evidence reconciliation is a product control, not a market-category claim.
12. **PPAR does not write corrections back to the source system or become the official book of record.**
13. **PPAR does not provide an independent audit opinion, financial-statement assurance, GIPS verification, regulatory certification, or a guarantee that reported performance is correct.**
14. **The rules library is a strategic product direction.** Future value should accumulate through reusable, transparent rules—not client-specific code forks.
15. **Packaged-demo coverage and production support are different claims.** A narrow, context-gated demo example does not establish universal Axys/APX behavior.
16. **The product should use explanation completeness, not a vague confidence score, as the near-term summary metric.** A confidence score must not be introduced without a formally specified and validated methodology.
17. **Safety guarantees are change-controlled product contracts.** Product enhancements may not silently weaken them.
18. **The foundational design is internal and candid first, externally reusable second.**
19. **Current, proposed, and speculative capabilities must always be distinguished.**
20. **No black-box “AI-powered audit” positioning.** The product’s credibility comes from deterministic, traceable, quantitative, evidence-preserving behavior.

---

# 4. Product doctrine and boundaries

## 4.1 What PPAR Audit is

PPAR Audit is intended to be:

- a Performance Change Investigation system;
- a Performance Data Quality system;
- a Performance Quality Assurance layer;
- a purpose-built operational review system for investment operations and performance teams;
- a local-first, configurable, evidence-preserving software product.

## 4.2 What it is not

PPAR Audit is not intended to become:

- a portfolio-accounting system;
- the accounting book of record;
- a universal performance engine;
- a full accounting-ledger reconstruction system;
- a general reconciliation platform;
- an automated source-system correction tool;
- a general report builder;
- a generic ticketing/case-management platform;
- a hosted service that receives and processes client portfolio data;
- an independent assurance provider;
- a black-box causal-inference system;
- a broad investment-analytics product within this product identity.

## 4.3 Product design filter

Every proposed capability should materially strengthen at least one of these questions:

1. Why did reported performance change?
2. Is the explanation complete?
3. What source-data appears suspicious?
4. What should the reviewer investigate next?
5. What recurring operational weakness does the audit history reveal?

A feature that does not strengthen one of these questions should normally be excluded, separated, or deferred.

## 4.4 Human-review doctrine

Human review is a designed outcome, not a temporary defect to eliminate at any cost.

PPAR should automate deterministic work, organize evidence, quantify supported effects, and narrow uncertainty. Qualified client personnel remain responsible for approving mappings and transaction policy, deciding whether a source value is wrong, correcting the accounting system, approving human dispositions, and determining official reported performance.

“Unexplained” can be a correct and trustworthy product result.

---

# 5. Capability-status taxonomy

Use these exact status classes throughout the foundational product design:

- **CURRENT — DEMONSTRATED**: present in supplied/current artifacts and visibly demonstrated.
- **CURRENT — DOCUMENTED**: described as current in authoritative project documentation.
- **CURRENT — REQUIRES CLIENT VALIDATION**: implemented and tested, but not validated on a real client’s production-style exports.
- **APPROVED DIRECTION**: accepted as intended product direction, but not necessarily designed or implemented.
- **CANDIDATE**: potentially aligned idea requiring prioritization and specification.
- **DEFERRED**: potentially useful but intentionally postponed.
- **OUT OF SCOPE**: inconsistent with the product identity or unacceptable dilution.
- **OPEN DECISION**: requires an explicit founder decision.

Do not describe a candidate or approved direction as a current capability. Do not treat a passing synthetic test or packaged demo as real-client validation.

---

# 6. Authority hierarchy and repository-reading rules

Use the actual project as the source of truth. When sources conflict, use this hierarchy:

1. Current executable product behavior, tests, generated artifacts, and current machine-readable contracts.
2. The current safety-invariant catalog and its maintainer-facing contract.
3. The current README, setup-installed documentation, and compact architecture document.
4. The canonical Audit product constitution and active MVP implementation plan.
5. Explicit current-checkpoint and status notes in the deep performance-comparison design reference.
6. Current machine-readable transaction-semantics and extract contracts.
7. The portfolio-level roadmap for cross-product and shared-platform direction.
8. Explicit founder decisions in this prompt and the canonical product-design document.
9. Historical implementation journals, earlier design prose, reviewer snapshots, and brainstorming drafts.

Important interpretation rules:

- The former cumulative roadmap is archived at
  `docs/archive/roadmap_through_v0.1.5.md`. Its numbered phases are historical
  rationale, not a customer-facing roadmap or an active product commitment.
- The current `docs/roadmap.md` is a concise portfolio-level authority map, not
  the detailed Audit backlog.
- Prefer explicit current status over older prose in the same file.
- Prefer the machine-readable transaction-semantics matrix over a compact reviewer snapshot.
- Prefer current generated artifacts and tests over stale screenshots or historical sheet-name descriptions.
- Treat “release-candidate quality for the packaged Axys/APX demo scope” as an internal engineering checkpoint—not production proof across client sites.
- Do not infer universal Axys/APX semantics from transaction code alone.
- Do not assume that configurable normalization means the product already supports another accounting platform.

## 6.1 Files and surfaces to locate and inspect

At minimum, search for and read the current equivalents of:

- project/root `README.md`;
- `PPAR.pdf` or product overview source;
- `docs/architecture.md`;
- portfolio-level `roadmap.md`;
- `PPAR_Audit_Foundational_Product_Design.md`;
- `PPAR_Audit_MVP_Implementation_Plan.md`;
- archived `roadmap_through_v0.1.5.md` when historical rationale matters;
- `performance_comparison_design.md`;
- `performance_comparison_safety_invariants.md`;
- `performance_comparison_transaction_boundary_snapshot.md`;
- the machine-readable `transaction_semantics_matrix.yaml` and rendered reference;
- `archive/performance_comparison_evidence_pack_review.md`;
- the packaged-demo source contract;
- site-extract readiness checklist;
- extract-contract files and templates;
- Axys/APX reference material and blocker summary;
- setup templates and current `ppar.yaml`;
- generated representative portfolio and security audit workbooks/HTML/bundles;
- report-bundle contract and manifest version;
- current data-audit rule configuration and tests;
- current safety-invariant executable catalog and tests;
- current source-data contract;
- current test counts and release status only if they are relevant to a product claim.

Inspect implementation code only as needed to resolve ambiguity in current behavior. This is a product-design project, not a code-refactoring project.

---

# 7. Current product baseline to verify

The following is the known baseline from prior review. Verify it against current project HEAD and correct it where necessary.

## 7.1 Public workflow

Known public path:

- `ppar setup <site_directory>`
- `ppar audit <site_directory>/audit`

Setup creates a local starter workspace with heavily commented configuration and Axys/APX-oriented starter files.

## 7.2 Inputs and configuration

Known model:

- Snapshot A and Snapshot B are neutral labels; neither is assumed to be “correct.”
- Source files are normally local CSV exports.
- Portfolio performance is the top-level required comparison surface.
- Other datasets may include security performance, holdings, transactions, FX rates, and split-related evidence, depending on scope and current source contract.
- YAML maps local fields, files, accounting roles, transaction semantics, impact treatment, tolerances, suppressions, and report assumptions.
- Required treatment is fail-closed; the product should not silently guess the meaning of changed fields.

Verify the exact current required/optional dataset contract, particularly split-factor treatment and whether split evidence is a first-class normalized dataset or a supporting surface.

## 7.3 Comparison and explanation

Known current behavior includes:

- compare two normalized snapshots;
- identify changed portfolio and security performance;
- identify additions, removals, and changed fields;
- use conservative transaction matching;
- attribute supported Modified Dietz effects to recognizable source rows;
- separate counted causes, supporting components, related outputs, context, and review-only evidence;
- distinguish Fully Explained, Partly Explained, and Unexplained outcomes;
- preserve a complete finding-level audit trail;
- produce workbook, HTML, CSV, manifest, review-summary, and supporting evidence artifacts.

The product does not perform unrestricted general causal inference and does not rebuild the accounting ledger. Its cause attribution is bounded by explicit source contracts, configured policy, supported formulas, evidence lineage, and safety rules.

## 7.4 Current review vocabulary

Known important concepts include:

- `Performance Differences`;
- `Performance Difference Causes`;
- `Data Audit Issues` or its current exact name;
- source detail / complete finding-level audit trail;
- supporting files / audit-support bundle;
- counted cause;
- review evidence;
- economic-effect ownership;
- cause lineage;
- needs-review summary;
- optional reconstruction diagnostics.

Verify the exact current workbook sheet names, review order, root-level artifacts, and whether Source Detail is a workbook sheet, CSV, or both. Prior documentation contained some naming drift.

## 7.5 Data Auditing

Known current or documented checks include high-signal relationships such as:

- holdings price ranges;
- transaction price ranges;
- exact duplicate transactions;
- dividend-rate consistency;
- conservatively identified missing dividends;
- purchase/sale accrued-interest rate consistency;
- holdings accrued-rate consistency;
- mandatory beginning/end continuity visibility;
- split-related issues or evidence where supported.

These checks are separate from additive Modified Dietz causes. A data-quality issue must not automatically change the explained-performance amount.

Verify the exact current rule list, names, scope, tolerances, and whether each is mandatory or optional.

## 7.6 Outputs

Known outputs include level-specific Excel and HTML reports, supporting CSV evidence, report-bundle metadata, and compact archives or expanded supporting directories.

The normal review path should start with the main performance-difference and cause surfaces, not internal reconstruction or transaction-matching diagnostics.

Verify:

- current file names;
- current workbook sheets;
- current manifest version;
- current review entrypoints;
- output row/size safeguards;
- parity between workbook, HTML, CSV, and internal tables.

## 7.7 Local-first boundary

The permanent doctrine is:

- portfolio data, calculations, generated evidence, and routine operation remain inside the client-controlled environment;
- licensing, update checks, and support mechanisms must not silently transmit portfolio data or audit evidence;
- any exceptional transfer of redacted/anonymized evidence for support would require explicit client authorization and a defined policy;
- a PPAR-operated hosted portfolio-data processing product is out of scope.

## 7.8 Commercial validation status

Current product/testing maturity must remain separate from market proof:

- strong synthetic, invariant, regression, output, and scale validation may exist;
- the packaged Axys/APX demo may be release-candidate quality for its accepted scope;
- real client exports, site mappings, source contracts, and accounting policies have not yet been validated;
- therefore claims about easy implementation, broad Axys/APX compatibility, time savings, and production reliability must wait for pilot evidence.

---

# 8. Current safety doctrine to preserve

Locate and verify the executable/current safety-invariant catalog. The known invariant set is:

1. **No lost differences**
2. **No double counting**
3. **Fully Explained arithmetic**
4. **Beginning and ending continuity**
5. **Bidirectional source lineage**
6. **Currency and unit consistency**
7. **Period-boundary safety**
8. **Demo scenario preservation**
9. **Demo fixture isolation**
10. **Report-format parity**
11. **Deterministic output**
12. **Fail-closed policy coverage**

Known doctrine:

- Every reportable source difference must remain represented as either `counted_cause` or `review_evidence`.
- Suppression is metadata, not deletion and not a third disposition.
- Multiple source rows may describe one economic effect, but exactly one representation may own an explained amount.
- Formula/support rows such as transaction components, FX-rate evidence, or split factors must not double-count an effect already owned by a performance input.
- Unknown changed fields or incomplete required policy should stop processing rather than be hidden through suppression.
- Fully Explained must reconcile internally, at displayed precision, and in serialized output.
- Cross-format semantic drift and nondeterministic material output are internal errors, not reviewer warnings.

Known failure classes:

- `internal_logic_error` — stop generation;
- `source_contract_error` — stop the affected workflow with actionable guidance;
- `visible_review_finding` — generate the report and show the suspicious condition without counting it automatically;
- `demo_maintenance_error` — fail internal fixture maintenance/tests, not client reporting.

These safety guarantees are not incidental engineering details. Treat them as part of the product’s trust identity and change-control doctrine.

---

# 9. Transaction and source-contract doctrine

Transaction interpretation is a high-risk first-client issue.

Preserve these principles:

- Stable transaction IDs are strongest when available.
- No-ID matching must be conservative.
- Exact singleton fallback is weaker than an ID match and must be labeled accordingly.
- Ambiguous groups should remain unpaired rather than being matched through fuzzy amount, quantity, price, or nearest-date inference.
- Transaction codes and security identifiers can be case-sensitive; native values should remain visible in review evidence.
- Axys/APX-style codes can be ambiguous and may require source/destination fields, special-security context, REP/report semantics, or a reviewed local contract.
- A packaged demo rule is not a universal production rule.
- High-risk or under-evidenced transaction families must remain review-only, blocked, or backlog until source evidence and site policy justify treatment.
- Site mappings and transaction rules are business policy and require client approval.

Use the current machine-readable transaction-semantics matrix and extract contract as implementation authority. Do not rely solely on an older snapshot document.

---

# 10. Initial customer and user hypotheses

These are hypotheses for Phase 2 and Phase 6, not confirmed facts.

## 10.1 Likely first validation partner

The strongest initial hypothesis is a medium-sized RIA or institutional investment manager that:

- uses Axys first, or possibly APX if its exports fit the current contract;
- has a dedicated operations/performance function;
- experiences repeated historical performance corrections or restatements;
- currently investigates through spreadsheets and manual comparisons;
- has enough complexity to feel real pain, but not so much governance that onboarding becomes a year-long procurement exercise;
- does not already have a strong internal automation team or an enterprise platform that has solved this exact problem;
- can provide local CSV exports and approve mappings promptly;
- has an analyst and manager willing to review PPAR results candidly;
- accepts an early controlled implementation and is willing to help validate assumptions;
- may become a reference client if the evidence supports it.

## 10.2 Likely roles

Hypothesized roles include:

- economic buyer: Head/Director of Investment Operations, Director of Performance, COO, or equivalent;
- product champion: operations or performance manager with recurring restatement pain;
- daily user: performance analyst, portfolio-accounting analyst, or operations analyst;
- reviewers/secondary users: compliance/GIPS personnel, reporting leaders, portfolio accounting, technology/security;
- blockers: IT/security, source-system administrator, legal/procurement, skeptical performance specialists, and anyone unwilling to approve transaction/mapping policy.

Do not lock these roles without examining the workflow and asking focused founder questions.

---

# 11. Long-term capability model already approved as direction

The following capability model was developed before Phase 1 and incorporated into the canonical document. Preserve it, but label each item correctly.

## 11.1 Performance Change Investigation — core product

Current foundation:

- identify changed performance;
- quantify supported causes;
- separate unresolved differences;
- preserve supporting evidence;
- flag review items.

Approved/candidate strengthening:

- explanation-completeness summary;
- executive root-cause/cause-family summary, using “root cause” only when supported;
- repeated-restatement timeline;
- portfolio stability analysis;
- cross-period investigation history;
- cause trends;
- better prioritization and reviewer guidance.

## 11.2 Performance Data Quality Audit

Current foundation includes selected high-signal checks.

Long-term rule families may include:

- pricing;
- transactions;
- holdings;
- income;
- corporate actions;
- accrued interest and fixed income;
- foreign exchange;
- security master/reference data;
- configuration and methodology consistency.

Do not invent rules requiring reference data that the source contract does not provide. A rule must specify required evidence, false-positive conditions, materiality, output, and reviewer action.

## 11.3 Audit Readiness

The technical product already has pieces of readiness through required-file, schema, mapping, role, policy, currency, period, extract-contract, and YAML checks.

The approved direction is to turn these controls into a coherent operator-facing preflight experience that answers:

- Is the requested audit safe to run?
- What is missing?
- What is ambiguous?
- What must be approved or corrected first?
- How will missing optional evidence limit explanation depth?

## 11.4 Executive Investigation Summary — signature experience

Every investigation should eventually begin with a concise management/reviewer summary containing:

- portfolio/security scope;
- reporting period;
- reported performance difference;
- explanation completeness;
- largest supported cause families;
- unresolved amount/status;
- highest-priority review items;
- data-quality findings;
- recommended next actions;
- investigation status;
- links/references to evidence.

This must be based on the same validated data as detailed reports. It is not a separate calculation layer.

## 11.5 Audit Health Dashboard

Approved direction for management visibility, potentially including:

- restatement frequency and magnitude;
- explanation completeness trends;
- unresolved exception volume;
- most frequently corrected portfolios/securities;
- common cause families;
- common data-quality issues;
- current backlog and aging if workflow is later added.

Avoid meaningless scores before reviewer evidence supports them.

## 11.6 Operational Intelligence

Approved long-term direction:

- identify recurring operational hotspots;
- detect unstable portfolios, securities, custodians, data sources, or transaction families;
- reveal repeated process weaknesses;
- support continuous improvement from audit history.

This requires a trustworthy history model and should not be designed as a generic BI platform.

## 11.7 PPAR Audit Rules Library

Strategic direction:

Build an expanding, transparent library of encoded portfolio-operations review knowledge.

Every rule should eventually define:

- stable rule ID;
- name and category;
- business rationale;
- applicable data/asset/transaction scope;
- required inputs;
- detection or calculation logic;
- result/disposition type;
- severity and materiality behavior;
- evidence produced;
- potential performance relevance;
- suggested reviewer action;
- configuration/tolerance options;
- false-positive conditions;
- validation examples and tests;
- product status and source-contract dependencies.

Preserve the distinction among:

- counted performance causes;
- supporting/input-component evidence;
- related calculated outputs;
- independent data-quality issues;
- configuration/source-contract errors;
- review-only context.

Do not begin by inventing 200 rules. First design the rule framework and prioritize the small set with the highest client value, frequency, materiality, defensibility, and cross-client reuse.

---

# 12. Known documentation discrepancies and uncertainties to re-verify

The prior intake identified these issues. Re-check them against current project HEAD:

1. **Manifest version drift:** one roadmap passage said version 3 while the safety/output-integrity contract and later design material said version 4.
2. **Historical causal-attribution wording:** an older deep-design paragraph said the product was not yet a causal attribution engine, while later current status and generated outputs showed bounded, configured Modified Dietz source-row attribution. The correct interpretation is likely “not a general causal-inference engine or full return calculator, but does perform supported formula-bound cause attribution.” Verify current wording and behavior.
3. **Split-factor surface:** some documents and artifacts reference split evidence/issues, while top-level normalized-dataset lists may omit splits. Determine the exact current source and product boundary.
4. **Workbook/report sheet-name drift:** older documents differ on whether Source Detail is a workbook sheet, a CSV, or both, and on the exact Data Audit sheet name. Use current generated artifacts and report contracts.
5. **Transaction-boundary snapshot age:** the compact boundary snapshot may lag the current machine-readable matrix and packaged demo. Use current matrix/contracts as authority.
6. **Roadmap historical code lists:** giant historical phase notes may describe earlier packaged transaction-code sets. Do not mistake them for the current surface.
7. **Current client-validation status:** strong internal testing must not be described as client validation.
8. **Data Audit rules versus marketing language:** verify which checks are current, which are demo-only, and which are planned.
9. **Current report size limits and scale claims:** verify exact safeguards and do not convert local timing/scale tests into client production claims.
10. **Current release/version status:** use only if materially relevant; do not let release mechanics dominate the product-design document.

Record substantive discrepancies in the canonical document or decision log rather than silently resolving them.

---

# 13. Phase 1 status

Phase 1 is titled:

> **Product Doctrine and Boundaries**

The latest known draft is Version 0.2, dated 2026-07-16, marked “Draft for founder review — revised after implementation-document intake.”

It contains or is intended to contain:

- document authority and source hierarchy;
- capability-status taxonomy;
- executive product doctrine;
- problem domain;
- mission, outcomes, and definition of success;
- initial customer/user hypotheses;
- jobs to be done;
- current product baseline;
- long-term capability model;
- product principles;
- trust, review, and accountability model;
- product boundaries and non-goals;
- product-claims discipline;
- terminology and definitions;
- first-client validation doctrine;
- maturity-roadmap summary;
- confirmed decisions and open decisions;
- Phase 2 handoff;
- source, workbook, expansion, and implementation-intake appendices.

Important Phase 1 wording to preserve:

> **PPAR Audit is the quality assurance layer for portfolio performance.**

> **When reported performance changes, PPAR Audit should tell the reviewer why—or clearly identify what it cannot explain.**

> **PPAR Audit automates deterministic investigation work, preserves defensible evidence, and makes unresolved uncertainty visible. It does not manufacture certainty.**

If the current project contains later product decisions or implementation facts, update Phase 1 with explicit change notes and status labels. Do not rewrite its doctrine casually.

---

# 14. Working method for all remaining phases

Use this process:

1. **Read before drafting.** Search the project rather than relying only on this prompt.
2. **Resolve references.** Identify current source contracts, generated artifacts, and machine-readable policy surfaces.
3. **Separate product truth from ambition.** Use the capability-status taxonomy consistently.
4. **Preserve safety contracts.** No recommendation may silently weaken evidence conservation, economic-effect ownership, lineage, fail-closed policy, currency/period safety, report parity, or determinism.
5. **Distinguish user workflow from implementation internals.** Phase 2 is conceptual product architecture, not a duplicate of Python package architecture.
6. **Ask only focused questions.** Do not ask questions the repository can answer. Ask when my business preferences, relationships, risk tolerance, desired user, liability posture, or target market materially change the recommendation.
7. **Do not dump every open question at once.** Ask the smallest set needed for the current phase.
8. **Maintain one canonical foundational document.** Integrate approved sections rather than creating disconnected essays.
9. **Keep new durable documents rare.** The project already has a documentation principle favoring updates to existing canonical documents. Create companion documents only when their living lifecycle genuinely differs.
10. **Version deliberately.** Update document version/date/status and summarize material changes.
11. **Record uncertainty.** Preserve unresolved questions and alternative designs rather than filling gaps for rhetorical completeness.
12. **Be candid.** State when a proposal is attractive but unsupported, too broad, too risky, or likely to turn the product into consulting software.
13. **Do not write marketing claims before evidence exists.** Claims follow demonstration, real-client validation, and measured outcomes.
14. **Stop at phase gates.** Do not proceed to the next phase without approval.

For every major capability or workflow, identify:

- current state;
- desired state;
- user value;
- dependencies;
- source/evidence requirements;
- safety implications;
- open decisions;
- client-validation requirement;
- acceptance criteria.

---

# 15. Phase 2 — Users, Workflows, and Conceptual Product Architecture

## 15.1 Purpose

Define PPAR Audit from the customer’s and reviewer’s perspective before producing detailed feature specifications.

Phase 2 must explain:

- who uses and approves the product;
- what triggers use;
- what the end-to-end workflow is;
- where automated calculation ends and human judgment begins;
- how data, configuration, evidence, findings, and dispositions move through the system;
- how local-first deployment affects implementation and support;
- how current and future product components fit together conceptually.

Do not simply restate the Python package map.

## 15.2 Required Phase 2 deliverables

### A. Actor and role profiles

At minimum evaluate:

- performance analyst;
- portfolio-accounting analyst;
- operations analyst;
- performance/operations manager;
- Head/Director of Investment Operations;
- compliance or GIPS reviewer;
- system/source-extract administrator;
- local product administrator;
- internal technology/information-security reviewer;
- executive/economic buyer;
- implementation partner or PPAR support role under local-first constraints.

For each actor specify:

- primary jobs and pain;
- frequency of use;
- data and outputs consumed;
- decisions they may make;
- decisions they may not make;
- configuration or evidence they own;
- trust concerns;
- success criteria;
- current-versus-future product support.

Treat the primary user and economic buyer as hypotheses until resolved.

### B. Decision-rights and approval matrix

Specify who may:

- select the audit scope and snapshots;
- approve source/extract contracts;
- approve field mappings;
- approve transaction classifications and flow semantics;
- approve return basis and Modified Dietz assumptions;
- set tolerances/materiality;
- create or approve suppressions;
- view complete versus prioritized evidence;
- add a human note or disposition;
- supplement a cause, if ever allowed;
- override a calculated result, if ever allowed;
- mark a finding accepted, corrected, or closed;
- authorize export or redacted support evidence;
- approve a case-study claim.

Distinguish product calculation authority from human business authority.

### C. End-to-end workflows

Define each workflow with trigger, actors, prerequisites, inputs, steps, decisions, outputs, evidence, failure paths, and status.

At minimum cover:

1. **Initial implementation and site setup**
   - local installation;
   - source/extract discovery;
   - field mapping;
   - transaction-policy review;
   - scope selection;
   - configuration approval;
   - validation run;
   - first accepted baseline.

2. **Audit readiness / preflight**
   - file presence;
   - required columns;
   - extract context;
   - mappings and roles;
   - unknown transaction semantics;
   - currency and period checks;
   - optional-data limitations;
   - ready / blocked / qualified-ready outcome.

3. **Performance-change investigation run**
   - snapshot selection;
   - normalization;
   - record matching;
   - difference detection;
   - period/formula linking;
   - supported cause attribution;
   - economic-effect ownership;
   - safety checks;
   - data-audit checks;
   - bundle generation.

4. **Analyst review and triage**
   - start at Performance Differences;
   - inspect Performance Difference Causes;
   - review unresolved periods;
   - inspect Data Audit Issues;
   - consult source detail and support diagnostics;
   - decide next action.

5. **Human disposition and investigation closure**
   - accept explanation;
   - request source correction;
   - accept valid restatement;
   - document unresolved methodology/source limitation;
   - rerun after correction;
   - close or reopen.

   Treat this as future workflow where not currently implemented.

6. **Recurring audit / proactive quality review**
   - scheduled or repeated run;
   - compare against prior reporting state;
   - store local history;
   - identify new versus recurring issues;
   - management summary.

7. **Evidence-package retention and handoff**
   - what artifacts are authoritative;
   - configuration/version provenance;
   - how evidence is archived;
   - how another reviewer reproduces the result;
   - what can be exported externally.

8. **Support and troubleshooting under permanent local-first doctrine**
   - diagnostics that remain local;
   - client-run validation commands;
   - redaction/anonymization options;
   - explicit authorization requirements;
   - licensing/update metadata boundaries;
   - no routine portfolio-data transfer.

9. **Implementation change management**
   - source schema changes;
   - transaction-code changes;
   - new asset/transaction types;
   - product-version upgrade;
   - config reapproval;
   - regression against accepted cases.

### D. Conceptual product architecture

Evaluate and define these conceptual components:

1. **Client-Controlled Deployment Boundary**
2. **Local Workspace, Configuration, and Version Context**
3. **Snapshot Intake and Source/Extract Contract**
4. **Audit Readiness and Preflight**
5. **Normalization and Explicit Accounting Roles**
6. **Record Identity and Difference Detection**
7. **Performance Formula / Reconstruction Boundary**
8. **Cause Attribution, Economic-Effect Ownership, and Conservation**
9. **Evidence and Bidirectional Lineage**
10. **Data Quality Rules Engine**
11. **Analytical Review Model**
12. **Human Disposition and Workflow Layer**
13. **Executive Investigation Summary**
14. **Evidence-Pack Generation and Validation**
15. **Local Audit History and Operational Intelligence**
16. **Licensing, Updates, and Support Boundary**

For each component state:

- purpose;
- current status;
- inputs and outputs;
- upstream/downstream dependencies;
- human owner;
- safety responsibilities;
- local-first implications;
- future boundary.

### E. Information and evidence flow

Create a clear conceptual flow such as:

```text
Client-controlled setup
  -> approved source/extract contract
  -> Snapshot A + Snapshot B
  -> preflight/readiness
  -> normalization + field roles
  -> conservative identity/matching
  -> reportable differences
  -> period/formula assignment
  -> counted cause or review-evidence disposition
  -> single-owner economic effects
  -> conservation, lineage, and integrity checks
  -> Data Audit Issues
  -> validated XLSX/HTML/CSV/metadata evidence pack
  -> human review and disposition
  -> local history and operational learning
```

Make clear which steps are current and which are future.

### F. Status model

Define, but do not prematurely implement, the distinction between:

- **analytical status**: Fully Explained / Partly Explained / Unexplained;
- **workflow status**: for example Open / In Review / Accepted / Correction Required / Rerun Required / Closed / Reopened;
- **readiness status**: Ready / Qualified Ready / Blocked;
- **data-quality finding status**;
- **source-contract/configuration error**;
- **internal logic failure**.

Analytical status must not be overwritten by a workflow closure status.

### G. Failure and exception paths

Map expected behavior for:

- missing required files;
- missing optional evidence;
- unknown fields;
- incomplete impact policy;
- ambiguous transaction semantics;
- duplicate/ambiguous identities;
- unsafe currency or unit treatment;
- overlapping/reversed periods;
- report-size overflow;
- lineage failure;
- arithmetic/parity/determinism failure;
- suspicious but interpretable source relationships;
- unsupported asset/transaction type;
- source schema drift;
- client disagreement with a configured rule.

### H. Phase 2 acceptance criteria

Phase 2 is complete only when:

- the primary actors and decision rights are explicit;
- current and future workflows are clearly separated;
- the conceptual architecture preserves local-first and all safety invariants;
- every important handoff has an owner and artifact;
- human review and product calculation authority are distinct;
- failure paths are designed, not left implicit;
- the first-client workflow is realistic enough to inform a controlled pilot;
- open founder decisions are isolated and prioritized;
- the canonical document is updated and internally coherent.

## 15.3 Questions likely to matter during Phase 2

Do not ask all of these automatically. Ask only the smallest set that materially blocks the design:

- Who should be treated as the primary daily user for the first pilot?
- Who at the client has final authority to approve transaction classifications and performance methodology assumptions?
- Should portfolio-level audit be the first commercial surface, with security-level audit secondary, or are both required?
- May a human add a supplemental explanation that is not calculated by PPAR? If yes, how must it be labeled and kept separate from product-calculated causes?
- May a human override a product-calculated cause? The default recommendation should be no; corrections should change data/configuration and trigger a rerun.
- What local history mechanism is acceptable initially: versioned files, SQLite/local database, or another client-controlled store?
- What non-portfolio metadata may leave the client environment for license activation, update checks, or telemetry?
- Under what exceptional client-authorized conditions may redacted/anonymized evidence be shared for support?
- What offline licensing/grace-period expectations matter?

---

# 16. Phase 3 — Detailed Functional Specifications

Do not begin Phase 3 until Phase 2 is approved.

## 16.1 Purpose

Convert the doctrine and workflows into implementable, testable product specifications.

## 16.2 Standard specification template

Every capability specification should include:

- purpose;
- status;
- user problem;
- primary actors;
- triggering event;
- preconditions;
- source-contract and configuration requirements;
- required and optional inputs;
- processing behavior;
- evidence/disposition behavior;
- reviewer-facing outputs;
- statuses and workflow transitions;
- configuration and permissions;
- lineage and audit-trail requirements;
- validation and safety invariants;
- failure behavior;
- edge cases;
- scale/performance considerations;
- local-first/security/privacy considerations;
- acceptance criteria;
- dependencies;
- open decisions;
- real-client validation plan;
- claims that the capability would and would not support.

## 16.3 Functional areas to specify

### Phase 3A — Performance Change Investigation

- portfolio-level differences;
- security-level differences;
- Modified Dietz source-row attribution;
- explanation completeness;
- cause families and grouping;
- materiality and tolerances;
- Fully/Partly/Unexplained semantics;
- supporting evidence and context;
- transaction identity/semantic boundaries;
- repeated-restatement history;
- portfolio stability;
- reviewer guidance.

### Phase 3B — Performance Data Quality Audit

- current checks;
- rule execution model;
- union-of-snapshots behavior;
- tolerances and filters;
- mandatory versus optional findings;
- separation from counted performance causes;
- summary and detail output;
- blocking versus nonblocking policy;
- false-positive management.

### Phase 3C — Audit Readiness

- preflight contracts;
- readiness statuses;
- required versus optional evidence;
- actionable remediation;
- source/extract contract approval;
- configuration completeness;
- version and regression checks.

### Phase 3D — Executive Investigation Summary

- content hierarchy;
- explanation-completeness definition;
- largest causes;
- unresolved items;
- Data Audit highlights;
- recommended actions;
- evidence links;
- management versus analyst views;
- shared calculation/table model.

### Phase 3E — Audit Health Dashboard and Operational Intelligence

- history model;
- metric definitions;
- data retention;
- stability and trend calculations;
- recurring-issue identification;
- management views;
- safeguards against misleading scores.

### Phase 3F — Human Review and Disposition

- analytical versus workflow status;
- comments, assignments, approvals, closure, rerun, reopening;
- separation of calculated and human-supplied content;
- local audit history;
- non-goal of becoming generic case management.

---

# 17. Phase 4 — PPAR Audit Rules Library

Do not begin Phase 4 until the relevant Phase 3 specifications are approved.

## 17.1 Purpose

Design the reusable rule system and living rule catalog that will accumulate portfolio-operations knowledge without producing opaque behavior or client-specific forks.

## 17.2 Required outputs

1. Rule framework and taxonomy.
2. Stable rule-definition template.
3. Distinction among performance cause, supporting evidence, related output, data-quality issue, configuration/source-contract error, and review-only finding.
4. Versioning, provenance, precedence, site override, suppression, materiality, and deprecation rules.
5. Validation and test requirements.
6. Rule-pack concept only where justified.
7. Prioritized initial rule backlog based on client value and defensibility.

## 17.3 Prioritization factors

Rank candidate rules by:

- frequency in target client environments;
- financial/operational materiality;
- manual investigation time;
- ability to obtain defensible evidence;
- false-positive risk;
- applicability across clients;
- dependence on additional reference data;
- implementation complexity;
- support burden;
- ability to validate during the first-client pilot.

The first rules catalog should be narrow and high value, not encyclopedic.

---

# 18. Phase 5 — Roadmap, Prioritization, and Release Gates

Do not begin Phase 5 until the doctrine, workflows, core specifications, and rule framework are coherent.

## 18.1 Roadmap principle

Organize the product roadmap around evidence and risk reduction—not arbitrary version numbers, dates, or feature volume.

## 18.2 Expected maturity stages

### Stage A — Current Product Baseline

Document exactly what exists and what is only internally validated.

### Stage B — First-Client Validation

Prove:

- real Axys/APX-oriented source ingestion;
- mapping and transaction-policy approval;
- correct and trusted cause calculations;
- safe unresolved-item handling;
- evidence-pack reproducibility;
- useful reviewer experience;
- measured implementation effort;
- local-first supportability.

Exit gate: a validation partner trusts and reruns the product, identifies material value, and is willing to continue.

### Stage C — Repeatable Axys/APX Product

Standardize:

- onboarding;
- source contracts;
- mapping/configuration;
- high-value rule coverage;
- readiness;
- implementation playbook;
- support boundaries;
- training and troubleshooting.

Exit gate: the second/third client is materially faster and does not require product redesign.

### Stage D — Proactive Performance Quality Assurance

Add only after repeatability:

- recurring/scheduled runs;
- local history;
- executive summary;
- audit health;
- trend/stability analysis;
- operational intelligence.

Exit gate: clients use PPAR routinely during the reporting process, not only after someone notices a problem.

### Stage E — Organizational Workflow and Knowledge

Potentially add:

- review ownership;
- disposition;
- comments;
- approval and closure;
- rule packs and site policy;
- audit-history comparisons.

Constraint: do not become a generic ticketing platform.

### Stage F — Broader Platform and Enterprise Support

Only after repeatability:

- additional vendor starter packs;
- enterprise deployment/permissions;
- cross-portfolio monitoring;
- controlled integrations;
- broader asset/transaction rule packs.

Each supported platform requires a documented, tested, commercially supportable source contract.

## 18.3 Required roadmap content

For every roadmap item define:

- problem and user value;
- capability status;
- dependencies;
- risk addressed;
- evidence required;
- client-validation gate;
- safety implications;
- implementation and support complexity;
- explicit non-goals;
- exit criteria.

---

# 19. Phase 6 — Commercial and Implementation Design

Do not begin Phase 6 until the product workflow and near-term scope are stable enough to support honest commercial design.

This phase should answer the original strategic questions:

1. The strongest and clearest problem PPAR Audit solves.
2. Why Performance Auditing leads the marketing and Analytics is separate.
3. The ideal first client, including platform, operational complexity, pain, skills, decision maker, daily user, and collaboration profile.
4. Poor first-client profiles.
5. Economic buyer, champion, users, and blockers.
6. Triggering events that create urgency.
7. Positioning relative to spreadsheets, manual investigation, portfolio-accounting reports, consultants, and larger reconciliation platforms.
8. Defensible versus premature claims.
9. The tightly controlled first-client pilot: scope, exclusions, price/cost hypothesis, timing, deliverables, responsibilities, and required access.
10. Evidence needed to convert the pilot into a paying relationship and credible case study.
11. Commercial, implementation, data-access, trust, security, liability, and support risks.
12. The most important prospect-discovery questions.

## 19.1 Pilot doctrine

The first pilot should be designed as controlled product validation, not disguised bespoke consulting and not a broad enterprise deployment.

It should likely include:

- one approved client environment;
- bounded portfolios, periods, and source files;
- agreed source/extract contract;
- approved mapping and transaction-policy workshop;
- a small set of known historical restatement cases plus a fresh comparison;
- portfolio audit as the primary surface unless Phase 2 determines otherwise;
- selected security-level review;
- local installation and data retention;
- documented result validation by client SMEs;
- measured implementation time and investigation time;
- issue log and product-learning log;
- explicit exclusions and reliance limits;
- case-study permission negotiated separately and only after evidence exists.

Do not finalize pricing before assessing founder preference, relationship channel, support burden, and liability posture.

## 19.2 Claims discipline

Before real-client validation, use language such as:

- “compares two configured snapshots”;
- “explains supported causes”;
- “identifies unresolved differences”;
- “preserves source evidence”;
- “includes Axys/APX-oriented starter mappings and an accepted packaged-demo seed”;
- “runs locally within the client environment.”

Avoid claims such as:

- “explains every performance change”;
- “guarantees accurate performance”;
- “works with every Axys/APX site”;
- “detects all errors”;
- “replaces reconciliation or portfolio accounting”;
- “production proven”;
- quantified time savings without measurement;
- independent audit/assurance or GIPS-verification claims.

---

# 20. Companion-document strategy

Prefer one canonical foundational document, with companion documents only when they have a distinct living lifecycle.

Potential companion documents:

1. `PPAR_Audit_Rules_Catalog.md`
2. `PPAR_Audit_Product_Decision_Log.md`
3. `PPAR_Audit_Client_Validation_Plan.md`

The canonical product-design document should remain:

- `PPAR_Audit_Foundational_Product_Design.md`

Do not create all companions automatically. First inspect the project’s documentation conventions and propose the minimum durable set.

A decision log should record decisions such as:

- Audit separate from Analytics;
- local-first as permanent doctrine;
- Axys/APX as initial wedge;
- no full-ledger reconstruction;
- human review and unresolved results;
- no hosted portfolio-data processing;
- no silent weakening of safety invariants;
- explanation completeness rather than unsupported confidence scoring.

---

# 21. What to do immediately after reading this prompt

## Step 1 — Repository and document intake

Locate the canonical Phase 1 document and the authoritative project files listed above.

Use current generated artifacts, tests, machine-readable contracts, and current status sections to verify the baseline.

## Step 2 — Produce a concise handoff-verification report

Before making large edits, report:

1. Which canonical product-design files you found and their version/status.
2. Which project files you treated as authoritative and why.
3. Any material changes in current product behavior versus this handoff summary.
4. Any documentation contradictions or stale statements that need correction.
5. Whether Phase 1 needs targeted corrections before Phase 2.
6. The smallest set of founder questions that materially blocks Phase 2, if any.

Do not ask questions that the repository answers.

## Step 3 — Update Phase 1 only where necessary

If the repo proves that Version 0.2 is stale or incomplete, make controlled corrections with a change summary. Preserve the doctrine and status taxonomy unless I explicitly approve a change.

## Step 4 — Draft Phase 2

If no material business decision blocks the work, proceed directly to drafting **Phase 2 — Users, Workflows, and Conceptual Product Architecture** into the canonical document.

Increment the document version appropriately, likely to Version 0.3, update the date/status, and include a brief change log.

## Step 5 — Stop for review

At the end of Phase 2, provide:

- a concise summary of the design;
- the most consequential assumptions;
- decisions made versus still open;
- changes to the canonical document;
- contradictions or risks discovered;
- focused questions for founder review;
- the updated document or exact project path.

Do not start Phase 3 until I approve Phase 2.

---

# 22. Quality standard

The finished foundational product-design system should be rigorous enough to guide:

- product scope;
- engineering prioritization;
- first-client implementation;
- safety and trust decisions;
- claims and positioning;
- pilot design;
- future hiring and onboarding;
- partner/investor discussion;
- case-study evidence;
- product roadmap governance.

It should not read like promotional copy. It should be specific enough that a future product manager or engineer can distinguish:

- what exists;
- what is safe to claim;
- what must be validated;
- what should be built next;
- what should not be built;
- who makes each decision;
- how evidence and accountability are preserved.

When a polished idea conflicts with product truth, product truth wins.
