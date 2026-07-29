# PPAR Audit Validation Partner Strategy

| Document field | Value |
| --- | --- |
| Assessment date | July 26, 2026 |
| Status | Advisory partner and channel strategy; not product or implementation authority |

## Executive Summary

PPAR Audit should be evaluated as a partner-operated product rather than as
software sold and supported directly to many individual Axys/APX firms.

The preferred model is an **authorized PPAR service partner** relationship:

- Empaxis, CSSI Solutions, Quartare, or another qualified partner implements
  PPAR Audit for its clients.
- The partner collects and maps client extracts, configures the YAML, reviews
  findings, communicates with the client, and delivers the resulting service.
- PPAR maintains the calculation engine, safety invariants, Axys/APX defaults,
  tested releases, and partner-facing technical documentation.
- Routine client questions remain with the partner. Only reproducible product
  defects and core-engine issues escalate to PPAR.

This model does not eliminate support. It concentrates support into a small
number of knowledgeable partners instead of requiring PPAR to support many
individual clients.

PPAR should not initially describe these organizations as ordinary software
resellers. They would be implementation and service operators selling a
partner-branded service powered by PPAR.

## Strategic Fit

PPAR Audit asks why previously reported portfolio performance changed. It
compares two portfolio-accounting snapshots, connects supported differences to
holdings, transactions, prices, FX, and related evidence, identifies Data
Issues, and retains unexplained residuals for human review.

The highly configurable YAML is potentially burdensome for an ordinary user but
valuable to an experienced Axys/APX implementer. A qualified partner should
already understand how to determine:

- local transaction meanings;
- file and column mappings;
- portfolio-accounting and performance policies;
- local performance conventions;
- client-specific suppressions and tolerances;
- which findings are operationally meaningful; and
- how to present conclusions without overstating assurance.

The partner also brings a customer relationship, operational staff, domain
credibility, and data-handling procedures that PPAR does not intend to build for
many individual clients.

The resulting business model is channel-first:

```text
Axys/APX client
    |
    v
Authorized PPAR service partner
    - obtains and maps approved client extracts
    - configures client YAML
    - operates PPAR Audit
    - reviews and delivers findings
    - provides client support
    |
    | reproducible product defects and core-engine questions only
    v
PPAR
    - maintains financial and safety invariants
    - maintains the calculation engine
    - maintains versioned Axys/APX defaults
    - produces tested releases and partner documentation
```

## Candidate Assessment

| Candidate | Best role | Assessment |
| --- | --- | --- |
| Empaxis | Lead managed-service and commercial delivery partner | Strongest initial channel fit |
| CSSI Solutions | Implementation and managed-service partner | Also very strong; broad technical and operational capability |
| Quartare / AdventGuru | Technical validator and boutique delivery partner | Exceptional Axys/APX review capability; delivery scale should be assessed |

### Empaxis

Empaxis appears to be the strongest first managed-service candidate. Its
published Axys/APX practice includes reconciliation, performance reporting,
middle- and back-office operations, process automation, and quality control.
Empaxis also describes a quality-assurance system that identifies reconciliation
breaks and reporting omissions or inaccuracies.

PPAR could help Empaxis investigate performance changes and serve additional
clients without increasing investigative labor proportionally. Its existing
outsourced operating model is closely aligned with a recurring partner-delivered
PPAR service.

Source:
[Empaxis Axys/APX outsourcing](https://www.empaxis.com/blog/advent-axys-apx-outsourcing-services)

### CSSI Solutions

CSSI is another strong implementation and service-channel candidate. Its
published services combine Axys/APX consulting, reconciliation, reporting,
operational cleanup, data migration, custom interfaces and extracts, custom
development, back-office services, and cloud hosting.

This breadth could expose PPAR to realistic histories, local practices, unusual
transactions, imperfect data, and operational review requirements. CSSI's
technical capability could also support deeper integration and repeatable
implementation. Because CSSI develops adjacent solutions, commercial and
intellectual-property boundaries should be explicit.

Source: [CSSI Solutions](https://cssisolutions.com/)

### Quartare / AdventGuru

Quartare appears particularly valuable as a technical and methodology
validator. Kevin Shea publicly describes decades of Advent experience involving
performance reports, reconciliation tools, data extracts, automation,
conversions, and SaaS capabilities for Axys/APX users.

Quartare may be the strongest candidate to challenge PPAR's assumptions,
extract authenticity, transaction interpretation, terminology, and edge-case
coverage. Its capacity to operate a multi-client service channel may be smaller
than Empaxis or CSSI and should be evaluated directly rather than assumed.

AdventGuru is a professional identity and contact channel associated with Kevin
Shea, not a separate validation relationship.

Source: [AdventGuru](https://adventguru.com/about/)

## Recommended Partner Model

The preferred relationship is an **Authorized PPAR Service Partner**, not an
unrestricted reseller or white-label licensee.

An authorized partner agreement should permit the partner to:

- install and operate PPAR for specifically authorized clients;
- assign authorized employees to operate and support the service;
- create and maintain client-specific configurations;
- process client-approved extracts in an authorized environment;
- deliver PPAR reports, evidence, and partner analysis as a paid service; and
- brand its professional service while retaining appropriate PPAR attribution.

The agreement should not permit the partner to:

- give PPAR or its source code to clients or other organizations;
- redistribute or sublicense PPAR generally;
- white-label PPAR as software developed or owned by the partner;
- use PPAR to create a competing or substitute product;
- share access with unapproved affiliates, contractors, or clients; or
- claim that PPAR provides an audit opinion, GIPS verification, attestation,
  certification, or other professional assurance.

The current public evaluation license prohibits hosted, service-bureau,
multi-user, and commercial use. A separate partner agreement would need to
grant the precise service-delivery rights required by the authorized partner
model.

## Commercial Structure

A practical commercial structure would combine:

- an annual partner or platform fee;
- an annual fee for each active end client;
- a minimum annual commitment after validation; and
- discounted or waived fees for a tightly controlled initial validation pilot.

PPAR should avoid perpetual partner licenses. Pricing should not depend on
auditing the partner's consulting hours or attempting to calculate a percentage
of each consulting engagement.

The agreement should be non-exclusive. PPAR should retain ownership of the
product, generally applicable improvements, safety behavior, and core
calculation logic. The partner should retain its client relationship,
confidential client information, and independently developed service materials.

## Support and Responsibility Boundary

### Partner Responsibilities

The partner should own:

- client qualification and contracting;
- client onboarding;
- client data authorization and security;
- extract generation and source review;
- YAML mapping and client-specific policy choices;
- normal operation and scheduling;
- first- and second-level support;
- investigation and interpretation of findings;
- communication and presentation to the client; and
- retention and deletion practices for client data and reports.

### PPAR Responsibilities

PPAR should own:

- reproducible product defects;
- core calculation correctness;
- financial and safety invariants;
- versioned Axys/APX defaults and configuration contracts;
- release testing and compatibility management;
- partner-facing implementation and technical documentation;
- documented escalation procedures; and
- third-level support for issues that cannot be resolved through documented
  partner procedures.

PPAR should not be copied on routine client support requests. A partner
escalation should include a reproducible, minimized support package that omits
client-confidential data unless the client has expressly authorized its use.

## Product Requirements for a Partner Channel

A partner-operated model still requires the product to reduce avoidable
configuration and support work. Important capabilities include:

- a versioned Axys/APX Audit configuration profile based on validation evidence;
- a clear separation between universal safety invariants, reusable Axys/APX
  defaults, required client decisions, and optional advanced settings;
- strict preflight validation with actionable error messages;
- an inspectable effective configuration retained with each run;
- stable input and output contracts;
- version and migration guidance for configuration changes;
- partner onboarding and readiness checklists;
- deterministic demonstrations and accepted regression scenarios;
- a minimized diagnostic bundle for partner escalation; and
- a documented responsibility matrix distinguishing product defects from
  client-specific configuration and interpretation.

The goal is not to hide client decisions. It is to make the configuration
surface appropriate for an expert implementer and prevent each partner from
having to rediscover PPAR's stable Axys/APX behavior.

## Phased Validation and Channel Approach

### Phase 1: Technical Validation

Engage Quartare or another strong Axys/APX subject-matter expert to review:

- source and extract authenticity;
- Axys/APX terminology and common conventions;
- transaction and security interpretation;
- performance-history and restatement behavior;
- YAML usability for an experienced implementer; and
- representative difficult cases.

This phase is a technical reality check, not a certification or endorsement.

### Phase 2: First Managed-Service Pilot

Select Empaxis or CSSI as the lead operating partner and run PPAR against one
authorized real-client environment. The partner should perform the source
mapping and YAML configuration while PPAR observes where the product, contracts,
documentation, or diagnostics require improvement.

Measure:

- implementation hours and assistance required;
- configuration questions;
- source-contract gaps;
- false positives and missed issues;
- explained and unexplained differences;
- reviewer usefulness;
- partner confidence in the evidence; and
- issues that reached PPAR rather than being resolved by the partner.

### Phase 3: Partner Independence Test

The lead partner should implement a second client without routine direct
participation from PPAR.

This is the critical scalability gate. If the second implementation still
requires extensive founder interpretation, PPAR is not yet ready for a partner
channel. The observed support should be used to distinguish missing product
capability from documentation gaps and legitimate client-specific decisions.

### Phase 4: Second Service Partner

After the lead partner has completed a repeat implementation, add a second
service provider. This determines whether PPAR is genuinely partner-operable or
merely dependent on one firm's institutional knowledge and custom practices.

No initial partner should receive exclusivity.

### Phase 5: Commercial Partner Release

Launch a formal partner edition only after the independence test succeeds. The
release should include:

- stable and versioned releases;
- partner installation and implementation documentation;
- the validated configuration profile;
- preflight and diagnostic tooling;
- commercial partner licensing;
- explicit support and escalation boundaries; and
- an agreed release and compatibility policy.

## Material Risks and Mitigations

### Support Is Concentrated, Not Eliminated

Partners will still require product support. The model succeeds only when
partners can resolve client-specific mapping, policy, operation, and
interpretation questions themselves.

Mitigation: enforce the responsibility boundary and use the second-client
independence test as a commercial gate.

### Partner Incentives May Conflict

Automation can reduce billable investigative labor. A partner may therefore
adopt PPAR only if it improves service margin, capacity, consistency, or client
retention.

Mitigation: position PPAR as a way to scale a recurring managed service, improve
quality control, and support reviewer-ready delivery rather than simply reduce
hours.

### Product Feedback May Become Indirect

PPAR may lose direct visibility into how end users understand and act on
reports.

Mitigation: require structured partner feedback, anonymized implementation
metrics, periodic product reviews, and permission to observe selected early
review sessions when the client agrees.

### Partner Dependence

A single partner could become the only source of clients, implementation
knowledge, and product feedback.

Mitigation: use non-exclusive agreements and add a second partner only after
the first operating model is repeatable.

### Adjacent Development Capability

The strongest partners already build automation, reconciliation, reporting, and
data tools. That makes them valuable validators but also capable of developing
adjacent or competing solutions.

Mitigation: use clear intellectual-property boundaries, retain PPAR attribution,
restrict competing derivative products, keep future differentiated engine logic
outside public repositories, and evaluate compiled delivery for the commercial
engine.

### Product Claims

Partners may be tempted to market PPAR as an audit, certification, or assurance
service beyond the product's demonstrated scope.

Mitigation: require approved product descriptions and preserve PPAR's current
boundary: it detects, compares, explains, and helps investigate supported
performance and source-data differences, but does not itself provide an audit
opinion, GIPS verification, attestation, certification, or assurance.

## Success Criteria

The partner model is ready for broader commercial use when:

- a qualified partner can configure and operate PPAR for a second client
  without routine founder involvement;
- the partner resolves normal source, YAML, policy, and interpretation questions;
- escalations are limited to reproducible product defects and core-engine
  questions;
- client data remains in an approved client- or partner-controlled environment;
- generated evidence is useful to both the partner and client reviewer;
- the configuration can be reproduced from its retained effective form;
- the partner is willing to deliver PPAR as a recurring service; and
- a second partner can adopt the product without inheriting the first partner's
  undocumented knowledge.

## Recommendation

PPAR should pursue a channel-first strategy built around a small number of
expert service partners rather than direct sales and support for many Axys/APX
firms.

The recommended sequence is:

1. use Quartare as a technical and domain reality check;
2. use Empaxis or CSSI as the first managed-service pilot partner;
3. require a second-client independence test;
4. add a second non-exclusive operating partner; and
5. launch a formal partner edition only after the support boundary has been
   demonstrated in practice.

This structure allows PPAR to remain a focused product operation while qualified
partners own client implementation, configuration, interpretation, and service
delivery.
