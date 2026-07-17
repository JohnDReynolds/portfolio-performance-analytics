# PPAR Portfolio Roadmap

This is the portfolio-level roadmap for PPAR. It keeps the two market-facing
products and their shared platform aligned without duplicating their detailed
plans, implementation contracts, or historical engineering notes.

PPAR has two products:

- **PPAR Audit** explains changed reported performance and surfaces independent
  Data Issues findings.
- **PPAR Analytics** produces attribution, contribution, cumulative-effect, and
  risk reporting.

Axys/APX is the initial shared integration and reference layer. It is not a
third PPAR product.

## Where Roadmap Decisions Live

| Area | Current authority | Scope |
| --- | --- | --- |
| PPAR Audit product direction | [Audit Product Constitution and Roadmap](audit/product_constitution.md) | Product identity, current truth, principles, boundaries, capability status, validation doctrine, and evidence-gated roadmap. |
| PPAR Audit active implementation | [Audit MVP Completion Plan](audit/mvp_plan.md) | Current MVP scope, implementation sequence, and acceptance gates. |
| PPAR Analytics | [Analytics Roadmap](analytics/roadmap.md) | Analytics product maintenance, evidence-gated improvements, and deferred cleanup. |
| Axys/APX integration knowledge | [Axys/APX Reference](axys_apx/README.md) | Vendor evidence, supported conclusions, explicit unknowns, and integration guidance. |
| Axys/APX evidence blockers | [Chapter 01: Overview](axys_apx/reference/Chapter_01_Overview.md#axys_apx-blockers) | Canonical evidence gaps that block broader automation or stronger vendor claims. |
| Shared platform | This document and [Architecture](architecture.md) | Cross-product onboarding, packaging, release, and architecture decisions. |
| Historical engineering work | [Roadmap through v0.1.5](archive/roadmap_through_v0.1.5.md) | Frozen implementation journal containing the former 98-phase cumulative roadmap. |

Detailed implementation truth remains in executable behavior, tests, generated
artifacts, safety invariants, and machine-readable contracts. A roadmap does not
override those sources.

## Current Portfolio Priorities

### 1. Complete the PPAR Audit MVP

The active product-development priority is the bounded Audit MVP described in
the [MVP plan](audit/mvp_plan.md). The
[Audit product constitution](audit/product_constitution.md) owns its four
founder-approved capability boundaries, including the MVP-blocking Axys/APX
transaction-semantics and demo work. The MVP plan owns their implementation
sequence, detailed acceptance, and current slice status rather than duplicating
that detail here.

This portfolio roadmap should not restate either document in detail.

### 2. Prepare for real-client validation

Release-candidate quality for the packaged Axys/APX demo is not evidence of
broad production readiness. After MVP completion, the next Audit gate is safe
validation against real client exports and approved local accounting policy.
The product should not claim universal Axys/APX compatibility before that work.

### 3. Maintain a clear Analytics product path

Keep the installed Analytics workflow, packaged Axys/APX starter data, README
story, generated images, and regression coverage aligned. Product improvements
and deferred maintainer-data cleanup belong in the
[Analytics roadmap](analytics/roadmap.md), not in Audit planning.

### 4. Strengthen integration evidence without overstating it

Use the Axys/APX reference chapters for supported conclusions and Unknowns. Use
the [transaction-semantics matrix](axys_apx/contracts/transaction_semantics_matrix.yaml)
for implemented transaction-policy boundaries. Expand vendor-specific behavior
only when evidence, source contracts, configuration, reporting, and tests agree.

Publishing an advanced Axys/APX integration reference remains a candidate. It
should be available to installed-package users when it is ready, but ordinary
setup should remain concise and should not copy the full research archive into
each starter workspace.

## Shared Platform Priorities

- Keep `ppar setup`, the root README, setup-installed documentation, and both
  product commands aligned around the same first-run path.
- Keep package contents, public commands, generated artifact names, and release
  records synchronized with executable contracts and tests.
- Keep inexpensive financial, conservation, lineage, and explanation-
  reconciliation invariants enabled in production.
- Run the 500x scale check after major cross-cutting, reporting, audit,
  safety-net, or performance changes.
- Validate the complete setup and report workflow on Windows before claiming
  Windows support.

## Candidate Cross-Product Work

These directions are not scheduled commitments:

- inspectable vendor YAML presets, beginning with the accepted Axys/APX seed;
- effective-dated security names and classifications shared by Analytics and
  Audit;
- additional vendor source contracts after Axys/APX implementation becomes
  repeatable; and
- optional report packaging when it improves real review handoffs without
  replacing the primary HTML, XLSX, CSV, and image artifacts.

## Maintenance Rule

Keep this document short. Update it only for portfolio-level priorities,
cross-product dependencies, shared-platform work, or a change in document
authority.

Put product-specific plans in the corresponding product directory. Put
Axys/APX facts and evidence gaps in the reference repository. Put executable
policy in machine-readable contracts. Preserve completed implementation trains
in the frozen journal rather than appending new phases to it.
