# PPAR Documentation

PPAR Audit is the current market-facing product. PPAR Analytics remains a
maintained additional module with its own product-ready documentation and
roadmap, but it is outside the current Audit validation program and default
onboarding workflow.

| Area | Location | Purpose |
| --- | --- | --- |
| PPAR Audit | [`audit/README.md`](audit/README.md) | Audit product doctrine, roadmap, active implementation, technical contracts, operator guidance, and historical references. |
| PPAR Analytics | [`analytics/README.md`](analytics/README.md) | Maintained Analytics product page, direction, demonstration, and refresh guidance. |
| Axys/APX integration | [`axys_apx/`](axys_apx/) | Shared Axys/APX research, evidence, contracts, reference material, and common-core export guidance. |
| Shared platform | This directory | Architecture, repository orientation, and repeatable maintenance guidance. |
| Documentation assets | [`images/`](images/) | Generated images and product-overview maintenance guidance. |

Start with:

- [`architecture.md`](architecture.md) for the shared system boundary.
- [`maintainer_guide.md`](maintainer_guide.md) for repeatable maintenance,
  validation, demo, and release workflows.
- [`audit/README.md`](audit/README.md) for the Audit documentation map and
  authority order.
- [`audit/roadmap.md`](audit/roadmap.md) for Audit product stages, evidence
  gates, and priorities.
- [`analytics/README.md`](analytics/README.md) for the maintained Analytics
  product page and documentation entry point.
- [`analytics/roadmap.md`](analytics/roadmap.md) for Analytics product direction.
- [`analytics/analytics_demo_refresh.md`](analytics/analytics_demo_refresh.md)
  for Analytics demo and README-asset maintenance.

New product-specific documentation should go directly into the corresponding
product directory. Shared architecture and integration material should remain outside
the product directories. Historical working notes belong under the relevant
product's `archive/` directory. Historical notes spanning multiple products
belong under [`archive/`](archive/).

## Documentation Ownership Rules

1. Keep the root [`README.md`](../README.md) focused on PPAR Audit and its
   validation-client onboarding path.
2. Keep shared system structure and boundaries in
   [`architecture.md`](architecture.md).
3. Keep repeatable repository operations in
   [`maintainer_guide.md`](maintainer_guide.md); keep detailed packaged-demo
   instructions beside the packaged demo.
4. Keep Audit doctrine and founder decisions in its product constitution,
   forward-looking direction in [`audit/roadmap.md`](audit/roadmap.md), and
   active implementation detail in its MVP plan.
5. Keep the self-contained Analytics product presentation in
   [`analytics/README.md`](analytics/README.md) and its direction in
   [`analytics/roadmap.md`](analytics/roadmap.md).
6. Keep Axys/APX facts, evidence, and implementation contracts under
   [`axys_apx/`](axys_apx/) according to that area's documented file roles.
7. Keep deep implementation rationale in its owning product directory and
   checkpoint or session notes in the relevant `archive/` directory.
8. Keep the former cumulative roadmap frozen in
   [`archive/roadmap_through_v0.1.5.md`](archive/roadmap_through_v0.1.5.md);
   do not append new phase trains.
9. Add a durable document only when it has a distinct audience, authority, and
   maintenance lifecycle that an existing document cannot serve clearly.
