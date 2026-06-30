# Site Extract Readiness Checklist

Use this checklist before running performance comparison against a site-specific
Axys/APX extract. The goal is to confirm the source files expose enough evidence
for Modified Dietz formula roles without turning ppar into a local accounting
system.

## Required Review

- Confirm the comparison YAML points to the intended snapshot directories and
  transaction file.
- Run `validate_config` before generating a report bundle.
- Keep `extract_contract.enforce_ambiguous_axys_flows: true` unless a local
  reviewer has approved a documented opt-out.
- Confirm every transaction code in performance-relevant data has either a
  recognized source category, a recognized transaction code, or explicit
  `transaction_rules`.

## IMEX With Context Fields

Use an IMEX-context contract when the transaction extract includes source fields
that conditional `transaction_rules` can inspect. For ambiguous Axys-style
`li`, `lo`, `dp`, and `wd` rows, useful context usually includes:

- transaction security type;
- source/destination type;
- source/destination symbol;
- special-security type;
- special-security symbol.

Having the columns is only the first gate. The row values still need reviewed
conditional rules before ppar treats an ambiguous code as an external flow,
transfer, fee/expense, or performance transaction.

## REP/Report Semantic Fallback

Use a REP/report or custom-report source when IMEX cannot expose enough row
context. The fallback source should carry reviewed semantics directly, such as:

- normalized transaction category;
- cash-flow sign;
- performance-flow sign;
- return-basis treatment for fees or expenses when relevant.

Do not weaken the `li`/`lo`/`dp`/`wd` guardrail only because IMEX is sparse.
Prefer a richer local extract that proves the Modified Dietz formula role.

## Code-Only Failure Mode

A code-only extract is expected to fail when it contains ambiguous Axys flow
codes and lacks the blocking context columns named in the extract contract. This
is a useful source-quality signal. Fix the extract, add a REP/report fallback,
or document a reviewed local opt-out before treating the rows as performance
inputs.

## Reviewed Local Opt-Out

The local opt-out exists for reviewed site workflows that already know how to
interpret code-only ambiguous rows outside ppar's extract contract. It should be
rare and explicit:

```yaml
extract_contract:
  enforce_ambiguous_axys_flows: false
```

When this setting is false, conditional context-column guards do not protect
`li`, `lo`, `dp`, and `wd` rows. The site's YAML rules and review process carry
the classification risk.

## Handoff Evidence

Generated report bundles include `manifest.json`. When the bundle writer knows
the comparison YAML path, the manifest records:

- comparison YAML path;
- extract-contract path;
- ambiguous-flow enforcement status;
- required transaction context columns;
- observed transaction codes;
- transaction semantics-source counts.

Use that manifest with the README and CSV artifacts when handing a review bundle
to another reviewer.
