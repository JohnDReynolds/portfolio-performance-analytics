# Project Coding Guidelines

Apply these conventions when modifying or creating code in this project.

## Style And Quality

- Follow PEP 8 unless an established project-specific convention intentionally differs.
- Limit lines to 99 characters.
- Keep code free of `pylint` and `pyright` errors. Evaluate warnings case by case.
- Prefer small, behavior-preserving changes unless a broader refactor has clear value.

## Typing And Naming

- Annotate public parameters, public return values, class attributes, and non-obvious
  local variables where annotations improve readability or type checking.
- Avoid unnecessary annotations for obvious local variables.
- Prefix module-level identifiers with `_` when they are intended only for use within
  that module.
- Do not underscore public APIs or intentionally imported package-internal names.

## Public APIs

- Use idiomatic Python conventions at public boundaries.
- Normalize compatibility sentinels and legacy conventions at public boundaries.
- Preserve public behavior unless an API change is explicitly requested.

## Comments And Financial Logic

- Comment non-obvious intent, business rules, financial interpretation, assumptions,
  sign conventions, and important edge cases.
- Avoid comments that simply paraphrase straightforward code.
- Favor explicit names and intermediate variables when they improve financial
  interpretability or auditability.

## Docstrings

Use consistently formatted Google-style docstrings for all public APIs and meaningful
internal classes and functions. Type annotations do not replace behavioral
documentation.

- Modules should include a concise summary and useful context where appropriate.
- Classes should document their purpose and meaningful public instance state using
  `Attributes:`.
- Nontrivial functions and methods should use applicable `Args:`, `Returns:`, and
  `Raises:` sections.
- Constructors should document their arguments either in the class docstring or in
  `__init__`, following a consistent project-wide approach.
- Use `Yields:` instead of `Returns:` for generators.
- Use `Examples:` when a public entry point is not obvious.
- Use `Notes:` for significant formulas, data-shape expectations, assumptions, or
  validation behavior.
- Use `Warnings:` only for genuine misuse risks or significant side effects.
- Use `References:` for financial methodologies or external specifications when
  useful.
- Use `See Also:` only when it meaningfully improves navigation.
- Trivial private helpers may retain concise one-line docstrings.
- When modifying an existing public API, bring its docstring into compliance as part
  of the same change.
