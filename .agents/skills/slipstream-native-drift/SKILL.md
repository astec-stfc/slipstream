---
name: slipstream-native-drift
description: Implement, extend or validate field-free native straight drift transport and its canonical-coordinate time of flight in Slipstream. Use when changing drift physics or its quantitative validation, not for routine tracking.
---

Locate the Slipstream package using pyproject.toml and src/slipstream/;
read applicable AGENTS.md and AGENTS.md instructions.

Read [the model contract](references/model.md) for verified equations,
units, validity limits and current integration points. Read
[validation](references/validation.md) for the independent quantitative
oracle, package paths and exact commands. Inspect current source before
assuming the recorded revision still describes the implementation.

Follow the requested scope. Preserve disabled-feature behavior, coordinate
conventions and CPU fallback. Keep native physics and portable tests in the
package. External-code comparisons belong to an optional parent workbench;
package work must remain possible without it.

Report actual numerical evidence, supported regime and unavailable checks.
Do not treat structural skill validation as proof of physics correctness.
