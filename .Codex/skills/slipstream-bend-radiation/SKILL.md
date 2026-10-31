---
name: slipstream-bend-radiation
description: Implement or validate local pure-dipole synchrotron loss and ISR in fixed-reference Slipstream LinacTracker runs. Use for changes to this model or its quantitative validation.
---

Read the package AGENTS.md, [model](references/model.md) and
[validation](references/validation.md). Inspect the current implementation
before relying on recorded evidence. Keep native physics and portable
tests in this package; executable comparisons belong in an optional workbench.
Preserve flag-off behavior and canonical momentum conventions. Radiation
lengths must be positive, independent of magnetic Yoshida weights. Never
silently ignore enabled flags or double count ring-average radiation.
Report measured evidence and the actual device/dtype/reference contract.

For smooth AD optimisation of expected beam moments, use the separate
`analysis.beamline_radiation.linear_radiation_transport` model described
in [the model](references/model.md). Its gradients are not stochastic
particle-tracker parameter gradients.
