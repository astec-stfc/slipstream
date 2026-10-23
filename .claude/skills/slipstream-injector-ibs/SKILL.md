---
name: slipstream-injector-ibs
description: Implement, extend or validate local binary Coulomb collisions for Slipstream photocathode injector studies, including Takizuka-Abe/Nanbu scattering, physical macro weights and thermal-frame conventions. Use for injector IBS changes or validation, not storage-ring growth rates or routine beamline runs.
---

Locate the Slipstream package using pyproject.toml and src/slipstream/;
read applicable AGENTS.md and CLAUDE.md instructions.

Read [the model contract](references/model.md) for verified equations,
units, validity limits and current integration points. Read
[validation](references/validation.md) for the independent quantitative
oracle, package paths and exact commands. Inspect current source before
assuming the recorded revision still describes the implementation.

Follow the requested scope. Preserve disabled-feature behavior, coordinate
conventions and CPU fallback. Keep native physics and portable tests in the
package. External-code comparisons belong to an optional parent workbench;
package work must remain possible without it.

Use `InjectorIBSSettings` and `Injector.ibs`. Keep physical charge weighting,
local emitted/surviving masks, private RNG state and elastic pair conservation.
The rate model assumes weakly coupled classical particles and nonrelativistic
pair-frame motion; exact Lorentz kinematics does not make the collision rate
fully relativistic. Unsupported modes must fail explicitly. Reuse the module
documentation's derivations and quantitative oracles before changing a factor,
time/density boost, pairing rule or cutoff. Check whether the requested study
needs strongly coupled cathode physics that this operator does not provide.

Use `backend="auto"` for optional float64 CUDA Warp acceleration and
`backend="torch"` for the portable reference. Both share grouping and private
random draws; compare impulses, physical rate and conservation before changing
the pair kernel. Backend parity alone is not a physical oracle.

Report actual numerical evidence, supported regime and unavailable checks.
Do not treat structural skill validation as proof of physics correctness.
