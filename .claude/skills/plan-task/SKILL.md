---
name: plan-task
description: Work one Nearpoint roadmap task from docs/PLAN.md end to end (restate acceptance, test, implement, measure, check off, changelog, commit). Use when starting, resuming, or finishing any PLAN.md task, or when asked "what's next".
---

# Work one PLAN.md task

1. **Pick.** Open `docs/PLAN.md`; take the first unchecked task unless the
   user named one. Read its phase's **Acceptance** line too.
2. **Restate** to the user, before writing code: the task id, what "done"
   means, and the phase acceptance numbers it contributes to.
3. **Test first or alongside.** Unit tests live next to the code. SIMD
   kernels get a proptest against their scalar reference. Storage code
   gets a corruption/truncation test.
4. **Implement** inside the crate `AGENTS.md` assigns. New dependency not
   in the root `[workspace.dependencies]`? Stop and ask.
5. **Gate:** `scripts/check.sh` must pass.
6. **Measure** if the task or its phase has a number: use the
   `record-result` skill. Release builds only.
7. **Design call not obvious from the code?** Write
   `docs/adr/NNNN-title.md` (next free number; context, decision,
   alternatives).
8. **Close out:**
   - tick the box in `docs/PLAN.md`;
   - add a `docs/CHANGELOG.md` entry (format at the top of that file);
   - commit: `<task id>: <what>`, e.g. `1.4: scalar f32/int8 distance kernels`.
9. **Missed a target?** Say so plainly, record it in RESULTS.md as missed,
   and propose the next step. Never retune the benchmark to pass.
