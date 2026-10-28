# Changelog

Agent-maintained log of every change. Newest first. One entry per completed
change. Record **what** changed and **why** so a future maintainer can
understand the decision without re-deriving it.

Format:

```
## YYYY-MM-DD — short title
- **What:** the change, in one or two lines (files/crates touched).
- **Why:** the reason it was needed. If it ties to a PLAN.md task or an ADR,
  link it (e.g. `docs/adr/0003-...`).
```

<!-- new entries go directly below this line -->

## 2026-10-28 — Dataset loaders (task 1.2)
- **What:** `crates/bench` gains a library with `dataset::{read_xvecs,
  read_bin, read_bin_ground_truth}` reading texmex and big-ann files straight
  into a row-major `Vectors<T>`; optional first-N limit. Added `.i8bin`
  (SPACEV is int8) and the big-ann ground-truth layout.
- **Why:** task 1.2. Every size is validated against the file length before
  allocating, so corrupt headers and wrong element types fail loudly, and a
  byte-range prefix of a 1B file loads with an explicit limit.

## 2026-10-28 — Scaffold workspace (task 1.1) and refine stack
- **What:** Cargo workspace with 7 stub crates; pinned toolchain,
  `target-cpu=native`, workspace lints, `scripts/check.sh`, GitHub Actions CI;
  doc stubs (`RESULTS.md`, `api.md`, `formats/`, ADR 0001); `README.md`;
  `agents.md` → `AGENTS.md` (+ `CLAUDE.md` import); project skills in
  `.claude/skills/`. PLAN.md: allowed `libc` and `tracing-subscriber`,
  `curl` instead of an HTTP client crate, compile-time SIMD dispatch,
  phys_footprint as the memory metric, macOS `F_FULLFSYNC` and SIGKILL
  crash-test notes.
- **Why:** task 1.1, plus gaps found reviewing the plan against macOS on Apple
  Silicon. See `docs/adr/0001-stack-refinements.md`.

## 2026-10-28 — Refine roadmap
- **What:** `docs/PLAN.md` — renamed vecdb→Nearpoint and `~/data/vecdb`→
  `~/data/Nearpoint`; recorded stack decisions (f32+int8 first, f16 deferred;
  JSON API + raw-binary bulk-upsert; clap/env config; plaintext Prometheus
  `/metrics`); folded doc stubs into 1.1; made FAISS/Qdrant comparison optional.
- **Why:** align the plan with `agents.md` naming and close the undecided
  framework gaps before implementation starts.
