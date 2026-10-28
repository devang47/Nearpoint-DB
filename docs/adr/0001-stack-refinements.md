# 0001: Stack refinements before Phase 1

Date: 2026-10-28. Status: accepted.

## Context
`AGENTS.md` fixes a small dependency list. Reviewing the roadmap against
the target machine (Apple Silicon Mac mini, macOS) turned up needs the
list didn't cover, and some rules that review alone would enforce.

## Decision
1. **`libc` allowed in `storage` only.** std has no `F_NOCACHE`
   (cold-cache benches, 8.2), `proc_pid_rusage`/`getrusage` (peak memory,
   1.7), or `madvise` beyond what memmap2 exposes. *Amended 2026-10-28:* first
   drafted as `storage` + `bench`, but calling libc needs `unsafe`, which
   `bench` may not use; bench reads memory through `storage::sys`.
2. **No HTTP client crate.** `tools/` shells out to the system `curl`,
   which already speaks the texmex FTP mirror, HTTP range requests and
   resume (`-C -`). Billion mode (9.2) pipes `curl` into the builder's
   stdin.
3. **`tracing-subscriber` in binaries only.** `tracing` alone emits
   nothing; the server and bench need a subscriber to print logs.
4. **Compile-time SIMD dispatch.** NEON is baseline on aarch64 and the
   project targets one machine, so `.cargo/config.toml` sets
   `target-cpu=native` and kernels use `cfg(target_feature)`. That drops
   the runtime-detection branch from every hot loop.
5. **Memory is measured as phys_footprint as well as RSS.** RSS counts
   clean mmap'd file pages that the OS can drop; the 10 GB budget is
   about memory pressure, which phys_footprint measures.
6. **Rules as lints.** `unsafe_code = deny` (kernels/storage opt out),
   `clippy::unwrap_used = deny` (tests exempt via `clippy.toml`),
   `clippy::undocumented_unsafe_blocks = deny`.
7. **Toolchain pinned** (`rust-toolchain.toml`) so codegen changes don't
   silently move benchmark numbers.

## Alternatives
- `ureq`/`reqwest` for downloads: a new dependency tree for what `curl`
  does already.
- `std::arch::is_aarch64_feature_detected!` runtime dispatch: needed
  only for portable binaries, which this project doesn't ship.
- `arc-swap` for reader snapshots: std `RwLock<Arc<Snapshot>>` held only
  for the clone is enough until a benchmark says otherwise.
- `fail` crate for fail points: a `failpoints` cargo feature with a few
  hand-rolled hooks covers 3.4 without a dependency.
