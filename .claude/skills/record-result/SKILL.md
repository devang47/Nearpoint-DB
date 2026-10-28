---
name: record-result
description: Run a Nearpoint benchmark (recall, latency, QPS, memory, disk) in release mode and append an honest row to docs/RESULTS.md. Use whenever a task or phase acceptance has a number to measure, or the user asks for benchmark results.
---

# Measure and record a result

1. **Release only.** `cargo build --release` first; never report debug
   numbers. Close other heavy apps; note anything that was running.
2. **Run** the harness, for example:
   `cargo run --release -p bench -- --dataset sift1m --index hnsw`.
   Kernel microbenches: `cargo bench -p kernels`.
3. **Collect** everything the RESULTS.md header asks for: date, task id,
   dataset (with N), index + params, RNG seed, recall@10 (tie-aware, with
   the id-match figure beside it; the harness prints both), p50/p99, QPS,
   peak RSS **and** peak phys_footprint, disk used, build time, and
   `git rev-parse --short HEAD`. Unmeasured column → `n/m`, not a guess.
4. **Cold vs warm.** Say which. Cold-cache runs use `F_NOCACHE` and
   `sudo purge` (ask before running `sudo`).
5. **Append** the row at the top of the table in `docs/RESULTS.md`.
6. **Compare to the target** in `docs/PLAN.md` and tell the user: met or
   missed, by how much. A miss stays recorded as a miss.
