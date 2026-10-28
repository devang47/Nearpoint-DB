# Nearpoint — a single-node vector database in Rust

## What this is
A real vector database (not just an index): durable writes (WAL), CRUD,
metadata filtering, HTTP API, concurrent reads/writes. Targets 1M–100M
vectors with all features, plus a compressed-only "billion mode" that
searches 1B vectors in <100 ms on a Mac mini.

The roadmap lives in `docs/PLAN.md`. Always check it before starting work.

## Hardware constraints (non-negotiable)
- Mac mini, Apple Silicon (aarch64, NEON), 16 GB RAM, 256 GB internal SSD,
  no external storage, no cloud.
- Never assume more than ~10 GB of RAM is available to the process.
- Datasets live in `~/data/Nearpoint/`, never inside the repo (gitignored).
- Before any download or file >5 GB: run `df -h`, state the size, and ask
  me first.
- Never store raw 1B datasets. Billion mode streams, encodes, and discards.

## Architecture (summary)
- Segment-based engine: WAL → mutable segment (in-RAM HNSW) → sealed
  immutable segments (HNSW for small, IVF-PQ for large) → background
  compaction.
- Queries fan out across all segments in parallel, skip tombstones,
  apply filter bitmaps during search, merge top-k.
- Metadata: per-segment columnar store; filters → roaring bitmaps.

## Workspace layout
- `crates/kernels`  distance functions (scalar + NEON, f32/f16/int8)
- `crates/quant`    k-means, product quantization, scalar quantization
- `crates/index`    `Index` trait; flat, HNSW, IVF-PQ implementations
- `crates/storage`  WAL, segment file format, mmap, checksums
- `crates/engine`   collections, segments, compaction, deletes, filters
- `crates/server`   axum HTTP API
- `crates/bench`    recall/latency harness, dataset loaders
- `tools/`          dataset downloader (curl + tar), streaming 1B builder

## Conventions
- Rust 1.98.1 (pinned in `rust-toolchain.toml`), edition 2024.
  `scripts/check.sh` (fmt, clippy `-D warnings`, tests) must pass.
- Builds use `target-cpu=native` (`.cargo/config.toml`): SIMD paths are
  chosen with `cfg(target_feature)`, not runtime detection.
- Core algorithms are written by hand. Do NOT add ANN/vector-search crates
  (no hnsw_rs, faiss bindings, usearch, etc.).
- Allowed dependencies: tokio, axum, serde, serde_json, memmap2, rayon,
  roaring, crc32fast, bytemuck, rand, thiserror, tracing;
  `libc` only in `storage`; anyhow, clap and tracing-subscriber
  only in binaries; criterion and proptest as dev-dependencies.
  Versions live in the root `Cargo.toml` `[workspace.dependencies]`;
  crates use `dep.workspace = true`. Ask before adding anything else.
  Tools use the system `curl`/`tar` instead of an HTTP client crate.
- `unsafe` only in `kernels` and `storage`, each block with a
  `// SAFETY:` comment (both enforced by workspace lints). Every SIMD
  kernel has a scalar reference implementation and a proptest comparing
  them.
- Library crates use `thiserror` error types; no `unwrap()` outside tests
  (enforced by `clippy::unwrap_used`).
- On-disk formats are versioned (magic bytes + version + CRC32).
  Document every format in `docs/formats/`.

## Commands
- Check:       `scripts/check.sh`
- Build:       `cargo build --release`
- Test:        `cargo test --workspace`
- Bench:       `cargo bench -p kernels`
- Recall test: `cargo run --release -p bench -- --dataset sift1m --index hnsw`
- Always benchmark with `--release`. Never report numbers from debug builds.

## How to work
1. Work one task from `docs/PLAN.md` at a time. Restate its acceptance
   criteria before starting.
2. Write tests first or alongside the code. A task is done only when its
   acceptance criteria are met and measured.
3. Check off the task in `docs/PLAN.md`, and record measured numbers in
   `docs/RESULTS.md` (date, dataset, params, recall, p50/p99, RAM, disk).
4. For non-obvious design decisions, write a short ADR in
   `docs/adr/NNNN-title.md` (context, decision, alternatives).
5. Add a `docs/CHANGELOG.md` entry for every change (what + why) before
   committing. See the format at the top of that file.
6. Commit after each completed task with a clear message.
7. If a measurement misses its target, report it honestly. Never tune
   the benchmark to hide it.

## Skills
Project skills in `.claude/skills/` hold the step-by-step procedures:
`plan-task` (work one PLAN.md task end to end), `record-result` (measure
and log to RESULTS.md), `fetch-dataset` (safe downloads), `disk-format`
(add or change an on-disk format).
