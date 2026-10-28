# Nearpoint

A single-node vector database in Rust, built from scratch for one Apple
Silicon Mac mini (16 GB RAM, 256 GB SSD). It's a database, not just an
index: durable writes (WAL), CRUD, metadata filtering, an HTTP API, and
concurrent reads and writes.

> **Status:** Phase 1, kernels and benchmark harness. The workspace is
> scaffolded and nothing is usable yet. Progress is tracked in
> [`docs/PLAN.md`](docs/PLAN.md).

## Targets

| Scale | Mode | Goal |
|-------|------|------|
| 1M–100M | all features | p99 < 100 ms, recall@10 ≥ 0.9, ≤ 10 GB memory |
| 1B | compressed, read-only ("billion mode") | p99 < 100 ms, ≤ 60 GB disk |

Every number is measured on the target machine and logged in
[`docs/RESULTS.md`](docs/RESULTS.md), including the misses.

## Architecture

```
            HTTP (axum) ─ JSON + raw-binary bulk upsert
                 │
   write ──► WAL ──► mutable segment (in-RAM HNSW)
                 │        │ seal
                 │        ▼
                 │   sealed segments: HNSW (small) │ IVF-PQ (large)
                 │        ▲ background compaction
   query ──► fan out across segments in parallel
             ├─ skip tombstones, apply filter bitmaps (roaring)
             └─ merge top-k
```

The ANN algorithms (HNSW, k-means, PQ, IVF) and the SIMD distance
kernels are written by hand. No vector-search crates.

## Layout

| Path | Contents |
|------|----------|
| `crates/kernels` | distance functions: scalar + NEON, f32 / int8 |
| `crates/quant` | k-means, product and scalar quantization |
| `crates/index` | `Index` trait; flat, HNSW, IVF-PQ |
| `crates/storage` | WAL, segment format, mmap, checksums |
| `crates/engine` | collections, segments, compaction, deletes, filters |
| `crates/server` | HTTP API binary |
| `crates/bench` | recall/latency harness, dataset loaders |
| `tools/` | dataset downloader, streaming 1B builder |
| `docs/` | plan, results, ADRs, on-disk formats, API, changelog |

## Quickstart

Requires the Rust toolchain pinned in `rust-toolchain.toml` (rustup
installs it automatically).

```sh
scripts/check.sh            # fmt + clippy + tests: the commit gate
cargo build --release
cargo bench -p kernels      # kernel microbenchmarks
```

Datasets live in `~/data/Nearpoint/`, never in the repo.

## Contributing

Read [`AGENTS.md`](AGENTS.md) first. It covers the hardware limits, the
allowed dependencies, the `unsafe` and error-handling rules, and the
per-task workflow. Design decisions are recorded in [`docs/adr/`](docs/adr/).
