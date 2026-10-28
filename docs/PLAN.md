# Nearpoint roadmap

Targets are for a release build on the Mac mini. Record every result in
docs/RESULTS.md.

Stack is fixed by `AGENTS.md` (tokio/axum/serde/memmap2/rayon/roaring/etc.);
ANN cores are hand-written. Decisions layered on top of that
(rationale in `docs/adr/0001-stack-refinements.md`):
- f32 + int8 kernels first; f16 deferred until a measured need.
- Kernels dispatch at **compile time**: NEON is baseline on aarch64 and
  `.cargo/config.toml` builds with `target-cpu=native`, so `dotprod`/`fp16`
  are `cfg(target_feature)` checks, not runtime detection.
- `libc` (storage only, since it needs `unsafe`) for what std lacks:
  `F_NOCACHE`, `proc_pid_rusage` (phys_footprint), `getrusage`. Bench gets
  memory numbers through `storage::sys`.
- No HTTP client crate: tools shell out to the system `curl` (handles the
  texmex FTP mirror and HTTP range requests) and `tar`.
- JSON API plus one raw-binary bulk-upsert path; config via clap flags +
  env only; `/metrics` as hand-written plaintext Prometheus.
- Memory is reported as **phys_footprint** (what macOS memory pressure
  counts) alongside RSS. Clean mmap'd file pages count toward RSS but are
  reclaimable, so RSS alone overstates usage for mmap-heavy segments.

## Phase 1: Kernels and benchmark harness
- [x] 1.1 Cargo workspace with all crates stubbed; CI script
      (`scripts/check.sh`: fmt, clippy, test) + GitHub Actions on an
      Apple Silicon runner. AGENTS.md rules enforced as workspace lints
      (`unsafe_code`, `clippy::unwrap_used`, `undocumented_unsafe_blocks`).
      Stub the docs `AGENTS.md` expects: `docs/RESULTS.md`, `docs/formats/`,
      `docs/adr/`, `docs/api.md`.
- [x] 1.2 Dataset loaders: fvecs/ivecs/bvecs (texmex) and
      .fbin/.u8bin/.i8bin/.ibin (big-ann-benchmarks: u32 count, u32 dim, data).
      Support reading the first N vectors only (the 1B files' header still
      says 1B, so N must override the header count).
      Plus the big-ann ground-truth .ibin (ids then distances).
- [x] 1.3 `tools/fetch-dataset.sh` (curl + tar) for SIFT10K (siftsmall)
      and SIFT1M into ~/data/Nearpoint/; checks `df -h` first, resumable
      (`curl -C -`), verifies sizes after extraction. `--mirror hf` for
      SIFT1M: the texmex FTP runs at ~30 KB/s (~1.5 h), HF takes ~2 min.
- [x] 1.4 Scalar L2, inner product, cosine (f32), plus int8 L2/dot. Scalar
      references are plain sequential loops (no manual unrolling), so they
      are the honest baseline for the NEON speedup.
- [x] 1.5 NEON versions via `std::arch::aarch64`, selected with
      `cfg(target_arch)` (scalar on other arches); int8 dot via `SDOT`
      behind `cfg(target_feature = "dotprod")`. Proptest vs scalar
      (tolerance 1e-4 relative for f32; exact for int8).
- [x] 1.6 Criterion benches for all kernels at dims 96, 128, 768.
- [x] 1.7 Bench harness: recall@k vs ground truth, p50/p95/p99 latency,
      QPS, peak RSS and peak phys_footprint. Fixed RNG seeds, recorded in
      every result row (from the first randomized index on). Recall is
      tie-aware, with id-match reported beside it (ADR 0002).
**Acceptance:** brute force on SIFT1M matches ground truth exactly
(recall@10 = 1.0); NEON ≥4× faster than scalar at dim 128.
**Status (2026-10-28): met.** Tie-aware recall@10 = 1.0000 (id match
0.9994, all ties); NEON f32 L2 4.4× scalar at dim 128.

## Phase 2: In-memory indexes
- [ ] 2.1 `Index` trait: build, insert, search(query, k, params) → ids + distances.
- [ ] 2.2 Flat (brute-force) index, multithreaded with rayon.
- [ ] 2.3 HNSW: layered graph, configurable M, ef_construction, ef_search;
      parallel build.
- [ ] 2.4 Parameter sweep tool: recall vs latency curves written as CSV.
**Acceptance:** HNSW on SIFT1M reaches recall@10 ≥ 0.95 at p50 ≤ 2 ms
single-threaded query; build ≤ 10 min.

## Phase 3: Durability
- [ ] 3.1 WAL: length-prefixed records (upsert/delete), CRC32 per record,
      configurable fsync policy (every write / batched / interval). On
      macOS plain `fsync` doesn't reach stable storage; std's `sync_data`
      issues `F_FULLFSYNC` (verify, and measure its latency; it's
      milliseconds). Batched group commit is the default.
- [ ] 3.2 Segment file format v1: header, vectors, ids, index blob,
      footer with checksums. Document in docs/formats/segment.md.
- [ ] 3.3 Recovery: on startup, load sealed segments, replay WAL,
      truncate torn tail records.
- [ ] 3.4 Crash tests, two layers: (a) hand-rolled fail points behind a
      `failpoints` cargo feature that abort or tear writes mid-record;
      (b) a harness that SIGKILLs a child server mid-load, restarts it,
      and checks every acknowledged write. (SIGKILL keeps the OS page
      cache, so (b) tests the process, not power loss; power-loss safety
      rests on 3.1's `F_FULLFSYNC`.)
**Acceptance:** 1,000 randomized crash/restart cycles with zero lost
acknowledged writes.

## Phase 4: Server and CRUD
Runs on one mutable segment + WAL per collection; Phase 6 generalizes to
many segments.
- [ ] 4.1 Collections: name, dim, metric, config; persisted manifest.
      Server config via clap flags + env vars (no config-file format yet).
- [ ] 4.2 Upsert, delete (tombstone bitmap), get-by-id.
- [ ] 4.3 Concurrency: many readers + one writer per collection;
      readers clone an `Arc<Snapshot>` of the segment list (std `RwLock`
      around the `Arc`, held only for the clone).
- [ ] 4.4 axum HTTP/JSON API: create/list/drop collection, upsert
      (batch), delete, search, get. OpenAPI-style doc in docs/api.md.
      Plus a raw-binary bulk-upsert endpoint (length-prefixed f32 blocks)
      so million-vector loads don't pay JSON parsing cost.
- [ ] 4.5 Minimal Python client in `clients/python/` (stdlib `urllib`
      only, no pip installs) for testing.
**Acceptance:** concurrent insert + search load test runs 30 min with
no errors; deleted ids never appear in results.

## Phase 5: Quantization and IVF-PQ
- [ ] 5.1 k-means (k-means++ init, multithreaded, trains on a sample).
- [ ] 5.2 Product quantization (m subquantizers, 8-bit codes); OPQ optional.
- [ ] 5.3 Scalar quantization (int8) for full-vector storage at lower cost.
- [ ] 5.4 IVF-PQ index: coarse quantizer, inverted lists, ADC with
      lookup tables, nprobe parameter.
- [ ] 5.5 4-bit fast-scan PQ kernel with NEON table lookups (`vqtbl1q_u8`).
- [ ] 5.6 Re-ranking top candidates with full or int8 vectors.
**Acceptance:** SIFT10M via IVF-PQ+rerank at recall@10 ≥ 0.9 using
<25% of HNSW's RAM (phys_footprint).

## Phase 6: Segments and compaction
- [ ] 6.1 Mutable segment (HNSW in RAM) sealing at configurable size.
- [ ] 6.2 Background builder: sealed segment → on-disk HNSW (small)
      or IVF-PQ (large) segment, swapped in atomically.
- [ ] 6.3 Compaction policy: merge small segments, drop tombstoned vectors.
- [ ] 6.4 Query fan-out across segments with parallel search and top-k merge.
**Acceptance:** 4-hour soak test with mixed insert/delete/search; recall
and p99 latency stay within 10% of the start.

## Phase 7: Metadata filtering
- [ ] 7.1 Per-segment payload store: typed columns (int, float, keyword, bool).
- [ ] 7.2 Filter expressions (eq, in, range, and/or/not) → roaring bitmaps.
- [ ] 7.3 Filter applied during graph traversal and IVF scanning.
- [ ] 7.4 Planner: brute-force over the bitmap when selectivity is very low.
**Acceptance:** recall@10 ≥ 0.9 at filter selectivity 1%, 10%, and 50%.

## Phase 8: Scale to 100M
- [ ] 8.1 Memory budget per component; enforce limits and expose metrics
      via a hand-written plaintext Prometheus `/metrics` endpoint (no new dep).
- [ ] 8.2 Cold-cache benchmarks using `F_NOCACHE` (via `libc`) and
      `sudo purge` between runs.
- [ ] 8.3 Load 100M vectors (BIGANN or SPACEV subset) through the API.
      BIGANN 100M u8 is ~12.8 GB: fetch only that prefix of the 1B file
      with an HTTP range request (ask before downloading).
**Acceptance:** 100M vectors, all features on, p99 < 100 ms,
recall@10 ≥ 0.9, phys_footprint ≤ 10 GB (RSS reported alongside).

## Phase 9: Billion mode
- [ ] 9.1 Train coarse quantizer + PQ on a 10M sample.
- [ ] 9.2 Streaming builder: `curl` range download piped into the builder's
      stdin → encode → append (list_id, code, id) to temp file → discard raw
      data. Resumable from the last committed byte offset.
- [ ] 9.3 Multi-pass on-disk counting sort into the final IVF list layout.
- [ ] 9.4 Optional refinement codes for second-stage re-ranking.
- [ ] 9.5 Load as a read-only compressed segment in the engine.
**Acceptance:** 1B vectors, p99 < 100 ms, peak disk ≤ 60 GB,
recall published honestly against the official 1B ground truth.

## Phase 10: Results and write-up
- [ ] 10.1 Benchmark table: 1M/10M/100M/1B × recall, p50/p99, QPS,
      RAM, disk, build time. Our own numbers are the deliverable;
      same-machine FAISS/Qdrant comparison is an optional stretch
      (needs Python + Docker on the mini).
- [ ] 10.2 README with architecture diagram, quickstart, and results.
