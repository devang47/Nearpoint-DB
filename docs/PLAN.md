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
- [x] 2.1 `Index` trait: build, insert, search(query, k, params) → ids + distances.
      As built: `add(batch)` covers build and insert (dyn-compatible);
      metrics L2 / inner product / cosine (normalized on insert).
- [x] 2.2 Flat (brute-force) index, multithreaded with rayon.
- [x] 2.3 HNSW: layered graph, configurable M, ef_construction, ef_search;
      parallel build. Design in `docs/adr/0003-hnsw-concurrency.md`.
- [x] 2.4 Parameter sweep tool: recall vs latency curves written as CSV
      (`bench --ef a,b,c --csv out.csv`; curves in `docs/results/`).
**Acceptance:** HNSW on SIFT1M reaches recall@10 ≥ 0.95 at p50 ≤ 2 ms
single-threaded query; build ≤ 10 min.
**Status (2026-10-28): met.** M=16, efc=200, ef=64: recall@10 0.9632 at
p50 0.106 ms; build 44.4 s on 10 threads.

## Phase 3: Durability
- [x] 3.1 WAL: length-prefixed records (upsert/delete), CRC32 per record,
      configurable fsync policy (every write / batched / interval). On
      macOS plain `fsync` doesn't reach stable storage; std's `sync_data`
      issues `F_FULLFSYNC` (verified: both `sync_data` and `sync_all` do;
      ~4 ms each on the M4). Batched group commit is the default. Any I/O
      failure poisons the WAL (ADR 0007). Format: docs/formats/wal.md.
- [x] 3.2 Segment file format v1: header, vectors, ids, index blob,
      footer with checksums. Document in docs/formats/segment.md.
      Plus `Hnsw`/`Flat` `to_bytes`/`from_bytes` for the index blob (ADR 0006).
- [x] 3.3 Recovery: on startup, load sealed segments, replay WAL,
      truncate torn tail records. As built: `engine::Collection`
      (WAL + mutable HNSW + sealed segments + tombstones); a versioned
      MANIFEST is the seal's commit point; payloads in a per-segment
      sidecar (ADR 0008, docs/formats/manifest.md, payloads.md).
- [x] 3.4 Crash tests, two layers: (a) hand-rolled fail points behind a
      `failpoints` cargo feature that abort or tear writes mid-record;
      (b) a harness that SIGKILLs a child server mid-load, restarts it,
      and checks every acknowledged write. (SIGKILL keeps the OS page
      cache, so (b) tests the process, not power loss; power-loss safety
      rests on 3.1's `F_FULLFSYNC`.)
**Acceptance:** 1,000 randomized crash/restart cycles with zero lost
acknowledged writes.
**Status (2026-10-28): met.** 1,000 SIGKILL cycles in release, 0 lost
acknowledged writes, 0 failures (138 s); plus fail points after every
seal step and on torn WAL appends.

## Phase 4: Server and CRUD
Runs on one mutable segment + WAL per collection; Phase 6 generalizes to
many segments.
- [x] 4.1 Collections: name, dim, metric, config; persisted manifest.
      Server config via clap flags + env vars (no config-file format yet).
- [x] 4.2 Upsert, delete (tombstone bitmap), get-by-id.
- [x] 4.3 Concurrency: many readers + one writer per collection;
      readers clone an `Arc<Snapshot>` of the segment list (std `RwLock`
      around the `Arc`, held only for the clone).
- [x] 4.4 axum HTTP/JSON API: create/list/drop collection, upsert
      (batch), delete, search, get. OpenAPI-style doc in docs/api.md.
      Plus a raw-binary bulk-upsert endpoint (length-prefixed f32 blocks)
      so million-vector loads don't pay JSON parsing cost.
- [x] 4.5 Minimal Python client in `clients/python/` (stdlib `urllib`
      only, no pip installs) for testing.
**Acceptance:** concurrent insert + search load test runs 30 min with
no errors; deleted ids never appear in results.
**Status (2026-10-28): met.** `loadtest` 1,800 s: 2 writers (1.7M
inserted, 34k replace and 34k delete requests), 4 readers (25.1M
searches, a third filtered), a seal every 60 s; 0 errors, 0 violations.
Design in ADR 0011. Server RSS grew ~20 MiB/min to 603 MiB: dead rows
stay in RAM until compaction (6.3).

## Phase 5: Quantization and IVF-PQ
- [x] 5.1 k-means (k-means++ init, multithreaded, trains on a sample).
- [x] 5.2 Product quantization (m subquantizers, 8-bit codes); OPQ optional.
- [x] 5.3 Scalar quantization (int8) for full-vector storage at lower cost.
      Quality on SIFT1M (ADR 0004, RESULTS.md): PQ ADC alone tops out at
      0.53 (m=16) / 0.73 (m=32) recall@10, so 5.4 must re-rank. PQ m=16 +
      top-1000 + SQ int8 re-rank reaches 0.98.
- [x] 5.4 IVF-PQ index: coarse quantizer, inverted lists, ADC with
      lookup tables, nprobe parameter.
- [x] 5.5 4-bit fast-scan PQ kernel with NEON table lookups (`vqtbl1q_u8`).
- [x] 5.6 Re-ranking top candidates with full or int8 vectors (int8,
      ADR 0009).
**Acceptance:** SIFT10M via IVF-PQ+rerank at recall@10 ≥ 0.9 using
<25% of HNSW's RAM (phys_footprint).
**Status (2026-10-28): met, the RAM half narrowly.** BIGANN 10M, nlist
4096: 8-bit m=16 nprobe 32 rerank 100 → 0.9175 at p50 0.59 ms; 4-bit
m=32 nprobe 32 rerank 200 → 0.9155 at p50 0.21 ms. IVF-PQ build + index
1,559 MiB vs HNSW's 9,438 MiB measured peak (16.5%) or ~6.6 GB
steady-state size (~24%); see RESULTS.md for why both are given.

## Phase 6: Segments and compaction
- [x] 6.1 Mutable segment (HNSW in RAM) sealing at configurable size.
- [x] 6.2 Background builder: sealed segment → on-disk HNSW (small)
      or IVF-PQ (large) segment, swapped in atomically.
      IVF-PQ only for L2/cosine for now: inner product reaches 0.84 at
      nprobe 64 / rerank 200 and ~0.94 even with rerank 5000 (int8
      re-rank ceiling; needs f32 re-rank, Phase 8).
- [x] 6.3 Compaction policy: merge small segments, drop tombstoned vectors.
- [x] 6.4 Query fan-out across segments with parallel search and top-k merge.
**Acceptance:** 4-hour soak test with mixed insert/delete/search; recall
and p99 latency stay within 10% of the start.
History (2026-10-29): not met (p99). 4-hour soak, 1M live dim-32
points, 13.3M inserts + 13.3M deletes: recall drift −0.2%, 0 violations, RSS
flat (dead rows are compacted away), but p99 rose 1.29 → 1.82 ms (+41%)
over the first ~2 h, then plateaued; 1 client-side transport error.
Diagnosis (2 h soak with segment logging): no degradation over time;
p99 tracks background builds (a full IVF-PQ retrain every 300 s under
uniform deletes: 1.98 vs 1.27 ms p99), and the drift check aliases with
that cycle. Reusing trained quantizers halved the spike windows (1 h);
HNSW merges still spike; running builds at utility QoS traded peaks for
duration (reverted). 4-hour re-soak with quantizer reuse: recall −0.5%,
0 errors/violations, RSS flat, hourly mean p99 flat, but p99 drift
+12.3% (30-window check; +14.6% first vs last hour): still not met. The
maintenance cycle (both IVF-PQ segments rewritten every 300 s) keeps ~2
of 5 minutes elevated.
2026-10-30: timed every background step; the expensive part was HNSW
tier merges (7–10 s) and fresh IVF-PQ trainings (14–19 s). Absorbing
3 small segments at a time into IVF-PQ while keeping its codes cuts
background work ~15× (30-minute soak: max p99 2.47 → 1.79 ms, mean
unchanged, recall −1.6% absolute). Predicted to pass the 4-hour check;
not run (by decision), so still unticked.
2026-10-31: nprobe scaled per segment with its nlist wins the recall
back and more (30 minutes: 0.749 → 0.792, mean p99 1.53 ms, −7.6%
throughput).
**Status (2026-10-31): met.** 4-hour soak at 71bbdd4, pre-registered
30-window check: recall 0.799 → 0.795 (−0.5%), p99 1.45 → 1.41 ms
(−2.3%); first vs last hour −0.0% / −2.5%. 0 errors, 0 violations,
0 reconnects, RSS flat (~440 MiB). RESULTS.md.

## Phase 7: Metadata filtering
- [x] 7.1 Per-segment payload store: typed columns (int, float, keyword, bool).
- [x] 7.2 Filter expressions (eq, in, range, and/or/not) → roaring bitmaps.
      JSON shape and semantics in docs/filters.md (ADR 0005).
- [x] 7.3 Filter applied during graph traversal and IVF scanning.
- [x] 7.4 Planner: brute-force over the bitmap when selectivity is very low
      (`filtered² ≤ 8·ef·len`, ADR 0010).
- [x] 7.5 Walk budget: stop a filtered walk after ~`filtered` distance
      computations and brute-force the bitmap instead, so correlated
      filters can't blow up p99 (measured: far 10% filter, p99 75 ms).
      As built: budget = `filtered` distances; far 10% p99 72 → 18 ms
      (3.1× brute force, not ≈2×: a walk distance costs ~2.3× a scan one).
**Acceptance:** recall@10 ≥ 0.9 at filter selectivity 1%, 10%, and 50%.
**Status (2026-10-28): met for HNSW on SIFT1M**: uniform filters
1.0000 / 0.9973 / 0.9810 at ef 64; correlated ("far") filters 1.0000 /
0.9991 / 0.9906. But a far 10% filter walks ~110k nodes: p99 75 ms vs
6 ms for brute force. Follow-up: cap the walk at `filtered` distances,
then brute-force (ADR 0010 addendum).

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
