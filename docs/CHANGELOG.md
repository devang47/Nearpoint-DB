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

## 2026-10-29 — Tried: background builds at utility QoS (reverted)
- **What:** `storage::sys::set_background_qos` on the build pool and the
  maintenance thread (4cf4a72), measured in a 1 h soak, then reverted.
- **Why reverted:** fewer worst windows (p99 > 1.8 ms: 11 → 7) but
  builds on the efficiency cores ran longer, raising mean and median p99
  (1.42 → 1.49, 1.35 → 1.44 ms) and cutting searches/s 5%. Not a clear
  win for new `unsafe` code. Numbers in RESULTS.md.

## 2026-10-29 — Compaction reuses IVF-PQ quantizers; soak p99 diagnosis
- **What:** `IvfPq::empty_clone` (same config + trained quantizers, no
  vectors); engine rewrites and merges with an IVF-PQ input build on the
  biggest one's quantizers unless nlist would more than double.
  `loadtest` retries once when a reused connection closes before any
  reply byte (counted as `reconnects`). Results in RESULTS.md.
- **Why:** a 2 h diagnostic soak showed p99 tracks background builds,
  not time: under uniform deletes the whole 1M set was rebuilt (and
  retrained) every 300 s, p99 1.98 vs 1.27 ms in quiet windows. Reuse
  halved the spike windows (20 → 11 per hour). The one error per run was
  a keep-alive connection the server had closed; every loadtest request
  is idempotent, and RFC 9112 §9.3.1 allows the retry.

## 2026-10-29 — Parallel round 4: segments and compaction, walk budget, soak harness
- **What:** scaffold commit (rayon in engine), three branches built in
  parallel worktrees and merged: Phase 6 (6.1–6.4), task 7.5, and a soak
  mode for `loadtest` (entries below). A read-only review of the Phase 6
  merge found edge-case bugs, fixed in one commit (entry below). After
  the merge, on a quiet machine under `caffeinate -dims`: the 7.5
  filtered sweep, then the 4-hour Phase 6 soak.
- **Why:** Phase 6 lives in `engine`, 7.5 in `index/hnsw.rs`, the soak
  harness in `server/src/bin/loadtest.rs`: disjoint, so they ran side by
  side; timings only after the merge. Soak result: recall held (−0.2%),
  p99 did not (+41%); Phase 6 acceptance not met yet (RESULTS.md).
  (Commit 2b96ab6 also carries these doc drafts, and its message's
  "hyper closes idle connections" guess was wrong: tested, it doesn't.)

## 2026-10-29 — Phase 6 review fixes
- **What:** upserts and queries reject components beyond ±1e15 and zero
  `cosine` vectors; `Sq::train` errors on a non-finite or zero scale and a
  failed IVF-PQ build falls back to HNSW; PQ m = largest divisor of dim
  ≤ max(16, dim/8), and dims with no split into ≤ 16-dim subspaces (primes)
  stay HNSW; merges also capped at 2 GiB of f32 input; `Config::validate`
  requires `seal_rows ≤ cap / 2`; a rewrite syncs the WAL before building;
  the Db thread catches a panicking `maintain`; `segments()` skips the
  mutable segment's lock (ADR 0012 addendum).
- **Why:** the worst case: one vector with a huge finite range made the
  int8 scale infinite, the IVF-PQ segment published fine and then refused
  to load, so the whole database would not open. The others: a poisoned
  WAL left a new orphan file pair per retry; configs that could never
  merge; prime/wide dims getting 1-byte or 48-dim PQ codes; wide-vector
  merges blowing the 10 GB budget; one panic stopping all maintenance.

## 2026-10-29 — Segments and compaction (tasks 6.1–6.4)
- **What:** `crates/engine`: the mutable segment seals at `seal_rows`
  (default 100k) with segment files written outside the writer lock; a
  background step (`Collection::maintain`, driven by one `Db` thread)
  rewrites sealed segments: drops rows when ≥ 25% are dead, rebuilds
  segments of ≥ `ivf_rows` (default 1M) as IVF-PQ for L2/cosine (f32
  kept in the file, served from the mmap), and merges 4 segments of a
  size tier up to `max_merge_rows` (4M); builds on a dedicated rayon pool
  (`--build-threads`, default half the cores). The id map points at
  stable segment keys, so a merge remaps only moved ids. Sealed segments
  are searched in parallel and merged to top k. Manifest v2 (per-segment
  file ids from a counter; v1 still read). `SearchOptions`/search JSON
  gain `nprobe` (32) and `rerank` (100); collection info lists segments.
  ADR 0012, docs/formats/manifest.md, docs/api.md.
- **Why:** Phase 6. Seals no longer stall writers for the file write;
  dead rows no longer stay in RAM forever (server RSS grew ~20 MiB/min in
  the Phase 4 load test); big segments become IVF-PQ instead of an HNSW
  that would not fit (10M swaps on 16 GB). Every manifest publish syncs
  the WAL first, so no tombstone or dropped row rests on a record that
  isn't durable.

## 2026-10-29 — Filtered walk budget (task 7.5)
- **What:** `index/hnsw.rs`: a filtered layer-0 walk stops after
  `filtered` (the bitmap's cardinality) distance computations of its own
  and brute-forces the bitmap instead; `Plan::FullWalk` keeps the old walk
  for benches; per-thread `hnsw::fallbacks()` counter; `filtered` bench
  prints fallbacks and takes `--plan walk`. ADR 0010 addendum.
- **Why:** a correlated 10% filter walked ~110k nodes (p99 75 ms vs 6 ms
  brute force) because the planner only sees cardinality. The budget
  bounds any filter without estimating correlation: measured p99 72 →
  18 ms on SIFT1M far 10%, recall unchanged (RESULTS.md).

## 2026-10-29 — Load test soak mode
- **What:** `loadtest --live N`: preload N points, then keep the live
  count at N (each round inserts and deletes `--batch`); every
  `--probe-secs` pause writers, check the server's live count, and score
  `--probe-queries` k=10 searches against exact brute-force top 10 over
  the client's model; per-window p50/p99, throughput, recall, server RSS,
  `--csv`; at the end compare medians of the first and last
  `--drift-windows` windows against `--max-drift`.
- **Why:** Phase 6 acceptance (recall and p99 within 10% over 4 hours).
  Medians, not single windows: a 3-minute smoke run showed one window's
  p99 swinging ±40% with whichever background build overlapped it.
  Later in the day: a fresh connection per probe; errors name the
  request and whether it reused an idle connection; each window logs the
  sealed segment layout (the 4-hour run's p99 rise couldn't be
  attributed without it).

## 2026-10-28 — Parallel round 3: server and CRUD, index/storage groundwork
- **What:** scaffold commit (server dependencies in Cargo.lock), then two
  branches built in parallel worktrees and merged: Phase 4 (4.1–4.5) and
  index/storage groundwork for Phases 6 and 8 (entries below). After the
  merge, on a quiet machine: 1,000-cycle crash run, SIFT1M bulk-loaded
  through the API (recall checked after a restart), HNSW and IVF-PQ
  (L2/IP/cosine) re-runs, correlated-filter sweep, the 30-minute Phase 4
  load test, and HNSW 10M after the memory fixes (peak 10.7 → 8.5 GB). New PLAN task 7.5 (walk
  budget) and a 6.2 note (IVF-PQ only for L2/cosine for now).
- **Why:** engine/server and index/storage are disjoint, so they ran side
  by side; timings were taken only after the merge.

## 2026-10-28 — Server and CRUD (tasks 4.1–4.5)
- **What:** `engine::Db` registry over a data directory (`LOCK` via
  `File::try_lock`, `collections/<name>/CONFIG` framed JSON v1, crash-safe
  create/drop by rename); `Collection` methods take `&self`: one writer,
  readers search an `Arc<Snapshot>` of per-segment views with
  copy-on-write tombstones and a visible-row cutoff; `upsert_batch`/
  `delete_batch` validate the whole batch and log it with one sync
  (`Wal::append_batch`); optional payload `schema` in the collection
  config. `crates/server`: axum JSON API (collections, upsert, binary bulk
  upsert, delete, get, search with ef/filter/payload/vector, seal, health),
  `Error::Unknown` → 503; `loadtest` binary. `clients/python/nearpoint.py`
  (stdlib) with live-server tests. Docs: `docs/api.md`,
  `docs/formats/manifest.md` (CONFIG), ADR 0011.
- **Why:** Phase 4 acceptance: the 30-minute concurrent load test passed
  with 0 errors and 0 deleted ids returned. Searches never wait on a
  writer's fsync; batches share one fsync (single durable writes cap at
  ~270/s). `docs/adr/0011-server-and-concurrency.md`.

## 2026-10-28 — IVF-PQ serialization and metrics, segment v2, HNSW memory, correlated filters
- **What:** `IvfPq::to_bytes`/`from_bytes` (blob `NPIV` v1, fully
  validated, random-corruption proptest); segment format v2 adds element
  type 1 (no f32 vector section) and blob kind IvfPq, v1 still read
  (fixtures for both). IVF-PQ supports cosine (normalize, L2, distance/2)
  and inner product (`<q,c> + Σ <q_j, r̂_j>` via `Pq::dot_table`, SQ re-rank
  by dot). `Hnsw::reserve` and 4096 striped node locks instead of a
  `Mutex` per node; the bench reserves before batched adds. Bench
  `--metric ip|cosine`; `filtered --filter far`. ADR 0003/0006/0009/0010
  addenda; docs/formats/segment.md.
- **Why:** Phase 6 seals large segments as IVF-PQ and must persist them;
  Phase 9 can't afford f32 vectors on disk. A per-node mutex cost ~80 B
  per vector on macOS and batched adds regrew every array (SIFT1M HNSW
  bench peak 1,922 → 1,231 MiB), which Phase 8's 10 GB budget can't carry.
  Correlated filters were 7.3's known gap: recall holds (≥ 0.99) but a
  far 10% filter's walk has a 75 ms p99 (task 7.5).

## 2026-10-28 — Parallel round 2: recovery, IVF-PQ, filtered search
- **What:** scaffold commit (`SearchParams` gains `filter`, `nprobe`,
  `rerank`; `ivfpq`/`collection` stubs; `index` → `quant`/`roaring`,
  `engine` → `index`/`storage`), then three tasks built in parallel
  worktrees and merged: 3.3–3.4, 5.4–5.6, 7.3–7.4 (entries below).
  SIFT10M (BIGANN 10M prefix, 1.29 GB) fetched from the big-ann-benchmarks
  mirror. After the merge, on a quiet machine: SIFT1M/10M IVF-PQ sweeps,
  the HNSW 10M baseline, filtered sweeps, an unfiltered HNSW A/B against
  the pre-round build, and the 1,000-cycle crash run. PLAN ticks, RESULTS
  rows and CSVs in `docs/results/`; ADRs 0008–0010 accepted.
- **Why:** the tasks touch disjoint modules, so they ran side by side;
  the shared API change was committed first so the branches merged
  without conflicts. Timings were taken only after the merge.

## 2026-10-28 — Filtered search and brute-force planner (tasks 7.3 HNSW/Flat, 7.4)
- **What:** `SearchParams::filter` is honoured by Flat (scans only the
  bitmap's ids) and HNSW (the layer-0 walk expands non-matching nodes but
  keeps only matches; `Hnsw::search` brute-forces the bitmap when
  `filtered² <= 8·max(ef,k)·len`). `Hnsw::plan`/`search_with` and a
  per-thread `hnsw::distances()` counter expose the choice; new
  `bench --bin filtered` reports filtered recall, distances and plan per
  selectivity. Proptests: results never leave the filter; filtered Flat
  equals brute force over the bitmap.
- **Why:** Phase 7 acceptance (recall@10 ≥ 0.9 at 1/10/50%): measured
  1.0000 / 0.9973 / 0.9810 at ef 64 on SIFT1M. The threshold is the
  distance-count crossover, confirmed by wall-clock (crossover between 2%
  and 5%); unfiltered latency unchanged. `docs/adr/0010-filtered-search.md`.

## 2026-10-28 — IVF-PQ index, 4-bit fast-scan, SQ re-rank (tasks 5.4–5.6)
- **What:** `index::IvfPq`: seeded k-means coarse quantizer, PQ on
  residuals, inverted lists, per-list ADC tables, `nprobe`, filter checked
  while scanning (IVF half of 7.3); L2 only. `nbits = 4` stores fast-scan
  blocks (32 vectors, nibble-interleaved) scanned by `kernels::pq4_scan`
  (scalar + NEON `vqtbl1q_u8`, bit-exact proptest) with u8-quantized
  tables. `rerank` re-scores the best candidates against int8 SQ codes and
  returns those distances. `quant::PqConfig.nbits`, transposed codebooks.
  Bench: `--index ivfpq` with `--nlist`, `--pq-m`, `--nbits`, `--nprobe`
  / `--rerank` sweeps and a `bigann10m` dataset; `fetch-dataset.sh
  bigann10m` range-fetches the 10M prefix.
- **Why:** Phase 5 acceptance: SIFT10M recall@10 0.9175 at p50 0.59 ms
  (8-bit) / 0.9155 at 0.21 ms (4-bit), using 16.5% of HNSW's measured
  peak (~24% of its steady-state size). PQ alone tops out near 0.57, so
  re-rank is required; int8 re-rank vectors (1.28 GB at 10M) fit where
  f32 (5.1 GB) can't. `docs/adr/0009-ivf-pq.md`.

## 2026-10-28 — Collection recovery and crash tests (tasks 3.3–3.4)
- **What:** `engine::collection::Collection` (WAL + mutable HNSW + sealed
  segments + id map + tombstones) with open/upsert/delete/get/search/seal.
  A versioned MANIFEST is the seal's commit point (covered LSN, live
  segments, per-segment tombstones); payloads go in a `<lsn>.payloads`
  sidecar; open replays only WAL records above the covered LSN and removes
  a crashed seal's leftovers. New `storage::framed` (magic/version/len/
  body/CRC32, atomic write) and `PayloadStore::check`. A WAL failure
  mid-write is reported as `Error::Unknown`. Crash tests: fail points
  after each seal step and on torn WAL appends, plus a SIGKILL harness
  (`crash-child` bin + `tests/crash.rs`).
- **Why:** Phase 3 acceptance: 1,000 SIGKILL cycles in release, 0 lost
  acknowledged writes (138 s). Writes survive crashes at any point,
  including mid-seal, without being lost or applied twice.
  `docs/adr/0008-collection-recovery.md`; formats manifest.md, payloads.md.

## 2026-10-28 — Parallel round: WAL, segments, quantization, filters
- **What:** four independent tasks built in parallel worktrees and merged:
  3.1 WAL, 3.2 segment format + index serialization, 5.1–5.3 quantization,
  7.1–7.2 payloads + filters (entries below). Then, on a quiet machine:
  WAL sync latency and a new group-commit throughput test, a re-run of the
  SIFT1M quantization check, and an HNSW regression run. ADRs 0006 (segment
  v1) and 0007 (WAL poisoning) written from the agents' drafts; formats
  index updated.
- **Why:** these tasks touch disjoint crates/modules, so running them
  side by side saved wall-clock time. Timings were taken only after the
  merge, so no number was measured while four builds shared the CPU.

## 2026-10-28 — Write-ahead log v1 (task 3.1)
- **What:** `storage::wal`: directory of LSN-named segment files; records
  (Upsert with opaque payload, Delete) with a 17-byte header (crc, len, kind,
  lsn) and CRC32 over everything but the crc; versioned file header
  (`NPNT_WAL`, v1, CRC) published atomically via tmp+rename+dir fsync;
  sync policies Always / Batch (group commit, default) / Interval; torn-tail
  recovery that truncates at the first bad record and drops later files;
  `rotate`/`remove_before`; `failpoints` feature for crash tests. Format in
  docs/formats/wal.md.
- **Why:** Phase 3 durability: acknowledged writes must survive a crash.
  Verified that Rust 1.98 std's `sync_data` already issues `F_FULLFSYNC` on
  macOS, so no libc call is needed.

## 2026-10-28 — Segment file format v1 and HNSW/Flat serialization (task 3.2)
- **What:** `storage::segment`: versioned, little-endian segment file (64 B
  header; vectors, ids and index blob at 64-byte-aligned offsets; 96 B footer
  with per-section CRC32s, section table, footer CRC, trailing magic), atomic
  publish (tmp, sync_data, rename, dir fsync), and `Segment::open` that mmaps,
  checks every CRC and the padding, and reads vectors and ids in place.
  `Hnsw`/`Flat` `to_bytes`/`from_bytes` with their own blob version and full
  validation on load. A committed v1 fixture, tests covering every bit flip
  and truncation, a proptest that corrupt blobs never panic, and an HNSW →
  segment → reopen → identical-search test. Docs in docs/formats/segment.md.
- **Why:** Phases 3 and 6 need sealed segments that survive restarts and
  that a reader can trust: damage gives a typed error, never a panic or an
  out-of-bounds read.

## 2026-10-28 — k-means, product quantization, int8 scalar quantization (tasks 5.1–5.3)
- **What:** `quant` crate: Lloyd's k-means with k-means++ seeding (rayon,
  sampled training, early stop, empty clusters re-seeded from the farthest
  points, bit-identical at any thread count); 8-bit PQ with batch encode,
  decode and ADC distance tables; int8 SQ (per-dim offset, shared scale)
  ranked with `kernels::l2_sq_i8`. An ignored SIFT1M quality test. ADR 0004.
- **Why:** building blocks for IVF-PQ + re-rank (Phase 5). The numbers show
  ADC alone can't reach 0.9 recall, but PQ candidates plus a re-rank can.

## 2026-10-28 — Payload store and filter expressions (tasks 7.1–7.2)
- **What:** `crates/engine`: `payload::PayloadStore` (one typed column per
  field; dictionary-encoded keywords with a bitmap of rows per term; dense
  numeric columns plus presence bitmaps), `filter::Filter` (eq/in/range/
  exists/and/or/not, serde JSON shape) and `evaluate` → `RoaringBitmap`, plus
  `selectivity`. Docs in `docs/filters.md`, ADR 0005.
- **Why:** Phase 7 metadata filtering. The bitmap is what 7.3 (filtered
  HNSW/IVF) and 7.4 (brute-force planner) consume. A proptest checks
  `evaluate` against a row-by-row reference evaluator.

## 2026-10-28 — Index trait, flat, HNSW, sweeps; Phase 2 accepted (2.1–2.4)
- **What:** `crates/index`: `Index` trait (`add`/`search`, three metrics),
  rayon-parallel `Flat`, and `Hnsw` (parallel build, selection heuristic,
  O(1)-reset visited set). `bench`: `--index hnsw`, `--m`, `--ef-construction`,
  `--ef` sweep, `--csv`. ADR 0003; curve in `docs/results/sift1m-phase2.csv`.
- **Why:** Phase 2. SIFT1M: recall@10 0.9632 at p50 0.106 ms (target ≥ 0.95
  at ≤ 2 ms), build 44.4 s (target ≤ 10 min). A test caught a race that left
  nodes unlinked on upper layers; fixed with hnswlib's approach (ADR 0003).

## 2026-10-28 — Bench harness; Phase 1 accepted (task 1.7)
- **What:** `bench` binary (`--dataset siftsmall|sift1m --index flat --k
  --queries`): recall@k (tie-aware + id match), p50/p95/p99, single-thread
  QPS, peak RSS and phys_footprint. `index::flat::search` (exact top-k heap),
  `storage::sys::peak_memory`. libc narrowed to `storage` only (needs
  `unsafe`; ADR 0001 amended). ADR 0002 for the recall definition.
- **Why:** task 1.7 and Phase 1 acceptance. SIFT1M brute force: recall@10
  1.0000 tie-aware, 0.9994 by id match; every mismatch is an equal-distance
  tie (ADR 0002). Results in `docs/RESULTS.md`.

## 2026-10-28 — Kernel benchmarks (task 1.6)
- **What:** `crates/kernels/benches/kernels.rs` (criterion): scalar vs NEON for
  all five kernels at dims 96/128/768. First numbers in `docs/RESULTS.md`.
- **Why:** task 1.6 and half of Phase 1's acceptance: f32 L2 is 4.4× scalar at
  dim 128 (target ≥4×). int8 gains are small because LLVM already vectorizes
  the integer reference loops; recorded as such.

## 2026-10-28 — NEON distance kernels (task 1.5)
- **What:** `crates/kernels/src/neon.rs`: f32 L2/dot (16 floats per iteration,
  4 accumulators) and cosine (via three `dot`s); int8 dot via SDOT and int8 L2
  via SABD + UDOT. The crate root re-exports the best version for the build
  (`cfg(target_arch)` / `cfg(target_feature = "dotprod")`); proptests against
  `scalar` (f32 within 1e-4 of the sum's scale, int8 exact).
- **Why:** task 1.5. Arithmetic intrinsics are safe inside
  `#[target_feature]` functions on 1.98, so `unsafe` is limited to loads and
  the one call into each such function.

## 2026-10-28 — Scalar distance kernels (task 1.4)
- **What:** `crates/kernels/src/scalar.rs`: `l2_sq`, `dot`, `cosine_distance`
  (f32) and `l2_sq_i8` → `u32`, `dot_i8` → `i32`. Plain sequential loops;
  panic on length mismatch; cosine with a zero vector is defined as 1.
- **Why:** task 1.4. They're the reference the NEON kernels (1.5) are
  proptested against and the baseline for the ≥4× speedup target. The int8
  return types are exact up to dim 65,536 (tested at the extremes).

## 2026-10-28 — Dataset downloader (task 1.3)
- **What:** `tools/fetch-dataset.sh siftsmall|sift1m [--mirror hf]` (curl +
  tar, resumable, checks every file's size); `#[ignore]`d test loading the
  real siftsmall. siftsmall and SIFT1M are in `~/data/Nearpoint/`.
- **Why:** task 1.3. The canonical texmex FTP runs at ~30 KB/s (SIFT1M ≈ 1.5 h),
  so `--mirror hf` pulls the extracted files from a third-party Hugging Face
  copy (`qbo-odp/sift1m`) in ~2 min. It's trusted only after the size checks
  and after brute force (1.7) reproduces its ground truth exactly.

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
