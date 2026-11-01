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

## 2026-11-01 — CI: gate the int8 test strategy on dotprod
- **What:** `i8_pair` in `crates/kernels/src/neon.rs` tests now carries
  `#[cfg(target_feature = "dotprod")]`, like the tests that use it.
- **Why:** GitHub's macOS runner VM doesn't expose dotprod under
  `target-cpu=native`, so those tests compiled out and the unused helper
  failed clippy `-D warnings`; every CI run had been red since round 3.

## 2026-11-01 — Phase 10: summary table and README
- **What:** RESULTS.md opens with a summary by scale (1M / 10M / 100M /
  1B: recall, p50 / p99, QPS for one client, RAM, disk, build time, from
  the recorded runs); README rewritten: status, results, architecture
  diagram (segments, billion mode, maintenance threads, prefetch),
  quickstart for the server, benchmarks and billion mode. PLAN 10.1–10.2
  ticked.
- **Why:** the roadmap's last phase: the numbers in one place, and a
  README that matches what the code does now (the old one stopped at
  Phase 2).

## 2026-11-01 — Phase 9 met: BIGANN 1B in billion mode
- **What:** streamed and encoded the BIGANN 1B base (one attempt, 7.7 h),
  sorted (7 passes, 154 s), verified, evaluated standalone, attached
  through the API and measured with 10M deletes. RESULTS.md, PLAN 9.2–9.4
  ticked, logs `docs/results/bigann1b-*.log`.
- **Why it matters:** p99 86.8 ms at nprobe 64 through the API, peak disk
  56.06 GB, recall@10 0.666 against the official ground truth, server
  phys_footprint ≤ 164 MiB. Recall is the 32-byte codes' ceiling.

## 2026-11-01 — 9.5: billion build as an engine segment
- **What:** `SegIndex::Billion` (quantizers + mapped files);
  `Collection::attach_billion` and `POST /collections/{name}/attach_billion`;
  implicit ids `0..n` resolved by `base_row` in get / delete / upsert /
  len / open; search with tombstones and filter as a skip predicate and
  prefetch hints for probed lists; `get` reconstructs from codes. `lists`
  gains an id → list section (for `get`); `bigload --filtered 0`.
- **Why:** the billion index is part of the database, not a side tool:
  one API, and deletes / replacements work on its rows without ~28 GB of
  per-row RAM. BIGANN 10M attached through the API: recall@10 0.727
  (nprobe 64, rerank 100) as standalone, p50 0.63 ms, phys_footprint 139
  MiB, 0 violations after 100k deletes.

## 2026-11-01 — Phase 9: billion-mode train / stream build / sort (9.1–9.3)
- **What:** `index::billion` (two-level k-means coarse quantizer with an
  HNSW over the centroids, PQ + refinement PQ, parallel encode, search
  over mapped lists with a refine re-rank); `storage::billion` (append-only
  `stream` / `refine` with CRC'd `progress` for resume, multi-pass
  counting sort into `lists`, footer-checked mapped reader); `billion`
  binary (train, status, build from stdin, sort, eval);
  `tools/billion-{stream,build}.sh` (chunked parallel range download piped
  in row order, resume loop). ADR 0014, docs/formats/billion.md.
- **Why:** 1B rows don't fit the engine's per-row RAM or its rewrite
  disk budget; this streams the 128 GB once, never stores it, and peaks
  at 54 GB on disk. BIGANN 10M through the whole pipeline: 74k rows/s
  encode, recall@10 0.727 at nprobe 64 / rerank 100 (p99 0.78 ms), 0.763
  at nprobe 256 (the 32-byte codes' ceiling).

## 2026-11-01 — 8.2: cold-cache benchmarks at 100M (Phase 8 complete)
- **What:** ran `.context/cold100m.sh` (purge before start and before
  each pass, dataset reads with `F_NOCACHE`) on the 100M data directory;
  results in RESULTS.md, 8.2 ticked in PLAN.md.
- **Why it matters:** cold and warm search are the same within noise
  (p50 ~47 ms, unfiltered p99 80–93 ms): the re-rank's random SQ reads
  miss the page cache either way at this size. Cold restart 59.8 s, bound
  by the blob CRC pass; that is the next thing to make lazy.

## 2026-10-31 — 8.2 prep: F_NOCACHE dataset reads, bigload --purge
- **What:** `storage::sys::set_nocache` (fcntl `F_NOCACHE`); `bigload`
  reads every dataset file with it and, with `--purge`, runs
  `sudo -n /usr/sbin/purge` before each measured pass.
- **Why:** 8.2 measures cold-cache search; `bigload` streams the 12.8 GB
  base for the filtered ground truth, which would otherwise refill the
  page cache with dataset pages and evict the segments under test.

## 2026-10-31 — 8.3 / Phase 8: BIGANN 100M through the API (met); re-rank prefetch
- **What:** `IvfPq` asks its shared buffer to prefetch every re-rank
  candidate's SQ row before scoring (`Shared::will_need`; the engine maps
  it to `Segment::will_need_blob`, `madvise(MADV_WILLNEED)` via memmap2);
  `nearpoint_rss_peak_bytes` in `/metrics`. Results in RESULTS.md (logs
  `docs/results/bigann{10m,100m}-*.log`), PLAN 8.1/8.3 ticked, ADR 0013
  addendum.
- **Why:** at 100M the first measurement missed p99 (123 ms; 194 ms after
  deletes): ~3,900 SQ rows per query faulted in one at a time from a map
  bigger than the page cache. With the prefetch: recall@10 0.977, p50 50
  ms, p99 92.9 ms, peak phys_footprint 8,965 MiB: acceptance met, p99
  with a thin margin and cold cache (8.2) still unmeasured.

## 2026-10-31 — 8.1: memory budget for 100M, /metrics, memory limit
- **What:** IVF-PQ SQ codes stay in the mapped segment file
  (`IvfPq::from_shared`) instead of being copied to RAM; collection option
  `exact_vectors` (default true; false = IVF-PQ segments without f32,
  `get` decodes the int8 code); one rewrite at a time across collections;
  `/metrics` (hand-written Prometheus text: RAM by component, mapped bytes,
  phys_footprint now/peak); `--memory-limit-mib` (upserts 503 above it).
  Builds: a sealed segment with no IVF-PQ input borrows an existing IVF-PQ
  segment's quantizers; with quantizers to reuse, IVF-PQ from the first
  tier merge (4 seals) up; `exact_vectors: false` rewrites skip decoding
  the rows whose codes they keep. `bigload` (API loader + recall/latency
  measure for big-ann files), `fetch-dataset.sh bigann100m`. ADR 0013.
- **Why:** at 100M × 128 the Phase 6 layout needed ~18 GB RAM (12.8 GB of
  it SQ codes) and 51 GB of f32 on disk. BIGANN 10M dry runs through the
  API: fresh IVF-PQ trainings (90–110 s) and HNSW tier merges (75 s) made
  seals wait and the in-RAM segment grow to 1.6M rows; borrowing
  quantizers cut those builds to 8–33 s.

## 2026-10-31 — Payloads v2: columnar binary section
- **What:** segment payload sidecars are written as v2: per column (name
  order) kind, present-rows bitmap, then typed values (i64/f64, a true-rows
  bitmap, or keyword dictionary + offsets + codes); keyword postings are
  rebuilt on open. v1 (JSON text per row) stays readable. `payload.rs`
  `encode`/`decode`, fixture `testdata/payloads-v2`,
  `docs/formats/payloads.md`.
- **Why:** every seal, rewrite and open serialized or re-parsed each
  row's JSON; at 100M rows that is minutes. Ad-hoc 1M-row run: encode
  520 → 4 ms, decode 505 → 33 ms, 42 → 23 MB. Side effect: an inferred
  column whose rows were all dropped now survives a reopen.

## 2026-10-31 — Segment open: lazy CRC check of the f32 vectors section
- **What:** `Segment::open` checks header, footer, ids and index blob
  CRCs but no longer the vectors section; `Segment::verify_vectors` does.
  HNSW segments verify at load (they copy the vectors anyway); IVF-PQ
  segments (vectors stay mmap'd) are checked by `maintain`, one per step,
  before planning, and an unverified or corrupt one is never a rewrite
  input. ADR 0012 addendum, `docs/formats/segment.md`.
- **Why:** known issue 5: a multi-GB segment was read whole at open. A
  rewrite copies f32 rows into a fresh file with valid CRCs, so it must
  never read unchecked ones.

## 2026-10-31 — Db: one maintenance thread per collection
- **What:** each collection gets its own background thread running
  `maintain` (spawned on open/create; joined on drop, and on Db drop
  before the LOCK is released). `db.rs`, ADR 0012 Threading.
- **Why:** one shared thread let a long build in one collection delay
  every other collection's seals. CPU stays capped by the shared build
  pool; RAM is not (two big merges in different collections can now
  overlap), noted in the ADR for Phase 8.1's memory budget.

## 2026-10-31 — Phase 6 met: 4-hour soak at 71bbdd4
- **What:** reran the Phase 6 soak (same settings, pre-registered
  30-window check) at the absorb + nprobe-per-nlist commit; results in
  RESULTS.md, CSV `docs/results/soak-4h-final.csv`; Phase 6 status set to
  met in PLAN.md.
- **Why it matters:** recall −0.5% and p99 −2.3% (first vs last hour
  −0.0% / −2.5%), 0 errors, 0 violations, 0 reconnects, RSS flat. Closes
  the p99 miss of the two earlier 4-hour runs (+41%, +12.3%).

## 2026-10-31 — IVF-PQ nprobe scaled per segment with its nlist
- **What:** `Collection::search` gives each IVF-PQ segment
  `ceil(nprobe · nlist / √ivf_rows)` probes (`segment_nprobe`, unit
  test); `nprobe` now means lists probed in an `ivf_rows`-row segment
  (docs/api.md). ADR 0012 addendum, RESULTS.md, CSV
  `docs/results/soak-nprobe-30m.csv`.
- **Why:** after absorb, steady state is one big IVF-PQ segment, and a
  fixed per-segment nprobe covered a smaller share of it (−1.6% recall).
  30-minute soak: recall 0.749 → 0.792 at mean p99 1.53 ms (target
  ≥ 0.765 at ≤ 1.6 ms), for 7.6% less throughput.

## 2026-10-30 — Compaction absorbs small segments into IVF-PQ, keeping codes
- **What:** `IvfPq::retain` (a copy with only the given rows' existing
  codes, 8-bit); rewrites keep the biggest IVF-PQ input's codes and encode
  only the rest; new plan step 0 folds 3 small sealed HNSW segments into
  the IVF-PQ segment with the most dead rows; INFO logs with per-phase
  timings for seals, rewrites and builds. ADR 0012 addendum, RESULTS.md.
- **Why:** the logs showed the soak's p99 windows came from HNSW tier
  merges (7–10 s) and fresh IVF-PQ trainings (14–19 s), not from dropping
  dead rows. Now ~1.2 s per 5 minutes. Trade-off: recall −1.6% absolute
  (one big IVF-PQ segment with per-segment nprobe), recorded with a
  predicted (not measured) 4-hour result.

## 2026-10-29 — 4-hour re-soak with quantizer reuse: Phase 6 still not met
- **What:** reran the Phase 6 soak at 3300a7d; results and per-window CSV
  in RESULTS.md (`docs/results/soak-4h-reuse.csv`).
- **Why it matters:** recall (−0.5%), memory and correctness hold (0
  errors, 0 violations, 0 reconnects), but p99 drift is +12.3% on the
  pre-registered 30-window check (+14.6% first vs last hour). Hourly mean
  p99 is flat; the median moved when the 300 s maintenance cycle settled
  into a phase covering 3 of 5 windows. Next fix: less background work.

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
