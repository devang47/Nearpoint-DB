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
