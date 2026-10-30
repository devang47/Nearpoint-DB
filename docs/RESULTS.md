# Results

Measured numbers only, release builds on the Mac mini, newest first.
Missed targets are recorded as missed. Never tune the benchmark to hide one.

recall@10 is tie-aware (a hit is a result whose true distance is <= the
10th ground-truth distance; `docs/adr/0002-tie-aware-recall.md`); the
id-match figure is always recorded beside it. Warm cache unless noted.

| Date | Task | Dataset | Index + params | Seed | recall@10 | p50 | p99 | QPS | Peak RSS | Peak phys_footprint | Disk | Build time | Commit |
|------|------|---------|----------------|------|-----------|-----|-----|-----|----------|---------------------|------|------------|--------|
| 2026-10-28 | HNSW memory | BIGANN 10M | hnsw M=16 efc=200 ef=64, reserve + striped locks (memory-bound, see note) | 42 | 0.9154 (0.9153) | *0.285 ms* | *6.26 ms* | *1045* | 7231 MiB | **8501 MiB** | – | 714.9 s | 8f67ec3 |
| 2026-10-28 | HNSW memory | BIGANN 10M | same, ef=128 | 42 | 0.9666 (0.9665) | *0.291 ms* | *4.47 ms* | *2082* | 7231 MiB | 8501 MiB | – | 714.9 s | 8f67ec3 |
| 2026-10-28 | 4.4 | SIFT1M via HTTP API (1k queries) | hnsw M=16 efc=200 ef=64, bulk-loaded, sealed, server restarted | 42 | id match 0.9574 | 0.266 ms (client) | 0.482 ms | – | 717 MiB (server after reopen) | – | – | load 44.2 s | 8f67ec3 |
| 2026-10-28 | HNSW memory | SIFT1M | hnsw M=16 efc=200 ef=64, reserve + striped locks | 42 | 0.9607 (0.9600) | 0.107 ms | 0.140 ms | 9420 | 1231 MiB | **1231 MiB** | – | 42.9 s | 8f67ec3 |
| 2026-10-28 | HNSW memory | SIFT1M | same, ef=128 | 42 | 0.9879 (0.9872) | 0.194 ms | 0.247 ms | 5305 | 1231 MiB | 1231 MiB | – | 42.9 s | 8f67ec3 |
| 2026-10-28 | 5.4–5.6 | BIGANN 10M (10k queries) | ivfpq nlist=4096 m=16 8-bit **nprobe=32 rerank=100** | 42 | **0.9175** (0.9175) | **0.593 ms** | 1.520 ms | 1486 | 2822 MiB | 2791 MiB | – | 159.9 s | dcabd5e |
| 2026-10-28 | 5.4–5.6 | BIGANN 10M | ivfpq nlist=4096 m=16 8-bit nprobe=64 rerank=200 | 42 | 0.9631 (0.9631) | 1.110 ms | 2.607 ms | 807 | 2822 MiB | 2791 MiB | – | 159.9 s | dcabd5e |
| 2026-10-28 | 5.5 | BIGANN 10M | ivfpq nlist=4096 m=32 **4-bit** nprobe=32 rerank=200 | 42 | **0.9155** (0.9154) | **0.214 ms** | 0.410 ms | 4286 | 2832 MiB | 2813 MiB | – | 131.0 s | dcabd5e |
| 2026-10-28 | 5.5 | BIGANN 10M | ivfpq nlist=4096 m=32 4-bit nprobe=64 rerank=200 | 42 | 0.9414 (0.9414) | 0.332 ms | 0.643 ms | 2773 | 2832 MiB | 2813 MiB | – | 131.0 s | dcabd5e |
| 2026-10-28 | 5 (baseline) | BIGANN 10M | hnsw M=16 efc=200 ef=64 (**swapped**, latency invalid) | 42 | 0.9163 (0.9162) | *1.038 ms* | *43.0 ms* | *261* | 7485 MiB | **10670 MiB** | – | 879.1 s | dcabd5e |
| 2026-10-28 | 5 (baseline) | BIGANN 10M | hnsw M=16 efc=200 ef=128 (swapped) | 42 | 0.9664 (0.9663) | *0.358 ms* | *7.65 ms* | *1164* | 7485 MiB | 10670 MiB | – | 879.1 s | dcabd5e |
| 2026-10-28 | 5.4–5.6 | SIFT1M | ivfpq nlist=1024 m=16 8-bit nprobe=16 rerank=100 | 42 | 0.9167 (0.9159) | 0.170 ms | 0.234 ms | 5720 | 969 MiB | 968 MiB | – | 25.8 s | dcabd5e |
| 2026-10-28 | 5.5 | SIFT1M | ivfpq nlist=1024 m=32 4-bit nprobe=32 rerank=200 | 42 | 0.9540 (0.9532) | 0.127 ms | 0.164 ms | 7738 | 970 MiB | 970 MiB | – | 17.1 s | dcabd5e |
| 2026-10-28 | 7.3 regression | SIFT1M | hnsw M=16 efc=200 ef=64, unfiltered | 42 | 0.9608 (0.9602) | 0.108 ms | 0.195 ms | 8934 | 1922 MiB | 1922 MiB | – | 42.8 s | dcabd5e |
| 2026-10-28 | 7.3 regression | SIFT1M | same, pre-round build (35e7c82), run back to back | 42 | 0.9634 (0.9627) | 0.109 ms | 0.165 ms | 8932 | 1257 MiB | 1256 MiB | – | 44.1 s | 35e7c82 |
| 2026-10-28 | 2.3 (rebuild) | SIFT1M | hnsw M=16 efc=200 ef=64 | 42 | 0.9602 (0.9595) | 0.108 ms | 0.145 ms | 9310 | 1257 MiB | 1256 MiB | – | 43.5 s | e96bb03 |
| 2026-10-28 | 2.3 | SIFT1M | hnsw M=16 efc=200 **ef=64** | 42 | **0.9632** (0.9625) | **0.106 ms** | 0.138 ms | 9512 | 1257 MiB | 1256 MiB | – | **44.4 s** | 5839104 |
| 2026-10-28 | 2.3 | SIFT1M | hnsw M=16 efc=200 ef=128 | 42 | 0.9894 (0.9888) | 0.192 ms | 0.244 ms | 5367 | 1257 MiB | 1256 MiB | – | 44.4 s | 5839104 |
| 2026-10-28 | 2.3 | SIFT1M | hnsw M=16 efc=200 ef=256 | 42 | 0.9972 (0.9967) | 0.346 ms | 0.449 ms | 2988 | 1257 MiB | 1256 MiB | – | 44.4 s | 5839104 |
| 2026-10-28 | 2.2 | SIFT1M | flat, exact, 10 threads per query | – | 1.0000 (0.9994) | 4.65 ms | 5.28 ms | 213 | 988 MiB | 988 MiB | – | 0.1 s | 5839104 |
| 2026-10-28 | 1.7 | SIFT1M (1M base, 10k queries) | flat, exact, 1 thread | – | **1.0000** (id match 0.9994) | 7.82 ms | 9.39 ms | 126.7 | 499 MiB | 499 MiB | – | – | 8c51523 |
| 2026-10-28 | 1.7 | siftsmall (10k base, 100 queries) | flat, exact, 1 thread | – | 1.0000 (id match 1.0000) | 0.16 ms | 0.28 ms | 5686 | 7 MiB | 6 MiB | – | – | 8c51523 |

- **2026-10-28 HNSW rebuild after the serialization merge:** 0.9602 vs
  0.9632 at ef=64. Same seed, so the node levels are identical, but parallel
  insertion order isn't, so each build is a slightly different graph.
  Expect about ±0.003 recall between builds; latency is unchanged.
- **2026-10-28 Phase 2:** target "recall@10 ≥ 0.95 at p50 ≤ 2 ms, build
  ≤ 10 min" **met** (0.9632 at 0.106 ms; 44.4 s). Full curve, ef 10–256:
  `docs/results/sift1m-phase2.csv`. Queries run one at a time on one
  thread; the build uses all 10.
- **Memory caveat:** the harness keeps its own copy of the base vectors
  (488 MiB) to recompute true distances, so the index's own share is peak
  minus ~0.5 GB: ~0.77 GB for HNSW (vectors 488 MiB + layer-0 links
  126 MiB + upper layers, locks, per-thread visited sets).
- **Flat is memory-bound:** 10 threads are only 1.7× faster than 1
  (7.82 → 4.65 ms). Each query streams 512 MB, and 512 MB / 4.65 ms ≈
  110 GB/s, close to the M4's ~120 GB/s. More cores won't help it;
  smaller vectors (int8, PQ) will.
- **2026-10-28 SIFT1M flat:** Phase 1 target "brute force matches ground
  truth exactly" **met**. All 56 queries with an id mismatch return exactly
  the ground truth's 10 distances; the two just break equal-distance ties
  differently. Also validates the HF mirror (`qbo-odp/sift1m`): its base,
  query and ground truth agree for all 10k queries. Mac mini M4, 16 GB.

- **2026-10-28 Phase 5 (IVF-PQ):** target "SIFT10M recall@10 ≥ 0.9 at
  < 25% of HNSW's phys_footprint" **met, the RAM half narrowly**. SIFT10M
  is the first 10M of BIGANN (u8) with big-ann-benchmarks' 10M ground
  truth. Recall ≥ 0.9 at 0.59 ms (8-bit) or 0.21 ms (4-bit fast-scan).
  RAM, like for like (peak minus the harness's 1,231 MiB u8 base):
  IVF-PQ 1,559 MiB vs HNSW 9,438 MiB = **16.5%**. But HNSW's peak is
  inflated by `Vec` regrowth while the bench adds 100k-vector batches
  (SIFT1M HNSW: 1,256 → 1,922 MiB peak after batching was introduced);
  its steady-state size is ~6.6 GB (vectors 5.12 GB + layer-0 links
  1.32 GB + upper layers and locks), estimated, not measured, giving
  **~24%**. Whole-process peaks (both include the harness data): 26%.
  Full sweeps: `docs/results/{sift1m,bigann10m}-ivfpq.csv`.
- **HNSW at 10M does not fit the budget:** whole-process phys_footprint
  peaked at 10,670 MiB (> 10 GB) and the machine swapped (6.7 GB swap
  in use), so its latencies above are invalid; recall stands (0.916 at
  ef 64, lower than SIFT1M's 0.96). Build 879 s. For ≥ 10M-vector
  segments IVF-PQ is the index; a `reserve` before batched `add` would
  remove the regrowth spike (Phase 8).
- **Unfiltered HNSW after 7.3:** no regression. Back to back with the
  pre-round build: p50 0.108 vs 0.109 ms, 8,934 vs 8,932 QPS. (An
  earlier post-merge run showed 0.116 ms under background load.) Peak
  memory differs only because the bench now adds in batches.
- The Mac slept during some runs (`pmset` log); `Instant` stops during
  sleep, so per-query timings are unaffected, but wall times in logs
  are not comparable.

- **2026-10-28 HNSW memory fixes** (`Hnsw::reserve`, 4096 striped locks
  instead of a mutex per node): SIFT1M peak 1,922 → **1,231 MiB** (build +
  index 732 MiB), recall and latency unchanged. BIGANN 10M: peak
  phys_footprint 10,670 → **8,501 MiB**, under the 10 GB budget (build +
  index 7,269 MiB vs ~6.7 GB estimated steady state at ~666 B/vector).
  Build 879 → 715 s. Its latencies are still memory-bound and only
  indicative: with ~8.5 GB in the process on a 16 GB machine, swap stayed
  at ~5.7 GB, and ef=128 (p50 0.291 ms) was barely slower than ef=64
  (0.285 ms) with multi-ms p99s. HNSW for 10M+ vectors stays impractical
  here; IVF-PQ is the large-segment index (Phase 6.2).
- **2026-10-28 API path:** SIFT1M bulk-loaded through `POST
  /points/bulk` (Python client, 50k batches) in 44.2 s (~22.6k vectors/s,
  HNSW built during the load), sealed in ~1 s, reopened in ~0.2 s.
  Recall over HTTP after the restart: id-match 0.9574 at ef 64, the same
  as in-process. Latency is client-side (HTTP + JSON + Python).

## Kernel microbenchmarks

`cargo bench -p kernels`, criterion point estimates per call (ns), two
vectors resident in L1. Speedup = scalar / NEON.

### 2026-10-28 — task 1.6, kernels at commit b2ced1a
Mac mini M4 (Mac16,10), 16 GB, macOS 26.6.1, Rust 1.98.1, `target-cpu=native`.

| Kernel | dim | scalar | NEON | speedup |
|--------|-----|--------|------|---------|
| l2_sq (f32) | 96 | 20.2 | 5.45 | 3.7× |
| l2_sq (f32) | **128** | 31.5 | 7.11 | **4.4×** |
| l2_sq (f32) | 768 | 340 | 49.2 | 6.9× |
| dot (f32) | 96 | 20.3 | 5.71 | 3.6× |
| dot (f32) | 128 | 34.9 | 6.91 | 5.0× |
| dot (f32) | 768 | 340 | 38.9 | 8.7× |
| cosine (f32) | 96 | 43.2 | 11.0 | 3.9× |
| cosine (f32) | 128 | 60.7 | 14.3 | 4.2× |
| cosine (f32) | 768 | 457 | 94.2 | 4.9× |
| l2_sq_i8 | 96 | 2.88 | 1.99 | 1.4× |
| l2_sq_i8 | 128 | 3.70 | 2.30 | 1.6× |
| l2_sq_i8 | 768 | 19.0 | 10.6 | 1.8× |
| dot_i8 | 96 | 2.52 | 1.74 | 1.4× |
| dot_i8 | 128 | 2.13 | 2.08 | 1.0× |
| dot_i8 | 768 | 10.9 | 11.0 | 1.0× |

- **Phase 1 target "NEON ≥4× scalar at dim 128": met** for f32 L2 (4.4×);
  dot 5.0×, cosine 4.2×. Below 4× at dim 96 (3.6–3.9×).
- **int8 "scalar" is not really scalar.** Integer addition is associative,
  so LLVM auto-vectorizes the reference loops (float sums it must leave
  sequential). The int8 speedups are hand-NEON vs compiler-NEON, not vs
  scalar. At dim 768 both int8 dots run at ~140 B/ns, about the L1 load
  bandwidth of two 16-byte streams, so neither can go much faster.
- `dot/scalar/128` had a wide interval (30.8–41.5 ns); re-measure before
  leaning on that single ratio.

## WAL durability (task 3.1)

`cargo test --release -p storage -- --ignored --nocapture sync_`. Mac mini
M4, internal SSD (APFS), quiet machine, commit e96bb03.

### 2026-10-28 — F_FULLFSYNC cost and group commit

| Policy | Writers | Durable appends/s | Notes |
|--------|---------|-------------------|-------|
| Always | 1 | ~270 | per append: mean 3.69 ms, p50 3.96 ms, p99 7.00 ms |
| Batch | 1 | 264 | one F_FULLFSYNC per append, same as Always |
| Batch | 2 | 394 | |
| Batch | 4 | 685 | |
| Batch | 8 | 1354 | |
| Batch | 16 | 2807 | |
| Batch | 32 | 5424 | 20.5× a single writer |

- A real flush to the SSD costs ~4 ms, so one durable writer is capped
  at ~270 writes/s however fast the code is. Group commit scales almost
  linearly with concurrent writers because they share each flush. The
  API should batch (bulk upsert) and accept concurrent requests rather
  than sync per vector.

## Filtered search (tasks 7.3–7.4)

`./target/release/filtered --dataset sift1m --ef 64,128 --plan
auto,graph,brute`. SIFT1M, HNSW M=16 efc=200 seed 42, first 1,000
queries, uniform random filters (seeded), exact filtered ground truth by
brute force over the bitmap. Quiet machine, commit dcabd5e. Full table:
`docs/results/sift1m-filtered.csv`.

### 2026-10-28

| Selectivity | ids | Plan (auto) | recall@10 ef 64 | p50 ef 64 | walk p50 ef 64 | brute p50 | walk recall ef 128 |
|-------------|-----|-------------|-----------------|-----------|----------------|-----------|--------------------|
| 0.1% | 998 | brute | 1.0000 | 0.018 ms | 27.5 ms | 0.018 ms | 0.9990 |
| 0.5% | 4,933 | brute | 1.0000 | 0.098 ms | 8.13 ms | 0.098 ms | 0.9989 |
| **1%** | 9,975 | brute | **1.0000** | 0.196 ms | 4.77 ms | 0.196 ms | 0.9986 |
| 2% | 20,012 | brute | 1.0000 | 0.392 ms | 2.71 ms | 0.392 ms | 0.9985 |
| 5% | 49,821 | walk | 0.9977 | 1.32 ms | 1.32 ms | 3.50 ms | 0.9987 |
| **10%** | 99,722 | walk | **0.9973** | 0.739 ms | 0.739 ms | 7.09 ms | 0.9986 |
| **50%** | 500,179 | walk | **0.9810** | 0.209 ms | 0.209 ms | 14.5 ms | 0.9952 |

- **Phase 7 target "recall@10 ≥ 0.9 at 1%, 10%, 50%": met** (HNSW,
  uniform filters). Correlated filters (matches far from the query) are
  not measured yet and are the known risk for the walk.
- The planner (`filtered² ≤ 8·ef·len`, from distance counts) picks the
  faster plan at every measured point; the time crossover is between 2%
  and 5% (ADR 0010). Without it, a 0.1% filter walks 205k nodes (27 ms).
- One filter serves every query, so small filters stay cache-resident
  and flatter brute force (20 ns per distance at 2%, 70 ns at 5%).
- First run of the 0.5 / ef 64 walk had p99 3.85 ms vs 0.27 ms on the
  repeat: a one-off stall, not reproducible.

### 2026-10-28 — correlated ("far") filters, commit 8f67ec3

`filtered --filter random,far --selectivity 0.01,0.1,0.5`. The far filter
keeps the base vectors farthest from the query set's centroid. ef 64,
1,000 queries. Full table: `docs/results/sift1m-filtered-far.csv`.

| Filter | Selectivity | Plan (auto) | recall@10 | p50 | p99 | Walk distances | Brute p50 / p99 |
|--------|-------------|-------------|-----------|-----|-----|----------------|-----------------|
| random | 1% | brute | 1.0000 | 0.175 ms | 0.20 ms | 45,797 | 0.175 / 0.20 ms |
| random | 10% | walk | 0.9990 | 0.735 ms | 1.04 ms | 8,023 | 7.30 / 7.66 ms |
| random | 50% | walk | 0.9811 | 0.197 ms | 0.26 ms | 2,187 | 14.3 / 17.4 ms |
| far | 1% | brute | 1.0000 | 0.152 ms | 0.18 ms | 293,740 | 0.152 / 0.18 ms |
| far | **10%** | walk | 0.9991 | 4.58 ms | **74.7 ms** | 110,794 | 5.54 / 6.01 ms |
| far | 50% | walk | 0.9906 | 0.223 ms | 10.6 ms | 10,829 | 12.4 / 14.3 ms |

- **Recall holds** for correlated filters (≥ 0.99 at 1/10/50%), so the
  Phase 7 target stands.
- **Latency doesn't:** the planner sizes by cardinality only, so at far
  10% it walks, and some queries visit most of the graph (p99 75 ms,
  12× brute force's). New task 7.5: cap the walk at ~`filtered` distance
  computations, then brute-force.

### 2026-10-29 — walk budget (task 7.5), commit 3bbb0b7

`filtered --dataset sift1m --selectivity 0.01,0.1,0.5 --ef 64,128 --plan
auto,walk,brute --filter far,random`; `auto` = the planner + budgeted walk
(`graph` when it walks), `walk` = the old unbudgeted walk. Same index and
queries as above; quiet machine, `caffeinate -dims`. Full table:
`docs/results/sift1m-filtered-budget.csv`.

| Filter | Sel. | ef | recall@10 auto | p50 auto | **p99 auto** | p99 unbudgeted | brute p99 | Fallbacks / 1000 | Mean dists auto (unbudgeted) |
|--------|------|----|---------------:|---------:|-------------:|---------------:|----------:|-----------------:|-----------------------------:|
| far | 10% | 64 | 0.9997 | 4.41 ms | **18.1 ms** | 71.5 ms | 5.74 ms | 349 | 86,775 (110,269) |
| far | 10% | 128 | 0.9998 | 6.57 ms | **18.1 ms** | 77.0 ms | 5.53 ms | 412 | 99,484 (128,389) |
| far | 50% | 64 | 0.9933 | 0.230 ms | 10.9 ms | 9.63 ms | 13.3 ms | 0 | 10,762 |
| random | 10% | 64 | 0.9986 | 0.713 ms | 1.02 ms | 1.03 ms | 7.30 ms | 0 | 7,965 |
| random | 50% | 64 | 0.9817 | 0.207 ms | 3.38 ms* | 0.258 ms | 14.8 ms | 0 | 2,169 |

- **Far 10% p99: 72 → 18 ms (4×)**, recall unchanged. 35–41% of those
  queries run out of budget and brute-force.
- **Bounded, but 3.1× brute force, not the ≈2× ADR 0010 predicted.** The
  budget counts distances, and a walk distance costs ~2.3× a scan
  distance (random access, heap, visited set): a fallback query spends
  ~12 ms walking 100k distances, then 5.5 ms scanning. Weighting the
  budget (≈ `filtered` / 2.3) would bring it to ~2×; follow-up, not
  tuned here.
- Uniform filters never fall back: same distances and recall as the
  unbudgeted walk. 1% filters are brute force up front, as before.
- \* random 50% ef 64 p99 3.38 ms vs 0.258 ms for the identical
  unbudgeted walk (0 fallbacks, same distances): a one-off stall like
  the 2026-10-28 one; ef 128 is 0.458 vs 0.451 ms.

## IVF-PQ metrics (round 3)

`bench --index ivfpq --metric l2|ip|cosine`, SIFT1M, nlist 1024, 10k
queries, ground truth by brute force with `Flat` for ip/cosine. Quiet
machine, commit 8f67ec3. Full sweep: `docs/results/sift1m-ivfpq-metrics.csv`.

### 2026-10-28

| Codes | nprobe / rerank | L2 | cosine | inner product | p50 (L2) |
|-------|-----------------|----|--------|---------------|----------|
| m=16 8-bit | 16 / 100 | 0.9167 | 0.9171 | 0.7161 | 0.172 ms |
| m=16 8-bit | 32 / 200 | 0.9657 | 0.9668 | 0.8301 | 0.338 ms |
| m=16 8-bit | 64 / 200 | 0.9797 | 0.9806 | 0.8410 | 0.605 ms |
| m=32 4-bit | 32 / 200 | 0.9540 | 0.9547 | 0.7183 | 0.128 ms |
| m=32 4-bit | 64 / 200 | 0.9661 | 0.9664 | 0.7245 | 0.194 ms |

- Cosine tracks L2, but SIFT's norms are nearly constant (506–511), so
  this doesn't stress cosine.
- **Inner product misses 0.9** at every swept setting; more probes barely
  help, more re-rank does (agent run, 1k queries: 0.9247 at 64 / 1000,
  0.9427 at 1024 / 5000). The int8 re-rank is the ceiling. IVF-PQ is for
  L2/cosine collections until an f32 re-rank exists (PLAN 6.2 note).

## Crash recovery (tasks 3.3–3.4)

`cargo test --release -p engine --test crash -- --ignored --nocapture`,
commit dcabd5e. A child process runs random upserts/deletes/seals over 64
ids under `SyncPolicy::Batch`, printing each op once acknowledged; the
parent SIGKILLs it at a random point, reopens, and checks every acked
write. Fresh collection every 25 cycles.

### 2026-10-28

| Cycles | Failures | Lost acked writes | Time | Seed |
|--------|----------|-------------------|------|------|
| 1,000 | 0 | 0 | 138 s | 1793188713750980000 |

- **Phase 3 target "1,000 randomized crash/restart cycles, zero lost
  acknowledged writes": met.** SIGKILL keeps the page cache, so this
  tests the process; power-loss safety rests on F_FULLFSYNC (3.1).

## Server load test (task 4.4, Phase 4 acceptance)

`server --data-dir ~/data/Nearpoint/db-accept` (sync batch) + `loadtest
--secs 1800 --seal-secs 60`, dim 32, 2 writers (each round: insert 50,
replace, delete) at 10 rounds/s, 4 readers (a third filtered). Commit
8f67ec3, quiet machine.

### 2026-10-28

| Duration | Writes | Searches | Errors | Violations | Search p50 / p99 / max | Server RSS |
|----------|--------|----------|--------|------------|------------------------|------------|
| 1,800 s | 34,130 insert, replace, delete requests each; 29 seals | 25.06M (8.35M filtered), ~13.9k/s | **0** | **0** | 0.27 / 0.66 / 34.2 ms | 3 → 603 MiB |

- **Phase 4 target "30 min, no errors, deleted ids never appear": met.**
  A violation is a hit whose delete was acknowledged before the search
  was sent; the agent showed the checker catches a disabled tombstone
  update (668k violations).
- RSS grew linearly (~20 MiB/min) with only ~40k live points: replaced
  and deleted rows stay in RAM until compaction (6.3). A 4-hour soak
  (Phase 6) needs compaction first.
- 1,000-cycle crash test on the merged code: passed, 121 s.

## Phase 6 soak (4 hours, steady state)

`loadtest --secs 14400 --live 1000000 --dim 32 --probe-secs 60
--probe-queries 1000 --drift-windows 30 --seal-secs 0`, collection config
`seal_rows 100000, ivf_rows 400000, max_merge_rows 1600000`, server
`--build-threads 4`, commit 3bbb0b7 + loadtest fixes. 1M live points
(uniform random, dim 32, L2) held constant: 2 writers × 10 rounds/s,
each inserting 50, deleting 50 and replacing 5; 4 readers (a third
filtered). Every 60 s a probe pauses the writers and scores 1,000 k=10
searches (default nprobe 32 / rerank 100, ef 64) against exact top 10
over the client's model. Quiet machine, `caffeinate -dims`. Per-window
CSV: `docs/results/soak-4h-1m-dim32.csv`.

### 2026-10-29

| Hours | p50 (median) | **p99 (median)** | searches/s | recall@10 | RSS median (max) |
|-------|-------------:|-----------------:|-----------:|----------:|-----------------:|
| 0–0.3 | 0.61 ms | 1.30 ms | 6,184 | 0.7783 | 561 (826) MiB |
| 0.3–1 | 0.62 ms | 1.21 ms | 6,110 | 0.7757 | 570 (979) MiB |
| 1–2 | 0.62 ms | 1.47 ms | 5,950 | 0.7740 | 575 (779) MiB |
| 2–3 | 0.65 ms | 1.65 ms | 5,600 | 0.7760 | 405 (836) MiB |
| 3–4 | 0.66 ms | 1.70 ms | 5,500 | 0.7780 | 420 (753) MiB |

- Totals: 13.3 M points inserted and 13.3 M deleted (~925/s each),
  1.3 M replaced, 80.6 M
  searches; **0 violations** (no acknowledged-deleted id returned, live
  count matched the model at all 240 probes); **1 error**: a client-side
  broken pipe at 7,982 s (unattributed; the loadtest now labels errors
  with the request and connection idle time).
- **Recall: met.** Median of the first vs last 30 windows 0.7791 →
  0.7772 (−0.2%). The absolute value is low because uniform random 32-d
  data is hard for both HNSW (ef 64, efc 100) and PQ; drift is the target.
- **p99: missed.** 1.29 → 1.82 ms (+41%, limit 10%). It rises over the
  first ~2 h and then plateaus; p50 only +11%, throughput −13%. Not yet
  attributed: the final segment layout was lost (loadtest drops the
  collection), so the loadtest now logs the sealed segments per window.
  Suspects: more sealed segments at steady state (each IVF-PQ segment
  pays nprobe 32 + rerank 100) and builds on 4 of 10 cores overlapping
  the windows.
- **Memory: fixed.** RSS flat to falling (561 → ~410 MiB median) over
  13.3 M deletes; the Phase 4 run grew ~20 MiB/min with dead rows. Peaks
  (≤ 979 MiB) are merges in flight.

### 2026-10-29 — p99 diagnosis and quantizer reuse

Same settings, `--drift-windows 15`, the loadtest now logging the sealed
segment layout per window. Diagnostic run: 2 h at commit 0b0a75e
(`docs/results/soak-diag-2h.csv`); fix run: 1 h at 9bfe09c
(`docs/results/soak-quantizer-reuse-1h.csv`).

- **Cause:** uniform deletes (~925/s) push every big IVF-PQ segment past
  the 25% dead-row rewrite threshold in ~4.5 min, so the whole 1M set
  was rebuilt every 300 s, for the full 2 h (the cycle never sped up),
  each time retraining k-means + PQ on 4 of 10 cores. p99 median 1.27 ms
  in quiet windows, 1.98 ms in windows overlapping such a build. Hourly
  medians were flat (1.28, 1.30 ms): no degradation over time. The
  drift check compares 15–30-window medians, and the share of build
  windows in each set depends on the cycle's phase and shape (+21% here,
  +41% in the 4-hour run).
- **Fix:** a rewrite or merge with an IVF-PQ input reuses its trained
  quantizers and only encodes (`IvfPq::empty_clone`), retraining only if
  nlist would more than double.

| First hour | Before (retrain) | After (reuse) |
|------------|-----------------:|--------------:|
| windows with p99 > 1.8 ms (of 59) | 20 | **11** |
| p99 mean / max | 1.58 / 2.42 ms | **1.42 / 2.25 ms** |
| p99 median | 1.28 ms | 1.35 ms |
| p50 median | 0.63 ms | 0.60 ms |
| searches/s median | 5,987 | 6,192 |
| recall@10 median | 0.7763 | 0.7846 |
| drift check | +21.0% (2 h) FAIL | −10.5% PASS |
| errors / reconnects | 1 / – | 0 / 0 |

- Spike windows halved, not gone. The rest line up with HNSW merges
  (4 × 100k → ~330k HNSW built from scratch) and the remaining IVF-PQ
  rewrites: background builds still share the CPU with searches.
- **Tried and reverted: builds at macOS utility QoS** (commit 4cf4a72,
  1 h, `docs/results/soak-qos-1h.csv`), so they favor the efficiency
  cores. Windows with p99 > 1.8 ms 11 → 7 and max 2.25 → 2.00 ms, but
  builds ran longer and windows > 1.5 ms went 20 → 24: p99 mean 1.42 →
  1.49, median 1.35 → 1.44 ms, searches/s −5%, drift +5.0% (PASS).
  Peaks traded for duration; not worth the extra `unsafe`.
- The diagnostic run's one error was a keep-alive connection closed
  0 ms after the previous reply (now retried once, counted as a
  reconnect; 0 in the fix run).

### 2026-10-29 — 4-hour re-soak with quantizer reuse, commit 3300a7d

Same settings as the first 4-hour run (`--drift-windows 30`, the
pre-registered check), plus per-window segment logging. Per-window CSV:
`docs/results/soak-4h-reuse.csv`.

| Hour | p99 median | p99 mean | p99 max | windows > 1.8 ms | p50 | searches/s | recall@10 | RSS median (max) |
|------|-----------:|---------:|--------:|-----------------:|----:|-----------:|----------:|-----------------:|
| 1 | 1.36 ms | 1.64 ms | 2.67 ms | 16 / 60 | 0.65 ms | 5,767 | 0.7692 | 463 (922) MiB |
| 2 | 1.31 ms | 1.56 ms | 2.47 ms | 16 / 61 | 0.62 ms | 5,996 | 0.7669 | 480 (795) MiB |
| 3 | 1.58 ms | 1.54 ms | 2.26 ms | 12 / 61 | 0.64 ms | 5,640 | 0.7646 | 496 (833) MiB |
| 4 | 1.56 ms | 1.58 ms | 2.15 ms | 24 / 61 | 0.62 ms | 5,817 | 0.7644 | 483 (766) MiB |

| Drift, first vs last N windows (loadtest's upper median) | recall@10 | p99 |
|-----------------------------------------------------------|----------:|----:|
| **N = 30 (pre-registered)** | 0.7691 → 0.7656 (−0.5%) | 1.36 → 1.53 ms (**+12.3%, FAIL**) |
| N = 60 (first vs last hour) | 0.7694 → 0.7656 (−0.5%) | 1.36 → 1.56 ms (+14.6%, FAIL) |

- 13.2 M inserts and as many deletes; **0 errors, 0 violations, 0
  reconnects**. RSS flat (~480 MiB), recall drift −0.5%.
- **p99 misses by both measures.** Medians of consecutive 30-window
  blocks: 1.36, 1.37, 1.33, 1.27, 1.29, then 1.60, 1.58, 1.49 ms.
- What changed at ~2.5 h is the alignment, not the work: from then on the
  layout settles into a strict 300 s cycle (two 100k seals, then both
  IVF-PQ segments rewritten), and the elevated stretch covers 3 of every
  5 windows (1.76, 2.09, 1.60, then 1.14, 1.26 ms) instead of ~2. Hourly
  mean p99 is flat (1.64, 1.56, 1.54, 1.58 ms) and max falls, so this is
  not degradation; but window medians, which the acceptance uses, move
  with that phase. Not redefining the check: background maintenance
  still costs ~2 of every 5 minutes, and the next fix is doing less of it.

### 2026-10-30 — absorb into IVF-PQ, keep codes (30-minute soaks)

Timings of every background step (new INFO logs) at 3300a7d: seals
70–540 ms; dead-row rewrites of IVF-PQ with reused quantizers 1.2–1.6 s;
but 3–4 × 100k seals merged into a ~330k HNSW (graph from scratch,
7–10 s) that later became a freshly trained IVF-PQ (14–19 s). Those two
were the p99 windows. Fix (ADR 0012 addendum 2026-10-30): `IvfPq::retain`
keeps a segment's live codes; small HNSW segments are absorbed into the
IVF-PQ segment (only new rows encoded). Same soak settings, 30 minutes
(`--drift-windows 15`); CSVs `docs/results/soak-absorb{1,3}-30m.csv`.

| Variant | rewrite cost in steady state | p99 median / mean / max | recall@10 median | drift (15 windows) |
|---------|-----------------------------:|------------------------:|-----------------:|-------------------:|
| reuse quantizers (3300a7d), 4 h run, hours 2–4 | 7–19 s builds every ~5 min | 1.46 / 1.56 / 2.47 ms | 0.7654 | (+12.3% at 30) |
| absorb every seal | ~0.5 s per seal (encode ~95k) | 1.72 / 1.77 / 2.52 ms | 0.7475 | p99 +0.6%, PASS |
| **absorb 3 seals at a time** (steady, t ≥ 9 min) | **~1.2 s per ~5 min** (encode ~260k) | **1.55 / 1.56 / 1.79 ms** | 0.7493 | p99 −10.4%, recall −0.7%, PASS |

- Background CPU per 5-minute cycle drops from ~25 s to ~1.5 s, and the
  worst steady-state window from 2.47 to 1.79 ms. Mean p99 is unchanged
  (1.56 ms); the median rises (1.46 → 1.55 ms) because the spikes are
  gone but the floor is a little higher.
- **Recall costs ~1.6% absolute** (0.765 → 0.749): steady state is one
  ~1M-row IVF-PQ segment (nlist 775 kept from its first training) plus
  0–3 HNSW segments, and `nprobe 32` per segment covers a smaller share
  of one big segment than of two half-size ones. Recall now cycles with
  the absorb (0.72 right after, 0.79 with three HNSW segments). Fix
  candidate: scale nprobe per segment with its nlist.
- **Prediction for the 4-hour check, not measured:** the cycle is ~5
  windows long with no multi-second builds, so 30-window medians of p99
  should sit at ~1.55 ± 0.05 ms at both ends (start a little higher from
  the warm-up trainings), and recall at ~0.75 ± 0.01: p99 drift within
  ±5% and recall within ±2%, a pass, at maybe 80% confidence. The run
  was skipped by decision (Devang, 2026-10-30); Phase 6 stays unticked
  until a 4-hour run measures it.

## Quantization quality (tasks 5.1–5.3)

`cargo test --release -p quant --test sift1m -- --ignored --nocapture`.
Trained on `sift_learn` (100k), ranking all 1M base vectors, 200 queries,
id-match recall@10. Quality only; it doesn't time the search. Reproduced
exactly on a second run (deterministic seeds). Commit e96bb03.

### 2026-10-28

| Method | Bytes/vector | recall@10 |
|--------|--------------|-----------|
| f32 (exact) | 512 | 1.0000 |
| SQ int8, full scan | 128 | 0.9805 |
| PQ m=16, ADC full scan | 16 | 0.5310 |
| PQ m=16, ADC top-100 → f32 re-rank | 16 + rerank | 0.9730 |
| PQ m=16, ADC top-100 → SQ re-rank (f32 query) | 16 + 128 | 0.9630 |
| PQ m=16, ADC top-1000 → f32 re-rank | 16 + rerank | 0.9995 |
| PQ m=16, ADC top-1000 → SQ re-rank (f32 query) | 16 + 128 | 0.9840 |
| PQ m=32, ADC full scan | 32 | 0.7280 |
| PQ m=32, ADC top-100 → f32 re-rank | 32 + rerank | 0.9980 |
| PQ m=32, ADC top-100 → SQ re-rank (f32 query) | 32 + 128 | 0.9830 |

- Reconstruction MSE as a share of total variance: PQ m=16 0.071, m=32
  0.024, SQ int8 0.00003.
- PQ alone can't reach the Phase 5 target (0.9). With a re-rank it clears
  it easily. Re-ranking with SQ int8 caps at ~0.98, the same as SQ alone,
  for a quarter of f32's memory. The Phase 5 design: PQ codes in RAM,
  candidates re-ranked by int8 (RAM or mmap), f32 optional on disk.

