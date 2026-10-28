# Results

Measured numbers only, release builds on the Mac mini, newest first.
Missed targets are recorded as missed. Never tune the benchmark to hide one.

recall@10 is tie-aware (a hit is a result whose true distance is <= the
10th ground-truth distance; `docs/adr/0002-tie-aware-recall.md`); the
id-match figure is always recorded beside it. Warm cache unless noted.

| Date | Task | Dataset | Index + params | Seed | recall@10 | p50 | p99 | QPS | Peak RSS | Peak phys_footprint | Disk | Build time | Commit |
|------|------|---------|----------------|------|-----------|-----|-----|-----|----------|---------------------|------|------------|--------|
| 2026-10-28 | 2.3 | SIFT1M | hnsw M=16 efc=200 **ef=64** | 42 | **0.9632** (0.9625) | **0.106 ms** | 0.138 ms | 9512 | 1257 MiB | 1256 MiB | – | **44.4 s** | 5839104 |
| 2026-10-28 | 2.3 | SIFT1M | hnsw M=16 efc=200 ef=128 | 42 | 0.9894 (0.9888) | 0.192 ms | 0.244 ms | 5367 | 1257 MiB | 1256 MiB | – | 44.4 s | 5839104 |
| 2026-10-28 | 2.3 | SIFT1M | hnsw M=16 efc=200 ef=256 | 42 | 0.9972 (0.9967) | 0.346 ms | 0.449 ms | 2988 | 1257 MiB | 1256 MiB | – | 44.4 s | 5839104 |
| 2026-10-28 | 2.2 | SIFT1M | flat, exact, 10 threads per query | – | 1.0000 (0.9994) | 4.65 ms | 5.28 ms | 213 | 988 MiB | 988 MiB | – | 0.1 s | 5839104 |
| 2026-10-28 | 1.7 | SIFT1M (1M base, 10k queries) | flat, exact, 1 thread | – | **1.0000** (id match 0.9994) | 7.82 ms | 9.39 ms | 126.7 | 499 MiB | 499 MiB | – | – | 8c51523 |
| 2026-10-28 | 1.7 | siftsmall (10k base, 100 queries) | flat, exact, 1 thread | – | 1.0000 (id match 1.0000) | 0.16 ms | 0.28 ms | 5686 | 7 MiB | 6 MiB | – | – | 8c51523 |

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
