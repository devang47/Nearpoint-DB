# 0009: IVF-PQ index, 4-bit fast-scan and re-ranking

Date: 2026-10-28. Status: draft (tasks 5.4–5.6).

## Context
Phase 5 acceptance: SIFT10M at recall@10 ≥ 0.9 with < 25% of HNSW's
phys_footprint. HNSW at 10M is about 6.5 GB (f32 vectors 5.12 GB,
layer-0 links 10M × 33 × 4 B = 1.32 GB, plus upper layers and locks), so
the budget is about 1.6 GB. Phases 8 (100M) and 9 (1B, compressed only)
reuse this index, and 7.3 needs filters applied while scanning lists.
ADR 0004 measured on SIFT1M that PQ ADC alone tops out at 0.53 (m=16) /
0.73 (m=32) recall@10, so re-ranking is required.

## Decision
- **Coarse quantizer:** seeded k-means (`quant::KMeans`) on a uniform
  sample of at most `max_train` vectors (default 262 144, i.e. 64 per
  cell at nlist 4096). Probing computes all `nlist` centroid distances
  and keeps the `nprobe` nearest.
- **PQ on residuals** (vector minus its cell centroid), one ADC table per
  probed list built from the query's residual. Measured on SIFT1M (nlist
  1024, m=16, 8-bit, no re-rank): residual 0.562 / 0.573 recall@10 at
  nprobe 16 / 64, raw vectors 0.535 / 0.545. The cost is a table per list
  instead of per query; with transposed codebooks (`[subspace][component]
  [centroid]`) the table build is element-wise and vectorizes, which
  matters at `dim / m` of 2–8.
- **List layout:** per cell, `ids: Vec<u32>` and codes in the same order.
  8-bit: `m` bytes per vector. 4-bit: fast-scan blocks (below). Lists are
  grown with exact reservations per `add` batch, so there is no amortized
  2x slack.
- **Fast-scan block layout (nbits = 4):** 32 vectors per block; a block is
  `m` groups of 16 bytes; in group `j`, byte `b` holds sub-code `j` of
  vector `b` (low nibble) and of vector `b + 16` (high nibble). Per group
  the NEON kernel does one load, a mask and a shift, two `vqtbl1q_u8`
  lookups into that sub-quantizer's 16-entry table, and four widening
  u16 adds. The per-list float table is quantized to u8: each row minus
  its minimum (the sum of minimums is a bias), one shared scale so that
  the widest row spans 0..=255. `m <= 256` is enforced, so a u16 sum
  (at most 256 × 255) never overflows. The adds are therefore plain
  wrapping adds, not saturating: nothing can saturate, and scalar and NEON
  match exactly (proptest, including m > 257 where both wrap).
- **Filter:** checked per candidate after the distance test (a candidate
  that wouldn't enter the heap never costs a bitmap lookup), so filtered
  ids are skipped while scanning, never post-filtered.
- **Re-rank storage: int8 SQ codes by id, always kept** (`quant::Sq`,
  trained on the same sample). With `rerank > 0` the best `max(k, rerank)`
  PQ candidates get the asymmetric distance (f32 query converted to code
  units once, then each code widened to f32 and one NEON `l2_sq`), and
  those distances are returned.
- **Metrics:** L2 only; `IvfPq::new` returns `InvalidConfig` otherwise.

Memory at SIFT10M (nlist 4096, m=16, 8-bit, 128 dims):

| Component | Size |
|-----------|------|
| SQ int8 codes (re-rank) | 10M × 128 B = 1.28 GB |
| PQ codes | 10M × 16 B = 160 MB |
| ids | 10M × 4 B = 40 MB |
| coarse centroids | 4096 × 512 B = 2 MB |
| PQ codebooks (+ transposed copy) | 16 × 256 × 8 × 4 B × 2 = 256 KB |
| **Total** | **≈ 1.48 GB (≈ 23% of HNSW's ≈ 6.5 GB)** |

4-bit m=32 has the same 16 B/vector. Training transients (sample,
residuals, PQ's sub-sample) are about 0.35 GB and are freed before the
lists fill.

## Alternatives
- **PQ on raw vectors:** one ADC table per query and comparable u8 tables
  across lists, but 0.03 lower recall at equal code size on SIFT1M.
- **Precomputed per-list table terms** (FAISS's `use_precomputed_table`):
  removes most per-list table work but costs `nlist × m × ksub` floats
  (64 MB at nlist 4096, m=16, 8-bit). Not needed while the transposed
  table build is cheap; revisit if probing many lists dominates latency.
- **f32 re-rank vectors:** best re-rank quality (ADR 0004: 0.9995 vs
  0.984 at top-1000), but 5.12 GB at 10M, far over budget. Re-ranking
  from an mmap'd f32 file on disk is the Phase 8/9 option.
- **SQ codes stored per list, next to the PQ codes:** better locality for
  the re-rank reads, but candidates would need to carry (list, offset) and
  each list would hold two code arrays. Re-ranking touches at most a few
  hundred vectors per query, so by-id random reads are fine.
- **FAISS's two-sub-quantizers-per-byte fast-scan layout:** fewer loads
  per block, but two table registers per group and a more involved
  repack. The one-table-per-group layout is simpler and keeps appends a
  nibble write.
- **Saturating u16 accumulation:** only matters if sums can exceed 65535;
  enforcing `m <= 256` rules that out, and saturated sums would mis-rank
  anyway.
- **Optional SQ codes (compressed-only mode):** needed for Phase 9 (1B ×
  128 B = 128 GB won't fit); add the switch then.
