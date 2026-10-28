# 0004: k-means and quantizer design

Date: 2026-10-28. Status: accepted.

## Context
Phase 5 needs k-means (PQ codebooks, later the IVF coarse quantizer),
8-bit PQ for candidate generation, and int8 SQ for re-ranking with 4x less
memory than f32. Training must be reproducible from a seed regardless of
the rayon thread count.

## Decision
- **k-means update is parallel over clusters**, not per-thread partial
  sums. Points are bucketed by cluster with a stable counting sort; each
  cluster sums its own points in f64, in index order. Fixed summation
  order means bit-identical centroids at any thread count, with O(n)
  scratch instead of O(chunks · k · dim) (64 MB per chunk for an IVF
  coarse quantizer with k = 65 536, dim = 128).
- **Empty clusters take the points farthest from their centroid** (from
  the last assignment, distinct points, ties by id). This can't raise the
  inertia, and it targets the worst-served region; splitting the largest
  cluster needs a second pass and a perturbation heuristic.
- **k-means++ runs on the training sample** (≤ `max_train` points), so
  seeding is O(max_train · k · dim) and the sample bounds it.
- **SQ uses a per-dimension offset and one shared scale** (largest
  per-dim range / 255), plain min/max on the sample, clamp outside. With
  a shared scale, `l2_sq_i8` on raw codes is exactly proportional to the
  L2 between reconstructions, so re-ranking uses the NEON int8 kernel
  directly. On SIFT1M the shared scale costs nothing (all dims span ~0–200).
- **PQ codes are `m` bytes per vector, stored back to back**; the ADC table
  is subspace-major (`j * 256 + c`), matching what a 4-bit fast-scan
  repack (task 5.5) reads.

## Alternatives
- Per-thread partial sums + ordered reduce: deterministic too, but the
  scratch grows with k and the chunk count.
- Per-dimension SQ scales: tighter for data with very uneven ranges, but
  the code distance becomes a weighted sum the int8 kernel can't do.
- Percentile clipping for SQ: helps heavy-tailed data; SIFT has no outliers
  worth clipping, and clamping already bounds out-of-range values.
