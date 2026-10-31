# 0013: Memory and disk budget for 100M vectors

Date: 2026-10-31. Status: accepted.

## Context
Phase 8 asks for 100M vectors with every feature on at phys_footprint
≤ 10 GB, on a box with ~66 GB of free disk. At BIGANN's dim 128, the
Phase 6 layout cost per row:

| Component | Where | Bytes/row | 100M |
|-----------|-------|----------:|-----:|
| f32 vectors of IVF-PQ segments | file (mapped) | 512 | 51.2 GB disk |
| int8 re-rank (SQ) codes | RAM (copied out of the blob on load) | 128 | 12.8 GB |
| IVF lists: u32 id + 16-byte PQ code | RAM | 20 | 2.0 GB |
| id map `HashMap<u64, (u32, u32)>` | RAM | ~19 (×1.5 while it doubles) | 1.9–3.4 GB |
| per-segment row → id `Vec<u64>` | RAM | 8 | 0.8 GB |
| `tag` keyword payload (codes + offsets) | RAM | ~8 | 0.8 GB |

RAM alone is ~18 GB, and the f32 section alone doesn't fit next to the
12.8 GB download. A rewrite also copies its rows as f32 (up to
`MERGE_BYTES`, 2 GiB), and since the per-collection maintenance threads
(2026-10-31) two collections could each hold one at once.

## Decision
- **SQ codes stay in the mapped file.** `IvfPq::from_shared` parses the
  blob of a mapped segment and keeps a range into it for the SQ codes
  instead of copying them (`SqCodes::Shared`); re-ranking reads
  `rerank` rows per segment per query from the page cache. Clean file
  pages aren't counted in phys_footprint, and macOS drops them under
  pressure. Freshly built IVF-PQ segments are reloaded from their file
  the same way. Search results are identical (test).
- **`exact_vectors` collection option** (default `true`, today's
  behaviour). `false`: IVF-PQ segments are written with element type 1
  (no vector section; the format has allowed it since segment v2), and
  `get` / `with_vector` / rewrites use the SQ code decoded (within half a
  step per component). The int8 codes already *are* the re-rank vectors,
  so this costs no search quality; it drops 4·dim bytes per row of disk
  and the write amplification of copying them on every rewrite. HNSW
  segments (small, in RAM) keep exact f32 either way; a row rewritten
  from a decoded IVF-PQ row stays decoded.
- **One rewrite at a time across collections** (a process-wide mutex
  around `rewrite`; seals don't take it), so at most one f32 merge copy
  plus one index build exists.
- **`--memory-limit-mib`**: upserts get `503` while phys_footprint is
  above it; searches, gets and deletes still run. A database at its
  memory cap should refuse writes, not swap the box.
- **`/metrics`** reports each component's estimate per collection, mapped
  bytes, phys_footprint now and at peak, and refused writes (docs/api.md).

Budget at 100M × dim 128 with `exact_vectors: false` (estimates; the
measured numbers are in RESULTS.md):

| Component | RAM |
|-----------|----:|
| IVF lists + quantizers | 2.0 GB |
| id map (steady / while doubling) | 1.9 / 3.4 GB |
| row → id arrays | 0.8 GB |
| payloads (one keyword) | ~0.8 GB |
| mutable segment + WAL copy (100k rows) | ~0.1 GB |
| one rewrite: f32 copy of ≤ 4M rows + its index | ≤ 2.2 GB |
| **total, worst case** | **~9.3 GB** |

Disk: ~170 bytes/row (SQ 128, PQ 16, ids 8 + 4, payloads ~10) ≈ 17 GB.

## Alternatives
- **f16 or u8 vector sections**: f16 halves the disk (25.6 GB) but is
  still 2× the SQ codes it duplicates; u8 is exact only for u8 sources.
- **Keep f32 and re-rank from it** (planned for the inner-product ceiling,
  ADR 0009): needs the 51 GB; left for collections that fit.
- **A per-component limit** (e.g. cap the id map): the components grow
  together with the row count, so one process-wide limit on what macOS
  actually counts is simpler and can't be gamed by a bad estimate.
- **Map the IVF lists too**: another 2 GB; not needed to fit, and the
  scan reads every probed list in full (page faults per query on a cold
  cache). Next step if a bigger collection needs it.

## Consequences and limits
- Cold-cache searches fault in `rerank` SQ rows per segment (8.2 measures
  that). Opening a segment still reads its whole blob for the CRC
  (~0.6 GB per 4M-row segment): ~17 GB read on a 100M restart.
- With `exact_vectors: false` a row's vector is approximate forever after
  its first IVF-PQ build, including if it is later rebuilt as HNSW.
