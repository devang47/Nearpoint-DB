# 0014: Billion mode: streamed IVF-PQ with on-disk lists

Date: 2026-11-01. Status: accepted.

## Context
Phase 9: search BIGANN 1B (128-dim u8, 128 GB raw) on the Mac mini, p99
< 100 ms, at most 60 GB of disk at any point, raw data never stored, RAM
≤ ~10 GB. The Phase 8 engine holds 30–90 bytes of RAM per row (id map,
row → id arrays, IVF lists) and rewrites segments as it compacts: at 1B
that is 30+ GB of RAM and more disk than the box has.

## Decision
- **One IVF-PQ, built once, read-only.** 65,536 lists (list ids fit a
  u16), PQ m = 16 on the residual to the coarse centroid, plus a 16-byte
  **refinement code**: PQ of what the first code missed. The scan ranks by
  ADC; the best `rerank` candidates are re-scored against
  `centroid + pq + refine`. 32 bytes per vector + 4 for the id.
- **Coarse quantizer from two-level k-means** (256 cells, then 256
  centroids in each) on the first 10M rows: minutes, where a flat
  65,536-way k-means on that sample takes hours. An **HNSW graph over
  the centroids** (the index's own `Hnsw`, m 32) assigns rows while
  streaming (~74k rows/s, so the download is the bottleneck) and picks the
  lists to probe.
- **Stream, encode, discard.** `tools/billion-stream.sh` range-fetches
  the 1B file in 64 MB chunks (4 at a time, the next group already
  downloading: 4.5 MB/s instead of 3.1 on one connection), pipes them in
  row order into `billion build`, and deletes each chunk once piped (at
  most ~0.5 GB raw on disk). `build` appends `list | code` to `stream`
  and the refinement code to `refine` (id order, its final layout), and
  commits every 1M rows (fsync + `progress` with CRCs); the driver loop
  resumes from the committed row.
- **Multi-pass counting sort** (`storage::billion::sort`): count per list,
  then each pass gathers a range of lists that fits a RAM budget (3 GiB:
  ~7 passes at 1B) from a full read of `stream` and writes it in place in
  `lists` (plus each id's list, u16 in id order, for `get`). Peak disk
  18 + 16 + 22 = 56 GB; `stream` is deleted after.
- **Search reads the files mapped**: lists for the probed cells, refine
  codes by id with `madvise(WILLNEED)` for all candidates first (as in
  ADR 0013).

## Alternatives
- **Refinement by int8 SQ (128 B/row)** or f32: 128+ GB, over the disk
  budget. A 32-byte refinement code: 70 GB peak, also over.
- **Flat k-means for the coarse quantizer**: better cells, hours of
  training; the HNSW assignment would be the same.
- **Engine segments all the way** (seal, merge into IVF-PQ): the id map
  and row → id arrays alone are ~28 GB of RAM at 1B, and every merge
  rewrites tens of GB.
- **Downloading in parallel into separate builds** and merging: faster
  only if the server limited per connection; the measured link limit is
  4.5 MB/s total.

## Consequences and limits
- Recall is bounded by the 32-byte codes: on BIGANN 10M with these
  quantizers it levels off at ~0.76 recall@10 (re-ranking 100 vs 1000
  candidates makes no difference); 1B is harder. Published as measured.
- The quantizers are trained on the first 10M rows only and never
  retrained.
- The build itself is read-only; the engine (9.5) makes it a segment.

## Addendum (2026-11-01): engine segment (9.5)
`Collection::attach_billion` (`POST /collections/{name}/attach_billion`)
checks a finished build directory, moves it into an empty L2 collection
as `<file>.billion` and publishes it as the first segment, holding ids
`0..n`. Its ids are implicit: never in the id map (1B entries ≈ 20 GB),
found by `base_row` (row = id unless tombstoned) wherever the map is
consulted: get, delete, upsert (which tombstones the billion row like any
replaced row), `len`, and open (no id may be live in both). Searches pass
the segment's tombstones and the filter as a skip predicate into the scan;
`get` reconstructs `c + pq + refine`. It has no payloads, so filters match
none of its rows; `vectors_ok` is false, so compaction never picks it.
- Alternatives: a separate read-only collection type (simpler, but no
  deletes, replacements or new rows next to the billion ones); ids in the
  id map (RAM); u64 row → id arrays like other segments (8 GB).
- Measured at 10M (RESULTS.md): recall identical to the standalone tool,
  server phys_footprint 139 MiB.
