# 0006: Segment file format v1

Date: 2026-10-28. Status: accepted.

## Context
Sealed segments have to be mmap'd safely, verified on open, and readable
by `storage` without `storage` depending on `index`.

## Decision
- One fixed layout per `(dim, count, blob_len)`: header, then vectors, ids
  and index blob at 64-byte-aligned offsets, then a footer with a CRC32 per
  section, a section table and a footer CRC. The footer table is re-checked
  against the computed layout, and padding must be zero.
- The index blob is opaque to `storage`. It carries its own magic and
  version, owned by `index`. The metric is stored as a raw code that
  `index` defines.
- `Hnsw::from_bytes` copies the vectors into the index, so one owned
  `Hnsw` type keeps accepting inserts. For SIFT1M that's 512 MB of heap on
  top of the reclaimable mapped pages.
- Neighbour lists are stored at full capacity (`1 + 2m` slots), so loading
  is a straight copy.
- `open` verifies every CRC, which means reading the whole file once.

## Alternatives
- Borrowing vectors from the map: no copy, but `Hnsw` would need two
  storage modes. Revisit in Phase 8, where RAM is the constraint.
- Compact lists (used slots only): about 20% smaller graph, but loading
  needs a rebuild step.
- Lazy or optional verification: postponed until segments are multi-GB.
  Adding it doesn't change the format.

## Addendum (2026-10-28): segment v2, segments without f32 vectors
IVF-PQ segments (Phase 6 builds them for large segments) carry their
whole index in the blob: PQ codes for the scan and int8 SQ codes for the
re-rank. The f32 vector section would be 4x the SQ codes (512 vs 128 B
per vector at dim 128), which Phase 9's compressed-only mode can't afford
on disk, and nothing in the index reads it.

- **v2 adds element type 1, "no vector section"**: the vector section
  has length 0 and `Segment::vectors()` is empty. `SegmentData` is
  unchanged (engine builds it with a struct literal): an empty `vectors`
  with non-empty `ids` means element type 1. Blob kind 2 is IVF-PQ.
- The writer always writes v2; the layout with element type 0 is
  byte-for-byte v1 apart from the version and the CRCs that cover it.
  v1 files still load (fixture); a v1 file with element type 1 is
  rejected, as is any version other than 1 or 2. Old builds reject v2
  files with `UnsupportedVersion(2)`.
- Whether an IVF-PQ segment keeps f32 vectors is the writer's choice:
  Phase 8 may want them (mapped, not in phys_footprint) for `get` or an
  f32 re-rank; Phase 9 won't.

Alternatives: an `Option` field in `SegmentData` (clearer, but breaks
every struct literal in engine); a separate vectors file (a second file
to publish atomically and keep in sync); keeping element type 1 under
version 1 (old readers would still reject it cleanly, but the version
number would no longer say what a file can contain).
