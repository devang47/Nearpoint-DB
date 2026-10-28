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
