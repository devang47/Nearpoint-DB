# Collection directory and manifest (v1)

Code: `crates/engine/src/collection.rs` (layout, manifest body, recovery),
`crates/storage/src/framed.rs` (framing). Design: ADR 0008. All integers
little-endian, explicit widths; no `usize` on disk.

## Directory

| Name | What |
|------|------|
| `MANIFEST` | live segments, their tombstones, covered LSN (below) |
| `wal/` | the WAL (`wal.md`) |
| `<lsn>.seg` | a sealed segment (`segment.md`), HNSW blob |
| `<lsn>.payloads` | that segment's payloads (`payloads.md`) |

`<lsn>` is the last LSN the seal covered, zero-padded to 20 digits. On
open, `*.tmp` files and `.seg`/`.payloads` files the manifest doesn't
list are deleted (leftovers of a crashed seal); other names are ignored.
No `MANIFEST` means no sealed segments (`covered = 0`).

## Framing (shared with `payloads.md`)

| Offset | Size | Field |
|-------:|-----:|-------|
| 0 | 8 | magic |
| 8 | 4 | version `u32` |
| 12 | 8 | body_len `u64`; must equal file length − 24 |
| 20 | body_len | body |
| 20+body_len | 4 | CRC32 (crc32fast) of every byte before it |

Read order: magic (`BadMagic`), length ≥ 24 (`Truncated`), version
(`UnsupportedVersion`, before any CRC so a newer file reports it),
body_len (`Truncated`), CRC (`Checksum`). Written atomically: `<name>.tmp`,
`sync_data` (F_FULLFSYNC), rename, fsync the directory.

## Manifest body (magic `NPMANIFS`, version 1)

| Size | Field |
|-----:|-------|
| 8 | covered `u64`: every write with LSN ≤ this is in the segments below |
| 4 | n `u32`: segment count |
| per segment: | |
| 8 | lsn `u64`; strictly increasing, 0 < lsn ≤ covered |
| 4 | dead_len `u32` |
| dead_len | tombstones: rows of this segment that are deleted or replaced, a RoaringBitmap in the standard portable format (`RoaringBitmap::serialize_into`) |

The body must be consumed exactly, and each bitmap's re-serialized size
must equal `dead_len`. A tombstone ≥ the segment's row count is corrupt.

## Recovery rules

1. Read the manifest (none: empty). Delete garbage files.
2. Open each listed segment (every CRC checked), check dim and metric
   against the config, load its HNSW and payloads, apply its tombstones.
   An id live in two rows is corrupt.
3. Open the WAL (torn tail truncated). Records with LSN ≤ covered are
   skipped; the next one must be exactly `covered + 1`, or the WAL must
   resume there if it has none. Otherwise WAL files are missing: corrupt.
4. Replay the rest, then delete WAL files wholly ≤ covered (finishes a
   seal that crashed after publishing its manifest).

Seal order and crash analysis: ADR 0008.

## Version history

| Version | Date | Change |
|---------|------|--------|
| 1 | 2026-10-28 | Initial (task 3.3). Fixture: `crates/engine/testdata/manifest-v1`. Phase 4.1 adds name/config with a version bump. |
