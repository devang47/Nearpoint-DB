# Collection directory and manifest (v2)

Code: `crates/engine/src/collection.rs` (layout, manifest body, recovery),
`crates/engine/src/db.rs` (data directory, `CONFIG`),
`crates/storage/src/framed.rs` (framing). Design: ADR 0008, ADR 0011,
ADR 0012 (file ids, compaction). All
integers little-endian, explicit widths; no `usize` on disk.

## Data directory (task 4.1)

| Name | What |
|------|------|
| `LOCK` | empty; held with an exclusive `flock` (`File::try_lock`) by the one process that has the directory open |
| `collections/<name>/` | one collection (below); `<name>` is 1–64 of `[A-Za-z0-9_-]` |
| `collections/.new-<name>/` | a create in progress: `CONFIG` is written here, then the directory is renamed to `<name>` (the commit point) |
| `collections/.drop-<name>/` | a drop in progress: `<name>` is renamed here (the commit point), then deleted |

On open, every `collections/.*` directory is deleted (a crashed create or
drop), and every other valid name is opened as a collection. Renames are
followed by an fsync of `collections/`.

## Directory

| Name | What |
|------|------|
| `CONFIG` | the collection's settings (below); written once at create |
| `MANIFEST` | live segments, their tombstones, covered LSN (below) |
| `wal/` | the WAL (`wal.md`) |
| `<file>.seg` | a sealed segment (`segment.md`): HNSW blob, or IVF-PQ blob with the f32 vectors kept (element type 0) |
| `<file>.payloads` | that segment's payloads (`payloads.md`) |

`<file>` is the segment's file id, zero-padded to 20 digits. Since
manifest v2 it comes from a counter (one past the largest id in the
manifest at open, then incremented per new segment); a v1 directory's
files are named by the LSN their seal covered, and those names stay
valid ids. On open, `*.tmp` files and `.seg`/`.payloads` files the
manifest doesn't list are deleted (leftovers of a crashed seal, build or
compaction, or inputs of a compaction that crashed before deleting
them); other names are ignored. No `MANIFEST` means no sealed segments
(`covered = 0`).

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

## CONFIG body (magic `NPCOLCFG`, version 1)

The body is the collection's `Config` as compact JSON (serde), e.g.

```
{"dim":128,"metric":"l2","hnsw":{"m":16,"ef_construction":200,"seed":42},"schema":{"price":"float"}}
```

| Key | Type | Meaning |
|-----|------|---------|
| `dim` | integer > 0 | vector dimension |
| `metric` | `"l2"`, `"inner_product"`, `"cosine"` | distance |
| `hnsw` | object; each key optional (defaults m 16, ef_construction 200, seed 42) | config for new mutable segments |
| `schema` | object, field → `"int"`, `"float"`, `"bool"`, `"keyword"`, `"keyword_list"`; optional | declared payload types (ADR 0011) |
| `seal_rows` | integer ≥ 1; optional, default 100000 | seal the mutable segment at this many rows (ADR 0012) |
| `ivf_rows` | integer ≥ 1024; optional, default 1000000 | segments this big become IVF-PQ (L2/cosine) |
| `max_merge_rows` | integer ≥ 1; optional, default 4000000 | largest segment a merge may produce |

The last three were added with Phase 6 as optional keys (still `CONFIG`
v1): older files load with the defaults, new ones are written with all
keys. Unknown keys are rejected. The WAL sync policy is not stored (a server
flag). A JSON body keeps the file readable and lets later versions add
optional keys; a breaking change bumps the version. Fixture:
`crates/engine/testdata/config-v1`.

## Manifest body (magic `NPMANIFS`, version 2; v1 still read)

| Size | Field |
|-----:|-------|
| 8 | covered `u64`: every write with LSN ≤ this is in the segments below |
| 4 | n `u32`: segment count |
| per segment: | |
| 8 | file `u64`: the file id; strictly increasing, > 0 (v1: the seal's LSN, also ≤ covered) |
| 4 | dead_len `u32` |
| dead_len | tombstones: rows of this segment that are deleted or replaced, a RoaringBitmap in the standard portable format (`RoaringBitmap::serialize_into`) |

The body must be consumed exactly, and each bitmap's re-serialized size
must equal `dead_len`. A tombstone ≥ the segment's row count is corrupt.

v1 and v2 bodies have the same layout. v2 changes what the id means: a
file id from a counter instead of an LSN, so a compaction can name a
merged segment, and ids may exceed `covered`. The bump makes an older
build fail with `UnsupportedVersion` instead of rejecting a v2 manifest
as corrupt. Readers read v2, or v1 with the extra `≤ covered` check.

**Tombstones may run ahead of `covered`.** Rows killed by WAL records
above `covered` (written while a seal or compaction ran) are tombstoned
too; the WAL is synced up to its end before each manifest is written, so
those records are durable and replay re-applies them idempotently: an
upsert re-inserts (its old row is already dead), a delete of an id with
no live row is a no-op. A compaction may likewise drop such rows. ADR
0012.

## Recovery rules

1. Read the manifest (v2, or v1; none: empty). Delete garbage files.
2. Open each listed segment (every CRC checked), check dim and metric
   against the config, load its index (HNSW: vectors copied into RAM;
   IVF-PQ: index in RAM, f32 vectors left in the mapped file) and
   payloads, apply its tombstones. A segment without f32 vectors, a flat
   blob, or an index whose length differs from the id count is corrupt.
   An id live in two rows is corrupt.
3. Open the WAL (torn tail truncated). Records with LSN ≤ covered are
   skipped; the next one must be exactly `covered + 1`, or the WAL must
   resume there if it has none. Otherwise WAL files are missing: corrupt.
4. Replay the rest, then delete WAL files wholly ≤ covered (finishes a
   seal that crashed after publishing its manifest).

Seal, build and compaction order and crash analysis: ADR 0008, ADR 0012.

## Version history

| Version | Date | Change |
|---------|------|--------|
| 1 | 2026-10-28 | Initial (task 3.3). Fixture: `crates/engine/testdata/manifest-v1`. |
| 2 | 2026-10-29 | Segment ids are file ids from a counter, not LSNs (may exceed `covered`); same layout. v1 still read. Fixture: `crates/engine/testdata/manifest-v2`. |

`CONFIG`: v1 (task 4.1); Phase 6 added optional keys (`seal_rows`,
`ivf_rows`, `max_merge_rows`) without a version change. The config lives
in its own file, written once, so seals never rewrite it, and the name is
the directory name.
