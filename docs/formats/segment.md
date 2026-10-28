# Segment file format

A sealed, immutable segment: its vectors, their external ids, and an
opaque index blob produced by the `index` crate. Code:
`crates/storage/src/segment.rs` (container), `crates/index/src/{flat,hnsw,blob}.rs`
(blobs).

All integers are little-endian with explicit widths. No `usize` on disk.

## File layout (v1)

| Part         | Offset                         | Length          |
|--------------|--------------------------------|-----------------|
| Header       | 0                              | 64              |
| Vectors      | 64                             | `count * dim * 4` |
| zero padding | to a multiple of 64            | 0–63            |
| Ids          | `align64(end of vectors)`      | `count * 8`     |
| zero padding | to a multiple of 64            | 0–63            |
| Index blob   | `align64(end of ids)`          | `blob_len`      |
| Footer       | `end of blob` = `file_len - 96`| 96              |

**Alignment rule.** Every section starts at a file offset that is a
multiple of 64. The file is mapped read-only (mmap is page-aligned), so
the vector section is viewed in place as `&[f32]` and the id section as
`&[u64]` (`bytemuck::try_cast_slice`, checked once on open). The footer
follows the blob directly and is not aligned.

v1 has exactly one layout for given `(dim, count, blob_len)`. The reader
recomputes it and requires the footer's section table and the file length
to match exactly; anything else is `Corrupt`.

### Header (64 bytes)

| Offset | Type    | Field |
|--------|---------|-------|
| 0      | [u8; 8] | magic `NPSEGMT\0` |
| 8      | u32     | format version (1) |
| 12     | u32     | element type: 0 = f32 (int8 reserved for later) |
| 16     | u32     | dim (> 0) |
| 20     | u32     | metric code: 0 = L2, 1 = inner product, 2 = cosine (`index::Metric::code`) |
| 24     | u64     | vector count |
| 32     | u32     | blob kind: 0 = flat, 1 = hnsw |
| 36     | [u8; 28]| reserved, zero |

### Sections

- **Vectors:** `count` rows of `dim` f32, row-major, in the index's stored
  form (unit length for cosine). Dense index id `i` is row `i`.
- **Ids:** `count` u64 external ids; `ids[i]` belongs to dense id `i`.
- **Index blob:** opaque to storage; layouts below.

### Footer (96 bytes)

| Offset | Type    | Field |
|--------|---------|-------|
| 0      | u32     | CRC32 of the header (bytes 0..64) |
| 4      | u32     | reserved, zero |
| 8      | 24 B    | vectors entry |
| 32     | 24 B    | ids entry |
| 56     | 24 B    | blob entry |
| 80     | u32     | reserved, zero |
| 84     | u32     | CRC32 of footer bytes 0..84 |
| 88     | [u8; 8] | magic `NPSEGMT\0` |

Section entry (24 bytes): `offset u64, length u64, crc32 u32, reserved u32`.

## Checksums and validation

CRC32 (`crc32fast`, IEEE). Coverage: the header, each section body, and the
footer itself (bytes 0..84). Padding is not under a CRC; the reader
requires it to be zero instead, so every single-bit flip anywhere in the
file is detected (tested exhaustively on a small file).

`Segment::open` checks, in order: header magic (`BadMagic`), length ≥ 160
(`Truncated`), version (`UnsupportedVersion`; read before any CRC so a
newer file gets a clear error), footer magic (`Truncated`), footer CRC, header CRC
(`Checksum`), element type (`UnsupportedElementType`), dim/blob kind/section
table/file length/padding (`Corrupt`), then every section CRC (`Checksum`).

**Cost.** v1 verifies every section on open, which reads the whole file
once: for SIFT1M HNSW (~658 MB) that is one sequential pass and pulls the
file into the page cache (clean, evictable pages, not counted in
phys_footprint). For multi-GB IVF-PQ segments (Phase 5/6) this becomes a
startup cost; the planned fix is an `open` that verifies header and footer
only, plus a lazy/background `verify()`. The format doesn't change for it.

## Publishing

`segment::write` writes `<name>.tmp`, `sync_data` (F_FULLFSYNC on macOS),
renames to `<name>`, then fsyncs the parent directory. A stale `.tmp`
left by a crash is overwritten by the next write and never opened as a
segment; on a write error the `.tmp` is removed. Published segments are
never modified in place, which is what makes mapping them safe.

## Index blobs

Every blob starts with a 16-byte prefix, then kind-specific fields. The
blob has its own version so index layouts can evolve without a segment
version bump. Blobs carry no CRC of their own: the segment's blob CRC
covers them.

| Offset | Type    | Field |
|--------|---------|-------|
| 0      | [u8; 4] | magic: `NPFL` (flat) or `NPHW` (hnsw) |
| 4      | u32     | blob version (1) |
| 8      | u32     | dim |
| 12     | u32     | metric code |

### Flat blob v1 (24 bytes)

| Offset | Type | Field |
|--------|------|-------|
| 16     | u64  | vector count |

### HNSW blob v1

| Offset | Type | Field |
|--------|------|-------|
| 16     | u32  | m |
| 20     | u32  | ef_construction |
| 24     | u64  | seed |
| 32     | u64  | n (node count) |
| 40     | u64  | entry point: `level << 32 \| id`, or `u64::MAX` if empty |
| 48     | n × u8 | level of each node (≤ 16) |
| 48+n   | n × (1+2m) × u32 | layer-0 lists, node order |
| …      | Σ level × (1+m) × u32 | upper lists: for each node in order, layers 1..=level |

Each list is written at full capacity exactly as in memory:
`[count, id_1, ..., id_cap]`; slots past `count` are ignored.

`Hnsw::from_bytes` / `Flat::from_bytes` reject (as `index::Error::Corrupt`
or `UnsupportedBlobVersion`): bad magic, unknown metric, dim 0, m < 2,
ef_construction 0, n > u32::MAX, vector count ≠ n × dim, level > 16, any
size overflow, truncation or trailing bytes, a list count ≥ capacity, a
link to id ≥ n or to a node not on that layer, and an entry point that
isn't a node on the top layer (or isn't `u64::MAX` exactly when n = 0).
Sizes come from bytes actually present, so a corrupt blob can't trigger a
huge allocation.

**Vectors on load.** `from_bytes` takes the vectors as an owned
`Vec<f32>` (callers pass `segment.vectors().to_vec()`). That is one copy:
for SIFT1M, 512 MB of heap on top of the (evictable) mapped pages. v1
accepts it so `Hnsw` keeps a single owned representation and stays
insertable; borrowing from the map is a later optimization.

### Size example: SIFT1M, HNSW M = 16

| Part | Bytes |
|------|-------|
| Header + footer | 160 |
| Vectors (1M × 128 × 4) | 512,000,000 |
| Ids (1M × 8) | 8,000,000 |
| Blob prefix + levels | 48 + 1,000,000 |
| Layer 0 (1M × 33 × 4) | 132,000,000 |
| Upper layers (≈ n/15 node-levels × 17 × 4) | ≈ 4,533,000 |
| Padding | 0 (all sections already 64-aligned) |
| **Total** | **≈ 657.5 MB** |

Overhead over raw vectors: ≈ 145.5 MB (+28.4%): ids 1.6%, graph 26.9%.
The graph stores unused list capacity; writing only `count` ids per list
would shrink it, at the cost of a rebuild step on load.

## Version history

| Version | Date | Change |
|---------|------|--------|
| 1 | 2026-10-28 | Initial format (task 3.2). Fixture: `crates/storage/testdata/segment-v1.seg`. |

HNSW and Flat blobs: version 1, same date.
