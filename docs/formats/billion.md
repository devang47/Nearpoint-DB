# Billion-mode build directory

Phase 9 (ADR 0014). Code: `crates/storage/src/billion.rs` (files),
`crates/index/src/billion.rs` (the `quant` body), `crates/bench/src/bin/billion.rs`
(train / build / sort / eval), `tools/billion-{stream,build}.sh` (the
streamed download). All integers little-endian with explicit widths;
CRC32 is `crc32fast` (IEEE).

| File | Written by | Lifetime |
|------|-----------|----------|
| `quant` | `billion train` | kept |
| `stream` | `billion build` (append) | deleted after `sort` |
| `refine.tmp` | `billion build` (append) | renamed to `refine` when the stream completes |
| `progress` | `billion build` (each commit) | kept (sort reads it) |
| `refine` | end of `billion build` | kept |
| `lists` | `billion sort` | kept |

Peak at 1B × m = 16: `stream` 18 GB + `refine` 16 GB + `lists` 22 GB
= 56 GB during the sort; 38 GB after.

## `quant` (framed, magic `NPBQUANT`, version 1)

A `storage::framed` file (`magic | version | body_len | body | crc32`,
docs/formats/manifest.md); the body is an index blob:

| Size | Field |
|-----:|-------|
| 16 | blob header: magic `NPBQ`, blob version u32 (1), dim u32, metric u32 (0 = L2; nothing else is accepted) |
| 4 | nlist u32 (1..=65,536) |
| 4 | m u32 (bytes per code; divides dim) |
| 4 | ksub u32 (256) |
| 4 | refine m u32 (= m) |
| 4·nlist·dim | coarse centroids, f32 row-major |
| 4·256·dim | PQ codebooks, m blocks of 256 × dim/m f32 |
| 4·256·dim | refinement codebooks, same shape |
| 8 | graph length u64 |
| … | HNSW blob over the centroids (segment.md, "HNSW blob"); its vectors are the centroids above |

## `stream` (no header) and `progress`

`stream`: one record per input row, in row (id) order: `list u16 | code
[m]` (18 bytes at m = 16). Row i is id i. `refine.tmp`: `code [m]` per
row, same order.

`progress` (framed, magic `NPBPROGR`, version 1), body 20 bytes:

| Size | Field |
|-----:|-------|
| 8 | rows u64: rows committed (both files hold exactly these) |
| 4 | m u32 |
| 4 | CRC32 of `stream`'s first `rows` records |
| 4 | CRC32 of `refine.tmp`'s first `rows` codes |

A commit flushes and `sync_data`s both files, then writes `progress`
atomically (tmp, sync, rename, fsync dir). On open, both files are cut to
`rows` records (a crash mid-batch leaves a torn tail), and hashing resumes
from the stored CRCs. Every sort pass re-reads `stream` and checks its
CRC; a list id ≥ nlist is `Corrupt`.

## `refine` (footer, magic `NPBREFIN`, version 1)

| Part | Length |
|------|--------|
| codes | n · m (id order) |
| footer | 32 |

Footer: magic `NPBREFIN` (8), version u32, m u32, n u64, CRC32 of the
codes u32, CRC32 of footer bytes 0..28 u32.

## `lists` (footer, magic `NPBLISTS`, version 1)

| Part | Offset | Length |
|------|--------|--------|
| offsets | 0 | 8 · (nlist + 1): u64 row offsets; list l is rows `offsets[l]..offsets[l+1]` |
| codes | 8 · (nlist + 1) | n · m, list by list |
| zero padding | | to a multiple of 4 |
| ids | | 4 · n: u32 id of each row, list by list (ascending within a list) |
| lists by id | | 2 · n: u16 list of each id, in id order (for `get`) |
| footer | file_len − 48 | 48 |

Footer: magic `NPBLISTS` (8), version u32, nlist u32, m u32, CRC32 of
lists-by-id u32, n u64, CRC32 of offsets, of codes, of ids (u32 each),
CRC32 of footer bytes 0..44 u32. Lists by id is written by the sort's
counting pass; ids ascend within a list, so an id's row is a binary search
in its list.

## Validation

`Billion::open` maps `lists` and `refine`, checks both footers (magic,
version before anything else, footer CRC), that m and n agree, that each
file's length is exactly what the footer implies, and the offsets
section's CRC and that it ascends from 0 to n (offsets are read on every
probe). The codes, ids, lists-by-id and refine sections (tens of GB) are checked by
`Billion::verify` (`billion sort --verify`), not on open. Tests: round
trip through a resumed stream and a multi-pass sort; every byte of both
files flipped is caught by open or verify; truncation fails open; a
damaged stream fails the sort.

## Version history

| Version | Date | Change |
|---------|------|--------|
| 1 (all files) | 2026-11-01 | First version (task 9.1–9.3). |
