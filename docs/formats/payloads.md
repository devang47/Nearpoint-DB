# Segment payload sidecar (v2)

`<file>.payloads` next to `<file>.seg` (file id, `manifest.md`): the
payloads of that segment's rows. Code: `PayloadStore::encode`/`decode`
in `crates/engine/src/payload.rs` (v2), `read_payloads` and
`decode_payloads_v1` in `crates/engine/src/collection.rs`. Framing
(magic, version, length, CRC32, atomic write) as in `manifest.md`. All
integers little-endian. Writers write v2; readers accept v2 and v1.

## Body (magic `NPPAYLDS`, version 2): columnar

The `PayloadStore` columns as they sit in memory, so encode and decode
are loops over typed arrays. Notation: `bytes` = len `u32` + that many
bytes; `bitmap` = len `u32` + a `RoaringBitmap` in the portable roaring
format; `words<T>` = count `u32` + count values of `T`.

| Size | Field |
|-----:|-------|
| 4 | rows `u32`; must equal the segment's vector count |
| 4 | columns `u32` |
| per column, names strictly ascending: | |
| bytes | name (UTF-8) |
| 1 | kind `u8`: 0 int, 1 float, 2 bool, 3 keyword, 4 keyword list |
| bitmap | present: rows that have a value (all `< rows`) |
| | then by kind: |
| words<`i64`> | int: value per row `0..n`, absent rows 0 |
| words<`f64`> | float: same, IEEE-754 bits |
| bitmap | bool: rows whose value is `true` (a subset of present) |
| `u32` + bytes each | keyword / list: terms (dictionary, code = position, unique) |
| words<`u32`> | keyword / list: offsets, `n + 1` entries, first 0, last = codes count |
| words<`u32`> | keyword / list: codes; row `r`'s are `codes[offsets[r]..offsets[r+1]]` |

`n <= rows` and every present row is `< n`. A keyword row has exactly one
code and a list row at least one iff the row is present. The keyword
postings (term -> rows) and the dictionary map are rebuilt on open, not
stored. Columns of the collection's schema always appear (possibly with
no present rows); so do inferred columns whose rows were all dropped, so
an inferred type survives a reopen (v1 lost those). The store's `len` is
one past the highest present row. Values are exactly what
`PayloadStore` holds: the store only keeps these five kinds (`null` and
`[]` mean missing; nested objects and non-string arrays are rejected on
insert), so there is no JSON fallback. A bad UTF-8 name or term, a
duplicate or out-of-order name, an unknown kind, a kind other than the
schema's, any length or bound above violated, a wrong row count, or
trailing bytes is corrupt.

Dead rows keep their payloads (the manifest's tombstones hide them).

## Body (magic `NPPAYLDS`, version 1): JSON per row, read-only legacy

| Size | Field |
|-----:|-------|
| 4 | rows `u32`; must equal the segment's vector count |
| per row, in row order: | |
| 4 | len `u32` |
| len | the payload as compact JSON text (UTF-8, an object), or nothing (len 0) for no payload |

Written from `PayloadStore::get` (values in stored form, e.g. an int in a
float column as a float). On open each row is re-inserted into a fresh
`PayloadStore`; a row that doesn't parse or conflicts with its column
type, a wrong row count, or trailing bytes is corrupt. Replaced by v2
because JSON per row cost minutes per seal/rewrite/open at 100M rows.

## Version history

| Version | Date | Change |
|---------|------|--------|
| 1 | 2026-10-28 | Initial (task 3.3). Fixture: `crates/engine/testdata/payloads-v1`. |
| 2 | 2026-10-31 | Columnar binary columns instead of JSON per row; v1 still read. Fixture: `crates/engine/testdata/payloads-v2`. |
