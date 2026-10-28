# 0005: Payload columns and filter semantics

Date: 2026-10-28. Status: accepted.

## Context
Tasks 7.1–7.2 need a per-segment metadata store and a filter language
whose result plugs into HNSW traversal and IVF scanning (7.3) and a
brute-force planner (7.4). Filters arrive as JSON from the HTTP API, so
they have to behave predictably for clients and never fail because of a
field name.

## Decision
- **One typed column per field, type fixed by the first value.** A
  conflicting value rejects the whole payload (`TypeConflict`), checked
  before anything is written. The only widening is int into a float
  column, because JSON from JavaScript can't tell `10.0` from `10`.
- **Keywords are dictionary-encoded with a posting bitmap per term**, so
  `eq`/`in` is one bitmap OR per value with no scan. Forward codes are kept
  in CSR form (`offsets`, `codes`) so `get` can rebuild the payload.
  Keyword lists (tags) share the column type, with a `list` flag; a scalar
  and a list in the same field conflict.
- **Numeric columns are a dense `Vec` plus a presence bitmap; range, eq and
  in on numbers are full scans.** A scan is O(rows) of sequential reads.
  A sorted index costs a second copy of the column and has to be kept up
  to date in the mutable segment. Revisit for sealed segments if numeric
  filters show up in profiles (marked `ponytail:` in `filter.rs`).
- **Two-valued logic.** Leaves never match a row missing the field; `not`
  is the complement over `0..universe`. This matches Elasticsearch and
  Qdrant `must_not`, De Morgan's laws and double negation hold, and SQL's
  `!=` is still available as `exists AND NOT eq`.
- **Unknown fields match nothing, not an error.** A field can exist in one
  segment and not another, so a per-segment error would make results
  depend on segment layout. Type mismatches (range on a keyword, string vs
  int) are errors, and all branches are evaluated so the error doesn't
  depend on the data.
- **JSON shape:** one operator key per object, and the leaves name one
  field (`{"eq": {"color": "red"}}`). Parsed through a private `Wire` enum
  with serde `try_from`/`into`, so the Rust AST stays `Eq(field, value)`.

## Alternatives
- **SQL three-valued logic:** `not eq` skips missing rows. That's
  surprising for tag filters ("everything not archived" would drop
  untagged rows), and you need `is null` to get them back.
- **A sorted numeric index now:** more code and RAM before there's a
  measurement that needs it.
- **Coercing conflicting types** (store `"12"` as 12, or ints as floats
  everywhere): this silently changes what matches, and every i64 above
  2^53 would lose precision.
- **Explicit schemas in the collection config:** the clean fix for the
  first-value-fixes-the-type ordering hazard. Deferred to collection
  config (4.1/6.x); the column types above are what a schema would declare.
