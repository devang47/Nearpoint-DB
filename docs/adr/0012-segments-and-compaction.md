# 0012: Sealing at a size, background builds, compaction and fan-out

Date: 2026-10-29. Status: accepted.

## Context
Phase 6 turns the one-mutable-segment collection (ADR 0008, 0011) into a
segment engine: seal at a configurable size (6.1), rebuild big sealed
segments as IVF-PQ (6.2), merge small segments and drop dead rows (6.3),
search segments in parallel (6.4). Until now dead rows stayed in RAM
forever (server RSS +~20 MiB/min under the load test), the seal held the
writer lock while writing the HNSW, segment files were named by LSN, and
`live` pointed at segment *positions*, which only worked because
segments were never removed.

## Decision

### Driver: `Collection::maintain`
One step per call, under a `maint` mutex (so only one seal, build or
compaction is ever in flight): finish a pending seal, else seal if the
mutable segment has `seal_rows` rows, else run one planned rewrite.
Returns whether it did anything. Tests call it directly; `Db` runs one
background thread that calls it on every collection in turn and waits
500 ms (condvar, so drop wakes it) after a pass with no work. Errors are
logged (`tracing::error!`) and retried on the next pass. `Collection::
close` (drop) waits for the step in flight and stops later ones, so no
build races the directory's deletion; `Db::drop` stops and joins the
thread. Explicit `seal()` (the `/seal` endpoint) takes the same mutex.

Lock order: maint, writer, live, snap, a segment.

### Seal protocol (6.1)
1. **Freeze** (writer lock): `wal.rotate()` at `last`; swap in a snapshot
   with a fresh mutable segment; remember the frozen one as *pending*
   with `covered = last` and a new file id. Writes continue right away.
   (Deletes only since the last seal: publish the tombstones here, done.)
2. **Write** `<file>.seg` + `<file>.payloads` with no lock. The frozen
   segment is never written again, so readers and writers don't care.
3. **Publish** (writer lock): set the view's file id, write the manifest
   with `covered = last` (below), swap the snapshot; then drop WAL files
   ≤ `last`.

A failed step leaves the seal pending; the next `maintain`/`seal`
retries from step 2 before doing anything else, so `covered` only grows.

### Durability rule for every manifest
A manifest may tombstone a row (or a compaction may drop it) only if the
WAL record that killed it is durable. With writes running during steps
2–3 and during rewrites, rows die from records above `covered`, which
under `--sync <n>ms` may not be on disk yet. **Every publish calls
`wal.sync()` under the writer lock before writing the manifest**, then
records the *current* tombstones. Replay of records above `covered` on
top of such a manifest is idempotent: upsert-replace re-inserts (the old
row is already dead, so the id isn't live), delete of an id with no live
row is a no-op, upsert-then-delete ends deleted. Tested by writes during
the seal's file write followed by a reopen (`seals_at_size_with_writes_
during_the_file_write`). One extra fsync per publish is noise next to the
segment write. The alternative, recording the bitmaps frozen at the
rotate, needs a second set of tombstones per segment and an old→new
translation of them for compactions.

### Stable segment keys and the id map
`live: id → (key, row)`, where `key` is a per-process `u32` given to each
segment when it is loaded or created and never reused. Removing or
merging segments therefore only touches the ids that moved; other
entries stay valid. A lookup finds the key's position with a linear scan
of the snapshot (tens of segments). `u32` keys keep entries at 16 bytes
(a `u64` file id would cost ~8 more bytes per live id, 0.8 GB at 100M).

### Manifest v2: file ids
Merged segments need names that aren't LSNs. v2 keeps the v1 layout but
the per-segment `u64` is a file id from a counter (one past the largest
listed id at open), strictly increasing in the file and allowed above
`covered`. v1 directories open unchanged (their LSN names are valid ids).
docs/formats/manifest.md.

### Rewrites: build and compaction (6.2, 6.3)
A rewrite turns 1+ sealed segments into one: copy their live rows (as of
a snapshot taken under `maint`, no other lock) into new ids, vectors and
payloads, build the index on the **build pool**, write the files (no
lock), then publish under the writer lock:
- For each copied row, if `live[id]` still is `(old key, old row)`,
  remap it to `(new key, new row)`; otherwise the row died meanwhile
  (deleted, or replaced by an upsert) and is tombstoned in the new
  segment. So an id is never live twice and never resurrected.
- New snapshot: inputs removed, new segment before the mutable one;
  sync WAL, write manifest (commit point), swap `live` and the snapshot
  together under `live`'s write lock, then delete the input files.
  Readers still holding old snapshots keep working: HNSW segments are in
  RAM and an unlinked mapped IVF-PQ file stays readable.

The new segment's payload store starts with the union of the inputs'
column types, so dropping rows can't change an inferred type. Segments
whose undeclared fields disagree on a type are never merged together.

**Index choice:** IVF-PQ when the rewrite has ≥ `ivf_rows` rows and the
metric is L2 or cosine, else HNSW. Inner product stays HNSW (IVF-PQ IP
recall is capped at ~0.84 by the int8 re-rank, RESULTS.md; Phase 8's f32
re-rank lifts that) and its merges are capped below `ivf_rows`.
IVF-PQ parameters: nlist = √n (the measured SIFT1M and BIGANN 10M points,
0.92 recall with nprobe 16/32 rerank 100, used ~1–1.3·√n; 4·√n with
nprobe 32 would probe a smaller fraction than anything measured), 8-bit
PQ with the largest m ≤ 16 dividing dim, `max_train = max(64·nlist,
262144)`. IVF-PQ segments keep their f32 vectors in the file (element
type 0) and serve `get`/`with_vector` from the map, so `get` stays exact
and costs no RAM. Search defaults: `nprobe 32`, `rerank 100`
(`SearchOptions`, search JSON).

**Policy**, in order, first match wins:
1. **Dead rows:** the sealed segment with the largest dead share, if
   that share is ≥ 25%, is rewritten alone. Bounds dead rows to ~25% of
   sealed rows (RAM and search waste).
2. **Build:** an HNSW segment with ≥ `ivf_rows` live rows (L2/cosine) is
   rebuilt as IVF-PQ (big seals, e.g. a bulk load, land here).
3. **Tiered merge:** tier(s) = how many times live rows can be divided
   by 4 while staying ≥ `4·seal_rows` (tier 0 is below `4·seal_rows`).
   In the lowest tier with ≥ 4 segments under the cap, merge up to 4 of
   them, smallest first, while the sum stays ≤ the cap (`max_merge_rows`;
   for inner product also < `ivf_rows`) and payload types agree, if that
   gives ≥ 2. Segment count stays O(log(n / seal_rows)) per tier width,
   and each row is rewritten O(log₄(n / seal_rows)) times.

Merging doesn't need adjacency: versions are resolved by `live`, not by
segment order, so any segments can merge.

### Threading
Builds run on a dedicated rayon pool (`set_build_threads`, server flag
`--build-threads`, default half the cores) via `pool.install`, so
`Hnsw::add` and IVF-PQ training parallelize there and leave the other
cores to searches. **Search fan-out (6.4):** sealed segments are searched
on the global rayon pool (`par_iter`) and their hits merged by distance
and truncated to `k`; the mutable segment is searched on the calling
thread afterwards. That split is required: the writer holds the mutable
segment's write lock while `Hnsw::add` waits on global-pool jobs, so a
pool worker blocked on that lock deadlocks the pool (found by the
concurrent test). Sealed segments are never write-locked. The "never a
dead row" re-check of every hit against the allowed bitmap stays.

### Crash safety
Fail points: 1 WAL rotated, 2 seal files written, 3 seal manifest
published, 4 rewrite files written, 5 rewrite manifest published. Tests
reopen after each (`crash_between_seal_steps`, `crash_between_rewrite_
steps`); crash-child (SIGKILL test) now seals at 16 rows and runs random
`maintain` steps. Before the manifest: orphans deleted on open, WAL holds
everything. After it: covered WAL files and compaction inputs are
leftovers that open deletes.

## Alternatives
- **Separate builder and compactor threads/queues:** more concurrency,
  but two background writers to one manifest need ordering rules. One
  step at a time is enough while builds take seconds.
- **Size-ratio "merge adjacent small segments":** simpler to say, but
  adjacency carries no meaning here and a pure size rule can re-merge a
  big segment with every small one (quadratic writes). Tiers bound it.
- **Compaction that rewrites all small segments at once:** fewer steps,
  unbounded build size; the cap and merge factor keep a step's RAM known.
- **Keep the position-based id map and renumber on compaction:** every
  live id of every later segment would move on each compaction.
- **Search all segments on rayon:** deadlocks (above). A separate search
  pool would also work but doubles the thread count for nothing.

## Consequences and limits
- `ponytail:` one background thread for all collections: a long build
  in one delays the others' seals (the mutable segment then grows past
  `seal_rows` meanwhile; it is in RAM).
- A merge holds its rows' f32 vectors, the new index and (while writing)
  its blob: ~2.6 GB at the default 4M-row cap and dim 128. Lower
  `max_merge_rows` for wide vectors.
- `seal()` waits for a background step in flight (seconds for a 1M-row
  IVF-PQ build).
- Every publish still rewrites all tombstones (ADR 0008's `ponytail:`).
