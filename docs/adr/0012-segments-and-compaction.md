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
background thread per collection (see Threading) that calls it in a
loop and waits 500 ms (condvar, so a drop wakes it) after a step with no
work. Errors and panics are logged (`tracing::error!`) and retried after
the wait. Dropping a collection stops and joins its thread (after the
step in flight), then `Collection::close` stops later steps, so no build
races the directory's deletion; `Db::drop` stops every thread, then
joins each. Explicit `seal()` (the `/seal` endpoint) takes the same mutex.

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
PQ with the largest m ≤ max(16, dim/8) dividing dim (16-byte codes up
to dim 128, ~8-dim subspaces beyond; a dim with no split into subspaces
of ≤ 16 dims, e.g. a prime, stays HNSW like inner product), `max_train = max(64·nlist,
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

**One maintenance thread per collection (2026-10-31).** It used to be
one Db thread looping over every collection, so a multi-second IVF-PQ
build in one collection held back every other collection's seal (its
mutable segment and WAL kept growing). Now each collection gets its own
thread, spawned on open/create and kept next to the collection in the
Db's map. The thread holds only its `Arc<Collection>` and stop flag, never
the collections map or the admin lock, so `drop_collection` joins it
(outside the map lock, under admin) without a lock-order risk, and
`Db::drop` signals all of them before joining any. Builds of several
collections can now run at once; they share the one build pool, so CPU
stays bounded by `--build-threads`, though each holds its own build RAM.
Alternative: a small worker pool pulling collections from a queue. It
caps threads but needs a queue, re-queueing after each step and a "this
collection is already being worked on" rule, all to save idle threads
that cost ~nothing at the expected tens of collections.

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
- `ponytail:` one OS thread per collection (mostly idle): fine to
  hundreds of collections; a worker pool with a queue past that.
  Concurrent builds in different collections add up in RAM (one merge
  each, below).
- A merge holds its rows' f32 vectors, the new index and (while writing)
  its blob: ~2.6 GB at the default 4M-row cap and dim 128. The cap is
  also limited to 2 GiB of f32 input, so wide vectors merge smaller.
- `seal()` waits for a background step in flight (seconds for a 1M-row
  IVF-PQ build).
- Every publish still rewrites all tombstones (ADR 0008's `ponytail:`).

## Addendum (2026-10-29): review fixes before the soak
- Upserts and queries reject components beyond ±1e15 and zero `cosine`
  vectors. A huge but finite range made IVF-PQ's int8 scale infinite: the
  segment was published, then refused to load, so the database would not
  open. `Sq::train` now errors on such a scale, and a failed IVF-PQ build
  falls back to HNSW instead of retrying forever.
- A rewrite syncs the WAL before building: with a poisoned WAL every
  retry used to write a new orphan file pair before failing to publish.
- `Config::validate` requires `seal_rows ≤ cap / 2`, or fresh segments
  start at the merge cap and the segment count grows without bound.
- The Db thread catches a panicking `maintain` and logs it, so one bad
  collection doesn't stop maintenance for all of them.

## Addendum (2026-10-30): absorb into IVF-PQ, keep codes
Soak timings (INFO logs per background step) showed the expensive work
was not dropping dead rows but a detour: 100k seals merged into a ~330k
HNSW tier (graph from scratch, 7–8 s), later merged again into a freshly
trained IVF-PQ (k-means + PQ from scratch, 14–17 s), every ~5 minutes.

- **Plan step 0, absorb** (before the dead-share rule): when an IVF-PQ
  segment with room exists and `MERGE_FACTOR − 1` (3) small sealed HNSW
  segments have accumulated, fold them into the IVF-PQ segment with the
  most dead rows, once the rewrite changes ≥ 1/`MERGE_FACTOR` of the
  result (new plus dropped rows). Waiting for 3 keeps recent rows in
  small HNSW segments for a while: absorbing every seal at once put all
  data in IVF-PQ and cost ~0.4 ms of p99 median and ~3% recall in a
  30-minute soak; and with 3 the tier merge (4) never fires. The threshold bounds write
  amplification: the segment's f32 section is rewritten whole, which
  matters for big, wide segments.
- **Keep codes:** a rewrite puts its biggest IVF-PQ input first and starts
  from `IvfPq::retain(live rows)`: the same lists, PQ and SQ codes minus
  the dead rows, renumbered. Only the other inputs' rows are encoded,
  with the same quantizers (the 2026-10-29 reuse); a fresh train happens
  only when nlist would more than double, or with no IVF-PQ input.
- Tier merges and big-HNSW → IVF-PQ builds still run when there is no
  IVF-PQ segment with room (initial load, inner product, growth past
  `max_merge_rows`).
- Alternatives: raising the dead-row threshold (fewer rewrites, but more
  dead codes scanned per query); appending to the IVF-PQ segment in place
  (needs a mutable segment file format); both left for later.

## Addendum (2026-10-31): nprobe scales with each segment's nlist
`nprobe` was applied per segment as is, so recall depended on how
compaction split the rows: after absorb, one ~1M segment (nlist 775)
probed 32 lists where two half-size ones had probed 32 each, −1.6%
recall. Now `nprobe` means the lists probed in a fresh `ivf_rows`-row
segment (nlist √ivf_rows), and every IVF-PQ segment probes the same share
of its lists: `ceil(nprobe · nlist / √ivf_rows)`, at least 1, capped at
nlist by the index. With the default `ivf_rows` (1M) a fresh 1M segment
still probes 32; a 4M merge (nlist 2000) probes 64. `rerank` stays a
per-segment count. 30-minute soak: recall 0.749 → 0.792, mean p99
1.56 → 1.53 ms, 7.6% less throughput (RESULTS.md).
- Alternatives: a reference of the segment's own √rows (nlist is kept
  across rewrites, so this drifts as dead rows go); a global
  probe budget split across segments by size (needs all segments'
  centroid distances before scanning, a bigger change).

## Addendum (2026-10-31): lazy CRC check of the f32 vectors section
`Segment::open` checked every section CRC, reading the whole file. An
IVF-PQ segment keeps its f32 vectors mapped and reads them only for
`get`/`with_vector` and as rewrite input; at 100M vectors (Phase 8) that
section is tens of GB, so opening would take minutes and page it all in.

- **Decision:** `open` checks everything but the vector section (header,
  footer, layout, padding, ids and blob CRCs: all read at load anyway).
  `Segment::verify_vectors` checks that section. HNSW segments call it on
  load (they copy the vectors into RAM, so it costs one more pass over
  pages being read anyway). Loaded IVF-PQ segments are checked by
  `Collection::maintain`, one segment per step, after a due seal and
  before planning. A segment not yet checked, or whose check failed, is
  never picked by `plan`: a rewrite copies its f32 rows into a new file
  with fresh CRCs and would launder the corruption. A failure is logged
  (`tracing::error!` with the path); the segment keeps serving search
  from its codes, and `get` returns whatever the page holds (as before:
  mapped pages were never re-checked after open). No repair path.
- **Alternatives:** verify on first touch (in `vectors()`): puts a
  multi-second read on whichever query or `get` comes first, and makes
  every accessor fallible. A background thread inside `storage`: a second
  scheduler next to the Db thread, with its own shutdown and error
  reporting, for work `maintain` already sequences. A per-row or per-block
  CRC (verify only the rows a rewrite copies): a format change, and a
  rewrite copies nearly every row anyway.
