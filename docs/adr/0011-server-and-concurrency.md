# 0011: Collection registry, snapshot concurrency and the HTTP API

Date: 2026-10-28. Status: accepted.

## Context
Phase 4 turns the single-owner `Collection` (ADR 0008) into a server:
named collections that survive restarts (4.1), batch CRUD (4.2), many
readers alongside a writer (4.3), and an HTTP API with a bulk path (4.4).
Constraints: the WAL's `Batch` policy costs one ~4 ms F_FULLFSYNC per
flush, so one-at-a-time durable upserts top out near 270/s; HNSW `add`
takes `&mut self` (ADR 0003); the acceptance test needs "deleted ids never
appear in results" under concurrent load; Phase 6 adds size-triggered
sealing, a background IVF-PQ builder, compaction and parallel fan-out on
top of whatever this round does.

## Decision
- **Registry layout.** `<data>/LOCK` (exclusive `flock` through std
  `File::try_lock`, stable since Rust 1.89, so no `libc`), and one
  directory per collection under `<data>/collections/<name>/`, holding a
  `CONFIG` file next to the collection's own files. `CONFIG` is framed
  (magic `NPCOLCFG`, version, CRC) with a JSON body: the serde form of
  `collection::Config` minus the sync policy. It is written once, so the
  manifest stays v1 and seals never touch it. The name is the directory
  name: 1–64 of `[A-Za-z0-9_-]`, never starting with `.`.
- **Crash-safe create and drop by rename.** Create writes `CONFIG` into
  `.new-<name>` and renames it to `<name>`; drop renames `<name>` to
  `.drop-<name>` and then deletes it. Each rename (plus a dir fsync) is
  the commit point; open deletes every `.`-directory. Drop first
  `close()`s the collection: it waits for the write in flight, and later
  writes through a stale handle fail with `Closed` (HTTP 404).
- **One writer, any number of readers.** A collection's `writer` mutex is
  held by each write for its whole run, WAL sync included. Readers never
  touch it. They clone the current `Arc<Snapshot>` (a `RwLock` held only
  for the clone) and search that.
- **Snapshot = per-segment views.** A snapshot is a list of
  `View { lsn, seg: Arc<RwLock<Seg>>, rows, dead: Arc<RoaringBitmap> }`:
  sealed segments, then the mutable one. `Seg` (HNSW, ids, payloads) is
  shared between snapshots. Sealed `Seg`s never change. The mutable one
  only grows, under its write lock, which only the writer takes, in
  chunks of 1,024 rows (`Hnsw::add` is parallel inside a chunk), so a
  search waits at most one chunk's insert, never an fsync. `rows` is how
  many of the segment's rows that snapshot may return; rows the writer
  has added but not yet published are filtered out like tombstones.
- **Publish order.** A write: (1) validates the whole batch, including
  payload types across the batch (`PayloadStore::check`); (2) logs it
  with `Wal::append_batch`, one write and one sync; (3) appends rows to
  the mutable segment (invisible: past `rows`); (4) under the `live`
  map's write lock, updates `live`, builds the next snapshot with
  copy-on-write tombstones (`Arc::make_mut` clones a segment's bitmap
  once per write) and the new `rows`, and swaps it in. The write returns
  after (4), so any search that starts after the acknowledgement sees the
  delete or the new version, and a search sees either the old row or the
  new one, never both. `get` reads `live` and the snapshot under `live`'s
  read lock, so the two agree. Lock order: `live`, `snap`, a segment.
- **Batch durability.** `Wal::append_batch` encodes every record before
  writing any (`RecordTooLarge` writes nothing), then writes and syncs
  once. On a later failure a prefix of the batch may be durable, which is
  what `Error::Unknown` already means. Replay on open applies runs of
  upserts as batches too, so restarting after a million-point bulk load
  without a seal rebuilds the HNSW in parallel.
- **Payload types: an optional schema.** `Config.schema` declares field
  types (`int`, `float`, `bool`, `keyword`, `keyword_list`); every
  segment's `PayloadStore` starts with those columns, so a declared float
  accepts `10` then `10.5` in any order and the type can't drift between
  segments. Undeclared fields keep the first-value rule. This is the fix
  ADR 0005 deferred to collection config.
- **HTTP.** axum, JSON everywhere except `POST .../points/bulk`, whose
  body is blocks of `n u32 | dim u32 | payload_len u32 | ids u64×n |
  vectors f32×n·dim | JSON lines`, little-endian, all blocks one batch
  (docs/api.md). Every engine call runs on `spawn_blocking`; a panic there
  is a 500. Bodies are capped at 256 MiB. Server config is flags plus env
  vars only: data dir, bind address, sync policy.
- **Error mapping.** `Error::Unknown` and a poisoned WAL are 503 with a
  body saying the write may or may not be durable and that a restart
  recovers. Never a 2xx: the handler only answers 2xx when
  `upsert_batch`/`delete_batch` returned `Ok`. Bad input (dim, NaN,
  payload type, filter type, names, config) is 400, missing collection or
  point 404, existing collection 409, I/O and corruption 500.
- **Seal stays synchronous** (`POST .../seal`): it holds the writer lock
  while writing the HNSW, so writes wait and searches don't.

## For Phase 6
- A background builder or compactor publishes the same way the writer
  does: take the `writer` lock (or split out a smaller publish lock),
  clone the snapshot, replace `View`s, swap. Segment indexes in `live`
  must be remapped when a compaction removes or merges segments (today
  they stay stable because segments are only appended).
- `Seg.index` is a concrete `Hnsw`; an IVF-PQ segment makes it an enum.
- Size-triggered seal can run the write half (`write_segment`) without
  the writer lock, since sealed rows don't change. Only the manifest
  publish and the WAL cut need it, but `wal.rotate()` must still happen
  at an exact LSN.
- Each request is one WAL sync and one writer-lock hold, so concurrent
  single-point clients don't share a group commit (`ponytail:` in
  `collection.rs`). A write-combining queue would fix that if it matters.

## Alternatives
- **`RwLock<Collection>`**: searches would wait behind every write's
  fsync, which is exactly what 4.3 rules out.
- **Concurrent HNSW insert during search** (`add(&self)`): would remove
  the chunk wait, but needs preallocated storage in `index` (another
  agent's crate this round) and ADR 0003 chose `&mut` on purpose. The
  `rows` cutoff already makes publish atomic, so it can come later
  without changing this design.
- **Apply before the WAL sync and publish after** (pipelining several
  writers through group commit): more throughput for single-point
  writes, but in-memory state then runs ahead of the log and every error
  path has to undo it. Batches and the bulk endpoint get the throughput
  without that.
- **Coercing ints into float columns** on the first float: silent type
  changes, lost precision above 2^53, and types could still differ
  between segments. ADR 0005 already rejected it.
- **Config in the manifest (v2):** the manifest is rewritten at every
  seal by the collection, which doesn't otherwise need to know about it.
  A separate write-once file kept v1 intact.
- **A name → directory map file:** another file to keep consistent; the
  directory name is already unique and atomic to rename.
