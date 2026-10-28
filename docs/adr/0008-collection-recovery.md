# 0008: Collection directory, LSN coverage and the seal protocol

Date: 2026-10-28. Status: accepted.

## Context
Task 3.3 recovers a collection from sealed segments plus the WAL; 3.4
crashes it at random points. That takes three things the WAL and segment
v1 formats don't answer on their own: which WAL records a sealed segment
already holds, which segment files are live, and where payloads and
tombstones of sealed rows live once the WAL that described them is
deleted. Phase 4.3 (readers clone an `Arc<Snapshot>`) and 6.1 (seal at a
size) build on the answers.

## Decision
- **One directory per collection:**
  `MANIFEST`, `wal/<lsn>.wal`, and per sealed segment `<lsn>.seg`
  (segment v1) plus `<lsn>.payloads`. `<lsn>` is 20 digits, like WAL
  files: the last LSN the seal covered.
- **The manifest is the commit point.** It lists the live segments, a
  tombstone bitmap per segment, and `covered`: every write with
  LSN <= `covered` is in those segments and tombstones, and no later one
  is. Rewritten whole, atomically (tmp, `sync_data`, rename, dir fsync).
  Files the manifest doesn't list are garbage and are deleted on open.
- **Seal protocol** (`Collection::seal`), each step durable before the next:
  1. `wal.rotate()`: syncs the current file and starts `last + 1` in a
     new one, so WAL files split exactly at the seal point.
  2. Write `<last>.seg` and `<last>.payloads` (skipped if the mutable
     segment has no rows: a seal of deletes only just persists tombstones).
  3. Publish the manifest with `covered = last`.
  4. `wal.remove_before(last + 1)`.

  A crash before 3 leaves orphan files (deleted on open) and the old
  manifest, and the WAL still holds everything. A crash after 3 leaves
  WAL files the manifest covers; open skips their records and finishes
  step 4. Each record is applied exactly once either way.
- **Recovery** (`Collection::open`): read the manifest, delete garbage,
  load its segments and tombstones, rebuild the id map, then replay WAL
  records with LSN > `covered`. The first replayed LSN must be
  `covered + 1` (or the WAL must be empty and resume there); anything else
  means lost WAL files and is `Corrupt`, never silently accepted.
- **Payloads: a sidecar file per segment**, JSON text per row, rebuilt
  into a `PayloadStore` on open. **Tombstones of sealed rows live in the
  manifest**: they change after the segment is sealed, and segment files
  are immutable.
- **Acknowledgement:** a write is applied in memory only after the WAL
  acknowledged it. Validation (dim, finite values, payload types via
  `PayloadStore::check`) happens before the append, so replay can't fail
  on a record that was accepted. A WAL error after the record may have
  been written returns `Error::Unknown` (ADR 0007); only
  `RecordTooLarge`, rejected before writing, is a plain failure.
- **Generic framing** (`storage::framed`): magic, version, length, body,
  CRC32, written atomically. Both new files use it with their own magic
  and version.

## Alternatives
- **LSN in segment file names, no manifest:** the "live set" would be
  whatever files exist, so a crash mid-seal leaves a half-committed
  segment indistinguishable from a real one, and there's still nowhere
  to put tombstones of older segments.
- **Segment v2 with a payload section:** one file per segment, but it
  changes a format with a committed fixture and puts JSON knowledge into
  `storage`. The sidecar can fold into a v2 later without changing
  recovery.
- **Per-segment tombstone files:** avoids rewriting all bitmaps every
  seal, but makes the commit point several files. Revisit when manifests
  get big (marked `ponytail:` in `collection.rs`).
- **A columnar payload section:** faster opens, much more format. JSON
  per row is enough until segment opens show up in profiles.

## Consequences
- One process owns a directory at a time (no lock file yet); open
  deletes `*.tmp` and unlisted segment files.
- Phase 4.3: sealed segments are immutable except their tombstone bitmap,
  so a snapshot can share them and copy-on-write `dead`.
- Phase 6: compaction drops a segment by publishing a manifest without
  it; the next open (or the compactor) deletes the files.
