# WAL format (v1)

Code: `crates/storage/src/wal.rs`. All integers little-endian, explicit
widths; no `usize` on disk.

## Directory

A WAL is a directory of files named by the LSN of their first record,
zero-padded to 20 digits: `00000000000000000001.wal`. File *i* holds
LSNs `[first_i, first_{i+1})`; the last file is the one being appended to.
LSNs start at 1, increase by exactly 1 per record, and continue across
files and reopens. `*.wal.tmp` files are leftovers of a crashed file
creation and are deleted on open; other names are ignored.

## File header (16 bytes)

| Offset | Size | Field   | Value                                  |
|-------:|-----:|---------|----------------------------------------|
| 0      | 8    | magic   | `NPNT_WAL` (ASCII)                     |
| 8      | 4    | version | `u32` = 1                              |
| 12     | 4    | crc     | `u32` CRC32 (crc32fast) of bytes 0..12 |

Files are published atomically: the header is written to `<name>.tmp`,
`sync_data`'d, renamed to `<name>`, then the directory is fsynced. So a
`.wal` file always has a complete header, and a bad one is an error, not
a torn write.

## Record (17-byte header + body)

| Offset | Size  | Field | Notes                                         |
|-------:|------:|-------|-----------------------------------------------|
| 0      | 4     | crc   | `u32` CRC32 of bytes 4..17+len (all below)    |
| 4      | 4     | len   | `u32` body length; > 64 MiB is invalid        |
| 8      | 1     | kind  | 1 = Upsert, 2 = Delete; others invalid in v1  |
| 9      | 8     | lsn   | `u64`; must equal previous LSN + 1            |
| 17     | len   | body  | per kind, below                               |

Upsert body (`len = 16 + 4*dim + payload_len`):

| Size        | Field       |
|------------:|-------------|
| 8           | id `u64`    |
| 4           | dim `u32`   |
| 4 × dim     | vector, `f32` each |
| 4           | payload_len `u32` |
| payload_len | payload (opaque bytes; metadata from Phase 7) |

Delete body (`len = 8`): id `u64`.

**Checksum coverage:** the record CRC covers len, kind, lsn and the whole
body, i.e. everything in the record except the CRC itself. The header
CRC covers magic and version.

## Recovery rules (on `Wal::open`)

Files are replayed in LSN order. A record is valid only if: the 17-byte
header is present, `len <= 64 MiB`, the full body is present, the CRC
matches, the LSN is the expected next one, the kind is known, and the
body parses to exactly `len` bytes.

At the first invalid record (or a file whose name isn't the expected next
LSN), replay stops: that file is truncated at the end of the last valid
record (`set_len` + `sync_all`), every later file is deleted (they can't
be trusted after a gap), and the directory is fsynced. The open result
reports `dropped_bytes` and `dropped_files` so the caller can log it.
`dropped_files > 0` means corruption before the tail, not just a torn
write. The next append continues at last valid LSN + 1.

File-level errors are typed and never panic: `BadMagic`,
`UnsupportedVersion` (any version other than 1, checked before the header
CRC so a future, longer header still reports its version), and
`CorruptHeader` (short header or bad header CRC).

After any failed write or fsync the WAL is **poisoned**: every later
call returns `Poisoned`, because the file may now hold a partial record
or lost pages (cf. PostgreSQL's fsync-retry bug). Reopen to recover.

## Sync policies

| Policy             | Append returns when            | fsyncs              |
|--------------------|--------------------------------|---------------------|
| `Always`           | record is durable              | one per append, under the log lock |
| `Batch` (default)  | record is durable              | group commit: one in-flight fsync at a time; appenders that arrive meanwhile wait and share the next one |
| `Interval(d)`      | record is in the page cache    | background thread every `d` |

`Interval` can lose up to the last interval of acknowledged appends on
power loss or kernel panic (a process crash loses nothing: the data is in
the OS page cache). `sync()` makes everything appended so far durable
under any policy; `rotate()` syncs the old file before switching;
dropping a `Wal` does a final best-effort sync.

## Durability on macOS

Plain `fsync(2)` on macOS does not flush the drive's write cache; only
`fcntl(fd, F_FULLFSYNC)` does. Checked in the Rust 1.98.1 std source
(`library/std/src/sys/fs/unix.rs`, read from the rust-docs `src/` pages
since rust-src isn't installed): on `target_vendor = "apple"`, both
`File::sync_all` (`fsync`) and `File::sync_data` (`datasync`) call
`libc::fcntl(fd, libc::F_FULLFSYNC)`. So the WAL uses `sync_data`
directly with no libc call. std has **no fallback** to `fsync` if
`F_FULLFSYNC` fails (e.g. on filesystems that don't support it); such an
error surfaces as `WalError::Io` and poisons the WAL. APFS supports it,
including on directory fds (used for directory fsync after creating,
renaming or deleting WAL files; covered by the tests).

## Version history

- **v1** (2026-10-28, task 3.1): initial format.
