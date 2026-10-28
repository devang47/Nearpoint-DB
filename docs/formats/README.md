# On-disk formats

Every file Nearpoint writes starts with magic bytes and a format version
and carries CRC32 checksums (see `AGENTS.md`). One document per format:

| Format | Doc | Status |
|--------|-----|--------|
| WAL | `wal.md` | v1 (task 3.1) |
| Segment | `segment.md` | v2 (v1: task 3.2; v2 adds vector-less IVF-PQ segments) |
| Data directory, collection directory, manifest, `CONFIG` | `manifest.md` | manifest v2 (v1: task 3.3; v2: file ids, task 6.3), `CONFIG` v1 (task 4.1) |
| Segment payload sidecar | `payloads.md` | v1 (task 3.3) |

The `disk-format` skill has the checklist for adding or changing one.
