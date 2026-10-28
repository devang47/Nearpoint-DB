# On-disk formats

Every file Nearpoint writes starts with magic bytes and a format version
and carries CRC32 checksums (see `AGENTS.md`). One document per format:

| Format | Doc | Status |
|--------|-----|--------|
| WAL | `wal.md` | planned (task 3.1) |
| Segment | `segment.md` | planned (task 3.2) |
| Collection manifest | `manifest.md` | planned (task 4.1) |

The `disk-format` skill has the checklist for adding or changing one.
