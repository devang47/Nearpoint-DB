---
name: disk-format
description: Checklist for adding or changing any Nearpoint on-disk format (WAL, segment, manifest, IVF lists, temp build files). Use whenever code writes bytes that a later process will read back.
---

# Add or change an on-disk format

1. **Header:** 4–8 magic bytes, a `u16`/`u32` format version, then the
   fields. Little-endian, explicit widths, no `usize` on disk.
2. **Checksums:** CRC32 (`crc32fast`) per record (WAL) or per section +
   footer (segments). Readers verify before trusting any length field.
3. **Reading:** unknown magic or a newer version → a typed error, never a
   panic. A torn tail (WAL) is truncated, not fatal.
4. **Atomic publish:** write to `<name>.tmp`, `sync_data`, rename,
   then fsync the parent directory. Readers never see half a file.
5. **Changing a format:** bump the version; keep reading the old version
   or document why not in an ADR.
6. **Document** it in `docs/formats/<name>.md`: byte layout table, version
   history, checksum coverage. Update the table in `docs/formats/README.md`.
7. **Test:** round-trip, a flipped bit is detected, a truncated file is
   detected, and an old-version fixture still loads.
