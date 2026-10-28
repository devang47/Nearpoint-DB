# Segment payload sidecar (v1)

`<file>.payloads` next to `<file>.seg` (file id, `manifest.md`): the JSON payload of each row of
that segment. Code: `crates/engine/src/collection.rs`
(`encode_payloads`/`decode_payloads`). Framing (magic, version, length,
CRC32, atomic write) as in `manifest.md`. All integers little-endian.

## Body (magic `NPPAYLDS`, version 1)

| Size | Field |
|-----:|-------|
| 4 | rows `u32`; must equal the segment's vector count |
| per row, in row order: | |
| 4 | len `u32` |
| len | the payload as compact JSON text (UTF-8, an object), or nothing (len 0) for no payload |

Rows are written from `PayloadStore::get`, so values are in their stored
form (e.g. an int in a float column reads back as a float) and keys are
in `serde_json::Map` order (sorted; readers don't depend on it). On open
each row is re-inserted into a fresh `PayloadStore`; a row that doesn't
parse or conflicts with its column type, a wrong row count, or trailing
bytes is corrupt. Dead rows keep their payloads (the manifest's
tombstones hide them).

Cost: JSON text, re-parsed on open. A columnar section would load faster;
deferred until segment opens show up in profiles (ADR 0008).

## Version history

| Version | Date | Change |
|---------|------|--------|
| 1 | 2026-10-28 | Initial (task 3.3). Fixture: `crates/engine/testdata/payloads-v1`. |
