# 0007: The WAL poisons itself on any write or fsync failure

Date: 2026-10-28. Status: accepted.

## Context
After a failed fsync the kernel may already have dropped the dirty pages,
so retrying can report success over lost data (PostgreSQL's "fsyncgate").
A failed partial write leaves bytes that stop replay at that point, which
would silently hide every record appended after it.

## Decision
Any write or fsync error poisons the `Wal`, and every later call returns
`Poisoned`. Recovery means reopening, which truncates the torn tail.

## Consequences
An append that returned an error may still have been persisted. The
caller sees an ambiguous outcome, never a lost acknowledged write. The
API layer must report such writes as "unknown" rather than "failed".

## Alternatives
- Retrying the fsync: unsafe for the reason above.
- Truncating back and continuing: more code, and still unsafe after a
  failed fsync.
