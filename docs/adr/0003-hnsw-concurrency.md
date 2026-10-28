# 0003: HNSW memory layout and concurrent build

Date: 2026-10-28. Status: accepted.

## Context
HNSW has to build 1M vectors in parallel (Phase 2 target ≤ 10 min) and
later serve as the mutable in-RAM segment (Phase 6). `unsafe` isn't
allowed in `index`, and no concurrency crates beyond rayon.

## Decision
- **Fixed-capacity atomic slots.** Each list is `[count, id...]` of
  `AtomicU32`: layer 0 (capacity 2M) is one flat array for all nodes, and
  upper layers (capacity M) are a boxed array per node that has them.
  Relaxed atomic loads compile to plain loads on aarch64, so searching
  costs no more than with `u32` slots.
- **Readers never lock.** A read racing a write can see a stale list.
  That costs a little build-time quality but can't be unsafe: every slot
  always holds some valid id.
- **Writers lock one node at a time** (a `Mutex<()>` per node), so they
  can't deadlock. A full list is re-pruned with the selection heuristic
  under that lock.
- **A `grow` lock serializes inserts above the current top layer.** The
  invariants test failed without it: two such nodes inserted at once each
  start from the old entry point, never see each other, and stay unlinked
  on the new layers. They're about 1 in M^top, so the lock costs nothing
  measurable. hnswlib does the same.
- **Level = hash(seed, id).** Levels are reproducible from the seed even
  though parallel insertion order isn't, and no RNG has to be shared
  between threads.
- **`search(&self)` and `add(&mut self)`.** The borrow checker guarantees
  queries never overlap inserts. Phase 6 adds concurrent reads during
  writes at the segment level (snapshots), not inside the graph.

## Alternatives
- `Mutex<Vec<u32>>` per list: a lock on every hop of every search, plus a
  heap allocation per list.
- Freezing to plain `u32` arrays after build: faster to read only in
  theory (atomics already compile to plain loads), and it breaks
  incremental `add`.
- A seeded `rand` RNG for levels: fine for one thread, but with a parallel
  build the order of draws depends on scheduling.
