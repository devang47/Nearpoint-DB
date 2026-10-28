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

## Addendum (2026-10-28): striped node locks, `reserve`
- **Locks are striped:** 4096 mutexes, node `i` takes `i % 4096`. On
  macOS a `Mutex<()>` is 16 B inline plus a 64 B pthread mutex boxed on
  first lock, and every node gets locked during a build, so a mutex per
  node was ~80 B per vector (76 MiB at 1M, ~8 GB at 100M), more than half
  of layer 0's links. A writer still holds one lock at a time, so stripes
  can't deadlock; two writers wanting the same stripe at once (≈ 10
  threads / 4096) just wait briefly.
- **`Hnsw::reserve(n)`:** `add` grows its arrays like any `Vec`, so a
  build fed in batches regrows (old + new copies at once, up to 2x slack).
  Callers that know the total (the bench, the Phase 6 segment builder)
  reserve it exactly first.
- Per-vector cost at M = 16, dim 128, after both: vectors 512 B, layer 0
  132 B, level 1 B, upper-layer box 16 B + ≈ 5 B of upper lists on
  average (1/15 node-levels × 68 B, rounded up by malloc), locks ≈ 0:
  ≈ 666 B, i.e. ≈ 154 B on top of the f32 vector (was ≈ 234 B). Searches
  add a 4 B-per-node visited array per searching thread.
