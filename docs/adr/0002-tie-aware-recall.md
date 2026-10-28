# 0002: Tie-aware recall

Date: 2026-10-28. Status: accepted.

## Context
Exact brute force on SIFT1M scored recall@10 = 0.9994 by id match. All 56
queries with a mismatch returned exactly the ground truth's 10 distances:
several base vectors sit at the same distance from the query at rank 10,
and the ground truth listed a different one of them than we did. SIFT
distances are integers below 2^24, so they're exact in f32; these are
real ties, not rounding.

## Decision
The headline recall@k counts a result as a hit when its **true** distance
(recomputed from the base vectors with the scalar kernel, never taken from
the index) is <= the distance of the k-th ground-truth neighbour. This is
big-ann-benchmarks' definition. The id-match recall is printed and
recorded beside it every time, so nothing is hidden.

## Alternatives
- Id match only: penalizes exact results for tie-breaking, so "exact"
  could never reach 1.0 on SIFT1M.
- Breaking ties by lowest id: matches the ground truth only if it was
  built that way, and it wasn't.
- Distances reported by the index: PQ indexes return approximate
  distances, which would inflate recall.
