# 0010: Filtered search in HNSW and Flat

Date: 2026-10-28. Status: accepted.

## Context
Tasks 7.3 and 7.4 apply a filter bitmap (ADR 0005) during search. The
acceptance is recall@10 ≥ 0.9 at selectivity 1%, 10% and 50%. In Phase 6
every segment gets its own bitmap, so the choice has to be made per
segment from what that segment knows: its size, the bitmap's cardinality
and `ef`. Unfiltered search must keep its results and cost.

## Decision
- **Flat** iterates the bitmap instead of checking every id: each rayon
  task walks `filter.range(rows)` for its row block.
- **HNSW walks through filtered-out nodes but only keeps matches.** The
  layer-0 beam search has two heaps: `todo` (what to expand) admits any
  node nearer than the worst kept match (or every node while fewer than
  `ef` matches are kept); `best` admits only matches. It stops when `ef`
  matches are all nearer than the nearest unexpanded node, or `todo` is
  empty. The greedy descent through upper layers is unfiltered. This is
  hnswlib's filtered search. The filter is a closure argument of
  `search_layer`; unfiltered callers pass `|_| true`, which the compiler
  removes, so the unfiltered path runs the same code as before.
- **Planner (7.4): brute force over the bitmap when it computes fewer
  distances.** `Hnsw::search` scans the filter's ids exactly when
  `filtered² <= 8 · max(ef, k) · len`; otherwise it walks.
  `Hnsw::plan` exposes the choice and `Hnsw::search_with` forces one,
  so the bench can measure both sides.

## How the threshold was chosen
Counted, not timed: other builds shared the CPU, and distance counts
don't depend on that. `hnsw::distances()` is a per-thread counter of
distances computed by searches (one per newly visited node; one per id
for brute force). SIFT1M, M=16, efc=200, 1000 queries, uniform random
filters, `filtered` bench binary, mean distances per query:

| selectivity | ids    | walk ef=64 | walk ef=128 | brute force |
|-------------|--------|-----------:|------------:|------------:|
| 0.1%        | 998    | 204,063    | 299,540     | 998         |
| 0.5%        | 4,933  | 73,031     | 116,198     | 4,933       |
| 1%          | 9,975  | 44,953     | 73,168      | 9,975       |
| 2%          | 20,012 | 26,854     | 44,494      | 20,012      |
| 5%          | 49,821 | 13,534     | 23,053      | 49,821      |
| 10%         | 99,722 | 7,866      | 13,576      | 99,722      |
| 50%         | 500,179| 2,147      | 3,778       | 500,179     |

Walk cost falls as about selectivity^-0.75 and grows about as √ef.
Interpolating in log-log, the curves cross at ~2.4% (ef 64, ~24k ids)
and ~3.2% (ef 128, ~32k ids). `filtered² = c · ef · len` puts the
crossover at `√(c · ef · len)`, and c = 8 matches both (22.6k and 32k).
The √len form also gives sensible small segments: below ~√(8·ef·len)
ids, scanning is cheaper than any walk, and a segment of a few hundred
vectors always scans.

Wall-clock check after the merge (quiet machine, `filtered --plan
auto,graph,brute`, p50 at ef 64): at 2% brute force 0.39 ms vs walk
2.71 ms; at 5% brute force 3.50 ms vs walk 1.32 ms. The time crossover
is between 2% and 5%, so the distance-count threshold stands
(`WALK_COST` = 8). Caveat: the bench uses one filter for every query, so
a small filter's vectors stay in cache (2%: 20 ns per scanned distance;
5%: 70 ns). With a different filter per query, brute force would be
slower and the crossover lower.

## Recall (SIFT1M, same setup, tie-aware recall@10 as in ADR 0002)
Walk recall is 0.998–0.999 from 0.1% to 10% and 0.981 (ef 64) /
0.995 (ef 128) at 50%; brute force is exact. Unfiltered recall at
ef 64 is unchanged.

## Alternatives
- **Post-filtering** (search unfiltered with a larger `ef`, drop
  non-matches): needs `ef ≈ k / selectivity` to keep k results, and
  returns fewer than k with no warning when the guess is short. The
  in-traversal filter costs the same when selectivity is high and
  degrades gracefully when it isn't.
- **ACORN-style 2-hop expansion** (expand only matching nodes, but look
  at neighbours' neighbours to stay connected): fewer distance
  computations at low selectivity, but needs denser graphs (ACORN-γ
  builds with more links) or more complex expansion, and the planner
  already handles the low-selectivity end exactly. Revisit if a
  workload has many filters in the 2–10% band where walks are costly.
- **A fixed selectivity cutoff** (say 1%): wrong for small segments,
  where scanning a few thousand ids beats any walk, and ignores `ef`.
- **A user-facing knob:** nobody has asked for one; the constant is in
  one place.

## Not covered
- Correlated filters (matches clustered away from the query) are
  harder for the walk than the uniform random filters measured here.
  The walk stays correct (it just visits more), and the planner only
  sees cardinality.
- IVF-PQ filtering (the other half of 7.3) is a separate change.

## Addendum (2026-10-28): correlated filters
`filtered --filter far` keeps the base vectors farthest (L2) from the
mean of the queries: the matches sit together at the edge of the data,
far from most queries. SIFT1M, M=16, efc=200, 1000 queries, forced walk
(`--plan graph`), tie-aware recall@10 and mean distances per query:

| selectivity | uniform ef 64 | far ef 64 | far ef 128 | far: distances ef 64 (uniform) | auto plan |
|-------------|--------------:|----------:|-----------:|-------------------------------:|-----------|
| 1%  | 0.9999 | 0.9999 | 0.9999 | 293,120 (45,393) | brute force (exact) |
| 10% | 0.9986 | 0.9995 | 0.9998 | 110,191 (7,941)  | walk |
| 50% | 0.9803 | 0.9920 | 0.9978 | 10,721 (2,158)   | walk |

- **Recall holds** (≥ 0.99 everywhere, slightly higher than uniform: the
  walk can't stop until it holds `ef` matches, so it ends up exploring
  the matching region thoroughly). No change to the walk.
- **Cost doesn't.** Reaching the matching region takes 5–14x the
  distances of a uniform filter. At 10% the walk computes 110k
  distances where brute force over the 100k ids would compute 100k, yet
  the planner walks: it only sees cardinality, so it is wrong by about
  1.1x (ef 64) to 1.3x (ef 128) on this filter, not catastrophically.
- **If it matters** (Phase 6 soak or real payloads show it): cap the
  walk at a distance budget of `filtered` and fall back to brute force
  when it runs out. That bounds any filter at ≈ 2x the cheaper plan
  without having to estimate correlation. Not done now: recall is met
  and the worst measured overshoot is 1.3x.
- Not measured: filters correlated *with* the query (matches near it),
  which only help the walk, and per-query filters (the bench uses one
  filter for all queries).
