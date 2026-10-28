//! HNSW (Malkov & Yashunin, 2016): a stack of proximity graphs, each layer
//! a sparser sample of the one below. Search descends greedily from the top
//! layer to find a good entry point, then runs a best-first beam search of
//! width `ef` on layer 0.
//!
//! Layout. Neighbour lists are fixed-capacity slots of `AtomicU32`:
//! `[count, id, id, ...]`. Layer 0 (every node, capacity `2 * m`) is one flat
//! array; upper layers (capacity `m`) are a small boxed array per node that
//! has them, so the ~94% of nodes living only on layer 0 cost nothing extra.
//!
//! Concurrency. `add` inserts a batch in parallel. A writer locks the one
//! node whose list it is changing (never two at once, so no deadlock). The
//! locks are striped (`LOCK_STRIPES` mutexes, node `i` uses `i % stripes`):
//! a `Mutex` per node would cost 16 B plus a 64 B pthread mutex boxed on
//! first use, ~80 B per vector.
//! Readers don't lock: every slot is atomic, so a read racing a write sees a
//! slightly stale list, which costs a little graph quality during build but
//! is never unsafe. `search` takes `&self` and `add` takes `&mut self`, so
//! queries never overlap with inserts and read a settled graph.
//!
//! One global lock (`grow`) serializes the rare inserts whose level is above
//! the current top (~1 in m^top). Without it, two such nodes inserted at
//! once each start from the old entry point, never see each other, and stay
//! unlinked on the new top layers. hnswlib does the same.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::ops::Range;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::*};
use std::sync::{Mutex, MutexGuard, PoisonError};

use rayon::prelude::*;

use crate::flat::top_k;
use crate::visited::VISITED;
use crate::{Error, Hit, Index, Metric, Result, SearchParams, batch_ids, blob, check_dim};

#[derive(Debug, Clone, Copy)]
pub struct Config {
    /// Neighbours per node on upper layers; layer 0 allows `2 * m`.
    pub m: usize,
    /// Beam width while inserting. Higher: better graph, slower build.
    pub ef_construction: usize,
    /// Seeds each node's level. A node's level depends only on (seed, id),
    /// so levels are reproducible even though parallel insertion order isn't.
    pub seed: u64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            m: 16,
            ef_construction: 200,
            seed: 42,
        }
    }
}

/// No node is on more than this many layers. With m = 16 a level-16 node
/// would need ~16^16 vectors.
const MAX_LEVEL: u8 = 16;
const NO_ENTRY: u64 = u64::MAX;
const MAGIC: &[u8; 4] = b"NPHW";
/// Node locks, shared by `id % LOCK_STRIPES`. With ~10 writers two rarely
/// want the same stripe at once, and it's a fixed 64 KiB (plus 64 B per
/// stripe ever locked) whatever the index size.
const LOCK_STRIPES: usize = 4096;

pub struct Hnsw {
    dim: usize,
    metric: Metric,
    cfg: Config,
    vectors: Vec<f32>,
    levels: Vec<u8>,
    /// `n * (1 + 2m)` slots.
    layer0: Vec<AtomicU32>,
    /// Per node: `level * (1 + m)` slots for layers `1..=level`.
    upper: Vec<Box<[AtomicU32]>>,
    locks: Box<[Mutex<()>]>,
    /// `level << 32 | id` of the entry point (a node on the top layer).
    /// Read lock-free; only changed by a holder of `grow`.
    entry: AtomicU64,
    grow: Mutex<()>,
}

impl Hnsw {
    pub fn new(dim: usize, metric: Metric, cfg: Config) -> Self {
        assert!(dim > 0 && cfg.m >= 2 && cfg.ef_construction >= 1);
        Hnsw {
            dim,
            metric,
            cfg,
            vectors: Vec::new(),
            levels: Vec::new(),
            layer0: Vec::new(),
            upper: Vec::new(),
            locks: stripes(),
            entry: AtomicU64::new(NO_ENTRY),
            grow: Mutex::new(()),
        }
    }

    pub fn config(&self) -> Config {
        self.cfg
    }

    /// Makes room for `additional` more vectors, exactly. `add` grows its
    /// arrays like any `Vec` (doubling, then copying), so a build fed in
    /// batches peaks at old + new copies of every array and keeps up to 2x
    /// slack; a caller that knows the total reserves it once instead.
    pub fn reserve(&mut self, additional: usize) {
        let m = self.cfg.m;
        self.vectors.reserve_exact(additional * self.dim);
        self.levels.reserve_exact(additional);
        self.layer0.reserve_exact(additional * (1 + 2 * m));
        self.upper.reserve_exact(additional);
    }

    /// The stored vectors, row-major, in stored form (unit length for
    /// cosine). This is what a segment's vector section holds.
    pub fn vectors(&self) -> &[f32] {
        &self.vectors
    }

    /// The index blob: config, levels, every neighbour list and the entry
    /// point, but not the vectors (they live in the segment's vector
    /// section). Lists are written at full capacity, `[count, ids...]`
    /// exactly as in memory. Layout in docs/formats/segment.md.
    pub fn to_bytes(&self) -> Vec<u8> {
        let slots = self.layer0.len() + self.upper.iter().map(|u| u.len()).sum::<usize>();
        let mut out = Vec::with_capacity(48 + self.levels.len() + 4 * slots);
        blob::put_header(&mut out, MAGIC, self.dim, self.metric);
        blob::put_u32(&mut out, self.cfg.m as u32);
        blob::put_u32(&mut out, self.cfg.ef_construction as u32);
        blob::put_u64(&mut out, self.cfg.seed);
        blob::put_u64(&mut out, self.levels.len() as u64);
        blob::put_u64(&mut out, self.entry.load(Acquire));
        out.extend_from_slice(&self.levels);
        for s in self
            .layer0
            .iter()
            .chain(self.upper.iter().flat_map(|u| u.iter()))
        {
            blob::put_u32(&mut out, s.load(Relaxed));
        }
        out
    }

    /// Rebuilds an index from `to_bytes` output and its vectors (in stored
    /// form, as `vectors()` returned them; they are not normalized again).
    /// Takes ownership of the vectors, so loading from a mapped segment
    /// costs one copy in RAM (v1 keeps `Hnsw` owning a `Vec<f32>`).
    ///
    /// Everything is validated, so a corrupt blob is an error here rather
    /// than a panic or an out-of-bounds read during a later search.
    pub fn from_bytes(bytes: &[u8], vectors: Vec<f32>) -> Result<Hnsw> {
        let mut r = blob::Reader::new(bytes);
        let (dim, metric) = r.header(MAGIC)?;
        let m = r.u32()? as usize;
        let ef_construction = r.u32()? as usize;
        let seed = r.u64()?;
        let n = r.u64()?;
        let entry = r.u64()?;
        if m < 2 || ef_construction == 0 {
            return Err(Error::Corrupt("bad config"));
        }
        if n > u64::from(u32::MAX) {
            return Err(Error::Corrupt("more nodes than u32 ids"));
        }
        blob::check_vectors(&vectors, n, dim)?;
        let n = n as usize;

        let levels = r.take(n)?.to_vec();
        if levels.iter().any(|&l| l > MAX_LEVEL) {
            return Err(Error::Corrupt("level above MAX_LEVEL"));
        }
        // Sizes come from the blob, so every allocation below is bounded
        // by bytes actually present.
        let atomics = |b: &[u8]| -> Box<[AtomicU32]> {
            b.as_chunks::<4>()
                .0
                .iter()
                .map(|&c| AtomicU32::new(u32::from_le_bytes(c)))
                .collect()
        };
        let layer0 = atomics(r.take_n(n, 4 * (1 + 2 * m))?).into_vec();
        let upper = levels
            .iter()
            .map(|&l| Ok(atomics(r.take_n(l as usize, 4 * (1 + m))?)))
            .collect::<Result<Vec<_>>>()?;
        r.finish()?;

        let h = Hnsw {
            dim,
            metric,
            cfg: Config {
                m,
                ef_construction,
                seed,
            },
            vectors,
            levels,
            layer0,
            upper,
            locks: stripes(),
            entry: AtomicU64::new(entry),
            grow: Mutex::new(()),
        };
        h.validate()?;
        Ok(h)
    }

    /// What search and insert rely on: every list's count fits its
    /// capacity, every link is a node that exists on that layer, and the
    /// entry point is a node on the top layer (or absent iff empty).
    fn validate(&self) -> Result<()> {
        let n = self.levels.len();
        for node in 0..n as u32 {
            for layer in 0..=self.levels[node as usize] {
                let slots = self.slots(node, layer);
                let count = slots[0].load(Relaxed) as usize;
                if count >= slots.len() {
                    return Err(Error::Corrupt("list count over capacity"));
                }
                for s in &slots[1..=count] {
                    let id = s.load(Relaxed) as usize;
                    if id >= n || self.levels[id] < layer {
                        return Err(Error::Corrupt("link to a node not on its layer"));
                    }
                }
            }
        }
        let top = self.levels.iter().max().copied();
        let ok = match self.entry() {
            None => n == 0,
            Some((id, level)) => {
                self.entry.load(Relaxed) == (u64::from(level) << 32 | u64::from(id))
                    && (id as usize) < n
                    && self.levels[id as usize] == level
                    && top == Some(level)
            }
        };
        if !ok {
            return Err(Error::Corrupt("bad entry point"));
        }
        Ok(())
    }

    fn vector(&self, id: u32) -> &[f32] {
        &self.vectors[id as usize * self.dim..][..self.dim]
    }

    fn dist(&self, q: &[f32], id: u32) -> f32 {
        self.metric.distance(q, self.vector(id))
    }

    fn hit(&self, q: &[f32], id: u32) -> Hit {
        Hit {
            id,
            dist: self.dist(q, id),
        }
    }

    /// The `[count, ids...]` slots of `node` on `layer`.
    fn slots(&self, node: u32, layer: u8) -> &[AtomicU32] {
        if layer == 0 {
            let w = 1 + 2 * self.cfg.m;
            &self.layer0[node as usize * w..][..w]
        } else {
            let w = 1 + self.cfg.m;
            &self.upper[node as usize][(layer as usize - 1) * w..][..w]
        }
    }

    fn entry(&self) -> Option<(u32, u8)> {
        match self.entry.load(Acquire) {
            NO_ENTRY => None,
            e => Some((e as u32, (e >> 32) as u8)),
        }
    }

    /// Best-first search of one layer from `entries`; returns up to `ef`
    /// nearest found that `keep` accepts, as a max-heap (farthest on top).
    ///
    /// Rejected nodes are still expanded like any other, so a filtered
    /// search walks through them; only the result set is filtered. It stops
    /// once it holds `ef` nodes all nearer than the nearest unexpanded one,
    /// or nothing is left to expand. Unfiltered callers pass `|_| true`,
    /// which compiles away.
    fn search_layer(
        &self,
        q: &[f32],
        entries: &[Hit],
        ef: usize,
        layer: u8,
        keep: impl Fn(u32) -> bool,
    ) -> BinaryHeap<Hit> {
        VISITED.with_borrow_mut(|visited| {
            visited.reset(self.levels.len());
            let mut best = BinaryHeap::with_capacity(ef + 1);
            let mut todo = BinaryHeap::new();
            for &e in entries {
                visited.insert(e.id);
                if keep(e.id) {
                    best.push(e);
                }
                todo.push(Reverse(e));
            }
            while best.len() > ef {
                best.pop();
            }
            while let Some(Reverse(c)) = todo.pop() {
                if best.len() >= ef && best.peek().is_some_and(|far: &Hit| c.dist > far.dist) {
                    break; // nearest unexplored is worse than everything kept
                }
                let slots = self.slots(c.id, layer);
                let count = slots[0].load(Acquire) as usize;
                for slot in &slots[1..=count] {
                    let id = slot.load(Relaxed);
                    if !visited.insert(id) {
                        continue;
                    }
                    visited.distances += 1;
                    let h = self.hit(q, id);
                    if best.len() < ef || best.peek().is_some_and(|far| h < *far) {
                        todo.push(Reverse(h));
                        if keep(id) {
                            best.push(h);
                            if best.len() > ef {
                                best.pop();
                            }
                        }
                    }
                }
            }
            best
        })
    }

    /// Greedy (beam width 1) descent from the entry point through every
    /// layer above `stop`; returns where it lands, the start for `stop`.
    fn descend(&self, q: &[f32], entry: (u32, u8), stop: u8) -> Hit {
        let mut cur = self.hit(q, entry.0);
        for layer in (stop + 1..=entry.1).rev() {
            if let Some(h) = self.search_layer(q, &[cur], 1, layer, |_| true).pop() {
                cur = h;
            }
        }
        cur
    }

    /// The HNSW neighbour-selection heuristic: walking candidates nearest
    /// first, keep one only if it's nearer the base point than to every
    /// neighbour already kept. That spreads links in different directions
    /// instead of clustering them, which is what keeps the graph navigable.
    fn select(&self, sorted: &[Hit], m: usize) -> Vec<u32> {
        let mut kept: Vec<u32> = Vec::with_capacity(m);
        for c in sorted {
            if kept.len() == m {
                break;
            }
            let v = self.vector(c.id);
            if kept.iter().all(|&s| self.dist(v, s) > c.dist) {
                kept.push(c.id);
            }
        }
        kept
    }

    /// Overwrites a list. Caller holds the node's lock.
    fn write(slots: &[AtomicU32], ids: &[u32]) {
        for (slot, &id) in slots[1..].iter().zip(ids) {
            slot.store(id, Relaxed);
        }
        slots[0].store(ids.len() as u32, Release);
    }

    fn lock(&self, node: u32) -> MutexGuard<'_, ()> {
        lock(&self.locks[node as usize % LOCK_STRIPES])
    }

    /// Adds `new` to `node`'s list on `layer`, re-selecting the list with
    /// the heuristic when it's full.
    fn link(&self, node: u32, new: u32, layer: u8) {
        let _guard = self.lock(node);
        let slots = self.slots(node, layer);
        let count = slots[0].load(Relaxed) as usize;
        let cap = slots.len() - 1;
        if count < cap {
            slots[1 + count].store(new, Relaxed);
            slots[0].store(count as u32 + 1, Release);
            return;
        }
        let base = self.vector(node);
        let mut cands: Vec<Hit> = slots[1..=count]
            .iter()
            .map(|s| s.load(Relaxed))
            .chain([new])
            .map(|id| self.hit(base, id))
            .collect();
        cands.sort_unstable();
        Self::write(slots, &self.select(&cands, cap));
    }

    fn insert(&self, node: u32) {
        let level = self.levels[node as usize];
        let packed = (u64::from(level) << 32) | u64::from(node);
        // The top only ever rises, so if `level` isn't above it now it never
        // will be during this insert, and the lock isn't needed.
        let _grow = self
            .entry()
            .is_none_or(|(_, top)| level > top)
            .then(|| lock(&self.grow));
        let Some(entry) = self.entry() else {
            self.entry.store(packed, Release); // first node
            return;
        };

        let q = self.vector(node);
        let mut cur = vec![self.descend(q, entry, level)];
        for layer in (0..=level.min(entry.1)).rev() {
            let mut found = self
                .search_layer(q, &cur, self.cfg.ef_construction, layer, |_| true)
                .into_sorted_vec();
            found.retain(|h| h.id != node);
            // m links out on every layer; layer 0's 2m capacity is headroom
            // for the links other nodes add back.
            let neighbours = self.select(&found, self.cfg.m);
            {
                let _guard = self.lock(node);
                Self::write(self.slots(node, layer), &neighbours);
            }
            for &n in &neighbours {
                self.link(n, node, layer);
            }
            cur = found;
        }

        if level > entry.1 {
            self.entry.store(packed, Release); // we hold `grow`
        }
    }

    /// The plan `search` picks: brute force when the filter is small
    /// enough that scanning it computes fewer distances than a graph walk
    /// would (ADR 0010).
    pub fn plan(&self, k: usize, params: &SearchParams) -> Plan {
        match params.filter {
            Some(f) if brute_force_wins(f.len(), params.ef.max(k), self.len()) => Plan::BruteForce,
            _ => Plan::Graph,
        }
    }

    /// `search` with the plan given rather than picked (benchmarks use it
    /// to measure both sides of the planner's threshold).
    pub fn search_with(
        &self,
        query: &[f32],
        k: usize,
        params: &SearchParams,
        plan: Plan,
    ) -> Result<Vec<Hit>> {
        check_dim(self.dim, query.len())?;
        let Some(entry) = self.entry().filter(|_| k > 0) else {
            return Ok(Vec::new());
        };
        let q = self.metric.prepare(query);
        let n = self.len() as u32;
        if plan == Plan::BruteForce {
            let hits = match params.filter {
                Some(f) => top_k(f.range(..n).map(|id| self.hit(&q, id)), k),
                None => top_k((0..n).map(|id| self.hit(&q, id)), k),
            };
            let count = params
                .filter
                .map_or(u64::from(n), |f| f.range_cardinality(..n));
            VISITED.with_borrow_mut(|v| v.distances += count);
            return Ok(hits);
        }
        let start = self.descend(&q, entry, 0);
        let ef = params.ef.max(k);
        let mut hits = match params.filter {
            None => self.search_layer(&q, &[start], ef, 0, |_| true),
            Some(f) => self.search_layer(&q, &[start], ef, 0, |id| f.contains(id)),
        }
        .into_sorted_vec();
        hits.truncate(k);
        Ok(hits)
    }
}

/// How a search finds its results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Plan {
    /// Walk the graph, keeping only nodes in the filter (if any).
    Graph,
    /// Exact: the distance to every id in the filter (or every id).
    BruteForce,
}

/// Brute force over a filter costs one distance per id; a filtered walk
/// costs more the fewer ids match. Brute force runs when
/// `filtered² <= WALK_COST * ef * len`. On SIFT1M (M=16) the distance
/// counts of the two cross at ~2.4% (ef 64) and ~3.2% (ef 128), which
/// this fits with 8 at both widths (ADR 0010).
const WALK_COST: f64 = 8.0;

/// Whether brute force over `filtered` of `len` ids computes fewer
/// distances than a walk of width `ef`.
fn brute_force_wins(filtered: u64, ef: usize, len: usize) -> bool {
    let f = filtered as f64;
    f * f <= WALK_COST * ef as f64 * len as f64
}

/// Distances computed by `Hnsw` searches (and inserts) on this thread so
/// far; brute-force plans count one per filtered id. Benchmarks diff it
/// around a search.
pub fn distances() -> u64 {
    VISITED.with_borrow(|v| v.distances)
}

fn stripes() -> Box<[Mutex<()>]> {
    (0..LOCK_STRIPES).map(|_| Mutex::new(())).collect()
}

/// The mutexes guard no data, so a poisoned one is still fine to use.
fn lock(m: &Mutex<()>) -> MutexGuard<'_, ()> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Deterministic 64-bit mix (splitmix64 finalizer).
pub(crate) fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Level for node `id`: geometric with P(level >= l) = m^-l.
fn level_for(seed: u64, id: u32, m: usize) -> u8 {
    let u = (splitmix64(seed ^ u64::from(id).wrapping_mul(0x2545_F491_4F6C_DD1D)) >> 11) as f64
        / (1u64 << 53) as f64; // uniform in [0, 1)
    let level = -(1.0 - u).ln() / (m as f64).ln();
    (level as u8).min(MAX_LEVEL)
}

impl Index for Hnsw {
    fn dim(&self) -> usize {
        self.dim
    }

    fn metric(&self) -> Metric {
        self.metric
    }

    fn len(&self) -> usize {
        self.levels.len()
    }

    fn add(&mut self, vectors: &[f32]) -> Result<Range<u32>> {
        let ids = batch_ids(self.dim, self.len(), vectors)?;
        let (m, n_new) = (self.cfg.m, ids.len());
        self.vectors.reserve(vectors.len());
        for v in vectors.chunks_exact(self.dim) {
            self.vectors.extend_from_slice(&self.metric.prepare(v));
        }
        for id in ids.clone() {
            let level = level_for(self.cfg.seed, id, m);
            self.levels.push(level);
            self.upper.push(
                (0..level as usize * (1 + m))
                    .map(|_| AtomicU32::new(0))
                    .collect(),
            );
        }
        self.layer0
            .extend((0..n_new * (1 + 2 * m)).map(|_| AtomicU32::new(0)));

        let this = &*self;
        ids.clone().into_par_iter().for_each(|id| this.insert(id));
        Ok(ids)
    }

    fn search(&self, query: &[f32], k: usize, params: &SearchParams) -> Result<Vec<Hit>> {
        self.search_with(query, k, params, self.plan(k, params))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Flat, test_data};
    use roaring::RoaringBitmap;

    fn build(n: usize, dim: usize, metric: Metric, batches: usize) -> (Hnsw, Flat, Vec<f32>) {
        let data = test_data::vectors(n, dim, 7);
        let mut h = Hnsw::new(
            dim,
            metric,
            Config {
                m: 8,
                ef_construction: 100,
                seed: 1,
            },
        );
        let mut f = Flat::new(dim, metric);
        for batch in data.chunks(dim * n.div_ceil(batches)) {
            h.add(batch).unwrap();
            f.add(batch).unwrap();
        }
        (h, f, data)
    }

    fn recall(h: &Hnsw, f: &Flat, dim: usize, ef: usize) -> f64 {
        let (queries, k) = (100, 10);
        let qs = test_data::vectors(queries, dim, 99);
        let p = SearchParams {
            ef,
            ..SearchParams::default()
        };
        let mut hits = 0;
        for q in qs.chunks_exact(dim) {
            let truth: Vec<u32> = f.search(q, k, &p).unwrap().iter().map(|h| h.id).collect();
            hits += h
                .search(q, k, &p)
                .unwrap()
                .iter()
                .filter(|h| truth.contains(&h.id))
                .count();
        }
        hits as f64 / (queries * k) as f64
    }

    #[test]
    fn high_recall_vs_exact() {
        let (h, f, _) = build(5_000, 16, Metric::L2, 1);
        let r = recall(&h, &f, 16, 64);
        assert!(r >= 0.95, "recall {r}");
        // A wider beam can't be (meaningfully) worse.
        assert!(recall(&h, &f, 16, 256) >= r - 0.01);
    }

    #[test]
    fn incremental_batches_and_cosine() {
        let (h, f, _) = build(3_000, 12, Metric::Cosine, 5);
        assert_eq!(h.len(), 3_000);
        let r = recall(&h, &f, 12, 64);
        assert!(r >= 0.95, "recall {r}");
    }

    #[test]
    fn reserve_then_batches_never_regrow() {
        let (n, dim) = (1_000, 8);
        let data = test_data::vectors(n, dim, 3);
        let mut h = Hnsw::new(dim, Metric::L2, Config::default());
        h.reserve(n);
        let ptrs = |h: &Hnsw| (h.vectors.as_ptr(), h.layer0.as_ptr(), h.upper.as_ptr());
        let before = ptrs(&h);
        for batch in data.chunks(dim * 300) {
            h.add(batch).unwrap();
        }
        assert_eq!(ptrs(&h), before);
        assert_eq!(h.vectors.capacity(), n * dim);
        assert_eq!(h.layer0.capacity(), h.layer0.len());
    }

    #[test]
    fn graph_invariants() {
        let (h, _, _) = build(2_000, 8, Metric::L2, 3);
        let n = h.len() as u32;
        for node in 0..n {
            for layer in 0..=h.levels[node as usize] {
                let slots = h.slots(node, layer);
                let count = slots[0].load(Relaxed) as usize;
                assert!(count < slots.len());
                assert!(count > 0, "node {node} isolated on layer {layer}");
                for s in &slots[1..=count] {
                    let id = s.load(Relaxed);
                    assert!(id < n && id != node);
                    assert!(
                        h.levels[id as usize] >= layer,
                        "link to a node not on layer"
                    );
                }
            }
        }
        let (e, top) = h.entry().unwrap();
        assert_eq!(top, *h.levels.iter().max().unwrap());
        assert_eq!(h.levels[e as usize], top);
    }

    #[test]
    fn levels_are_geometric_and_reproducible() {
        let levels: Vec<u8> = (0..100_000).map(|id| level_for(42, id, 16)).collect();
        let above0 = levels.iter().filter(|&&l| l > 0).count() as f64 / 1e5;
        assert!((above0 - 1.0 / 16.0).abs() < 0.005, "{above0}");
        assert_eq!(levels[123], level_for(42, 123, 16));
    }

    fn search_all(h: &Hnsw, dim: usize) -> Vec<Vec<Hit>> {
        let p = SearchParams {
            ef: 48,
            ..SearchParams::default()
        };
        test_data::vectors(300, dim, 5)
            .chunks_exact(dim)
            .map(|q| h.search(q, 10, &p).unwrap())
            .collect()
    }

    #[test]
    fn blob_round_trip_searches_identically() {
        for metric in [Metric::L2, Metric::Cosine] {
            let (h, _, _) = build(2_000, 16, metric, 2);
            let bytes = h.to_bytes();
            let mut h2 = Hnsw::from_bytes(&bytes, h.vectors().to_vec()).unwrap();
            assert_eq!(h2.to_bytes(), bytes);
            assert_eq!((h2.len(), h2.dim(), h2.metric()), (h.len(), 16, metric));
            assert_eq!(search_all(&h2, 16), search_all(&h, 16));
            // A loaded index keeps accepting inserts.
            h2.add(&test_data::vectors(100, 16, 8)).unwrap();
            assert_eq!(h2.len(), 2_100);
        }
        let empty = Hnsw::new(3, Metric::L2, Config::default());
        let e2 = Hnsw::from_bytes(&empty.to_bytes(), Vec::new()).unwrap();
        assert!(e2.is_empty());
    }

    #[test]
    fn end_to_end_through_a_segment_file() {
        use storage::segment::{self, BlobKind, Segment, SegmentData};
        let (h, _, _) = build(1_500, 16, Metric::L2, 1);
        let ids: Vec<u64> = (0..h.len() as u64).map(|i| 1_000_000 + i).collect();
        let dir = std::env::temp_dir().join(format!("nearpoint-hnsw-e2e-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("hnsw.seg");
        segment::write(
            &path,
            &SegmentData {
                dim: h.dim(),
                metric: h.metric().code(),
                vectors: h.vectors(),
                ids: &ids,
                blob_kind: BlobKind::Hnsw,
                blob: &h.to_bytes(),
            },
        )
        .unwrap();

        let seg = Segment::open(&path).unwrap();
        assert_eq!(seg.blob_kind(), BlobKind::Hnsw);
        let metric = Metric::from_code(seg.metric()).unwrap();
        assert_eq!((seg.dim(), metric), (16, Metric::L2));
        let h2 = Hnsw::from_bytes(seg.index_blob(), seg.vectors().to_vec()).unwrap();
        assert_eq!(search_all(&h2, 16), search_all(&h, 16));
        let hit = h2
            .search(h.vector(42), 1, &SearchParams::default())
            .unwrap()[0];
        assert_eq!(seg.ids()[hit.id as usize], 1_000_042);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    fn small_blob() -> (Vec<u8>, Vec<f32>) {
        let (h, _, _) = build(200, 4, Metric::L2, 1);
        (h.to_bytes(), h.vectors().to_vec())
    }

    #[test]
    fn corrupt_blobs_are_errors() {
        let (bytes, vecs) = small_blob();
        let load = |b: &[u8]| Hnsw::from_bytes(b, vecs.clone());
        for len in 0..bytes.len() {
            assert!(load(&bytes[..len]).is_err(), "truncated to {len}");
        }
        let mut longer = bytes.clone();
        longer.push(0);
        assert!(load(&longer).is_err());
        assert!(Hnsw::from_bytes(&bytes, vecs[4..].to_vec()).is_err());

        let at = |b: &mut Vec<u8>, off: usize, v: &[u8]| b[off..off + v.len()].copy_from_slice(v);
        let mut v2 = bytes.clone();
        at(&mut v2, 4, &2u32.to_le_bytes());
        assert_eq!(load(&v2).err(), Some(Error::UnsupportedBlobVersion(2)));

        let (m, n) = (8, 200);
        let layer0 = 48 + n;
        let w0 = 4 * (1 + 2 * m);
        for (off, val, what) in [
            (layer0, 2 * m as u32 + 1, "list count over capacity"),
            (layer0 + 4, n as u32, "link to a node not on its layer"),
            (40, 7, "bad entry point"), // entry level 0, but higher layers exist
        ] {
            let mut b = bytes.clone();
            at(&mut b, off, &val.to_le_bytes());
            if off == 40 {
                at(&mut b, 44, &0u32.to_le_bytes());
            }
            assert_eq!(load(&b).err(), Some(Error::Corrupt(what)));
        }
        // A layer-0 node linked from an upper layer.
        let low = bytes[48..48 + n].iter().position(|&l| l == 0).unwrap() as u32;
        let first_upper = layer0 + n * w0;
        let mut b = bytes.clone();
        at(&mut b, first_upper, &1u32.to_le_bytes());
        at(&mut b, first_upper + 4, &low.to_le_bytes());
        assert_eq!(
            load(&b).err(),
            Some(Error::Corrupt("link to a node not on its layer"))
        );
    }

    proptest::proptest! {
        #[test]
        fn random_corruption_never_panics(
            flips in proptest::collection::vec((0usize..1 << 20, 1u8..=255), 1..8),
            cut in 0usize..1 << 20,
        ) {
            static BLOB: std::sync::LazyLock<(Vec<u8>, Vec<f32>)> =
                std::sync::LazyLock::new(small_blob);
            let (bytes, vecs) = &*BLOB;
            let mut b = bytes.clone();
            for (pos, x) in flips {
                let i = pos % b.len();
                b[i] ^= x;
            }
            if cut % 4 == 0 {
                b.truncate(cut % b.len());
            }
            if let Ok(h) = Hnsw::from_bytes(&b, vecs.clone()) {
                // Whatever loads must be searchable without panicking.
                for q in test_data::vectors(5, 4, 3).as_chunks::<4>().0 {
                    let _ = h.search(q, 10, &SearchParams {
                        ef: 32,
                        ..SearchParams::default()
                    });
                }
            }
        }
    }

    /// Every `step`-th id (offset by `offset`), plus ids past the end.
    fn every(step: u32, offset: u32, n: usize) -> RoaringBitmap {
        (0..n as u32 + 100)
            .filter(|i| (i + offset).is_multiple_of(step))
            .collect()
    }

    #[test]
    fn filtered_recall_both_plans() {
        let (h, f, _) = build(5_000, 16, Metric::L2, 1);
        let qs = test_data::vectors(50, 16, 99);
        // 1%, 10%, 50%.
        for step in [100, 10, 2] {
            let filter = every(step, 3, h.len());
            let p = SearchParams {
                ef: 64,
                filter: Some(&filter),
                ..SearchParams::default()
            };
            for plan in [Plan::Graph, Plan::BruteForce] {
                let mut hits = 0;
                for q in qs.as_chunks::<16>().0 {
                    let truth = f.search(q, 10, &p).unwrap();
                    let got = h.search_with(q, 10, &p, plan).unwrap();
                    assert!(got.iter().all(|g| filter.contains(g.id)));
                    if plan == Plan::BruteForce {
                        assert_eq!(got, truth);
                    }
                    hits += got.iter().filter(|g| truth.contains(g)).count();
                }
                let r = hits as f64 / 500.0;
                assert!(r >= 0.95, "step {step} {plan:?}: recall {r}");
            }
        }
    }

    #[test]
    fn planner_and_distance_counter() {
        let (h, _, _) = build(5_000, 16, Metric::L2, 1);
        let q = test_data::vectors(1, 16, 9);
        let p = |filter| SearchParams {
            ef: 64,
            filter,
            ..SearchParams::default()
        };
        assert_eq!(h.plan(10, &p(None)), Plan::Graph);
        let (small, big) = (every(100, 0, h.len()), every(2, 0, h.len()));
        assert_eq!(h.plan(10, &p(Some(&small))), Plan::BruteForce);
        assert_eq!(h.plan(10, &p(Some(&big))), Plan::Graph);
        // A bigger k widens the walk, so brute force wins for longer.
        assert_eq!(h.plan(2_500, &p(Some(&big))), Plan::BruteForce);

        let before = distances();
        h.search(&q, 10, &p(Some(&small))).unwrap();
        assert_eq!(distances() - before, 50); // ids past the end don't count
        let before = distances();
        h.search(&q, 10, &p(None)).unwrap();
        assert!((64..5_000).contains(&(distances() - before)));
    }

    proptest::proptest! {
        #[test]
        fn filtered_search_stays_in_the_filter(
            picks in proptest::collection::vec(0u32..2_100, 0..300),
            k in 0usize..30,
            seed in 0u64..1000,
        ) {
            static INDEX: std::sync::LazyLock<(Hnsw, Flat, Vec<f32>)> =
                std::sync::LazyLock::new(|| build(2_000, 8, Metric::L2, 1));
            let (h, f, _) = &*INDEX;
            let filter: RoaringBitmap = picks.into_iter().collect();
            let q = test_data::vectors(1, 8, seed);
            let p = SearchParams { ef: 32, filter: Some(&filter), ..SearchParams::default() };
            let exact = f.search(&q, k, &p).unwrap();
            proptest::prop_assert_eq!(&h.search_with(&q, k, &p, Plan::BruteForce).unwrap(), &exact);
            let walk = h.search_with(&q, k, &p, Plan::Graph).unwrap();
            proptest::prop_assert!(walk.len() <= exact.len());
            proptest::prop_assert!(walk.iter().all(|g| filter.contains(g.id)));
            proptest::prop_assert!(walk.is_sorted());
        }
    }

    #[test]
    fn edge_cases() {
        let p = SearchParams::default();
        let mut h = Hnsw::new(2, Metric::L2, Config::default());
        assert!(h.search(&[0.0, 0.0], 5, &p).unwrap().is_empty());
        h.add(&[1.0, 1.0]).unwrap();
        let hits = h.search(&[0.0, 0.0], 5, &p).unwrap();
        assert_eq!((hits.len(), hits[0].id, hits[0].dist), (1, 0, 2.0));
        assert!(h.search(&[0.0, 0.0], 0, &p).unwrap().is_empty());
        assert!(h.search(&[0.0], 1, &p).is_err());
        assert!(h.add(&[1.0, 2.0, 3.0]).is_err());
    }
}
