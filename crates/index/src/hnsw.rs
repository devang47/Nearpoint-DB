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
//! node whose list it is changing (never two at once, so no deadlock).
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

use crate::visited::VISITED;
use crate::{Hit, Index, Metric, Result, SearchParams, batch_ids, check_dim};

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
    locks: Vec<Mutex<()>>,
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
            locks: Vec::new(),
            entry: AtomicU64::new(NO_ENTRY),
            grow: Mutex::new(()),
        }
    }

    pub fn config(&self) -> Config {
        self.cfg
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
    /// nearest found, as a max-heap (farthest on top).
    fn search_layer(&self, q: &[f32], entries: &[Hit], ef: usize, layer: u8) -> BinaryHeap<Hit> {
        VISITED.with_borrow_mut(|visited| {
            visited.reset(self.levels.len());
            let mut best = BinaryHeap::with_capacity(ef + 1);
            let mut todo = BinaryHeap::new();
            for &e in entries {
                visited.insert(e.id);
                best.push(e);
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
                    let h = self.hit(q, id);
                    if best.len() < ef || best.peek().is_some_and(|far| h < *far) {
                        todo.push(Reverse(h));
                        best.push(h);
                        if best.len() > ef {
                            best.pop();
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
            if let Some(h) = self.search_layer(q, &[cur], 1, layer).pop() {
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
        lock(&self.locks[node as usize])
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
                .search_layer(q, &cur, self.cfg.ef_construction, layer)
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
        self.locks.extend((0..n_new).map(|_| Mutex::new(())));

        let this = &*self;
        ids.clone().into_par_iter().for_each(|id| this.insert(id));
        Ok(ids)
    }

    fn search(&self, query: &[f32], k: usize, params: &SearchParams) -> Result<Vec<Hit>> {
        check_dim(self.dim, query.len())?;
        let Some(entry) = self.entry().filter(|_| k > 0) else {
            return Ok(Vec::new());
        };
        let q = self.metric.prepare(query);
        let start = self.descend(&q, entry, 0);
        let mut hits = self
            .search_layer(&q, &[start], params.ef.max(k), 0)
            .into_sorted_vec();
        hits.truncate(k);
        Ok(hits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Flat, test_data};

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
        let p = SearchParams { ef };
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
