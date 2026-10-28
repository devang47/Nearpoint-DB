//! The `Index` trait and its flat, HNSW and IVF-PQ implementations.
//!
//! Indexes own their vectors and hand out dense ids `0..len()` in insertion
//! order; mapping to user-facing ids is the engine's job.

use std::borrow::Cow;
use std::cmp::Ordering;
use std::ops::Range;

pub mod flat;
pub mod hnsw;
mod visited;

pub use flat::Flat;
pub use hnsw::Hnsw;

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum Error {
    #[error("vector has {got} components, index dim is {expected}")]
    DimMismatch { expected: usize, got: usize },
    #[error("index is full (u32 ids)")]
    Full,
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// Squared Euclidean distance.
    L2,
    /// Negated inner product, so smaller is nearer like the others.
    InnerProduct,
    /// `1 - cos`. Vectors are normalized on insert and queries on search,
    /// so this is one dot product per comparison.
    Cosine,
}

impl Metric {
    #[inline]
    fn distance(self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Metric::L2 => kernels::l2_sq(a, b),
            Metric::InnerProduct => -kernels::dot(a, b),
            Metric::Cosine => 1.0 - kernels::dot(a, b),
        }
    }

    /// The form a vector is stored and compared in: unit length for
    /// cosine (a zero vector stays zero), untouched otherwise.
    fn prepare(self, v: &[f32]) -> Cow<'_, [f32]> {
        if self != Metric::Cosine {
            return Cow::Borrowed(v);
        }
        let norm = kernels::dot(v, v).sqrt();
        if norm == 0.0 {
            return Cow::Borrowed(v);
        }
        Cow::Owned(v.iter().map(|x| x / norm).collect())
    }
}

/// A search result: vector id and its distance under the index's metric.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    pub id: u32,
    pub dist: f32,
}

// Nearer first; equal distances order by id. Max-heaps of `Hit` therefore
// evict the farthest (and, on ties, the larger id) first.
impl Eq for Hit {}
impl Ord for Hit {
    fn cmp(&self, o: &Self) -> Ordering {
        self.dist.total_cmp(&o.dist).then(self.id.cmp(&o.id))
    }
}
impl PartialOrd for Hit {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SearchParams {
    /// HNSW candidate-list size (`ef_search`); raised to `k` if smaller.
    /// Ignored by exact indexes.
    pub ef: usize,
}

impl Default for SearchParams {
    fn default() -> Self {
        SearchParams { ef: 64 }
    }
}

pub trait Index: Send + Sync {
    fn dim(&self) -> usize;
    fn metric(&self) -> Metric;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Adds `vectors` (row-major, a multiple of `dim()` long) and returns
    /// their ids. Building an index is `add` on an empty one; batches are
    /// inserted in parallel.
    fn add(&mut self, vectors: &[f32]) -> Result<Range<u32>>;
    /// The `k` nearest vectors to `query`, nearest first.
    fn search(&self, query: &[f32], k: usize, params: &SearchParams) -> Result<Vec<Hit>>;
}

fn check_dim(expected: usize, got: usize) -> Result<()> {
    if got != expected {
        return Err(Error::DimMismatch { expected, got });
    }
    Ok(())
}

/// Validates a batch for `add` and returns the id range it will get.
fn batch_ids(dim: usize, len: usize, vectors: &[f32]) -> Result<Range<u32>> {
    if !vectors.len().is_multiple_of(dim) {
        return Err(Error::DimMismatch {
            expected: dim,
            got: vectors.len() % dim,
        });
    }
    let end = len + vectors.len() / dim;
    if end > u32::MAX as usize {
        return Err(Error::Full);
    }
    Ok(len as u32..end as u32)
}

#[cfg(test)]
pub(crate) mod test_data {
    /// Deterministic pseudo-random vectors in [-1, 1) (splitmix64).
    pub fn vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        (0..(n * dim) as u64)
            .map(|i| (crate::hnsw::splitmix64(seed ^ i) >> 40) as f32 / (1u64 << 23) as f32 - 1.0)
            .collect()
    }
}
