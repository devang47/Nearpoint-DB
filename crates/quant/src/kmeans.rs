//! Lloyd's k-means with k-means++ seeding.
//!
//! - **Sample.** Training uses at most `max_train` points, drawn uniformly
//!   without replacement with the seeded RNG. k-means++ also runs on that
//!   sample only, so seeding costs O(max_train · k · dim).
//! - **Assignment** is parallel over fixed-size chunks of points; each chunk
//!   returns (changed, inertia) and the chunk results are summed in order.
//! - **Update** is parallel over clusters: points are bucketed by cluster
//!   (a stable counting sort), then each cluster sums its own points in f64
//!   in index order. Every sum has a fixed order, so the result doesn't
//!   depend on the thread count, and it needs O(n) scratch instead of the
//!   O(threads · k · dim) of per-thread partial sums (which matters for a
//!   coarse quantizer with large k).
//! - **Empty clusters** are re-seeded with the points farthest from their
//!   centroid (largest distance first, distinct points). This never raises
//!   the inertia: the moved point's distance drops to 0, others are as
//!   before.
//! - **Stop** after `max_iter` iterations, or when fewer than 0.1% of the
//!   points change cluster.

use std::borrow::Cow;

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

use crate::{Error, Result, rows};

/// Points per parallel work item in the assignment step. Fixed (not tied to
/// the thread count) so the inertia sum has the same order everywhere.
const CHUNK: usize = 1024;
/// Stop once fewer than this fraction of points change cluster.
const STOP_FRACTION: f64 = 1e-3;

#[derive(Debug, Clone, Copy)]
pub struct KMeans {
    pub k: usize,
    pub dim: usize,
    pub max_iter: usize,
    /// Train on a random sample of at most this many points.
    pub max_train: usize,
    pub seed: u64,
}

impl KMeans {
    /// 25 iterations, 256 training points per centroid, seed 42.
    pub fn new(k: usize, dim: usize) -> Self {
        KMeans {
            k,
            dim,
            max_iter: 25,
            max_train: 256 * k,
            seed: 42,
        }
    }

    pub fn train(&self, data: &[f32]) -> Result<Centroids> {
        self.train_with_history(data).map(|(c, _)| c)
    }

    /// Also returns the inertia (sum of squared distances to the assigned
    /// centroid, over the training sample) after each assignment step.
    pub fn train_with_history(&self, data: &[f32]) -> Result<(Centroids, Vec<f64>)> {
        let n = rows(data, self.dim)?;
        if self.k == 0 {
            return Err(Error::Invalid("k must be > 0"));
        }
        if self.k > u32::MAX as usize {
            return Err(Error::Invalid("k must fit in u32"));
        }
        let got = n.min(self.max_train);
        if got < self.k {
            return Err(Error::TooFewPoints { need: self.k, got });
        }
        let mut rng = StdRng::seed_from_u64(self.seed);
        let x = sample_rows(data, self.dim, self.max_train, &mut rng);
        let n = x.len() / self.dim;

        let mut c = Centroids {
            dim: self.dim,
            data: plus_plus(&x, self.dim, self.k, &mut rng),
        };
        let mut assign = vec![u32::MAX; n];
        let mut dist = vec![0f32; n];
        let mut history = Vec::new();
        for _ in 0..self.max_iter {
            let (changed, inertia) = assign_step(&x, &c, &mut assign, &mut dist);
            history.push(inertia);
            if history.len() > 1 && (changed as f64) < STOP_FRACTION * n as f64 {
                break;
            }
            update(&x, self.dim, self.k, &assign, &dist, &mut c.data);
        }
        Ok((c, history))
    }
}

/// `k` centroids of `dim` floats, row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct Centroids {
    dim: usize,
    data: Vec<f32>,
}

impl Centroids {
    /// From raw row-major centroids (e.g. read back from disk).
    pub fn from_vec(dim: usize, data: Vec<f32>) -> Result<Self> {
        if rows(&data, dim)? == 0 {
            return Err(Error::Invalid("need at least one centroid"));
        }
        Ok(Centroids { dim, data })
    }

    pub fn k(&self) -> usize {
        self.data.len() / self.dim
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn get(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..][..self.dim]
    }

    /// Nearest centroid and its squared L2 distance (lowest id on ties).
    /// Panics if `v.len() != dim`.
    pub fn assign(&self, v: &[f32]) -> (u32, f32) {
        let mut best = (0u32, f32::INFINITY);
        for (j, c) in self.data.chunks_exact(self.dim).enumerate() {
            let d = kernels::l2_sq(v, c);
            if d < best.1 {
                best = (j as u32, d);
            }
        }
        best
    }

    /// `assign` for every row of `data`, in parallel.
    pub fn assign_all(&self, data: &[f32]) -> Result<Vec<(u32, f32)>> {
        rows(data, self.dim)?;
        Ok(data
            .par_chunks_exact(self.dim)
            .map(|v| self.assign(v))
            .collect())
    }
}

/// `data` itself if it has at most `max` rows, else `max` random rows (in
/// their original order, for locality).
pub(crate) fn sample_rows<'a>(
    data: &'a [f32],
    dim: usize,
    max: usize,
    rng: &mut StdRng,
) -> Cow<'a, [f32]> {
    let n = data.len() / dim;
    if n <= max {
        return Cow::Borrowed(data);
    }
    let mut ids = rand::seq::index::sample(rng, n, max).into_vec();
    ids.sort_unstable();
    let mut out = Vec::with_capacity(max * dim);
    for i in ids {
        out.extend_from_slice(&data[i * dim..][..dim]);
    }
    Cow::Owned(out)
}

/// k-means++: each next seed is a point drawn with probability proportional
/// to its squared distance to the nearest seed so far.
fn plus_plus(x: &[f32], dim: usize, k: usize, rng: &mut StdRng) -> Vec<f32> {
    let n = x.len() / dim;
    let mut c = Vec::with_capacity(k * dim);
    let first = rng.random_range(0..n);
    c.extend_from_slice(&x[first * dim..][..dim]);
    let mut d: Vec<f32> = x
        .par_chunks_exact(dim)
        .map(|p| kernels::l2_sq(p, &c[..dim]))
        .collect();
    for _ in 1..k {
        let total: f64 = d.iter().map(|&v| f64::from(v)).sum();
        let pick = if total > 0.0 {
            let target = rng.random::<f64>() * total;
            let mut acc = 0.0;
            // Fallback for rounding: the last point with nonzero weight.
            let mut pick = d.iter().rposition(|&v| v > 0.0).unwrap_or(0);
            for (i, &v) in d.iter().enumerate() {
                acc += f64::from(v);
                if acc > target {
                    pick = i;
                    break;
                }
            }
            pick
        } else {
            // Fewer distinct points than k: any point will do.
            rng.random_range(0..n)
        };
        let start = c.len();
        c.extend_from_slice(&x[pick * dim..][..dim]);
        let new = &c[start..];
        d.par_iter_mut()
            .zip(x.par_chunks_exact(dim))
            .for_each(|(di, p)| *di = di.min(kernels::l2_sq(p, new)));
    }
    c
}

/// Assigns every point; returns (points that changed cluster, inertia).
fn assign_step(x: &[f32], c: &Centroids, assign: &mut [u32], dist: &mut [f32]) -> (usize, f64) {
    let dim = c.dim;
    let parts: Vec<(usize, f64)> = x
        .par_chunks(CHUNK * dim)
        .zip(assign.par_chunks_mut(CHUNK))
        .zip(dist.par_chunks_mut(CHUNK))
        .map(|((xs, a), d)| {
            let (mut changed, mut sum) = (0, 0.0);
            for ((p, a), d) in xs.chunks_exact(dim).zip(a).zip(d) {
                let (j, dd) = c.assign(p);
                changed += usize::from(j != *a);
                *a = j;
                *d = dd;
                sum += f64::from(dd);
            }
            (changed, sum)
        })
        .collect();
    parts
        .iter()
        .fold((0, 0.0), |(c, s), &(c2, s2)| (c + c2, s + s2))
}

/// Moves each centroid to the mean of its points; re-seeds empty clusters.
fn update(x: &[f32], dim: usize, k: usize, assign: &[u32], dist: &[f32], cent: &mut [f32]) {
    // Stable counting sort of point ids by cluster.
    let mut start = vec![0usize; k + 1];
    for &a in assign {
        start[a as usize + 1] += 1;
    }
    for j in 0..k {
        start[j + 1] += start[j];
    }
    let mut pos = start.clone();
    let mut order = vec![0u32; assign.len()];
    for (i, &a) in assign.iter().enumerate() {
        order[pos[a as usize]] = i as u32;
        pos[a as usize] += 1;
    }

    cent.par_chunks_exact_mut(dim)
        .enumerate()
        .for_each(|(j, cj)| {
            let ids = &order[start[j]..start[j + 1]];
            if ids.is_empty() {
                return;
            }
            let mut sum = vec![0f64; dim];
            for &i in ids {
                for (s, &v) in sum.iter_mut().zip(&x[i as usize * dim..][..dim]) {
                    *s += f64::from(v);
                }
            }
            let inv = 1.0 / ids.len() as f64;
            for (c, s) in cj.iter_mut().zip(sum) {
                *c = (s * inv) as f32;
            }
        });

    let empty: Vec<usize> = (0..k).filter(|&j| start[j] == start[j + 1]).collect();
    if empty.is_empty() {
        return;
    }
    // ponytail: full sort, O(n log n), only on iterations with empty clusters.
    let mut far: Vec<u32> = (0..assign.len() as u32).collect();
    far.sort_unstable_by(|&a, &b| {
        dist[b as usize]
            .total_cmp(&dist[a as usize])
            .then(a.cmp(&b))
    });
    for (&j, &i) in empty.iter().zip(&far) {
        cent[j * dim..][..dim].copy_from_slice(&x[i as usize * dim..][..dim]);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// `per` points around each of `centers` (spread `spread`), labelled.
    pub(crate) fn blobs(
        centers: usize,
        per: usize,
        dim: usize,
        spread: f32,
        seed: u64,
    ) -> (Vec<f32>, Vec<usize>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let c: Vec<f32> = (0..centers * dim)
            .map(|_| rng.random_range(-100.0..100.0))
            .collect();
        let (mut x, mut labels) = (Vec::new(), Vec::new());
        for _ in 0..per {
            for b in 0..centers {
                for t in 0..dim {
                    x.push(c[b * dim + t] + spread * gauss(&mut rng));
                }
                labels.push(b);
            }
        }
        (x, labels)
    }

    /// Standard normal via Box-Muller.
    pub(crate) fn gauss(rng: &mut StdRng) -> f32 {
        let u: f64 = 1.0 - rng.random::<f64>();
        let v: f64 = rng.random();
        ((-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()) as f32
    }

    #[test]
    fn recovers_blobs() {
        let (x, labels) = blobs(10, 200, 16, 1.0, 1);
        let c = KMeans::new(10, 16).train(&x).unwrap();
        let a = c.assign_all(&x).unwrap();
        // Same blob <=> same cluster.
        let mut map = [None; 10];
        for (&(j, _), &l) in a.iter().zip(&labels) {
            assert_eq!(*map[l].get_or_insert(j), j, "blob {l} split");
        }
        let mut seen: Vec<u32> = map.iter().map(|m| m.unwrap()).collect();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), 10, "two blobs merged");
    }

    #[test]
    fn inertia_never_increases() {
        let (x, _) = blobs(30, 100, 8, 20.0, 2);
        let km = KMeans {
            max_iter: 50,
            ..KMeans::new(20, 8)
        };
        let (_, h) = km.train_with_history(&x).unwrap();
        assert!(h.len() > 2, "history {h:?}");
        for w in h.windows(2) {
            assert!(w[1] <= w[0] * (1.0 + 1e-6), "inertia rose: {h:?}");
        }
    }

    #[test]
    fn same_seed_same_centroids_any_thread_count() {
        let (x, _) = blobs(20, 300, 12, 15.0, 3);
        let km = KMeans {
            max_train: 4000,
            ..KMeans::new(32, 12)
        };
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| km.train(&x).unwrap())
        };
        let one = run(1);
        assert_eq!(one, run(7));
        assert_eq!(one, km.train(&x).unwrap());
        let other = KMeans { seed: 7, ..km }.train(&x).unwrap();
        assert_ne!(one, other);
    }

    #[test]
    fn empty_cluster_takes_farthest_point() {
        // 4 points in 1-D, all assigned to cluster 0; cluster 1 is empty.
        let x = [0.0, 1.0, 10.0, 2.0];
        let assign = [0, 0, 0, 0];
        let dist = [9.0, 4.0, 49.0, 1.0];
        let mut cent = [0.0, 0.0];
        update(&x, 1, 2, &assign, &dist, &mut cent);
        assert_eq!(cent, [13.0 / 4.0, 10.0]);
    }

    #[test]
    fn duplicates_fewer_distinct_than_k() {
        let x: Vec<f32> = (0..100).map(|i| (i % 2) as f32).collect();
        let c = KMeans::new(5, 1).train(&x).unwrap();
        assert!(c.as_slice().iter().all(|v| v.is_finite()));
        let a = c.assign_all(&x).unwrap();
        assert!(a.iter().all(|&(_, d)| d == 0.0));
    }

    #[test]
    fn rejects_bad_input() {
        let km = KMeans::new(4, 2);
        assert_eq!(km.train(&[0.0; 5]), Err(Error::Ragged { len: 5, dim: 2 }));
        assert_eq!(
            km.train(&[0.0; 6]),
            Err(Error::TooFewPoints { need: 4, got: 3 })
        );
    }
}
