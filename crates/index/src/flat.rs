//! Exact brute-force index. Each query is split across threads with rayon;
//! it's the exactness baseline for every approximate index.

use std::collections::BinaryHeap;
use std::ops::Range;

use rayon::prelude::*;

use crate::{Error, Hit, Index, Metric, Result, SearchParams, batch_ids, blob, check_dim};

/// Rows per rayon task: big enough to amortize scheduling, small enough
/// that 1M rows give every core plenty of work (~61 tasks).
const ROWS_PER_TASK: usize = 16_384;

pub struct Flat {
    dim: usize,
    metric: Metric,
    data: Vec<f32>,
}

impl Flat {
    pub fn new(dim: usize, metric: Metric) -> Self {
        assert!(dim > 0);
        Flat {
            dim,
            metric,
            data: Vec::new(),
        }
    }

    /// The stored vectors, row-major, in stored form (unit length for
    /// cosine). This is what a segment's vector section holds.
    pub fn vectors(&self) -> &[f32] {
        &self.data
    }

    /// The index blob: config only, since the vectors live in the
    /// segment's vector section. Layout in docs/formats/segment.md.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(24);
        blob::put_header(&mut out, MAGIC, self.dim, self.metric);
        blob::put_u64(&mut out, self.len() as u64);
        out
    }

    /// Rebuilds a `Flat` from its blob and its vectors (in stored form, as
    /// `vectors()` returned them; they are not normalized again). Takes
    /// ownership, so loading from a mapped segment means one copy.
    pub fn from_bytes(bytes: &[u8], vectors: Vec<f32>) -> Result<Flat> {
        let mut r = blob::Reader::new(bytes);
        let (dim, metric) = r.header(MAGIC)?;
        let n = r.u64()?;
        r.finish()?;
        blob::check_vectors(&vectors, n, dim)?;
        if n > u64::from(u32::MAX) {
            return Err(Error::Corrupt("more vectors than u32 ids"));
        }
        Ok(Flat {
            dim,
            metric,
            data: vectors,
        })
    }
}

const MAGIC: &[u8; 4] = b"NPFL";

impl Index for Flat {
    fn dim(&self) -> usize {
        self.dim
    }

    fn metric(&self) -> Metric {
        self.metric
    }

    fn len(&self) -> usize {
        self.data.len() / self.dim
    }

    fn add(&mut self, vectors: &[f32]) -> Result<Range<u32>> {
        let ids = batch_ids(self.dim, self.len(), vectors)?;
        self.data.reserve(vectors.len());
        for v in vectors.chunks_exact(self.dim) {
            self.data.extend_from_slice(&self.metric.prepare(v));
        }
        Ok(ids)
    }

    fn search(&self, query: &[f32], k: usize, _: &SearchParams) -> Result<Vec<Hit>> {
        check_dim(self.dim, query.len())?;
        let q = self.metric.prepare(query);
        Ok(self
            .data
            .par_chunks(self.dim * ROWS_PER_TASK)
            .enumerate()
            .map(|(t, rows)| {
                top_k(
                    rows,
                    self.dim,
                    self.metric,
                    &q,
                    k,
                    (t * ROWS_PER_TASK) as u32,
                )
            })
            .reduce(Vec::new, |mut a, b| {
                a.extend(b);
                a.sort_unstable();
                a.truncate(k);
                a
            }))
    }
}

/// The `k` nearest of `rows` to `q`, nearest first; ids start at `first_id`.
fn top_k(rows: &[f32], dim: usize, metric: Metric, q: &[f32], k: usize, first_id: u32) -> Vec<Hit> {
    let mut heap = BinaryHeap::with_capacity(k + 1);
    for (i, row) in rows.chunks_exact(dim).enumerate() {
        let hit = Hit {
            id: first_id + i as u32,
            dist: metric.distance(q, row),
        };
        if heap.len() < k {
            heap.push(hit);
        } else if let Some(mut far) = heap.peek_mut()
            && hit < *far
        {
            *far = hit;
        }
    }
    heap.into_sorted_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Error, test_data};

    fn ids(hits: &[Hit]) -> Vec<u32> {
        hits.iter().map(|h| h.id).collect()
    }

    #[test]
    fn nearest_first_and_ties_keep_smaller_id() {
        let mut f = Flat::new(1, Metric::L2);
        assert_eq!(f.add(&[0.0, 3.0, 1.0, 1.0, 5.0]).unwrap(), 0..5);
        let p = SearchParams::default();
        assert_eq!(ids(&f.search(&[0.0], 3, &p).unwrap()), [0, 2, 3]);
        assert_eq!(ids(&f.search(&[0.0], 2, &p).unwrap()), [0, 2]); // id 3 ties, loses
        assert_eq!(ids(&f.search(&[0.0], 10, &p).unwrap()), [0, 2, 3, 1, 4]);
        assert!(f.search(&[0.0], 0, &p).unwrap().is_empty());
    }

    #[test]
    fn parallel_search_matches_a_full_sort_across_tasks() {
        // > ROWS_PER_TASK rows so the reduce actually merges several tasks.
        let (dim, n) = (8, ROWS_PER_TASK * 2 + 123);
        let data = test_data::vectors(n, dim, 1);
        let q = test_data::vectors(1, dim, 2);
        let mut f = Flat::new(dim, Metric::L2);
        f.add(&data).unwrap();
        let mut all: Vec<Hit> = data
            .chunks_exact(dim)
            .enumerate()
            .map(|(id, r)| Hit {
                id: id as u32,
                dist: kernels::scalar::l2_sq(&q, r),
            })
            .collect();
        all.sort();
        let got = f.search(&q, 25, &SearchParams::default()).unwrap();
        assert_eq!(ids(&got), ids(&all[..25]));
    }

    #[test]
    fn metrics() {
        let p = SearchParams::default();
        let mut ip = Flat::new(2, Metric::InnerProduct);
        ip.add(&[1.0, 0.0, 10.0, 0.0, -5.0, 0.0]).unwrap();
        assert_eq!(ids(&ip.search(&[1.0, 0.0], 3, &p).unwrap()), [1, 0, 2]);

        // Cosine ignores magnitude: [10, 0.1] is nearly parallel to [1, 0].
        let mut cos = Flat::new(2, Metric::Cosine);
        cos.add(&[0.0, 1.0, 10.0, 0.1, 0.0, 0.0]).unwrap();
        let hits = cos.search(&[3.0, 0.0], 3, &p).unwrap();
        assert_eq!(ids(&hits), [1, 0, 2]);
        assert!(hits[0].dist < 1e-3);
        assert!((hits[2].dist - 1.0).abs() < 1e-6); // zero vector: distance 1
    }

    #[test]
    fn blob_round_trip_and_corruption() {
        let mut f = Flat::new(3, Metric::Cosine);
        f.add(&test_data::vectors(50, 3, 4)).unwrap();
        let bytes = f.to_bytes();
        let g = Flat::from_bytes(&bytes, f.vectors().to_vec()).unwrap();
        assert_eq!(
            (g.dim(), g.metric(), g.vectors()),
            (3, Metric::Cosine, f.vectors())
        );
        let q = [0.3, -0.2, 0.9];
        let p = SearchParams::default();
        assert_eq!(g.search(&q, 5, &p).unwrap(), f.search(&q, 5, &p).unwrap());

        for len in 0..bytes.len() {
            assert!(Flat::from_bytes(&bytes[..len], f.vectors().to_vec()).is_err());
        }
        assert!(Flat::from_bytes(&bytes, f.vectors()[3..].to_vec()).is_err());
        let mut b = bytes.clone();
        b[12] = 9; // metric code
        assert_eq!(
            Flat::from_bytes(&b, f.vectors().to_vec()).err(),
            Some(Error::Corrupt("unknown metric"))
        );
    }

    #[test]
    fn rejects_wrong_dims() {
        let mut f = Flat::new(3, Metric::L2);
        assert_eq!(
            f.add(&[1.0; 4]),
            Err(Error::DimMismatch {
                expected: 3,
                got: 1
            })
        );
        assert!(f.search(&[1.0; 2], 1, &SearchParams::default()).is_err());
    }
}
