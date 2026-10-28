//! Product quantization (Jégou et al., 2011), 8-bit codes.
//!
//! The vector is split into `m` contiguous sub-vectors of `dim / m` floats;
//! each is replaced by the id of its nearest of 256 centroids, trained by
//! k-means per subspace. A code is `m` bytes, and `encode_all` stores codes
//! back to back (`n * m` bytes), which a later 4-bit fast-scan can repack.
//!
//! ADC (asymmetric distance): for a query, `distance_table` holds the
//! squared L2 from each query sub-vector to each centroid (`m * 256`
//! floats, subspace-major); the distance to a code is then `m` lookups.

use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;

use crate::kmeans::sample_rows;
use crate::{Centroids, Error, KMeans, Result, rows};

/// Centroids per subspace (8-bit codes).
pub const KSUB: usize = 256;

#[derive(Debug, Clone, Copy)]
pub struct PqConfig {
    pub dim: usize,
    /// Number of subspaces = bytes per code. Must divide `dim`.
    pub m: usize,
    pub max_iter: usize,
    pub max_train: usize,
    pub seed: u64,
}

impl PqConfig {
    /// 25 k-means iterations on up to 100k points, seed 42.
    pub fn new(dim: usize, m: usize) -> Self {
        PqConfig {
            dim,
            m,
            max_iter: 25,
            max_train: 100_000,
            seed: 42,
        }
    }

    pub fn train(&self, data: &[f32]) -> Result<Pq> {
        rows(data, self.dim)?;
        if self.m == 0 || !self.dim.is_multiple_of(self.m) {
            return Err(Error::NotDivisible {
                dim: self.dim,
                m: self.m,
            });
        }
        let dsub = self.dim / self.m;
        // One sample shared by all subspaces.
        let mut rng = StdRng::seed_from_u64(self.seed);
        let x = sample_rows(data, self.dim, self.max_train, &mut rng);
        let books = (0..self.m)
            .map(|j| {
                let sub: Vec<f32> = x
                    .chunks_exact(self.dim)
                    .flat_map(|v| &v[j * dsub..][..dsub])
                    .copied()
                    .collect();
                KMeans {
                    k: KSUB,
                    dim: dsub,
                    max_iter: self.max_iter,
                    max_train: usize::MAX,
                    seed: self.seed.wrapping_add(1 + j as u64),
                }
                .train(&sub)
            })
            .collect::<Result<_>>()?;
        Ok(Pq {
            dim: self.dim,
            books,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Pq {
    dim: usize,
    /// One codebook of `KSUB` centroids per subspace.
    books: Vec<Centroids>,
}

impl Pq {
    /// From per-subspace codebooks (e.g. read back from disk).
    pub fn from_codebooks(books: Vec<Centroids>) -> Result<Self> {
        let Some(first) = books.first() else {
            return Err(Error::Invalid("need at least one codebook"));
        };
        let dsub = first.dim();
        if books.iter().any(|b| b.dim() != dsub || b.k() != KSUB) {
            return Err(Error::Invalid("codebooks must all be 256 x dim/m"));
        }
        Ok(Pq {
            dim: dsub * books.len(),
            books,
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Bytes per code.
    pub fn m(&self) -> usize {
        self.books.len()
    }

    pub fn codebooks(&self) -> &[Centroids] {
        &self.books
    }

    fn dsub(&self) -> usize {
        self.dim / self.m()
    }

    /// Writes the `m`-byte code of `v` into `out`. Panics on wrong lengths.
    pub fn encode_into(&self, v: &[f32], out: &mut [u8]) {
        assert_eq!(v.len(), self.dim);
        assert_eq!(out.len(), self.m());
        for ((b, sub), o) in self.books.iter().zip(v.chunks_exact(self.dsub())).zip(out) {
            *o = b.assign(sub).0 as u8;
        }
    }

    pub fn encode(&self, v: &[f32]) -> Vec<u8> {
        let mut out = vec![0; self.m()];
        self.encode_into(v, &mut out);
        out
    }

    /// Codes of every row of `data`, back to back (`n * m` bytes).
    pub fn encode_all(&self, data: &[f32]) -> Result<Vec<u8>> {
        let n = rows(data, self.dim)?;
        let mut out = vec![0; n * self.m()];
        data.par_chunks_exact(self.dim)
            .zip(out.par_chunks_exact_mut(self.m()))
            .for_each(|(v, o)| self.encode_into(v, o));
        Ok(out)
    }

    /// The reconstruction: each sub-vector replaced by its centroid.
    pub fn decode(&self, code: &[u8]) -> Vec<f32> {
        assert_eq!(code.len(), self.m());
        self.books
            .iter()
            .zip(code)
            .flat_map(|(b, &c)| b.get(c as usize))
            .copied()
            .collect()
    }

    /// `m * 256` squared L2 distances from each query sub-vector to each
    /// centroid of its subspace; entry `j * 256 + c` is subspace j, code c.
    pub fn distance_table(&self, query: &[f32]) -> Vec<f32> {
        assert_eq!(query.len(), self.dim);
        let mut t = Vec::with_capacity(self.m() * KSUB);
        for (b, q) in self.books.iter().zip(query.chunks_exact(self.dsub())) {
            t.extend(
                b.as_slice()
                    .chunks_exact(b.dim())
                    .map(|c| kernels::l2_sq(q, c)),
            );
        }
        t
    }
}

/// ADC: squared L2 from the table's query to `decode(code)`.
pub fn adc_distance(table: &[f32], code: &[u8]) -> f32 {
    debug_assert_eq!(table.len(), code.len() * KSUB);
    table
        .as_chunks::<KSUB>()
        .0
        .iter()
        .zip(code)
        .map(|(t, &c)| t[c as usize])
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kmeans::tests::blobs;
    use proptest::prelude::*;
    use std::sync::LazyLock;

    /// 4000 points of structured data, dim 16, plus a PQ with m = 8.
    static FIXTURE: LazyLock<(Vec<f32>, Pq)> = LazyLock::new(|| {
        let (x, _) = blobs(40, 100, 16, 5.0, 11);
        let pq = PqConfig {
            max_iter: 10,
            ..PqConfig::new(16, 8)
        }
        .train(&x)
        .unwrap();
        (x, pq)
    });

    fn variance(x: &[f32], dim: usize) -> f64 {
        let n = (x.len() / dim) as f64;
        let mut mean = vec![0f64; dim];
        for v in x.chunks_exact(dim) {
            for (m, &a) in mean.iter_mut().zip(v) {
                *m += f64::from(a) / n;
            }
        }
        x.chunks_exact(dim)
            .flat_map(|v| {
                v.iter()
                    .zip(&mean)
                    .map(|(&a, m)| (f64::from(a) - m).powi(2))
            })
            .sum::<f64>()
            / x.len() as f64
    }

    #[test]
    fn reconstruction_error_far_below_variance() {
        let (x, pq) = &*FIXTURE;
        let codes = pq.encode_all(x).unwrap();
        let mse = x
            .as_chunks::<16>()
            .0
            .iter()
            .zip(codes.as_chunks::<8>().0)
            .map(|(v, c)| f64::from(kernels::l2_sq(v, &pq.decode(c))))
            .sum::<f64>()
            / x.len() as f64;
        let var = variance(x, 16);
        assert!(mse < 0.05 * var, "mse {mse} vs variance {var}");
    }

    #[test]
    fn rejects_indivisible_dim() {
        let x = vec![0.0; 300 * 10];
        assert_eq!(
            PqConfig::new(10, 4).train(&x),
            Err(Error::NotDivisible { dim: 10, m: 4 })
        );
    }

    #[test]
    fn codebooks_round_trip() {
        let (_, pq) = &*FIXTURE;
        assert_eq!(&Pq::from_codebooks(pq.codebooks().to_vec()).unwrap(), pq);
    }

    proptest! {
        #[test]
        fn adc_equals_exact_distance_to_decoded(
            q in prop::collection::vec(-120f32..120.0, 16),
            i in 0usize..4000,
        ) {
            let (x, pq) = &*FIXTURE;
            let code = pq.encode(&x[i * 16..][..16]);
            let adc = adc_distance(&pq.distance_table(&q), &code);
            let exact = kernels::l2_sq(&q, &pq.decode(&code));
            prop_assert!((adc - exact).abs() <= 1e-4 * exact.max(1.0), "{adc} vs {exact}");
        }
    }
}
