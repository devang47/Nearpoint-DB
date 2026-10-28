//! k-means, product quantization, scalar quantization.
//!
//! Vectors are row-major `&[f32]` slices of `n * dim` floats, as in `index`.
//! Training is reproducible: same data + same seed gives bit-identical
//! output whatever the rayon thread count.

pub mod kmeans;
pub mod pq;
pub mod sq;

use std::borrow::Cow;

use rand::SeedableRng;
use rand::rngs::StdRng;

pub use kmeans::{Centroids, KMeans};
pub use pq::{Pq, PqConfig, adc_distance};
pub use sq::Sq;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("data length {len} is not a multiple of dim {dim}")]
    Ragged { len: usize, dim: usize },
    #[error("need at least {need} training points, got {got}")]
    TooFewPoints { need: usize, got: usize },
    #[error("dim {dim} is not divisible by m = {m}")]
    NotDivisible { dim: usize, m: usize },
    #[error("invalid parameter: {0}")]
    Invalid(&'static str),
}

pub type Result<T> = std::result::Result<T, Error>;

/// `data` itself if it has at most `max` rows, else `max` rows drawn
/// uniformly without replacement (seeded), in their original order.
pub fn sample(data: &[f32], dim: usize, max: usize, seed: u64) -> Cow<'_, [f32]> {
    kmeans::sample_rows(data, dim, max, &mut StdRng::seed_from_u64(seed))
}

/// Number of rows in `data`, or an error if it isn't whole rows.
pub(crate) fn rows(data: &[f32], dim: usize) -> Result<usize> {
    if dim == 0 {
        return Err(Error::Invalid("dim must be > 0"));
    }
    if !data.len().is_multiple_of(dim) {
        return Err(Error::Ragged {
            len: data.len(),
            dim,
        });
    }
    Ok(data.len() / dim)
}
