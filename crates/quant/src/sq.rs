//! Scalar quantization to int8.
//!
//! Each component maps to `round((x - min[i]) / scale) - 128`, clamped to
//! int8. `min` is per dimension, but `scale` is shared: the largest
//! per-dimension range (max - min over the training sample) divided by 255.
//! A shared scale makes the squared L2 between two codes exactly
//! proportional to the L2 between their reconstructions (the offsets
//! cancel), so `kernels::l2_sq_i8` on raw codes ranks correctly; per-dim
//! scales would need a weighted sum. Plain min/max, no percentile clip:
//! values outside the trained range are clamped.
//!
//! Round-trip error is at most `scale / 2` per in-range component.

use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;

use crate::kmeans::sample_rows;
use crate::{Error, Result, rows};

#[derive(Debug, Clone, PartialEq)]
pub struct Sq {
    min: Vec<f32>,
    scale: f32,
}

impl Sq {
    /// Trains min/max on a random sample of at most `max_train` rows.
    pub fn train(data: &[f32], dim: usize, max_train: usize, seed: u64) -> Result<Sq> {
        if rows(data, dim)? == 0 || max_train == 0 {
            return Err(Error::TooFewPoints { need: 1, got: 0 });
        }
        let x = sample_rows(data, dim, max_train, &mut StdRng::seed_from_u64(seed));
        let mut min = vec![f32::INFINITY; dim];
        let mut max = vec![f32::NEG_INFINITY; dim];
        for v in x.chunks_exact(dim) {
            for ((lo, hi), &a) in min.iter_mut().zip(&mut max).zip(v) {
                *lo = lo.min(a);
                *hi = hi.max(a);
            }
        }
        let range = min
            .iter()
            .zip(&max)
            .map(|(lo, hi)| hi - lo)
            .fold(0f32, f32::max);
        let scale = if range > 0.0 { range / 255.0 } else { 1.0 };
        Ok(Sq { min, scale })
    }

    /// From stored parameters (e.g. read back from disk).
    pub fn from_parts(min: Vec<f32>, scale: f32) -> Result<Sq> {
        if min.is_empty() || !(scale > 0.0 && scale.is_finite()) {
            return Err(Error::Invalid("need dim > 0 and a finite scale > 0"));
        }
        Ok(Sq { min, scale })
    }

    pub fn dim(&self) -> usize {
        self.min.len()
    }

    pub fn min(&self) -> &[f32] {
        &self.min
    }

    /// Step between adjacent code values (shared by all dimensions).
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Panics on wrong lengths.
    pub fn encode_into(&self, v: &[f32], out: &mut [i8]) {
        assert_eq!(v.len(), self.dim());
        assert_eq!(out.len(), self.dim());
        let inv = 1.0 / self.scale;
        for ((o, &a), &lo) in out.iter_mut().zip(v).zip(&self.min) {
            *o = (((a - lo) * inv).round().clamp(0.0, 255.0) as i32 - 128) as i8;
        }
    }

    pub fn encode(&self, v: &[f32]) -> Vec<i8> {
        let mut out = vec![0; self.dim()];
        self.encode_into(v, &mut out);
        out
    }

    /// Codes of every row of `data`, back to back (`n * dim` bytes).
    pub fn encode_all(&self, data: &[f32]) -> Result<Vec<i8>> {
        let n = rows(data, self.dim())?;
        let mut out = vec![0; n * self.dim()];
        data.par_chunks_exact(self.dim())
            .zip(out.par_chunks_exact_mut(self.dim()))
            .for_each(|(v, o)| self.encode_into(v, o));
        Ok(out)
    }

    pub fn decode(&self, code: &[i8]) -> Vec<f32> {
        assert_eq!(code.len(), self.dim());
        code.iter()
            .zip(&self.min)
            .map(|(&c, &lo)| (i32::from(c) + 128) as f32 * self.scale + lo)
            .collect()
    }

    /// Squared L2 between the reconstructions of two codes (exact up to
    /// f32 rounding), via the int8 kernel.
    pub fn l2_sq(&self, a: &[i8], b: &[i8]) -> f32 {
        kernels::l2_sq_i8(a, b) as f32 * self.scale * self.scale
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn round_trip_error_bounded(
            data in (1usize..9).prop_flat_map(|d| {
                (Just(d), prop::collection::vec(-1000f32..1000.0, d..=d * 50))
            }),
        ) {
            let (dim, mut x) = data;
            x.truncate(x.len() / dim * dim);
            let sq = Sq::train(&x, dim, usize::MAX, 1).unwrap();
            for v in x.chunks_exact(dim) {
                let r = sq.decode(&sq.encode(v));
                for (a, b) in v.iter().zip(&r) {
                    // (max - min) / 255 / 2, plus f32 rounding.
                    let bound = sq.scale() / 2.0 * 1.001 + a.abs().max(1.0) * 1e-6;
                    prop_assert!((a - b).abs() <= bound, "{a} -> {b}, bound {bound}");
                }
            }
        }

        #[test]
        fn code_distance_matches_decoded(
            a in prop::collection::vec(-50f32..50.0, 8),
            b in prop::collection::vec(-50f32..50.0, 8),
        ) {
            let train: Vec<f32> = (0..8 * 4).map(|i| (i as f32 * 7.3) % 100.0 - 50.0).collect();
            let sq = Sq::train(&train, 8, usize::MAX, 1).unwrap();
            let (ca, cb) = (sq.encode(&a), sq.encode(&b));
            let want = kernels::l2_sq(&sq.decode(&ca), &sq.decode(&cb));
            let got = sq.l2_sq(&ca, &cb);
            prop_assert!((got - want).abs() <= 1e-3 * want.max(1.0), "{got} vs {want}");
        }
    }

    #[test]
    fn clamps_out_of_range() {
        let sq = Sq::train(&[0.0, 10.0, 255.0, 20.0], 2, 10, 0).unwrap();
        assert_eq!(sq.encode(&[-5.0, 1000.0]), vec![-128, 127]);
    }
}
