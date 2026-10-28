//! Scalar reference kernels.
//!
//! Deliberately plain sequential loops: float addition isn't associative,
//! so the compiler won't vectorize these reductions on its own. That makes
//! them the honest baseline for the NEON speedup and the ground truth the
//! SIMD kernels are property-tested against.
//!
//! All kernels panic if the slices differ in length.

/// Squared Euclidean distance. The square root is skipped: it doesn't
/// change the nearest-neighbour order.
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut sum = 0.0;
    for (&x, &y) in a.iter().zip(b) {
        let d = x - y;
        sum += d * d;
    }
    sum
}

/// Inner product.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut sum = 0.0;
    for (&x, &y) in a.iter().zip(b) {
        sum += x * y;
    }
    sum
}

/// Cosine distance, `1 - cos(a, b)`, in `[0, 2]`. A zero vector has no
/// direction, so its distance to anything is defined as 1.
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    if aa == 0.0 || bb == 0.0 {
        return 1.0;
    }
    1.0 - ab / (aa * bb).sqrt()
}

/// Squared Euclidean distance over int8. Exact for `len <= 65_536`
/// (each term is at most 255², and the sum must fit a `u32`).
pub fn l2_sq_i8(a: &[i8], b: &[i8]) -> u32 {
    assert_eq!(a.len(), b.len());
    debug_assert!(a.len() <= 65_536);
    let mut sum = 0u32;
    for (&x, &y) in a.iter().zip(b) {
        let d = i32::from(x) - i32::from(y);
        sum += (d * d) as u32;
    }
    sum
}

/// Inner product over int8. Exact for `len <= 65_536` (each term is at
/// most 128² in magnitude, and the sum must fit an `i32`).
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    assert_eq!(a.len(), b.len());
    debug_assert!(a.len() <= 65_536);
    let mut sum = 0i32;
    for (&x, &y) in a.iter().zip(b) {
        sum += i32::from(x) * i32::from(y);
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_known_values() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 6.0, 3.0];
        assert_eq!(l2_sq(&a, &b), 9.0 + 16.0);
        assert_eq!(dot(&a, &b), 4.0 + 12.0 + 9.0);
        assert_eq!(l2_sq(&a, &a), 0.0);
        assert_eq!(l2_sq(&[], &[]), 0.0);
    }

    #[test]
    fn cosine() {
        assert!(cosine_distance(&[1.0, 0.0], &[5.0, 0.0]).abs() < 1e-6);
        assert!((cosine_distance(&[1.0, 0.0], &[0.0, 3.0]) - 1.0).abs() < 1e-6);
        assert!((cosine_distance(&[1.0, 1.0], &[-2.0, -2.0]) - 2.0).abs() < 1e-6);
        assert_eq!(cosine_distance(&[0.0, 0.0], &[1.0, 2.0]), 1.0);
    }

    #[test]
    fn int8_known_values() {
        assert_eq!(l2_sq_i8(&[1, -2, 3], &[4, 2, 3]), 9 + 16);
        assert_eq!(dot_i8(&[1, -2, 5], &[4, 2, 3]), 15); // 4 - 4 + 15
    }

    #[test]
    fn int8_extremes_at_max_len_dont_overflow() {
        let lo = vec![i8::MIN; 65_536];
        let hi = vec![i8::MAX; 65_536];
        assert_eq!(l2_sq_i8(&lo, &hi), 65_536 * 255 * 255);
        assert_eq!(dot_i8(&lo, &lo), 65_536 * 128 * 128); // 2^30
        assert_eq!(dot_i8(&lo, &hi), -65_536 * 128 * 127);
    }

    #[test]
    #[should_panic]
    fn length_mismatch_panics() {
        l2_sq(&[1.0], &[1.0, 2.0]);
    }
}
