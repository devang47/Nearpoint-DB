//! NEON kernels. Same contracts as [`crate::scalar`]; f32 results differ
//! from it only in summation order.
//!
//! f32 kernels take 16 floats per iteration into 4 independent
//! accumulators so consecutive FMAs don't wait on each other. The int8
//! kernels need `dotprod` (SDOT/UDOT), which every Apple Silicon chip has.
//! Without it the crate root falls back to the scalar int8 kernels.

use std::arch::aarch64::*;

use crate::scalar;

pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    // SAFETY: NEON is part of the aarch64 baseline, so it's always available.
    unsafe { l2_sq_f32(a, b) }
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    // SAFETY: NEON is part of the aarch64 baseline, so it's always available.
    unsafe { dot_f32(a, b) }
}

/// `1 - cos(a, b)`; a zero vector's distance to anything is 1.
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    // ponytail: three passes over two L1-resident vectors. Fuse into one
    // loop if it shows up in a profile; the hot path should store
    // normalized vectors and use `dot` anyway.
    let (ab, aa, bb) = (dot(a, b), dot(a, a), dot(b, b));
    if aa == 0.0 || bb == 0.0 {
        return 1.0;
    }
    1.0 - ab / (aa * bb).sqrt()
}

#[cfg(target_feature = "dotprod")]
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    assert_eq!(a.len(), b.len());
    debug_assert!(a.len() <= 65_536);
    // SAFETY: NEON is baseline on aarch64, and this function only exists
    // when the build enables `dotprod`.
    unsafe { dot_i8_sdot(a, b) }
}

#[cfg(target_feature = "dotprod")]
pub fn l2_sq_i8(a: &[i8], b: &[i8]) -> u32 {
    assert_eq!(a.len(), b.len());
    debug_assert!(a.len() <= 65_536);
    // SAFETY: as in `dot_i8`.
    unsafe { l2_sq_i8_udot(a, b) }
}

/// Same contract as [`scalar::pq4_scan`], bit-exact.
pub fn pq4_scan(blocks: &[u8], lut: &[u8], out: &mut [u16]) {
    let m = lut.len() / 16;
    assert!(m > 0 && lut.len() == m * 16);
    assert_eq!(blocks.len() / (m * 16) * scalar::PQ4_BLOCK, out.len());
    assert!(blocks.len().is_multiple_of(m * 16));
    // SAFETY: NEON is part of the aarch64 baseline, so it's always available.
    unsafe { pq4_scan_tbl(blocks, lut, out) }
}

#[target_feature(enable = "neon")]
fn l2_sq_f32(a: &[f32], b: &[f32]) -> f32 {
    let (ca, ta) = a.as_chunks::<16>();
    let (cb, tb) = b.as_chunks::<16>();
    let mut acc = [vdupq_n_f32(0.0); 4];
    for (x, y) in ca.iter().zip(cb) {
        // SAFETY: x and y are [f32; 16], exactly four 4-lane registers.
        let (x, y) = unsafe { (vld1q_f32_x4(x.as_ptr()), vld1q_f32_x4(y.as_ptr())) };
        let d = [
            vsubq_f32(x.0, y.0),
            vsubq_f32(x.1, y.1),
            vsubq_f32(x.2, y.2),
            vsubq_f32(x.3, y.3),
        ];
        for j in 0..4 {
            acc[j] = vfmaq_f32(acc[j], d[j], d[j]);
        }
    }
    sum_f32(acc) + scalar::l2_sq(ta, tb)
}

#[target_feature(enable = "neon")]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let (ca, ta) = a.as_chunks::<16>();
    let (cb, tb) = b.as_chunks::<16>();
    let mut acc = [vdupq_n_f32(0.0); 4];
    for (x, y) in ca.iter().zip(cb) {
        // SAFETY: x and y are [f32; 16], exactly four 4-lane registers.
        let (x, y) = unsafe { (vld1q_f32_x4(x.as_ptr()), vld1q_f32_x4(y.as_ptr())) };
        acc[0] = vfmaq_f32(acc[0], x.0, y.0);
        acc[1] = vfmaq_f32(acc[1], x.1, y.1);
        acc[2] = vfmaq_f32(acc[2], x.2, y.2);
        acc[3] = vfmaq_f32(acc[3], x.3, y.3);
    }
    sum_f32(acc) + scalar::dot(ta, tb)
}

// Per 16-byte group: split the nibbles, look both halves up in the
// sub-quantizer's 16-entry table with TBL, and widen-add into four u16x8
// accumulators (vectors 0-7, 8-15, 16-23, 24-31).
#[target_feature(enable = "neon")]
fn pq4_scan_tbl(blocks: &[u8], lut: &[u8], out: &mut [u16]) {
    let m = lut.len() / 16;
    let tables = lut.as_chunks::<16>().0;
    let low = vdupq_n_u8(0xF);
    let outs = out.as_chunks_mut::<32>().0;
    for (block, o) in blocks.chunks_exact(m * 16).zip(outs) {
        let mut acc = [vdupq_n_u16(0); 4];
        for (g, t) in block.as_chunks::<16>().0.iter().zip(tables) {
            // SAFETY: g and t are [u8; 16], exactly one register each.
            let (g, t) = unsafe { (vld1q_u8(g.as_ptr()), vld1q_u8(t.as_ptr())) };
            let lo = vqtbl1q_u8(t, vandq_u8(g, low));
            let hi = vqtbl1q_u8(t, vshrq_n_u8::<4>(g));
            acc[0] = vaddw_u8(acc[0], vget_low_u8(lo));
            acc[1] = vaddw_high_u8(acc[1], lo);
            acc[2] = vaddw_u8(acc[2], vget_low_u8(hi));
            acc[3] = vaddw_high_u8(acc[3], hi);
        }
        for (dst, a) in o.as_chunks_mut::<8>().0.iter_mut().zip(acc) {
            // SAFETY: dst is [u16; 8], exactly one register.
            unsafe { vst1q_u16(dst.as_mut_ptr(), a) };
        }
    }
}

#[target_feature(enable = "neon")]
fn sum_f32(acc: [float32x4_t; 4]) -> f32 {
    vaddvq_f32(vaddq_f32(
        vaddq_f32(acc[0], acc[1]),
        vaddq_f32(acc[2], acc[3]),
    ))
}

// int8: 64 bytes per iteration into 4 accumulators, then 16-byte steps,
// then scalar for the last < 16. Dim 96 is 64 + 32, so the middle loop
// keeps it off the scalar path.

#[cfg(target_feature = "dotprod")]
#[target_feature(enable = "neon,dotprod")]
fn dot_i8_sdot(a: &[i8], b: &[i8]) -> i32 {
    let (ca, ta) = a.as_chunks::<64>();
    let (cb, tb) = b.as_chunks::<64>();
    let mut acc = [vdupq_n_s32(0); 4];
    for (x, y) in ca.iter().zip(cb) {
        // SAFETY: x and y are [i8; 64], exactly four 16-lane registers.
        let (x, y) = unsafe { (vld1q_s8_x4(x.as_ptr()), vld1q_s8_x4(y.as_ptr())) };
        acc[0] = vdotq_s32(acc[0], x.0, y.0);
        acc[1] = vdotq_s32(acc[1], x.1, y.1);
        acc[2] = vdotq_s32(acc[2], x.2, y.2);
        acc[3] = vdotq_s32(acc[3], x.3, y.3);
    }
    let (ca, ta) = ta.as_chunks::<16>();
    let (cb, tb) = tb.as_chunks::<16>();
    for (x, y) in ca.iter().zip(cb) {
        // SAFETY: x and y are [i8; 16], exactly one register.
        let (x, y) = unsafe { (vld1q_s8(x.as_ptr()), vld1q_s8(y.as_ptr())) };
        acc[0] = vdotq_s32(acc[0], x, y);
    }
    let acc = vaddq_s32(vaddq_s32(acc[0], acc[1]), vaddq_s32(acc[2], acc[3]));
    vaddvq_s32(acc) + scalar::dot_i8(ta, tb)
}

#[cfg(target_feature = "dotprod")]
#[target_feature(enable = "neon,dotprod")]
fn l2_sq_i8_udot(a: &[i8], b: &[i8]) -> u32 {
    // |x - y| <= 255 always fits a u8, so SABD's 8-bit result read as
    // unsigned is exact, and UDOT squares and sums it into u32 lanes.
    let (ca, ta) = a.as_chunks::<64>();
    let (cb, tb) = b.as_chunks::<64>();
    let mut acc = [vdupq_n_u32(0); 4];
    for (x, y) in ca.iter().zip(cb) {
        // SAFETY: x and y are [i8; 64], exactly four 16-lane registers.
        let (x, y) = unsafe { (vld1q_s8_x4(x.as_ptr()), vld1q_s8_x4(y.as_ptr())) };
        let d = [
            vreinterpretq_u8_s8(vabdq_s8(x.0, y.0)),
            vreinterpretq_u8_s8(vabdq_s8(x.1, y.1)),
            vreinterpretq_u8_s8(vabdq_s8(x.2, y.2)),
            vreinterpretq_u8_s8(vabdq_s8(x.3, y.3)),
        ];
        for j in 0..4 {
            acc[j] = vdotq_u32(acc[j], d[j], d[j]);
        }
    }
    let (ca, ta) = ta.as_chunks::<16>();
    let (cb, tb) = tb.as_chunks::<16>();
    for (x, y) in ca.iter().zip(cb) {
        // SAFETY: x and y are [i8; 16], exactly one register.
        let (x, y) = unsafe { (vld1q_s8(x.as_ptr()), vld1q_s8(y.as_ptr())) };
        let d = vreinterpretq_u8_s8(vabdq_s8(x, y));
        acc[0] = vdotq_u32(acc[0], d, d);
    }
    let acc = vaddq_u32(vaddq_u32(acc[0], acc[1]), vaddq_u32(acc[2], acc[3]));
    vaddvq_u32(acc) + scalar::l2_sq_i8(ta, tb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::collection::vec;
    use proptest::prelude::*;

    // Lengths up to 300 cover every tail size for both chunk widths.
    fn f32_pair() -> impl Strategy<Value = (Vec<f32>, Vec<f32>)> {
        (0usize..300).prop_flat_map(|n| (vec(-1e3f32..1e3, n), vec(-1e3f32..1e3, n)))
    }

    fn i8_pair() -> impl Strategy<Value = (Vec<i8>, Vec<i8>)> {
        (0usize..300).prop_flat_map(|n| (vec(any::<i8>(), n), vec(any::<i8>(), n)))
    }

    /// Within 1e-4 of the scale of the sum. For dot products that's the sum
    /// of |terms|, since cancellation can make the result itself near zero.
    fn close(got: f32, want: f32, scale: f32) -> bool {
        (got - want).abs() <= 1e-4 * scale.max(1e-6)
    }

    proptest! {
        #[test]
        fn l2_sq_matches_scalar((a, b) in f32_pair()) {
            let want = scalar::l2_sq(&a, &b);
            prop_assert!(close(l2_sq(&a, &b), want, want));
        }

        #[test]
        fn dot_matches_scalar((a, b) in f32_pair()) {
            let scale: f32 = a.iter().zip(&b).map(|(x, y)| (x * y).abs()).sum();
            prop_assert!(close(dot(&a, &b), scalar::dot(&a, &b), scale));
        }

        #[test]
        fn cosine_matches_scalar((a, b) in f32_pair()) {
            let (got, want) = (cosine_distance(&a, &b), scalar::cosine_distance(&a, &b));
            prop_assert!(close(got, want, 1.0), "{got} vs {want}");
        }

        // m up to 300 includes sums that wrap u16 (m > 257), which both
        // must do identically.
        #[test]
        fn pq4_scan_is_exact(
            (blocks, lut) in (1usize..300, 0usize..4).prop_flat_map(|(m, nb)| {
                (vec(any::<u8>(), nb * m * 16), vec(any::<u8>(), m * 16))
            }),
        ) {
            let n = blocks.len() / lut.len() * 32;
            let (mut got, mut want) = (vec![1u16; n], vec![2u16; n]);
            pq4_scan(&blocks, &lut, &mut got);
            scalar::pq4_scan(&blocks, &lut, &mut want);
            prop_assert_eq!(got, want);
        }
    }

    #[cfg(target_feature = "dotprod")]
    proptest! {
        #[test]
        fn dot_i8_is_exact((a, b) in i8_pair()) {
            prop_assert_eq!(dot_i8(&a, &b), scalar::dot_i8(&a, &b));
        }

        #[test]
        fn l2_sq_i8_is_exact((a, b) in i8_pair()) {
            prop_assert_eq!(l2_sq_i8(&a, &b), scalar::l2_sq_i8(&a, &b));
        }
    }

    #[test]
    #[cfg(target_feature = "dotprod")]
    fn int8_extremes_at_max_len() {
        let lo = vec![i8::MIN; 65_536];
        let hi = vec![i8::MAX; 65_536];
        assert_eq!(l2_sq_i8(&lo, &hi), 65_536 * 255 * 255);
        assert_eq!(dot_i8(&lo, &lo), 65_536 * 128 * 128);
        assert_eq!(dot_i8(&lo, &hi), -65_536 * 128 * 127);
    }
}
