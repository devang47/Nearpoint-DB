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

/// Vectors per 4-bit fast-scan block.
pub const PQ4_BLOCK: usize = 32;

/// Writes the 4-bit PQ `code` (`m` values < 16) of vector `i` into the
/// fast-scan layout `blocks`, which must already hold block `i / 32`.
///
/// Layout: vectors in blocks of 32; a block is `m` groups of 16 bytes, one
/// per sub-quantizer. In group `j`, byte `b` holds sub-code `j` of vector
/// `b` in its low nibble and of vector `b + 16` in its high nibble, so one
/// 16-byte load and one table lookup per nibble cover the whole block.
pub fn pq4_set(blocks: &mut [u8], i: usize, code: &[u8]) {
    let m = code.len();
    let block = &mut blocks[i / PQ4_BLOCK * m * 16..][..m * 16];
    let (b, shift) = (i % 16, if i % PQ4_BLOCK < 16 { 0 } else { 4 });
    for (group, &c) in block.as_chunks_mut::<16>().0.iter_mut().zip(code) {
        debug_assert!(c < 16);
        group[b] = group[b] & !(0xF << shift) | (c & 0xF) << shift;
    }
}

/// 4-bit PQ fast-scan: for every vector in `blocks` (layout in
/// [`pq4_set`]), the sum over sub-quantizers `j` of `lut[j * 16 + code_j]`,
/// into `out[i]`. `lut` is `m * 16` quantized distances; sums wrap at
/// `u16`, which can't happen for `m <= 257`. Panics unless `out` has 32
/// slots per block.
pub fn pq4_scan(blocks: &[u8], lut: &[u8], out: &mut [u16]) {
    let m = lut.len() / 16;
    assert!(m > 0 && lut.len() == m * 16);
    assert_eq!(blocks.len() / (m * 16) * PQ4_BLOCK, out.len());
    assert!(blocks.len().is_multiple_of(m * 16));
    for (block, o) in blocks
        .chunks_exact(m * 16)
        .zip(out.as_chunks_mut::<PQ4_BLOCK>().0)
    {
        o.fill(0);
        for (group, t) in block
            .as_chunks::<16>()
            .0
            .iter()
            .zip(lut.as_chunks::<16>().0)
        {
            for (b, &byte) in group.iter().enumerate() {
                o[b] = o[b].wrapping_add(u16::from(t[usize::from(byte & 0xF)]));
                o[b + 16] = o[b + 16].wrapping_add(u16::from(t[usize::from(byte >> 4)]));
            }
        }
    }
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
    fn pq4_scan_sums_each_vectors_entries() {
        // 40 vectors (2 blocks, the second partly empty), m = 3.
        let m = 3;
        let codes: Vec<Vec<u8>> = (0..40usize)
            .map(|i| (0..m).map(|j| ((i * 7 + j * 5) % 16) as u8).collect())
            .collect();
        let mut blocks = vec![0u8; 2 * m * 16];
        for (i, c) in codes.iter().enumerate() {
            pq4_set(&mut blocks, i, c);
        }
        let lut: Vec<u8> = (0..m * 16).map(|e| (e * 37 % 251) as u8).collect();
        let mut out = vec![0u16; 64];
        pq4_scan(&blocks, &lut, &mut out);
        for (i, c) in codes.iter().enumerate() {
            let want: u16 = c
                .iter()
                .enumerate()
                .map(|(j, &c)| u16::from(lut[j * 16 + c as usize]))
                .sum();
            assert_eq!(out[i], want, "vector {i}");
        }
        // Unused slots of the last block read code 0 everywhere.
        assert_eq!(out[63], (0..m).map(|j| u16::from(lut[j * 16])).sum());
    }

    #[test]
    #[should_panic]
    fn length_mismatch_panics() {
        l2_sq(&[1.0], &[1.0, 2.0]);
    }
}
