//! Scalar vs NEON for every kernel at dims 96 (Deep), 128 (SIFT) and
//! 768 (text embeddings). Run with `cargo bench -p kernels`.

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use kernels::{neon, scalar};

const DIMS: [usize; 3] = [96, 128, 768];

/// Benchmarks `scalar` and `neon` side by side. Taking `impl Fn` (not fn
/// pointers) lets each kernel inline, so call overhead doesn't skew the ratio.
fn pair<T, R>(
    c: &mut Criterion,
    name: &str,
    gen_vec: impl Fn(usize, u32) -> Vec<T>,
    scalar: impl Fn(&[T], &[T]) -> R,
    neon: impl Fn(&[T], &[T]) -> R,
) {
    let mut g = c.benchmark_group(name);
    g.warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));
    for dim in DIMS {
        let (a, b) = (gen_vec(dim, 1), gen_vec(dim, 2));
        g.throughput(Throughput::Elements(dim as u64));
        g.bench_with_input(BenchmarkId::new("scalar", dim), &dim, |bch, _| {
            bch.iter(|| scalar(black_box(&a), black_box(&b)))
        });
        g.bench_with_input(BenchmarkId::new("neon", dim), &dim, |bch, _| {
            bch.iter(|| neon(black_box(&a), black_box(&b)))
        });
    }
    g.finish();
}

// Deterministic, non-degenerate values; the kernels' speed doesn't depend
// on them (no denormals in this range).
fn f32s(n: usize, seed: u32) -> Vec<f32> {
    (0..n as u32)
        .map(|i| (i.wrapping_mul(2_654_435_761).wrapping_add(seed) % 2001) as f32 / 10.0 - 100.0)
        .collect()
}

fn i8s(n: usize, seed: u32) -> Vec<i8> {
    (0..n as u32)
        .map(|i| i.wrapping_mul(2_654_435_761).wrapping_add(seed) as i8)
        .collect()
}

fn benches(c: &mut Criterion) {
    pair(c, "l2_sq", f32s, scalar::l2_sq, neon::l2_sq);
    pair(c, "dot", f32s, scalar::dot, neon::dot);
    pair(
        c,
        "cosine",
        f32s,
        scalar::cosine_distance,
        neon::cosine_distance,
    );
    #[cfg(target_feature = "dotprod")]
    {
        pair(c, "l2_sq_i8", i8s, scalar::l2_sq_i8, neon::l2_sq_i8);
        pair(c, "dot_i8", i8s, scalar::dot_i8, neon::dot_i8);
    }
}

criterion_group!(kernel_benches, benches);
criterion_main!(kernel_benches);
