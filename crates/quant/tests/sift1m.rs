//! Quantization quality on SIFT1M (recall only, no timings). Needs the
//! dataset in ~/data/Nearpoint/sift1m/. Run with:
//! `cargo test --release -p quant --test sift1m -- --ignored --nocapture`

use std::path::PathBuf;

use quant::{PqConfig, Sq, adc_distance};
use rayon::prelude::*;

const QUERIES: usize = 200;
const DIM: usize = 128;

fn path(name: &str) -> PathBuf {
    let home = std::env::var("HOME").expect("HOME");
    PathBuf::from(home).join("data/Nearpoint/sift1m").join(name)
}

/// Reads .fvecs / .ivecs: each record is `[d: i32][d x 4 bytes]`. Returns
/// the raw 4-byte words, row-major, and d.
fn read_vecs(name: &str) -> (Vec<[u8; 4]>, usize) {
    let bytes = std::fs::read(path(name)).expect("read dataset");
    let d = i32::from_le_bytes(bytes[..4].try_into().expect("header")) as usize;
    let rec = 4 + 4 * d;
    assert_eq!(bytes.len() % rec, 0);
    let words = bytes
        .chunks_exact(rec)
        .flat_map(|r| r[4..].as_chunks::<4>().0.iter().copied())
        .collect();
    (words, d)
}

fn fvecs(name: &str) -> Vec<f32> {
    let (w, d) = read_vecs(name);
    assert_eq!(d, DIM);
    w.into_iter().map(f32::from_le_bytes).collect()
}

/// Ids of the `k` smallest distances (ties by id).
fn top_k(dist: impl Iterator<Item = (f32, u32)>, k: usize) -> Vec<u32> {
    let mut v: Vec<(f32, u32)> = dist.collect();
    let cmp = |a: &(f32, u32), b: &(f32, u32)| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1));
    if v.len() > k {
        v.select_nth_unstable_by(k, cmp);
        v.truncate(k);
    }
    v.sort_unstable_by(cmp);
    v.into_iter().map(|p| p.1).collect()
}

/// Mean over queries of |found[..10] ∩ gt[..10]| / 10.
fn recall10(found: &[Vec<u32>], gt: &[Vec<u32>]) -> f64 {
    let hits: usize = found
        .iter()
        .zip(gt)
        .map(|(f, g)| f.iter().take(10).filter(|id| g[..10].contains(id)).count())
        .sum();
    hits as f64 / (10 * found.len()) as f64
}

/// Fraction of the true top 10 found anywhere in `found`.
fn recall10_at(found: &[Vec<u32>], gt: &[Vec<u32>]) -> f64 {
    let hits: usize = found
        .iter()
        .zip(gt)
        .map(|(f, g)| g[..10].iter().filter(|id| f.contains(id)).count())
        .sum();
    hits as f64 / (10 * found.len()) as f64
}

fn mse(x: &[f32], decode: impl Fn(usize) -> Vec<f32> + Sync) -> f64 {
    let n = x.len() / DIM;
    (0..n)
        .into_par_iter()
        .map(|i| f64::from(kernels::l2_sq(&x[i * DIM..][..DIM], &decode(i))))
        .sum::<f64>()
        / n as f64
}

#[test]
#[ignore = "needs SIFT1M on disk; run in --release"]
fn sift1m_quality() {
    let base = fvecs("sift_base.fvecs");
    let learn = fvecs("sift_learn.fvecs");
    let queries = fvecs("sift_query.fvecs");
    let (gtw, _) = read_vecs("sift_groundtruth.ivecs");
    let gt: Vec<Vec<u32>> = gtw
        .as_chunks::<100>()
        .0
        .iter()
        .take(QUERIES)
        .map(|r| r.iter().map(|&w| u32::from_le_bytes(w)).collect())
        .collect();
    let queries = &queries[..QUERIES * DIM];
    let n = base.len() / DIM;
    let row = |i: usize| &base[i * DIM..][..DIM];
    println!("base {n}, learn {}, queries {QUERIES}", learn.len() / DIM);

    // Per-vector total variance (sum over dims), for scale.
    let mean: Vec<f64> = (0..DIM)
        .map(|t| (0..n).map(|i| f64::from(base[i * DIM + t])).sum::<f64>() / n as f64)
        .collect();
    let var: f64 = (0..n)
        .into_par_iter()
        .map(|i| {
            row(i)
                .iter()
                .zip(&mean)
                .map(|(&a, m)| (f64::from(a) - m).powi(2))
                .sum::<f64>()
        })
        .sum::<f64>()
        / n as f64;
    println!("total variance per vector: {var:.0}");

    let sq = Sq::train(&learn, DIM, 100_000, 42).expect("sq train");
    let sq_codes = sq.encode_all(&base).expect("sq encode");
    let sq_row = |i: usize| &sq_codes[i * DIM..][..DIM];
    let sq_q: Vec<Vec<i8>> = queries
        .as_chunks::<DIM>()
        .0
        .iter()
        .map(|q| sq.encode(q))
        .collect();
    println!(
        "SQ int8: scale {:.3}, MSE {:.1} ({:.5} of variance)",
        sq.scale(),
        mse(&base, |i| sq.decode(sq_row(i))),
        mse(&base, |i| sq.decode(sq_row(i))) / var
    );
    let found: Vec<Vec<u32>> = sq_q
        .par_iter()
        .map(|q| top_k((0..n).map(|i| (sq.l2_sq(q, sq_row(i)), i as u32)), 10))
        .collect();
    println!("SQ int8 full scan: recall@10 {:.4}", recall10(&found, &gt));

    for m in [16, 32] {
        let pq = PqConfig::new(DIM, m).train(&learn).expect("pq train");
        let codes = pq.encode_all(&base).expect("pq encode");
        let code = |i: usize| &codes[i * m..][..m];
        let e = mse(&base, |i| pq.decode(code(i)));
        println!("PQ m={m}: MSE {e:.0} ({:.4} of variance)", e / var);

        let cand: Vec<Vec<u32>> = queries
            .par_chunks_exact(DIM)
            .map(|q| {
                let t = pq.distance_table(q);
                top_k((0..n).map(|i| (adc_distance(&t, code(i)), i as u32)), 1000)
            })
            .collect();
        println!("  ADC full scan: recall@10 {:.4}", recall10(&cand, &gt));
        for depth in [100, 1000] {
            let c: Vec<&[u32]> = cand.iter().map(|c| &c[..depth]).collect();
            let ceil: Vec<Vec<u32>> = c.iter().map(|c| c.to_vec()).collect();
            let f32_rr: Vec<Vec<u32>> = c
                .par_iter()
                .zip(queries.par_chunks_exact(DIM))
                .map(|(c, q)| {
                    let d = c.iter().map(|&i| (kernels::l2_sq(q, row(i as usize)), i));
                    top_k(d, 10)
                })
                .collect();
            let sq_rr: Vec<Vec<u32>> = c
                .par_iter()
                .zip(&sq_q)
                .map(|(c, q)| {
                    let d = c.iter().map(|&i| (sq.l2_sq(q, sq_row(i as usize)), i));
                    top_k(d, 10)
                })
                .collect();
            // Asymmetric: f32 query against the decoded int8 vector.
            let sq_asym: Vec<Vec<u32>> = c
                .par_iter()
                .zip(queries.par_chunks_exact(DIM))
                .map(|(c, q)| {
                    let d = c
                        .iter()
                        .map(|&i| (kernels::l2_sq(q, &sq.decode(sq_row(i as usize))), i));
                    top_k(d, 10)
                })
                .collect();
            println!(
                "  top-{depth}: true top-10 in candidates {:.4}, f32 re-rank recall@10 {:.4}, \
                 SQ int8 re-rank recall@10 {:.4} (symmetric) {:.4} (f32 query)",
                recall10_at(&ceil, &gt),
                recall10(&f32_rr, &gt),
                recall10(&sq_rr, &gt),
                recall10(&sq_asym, &gt)
            );
        }
    }
}
