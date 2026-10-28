//! Filtered-search harness (tasks 7.3/7.4): builds HNSW once, then for each
//! filter selectivity draws a seeded uniform random subset of the base ids,
//! computes exact filtered ground truth by brute force over it, and runs the
//! queries one at a time with that filter. Reports recall, the distances
//! each search computed and which plan ran, per (selectivity, ef, plan).
//!
//! `cargo run --release -p bench --bin filtered -- --dataset sift1m \
//!     --selectivity 0.01,0.1,0.5 --ef 64,128 --plan auto,graph`

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use bench::dataset::{Vectors, read_xvecs};
use clap::{Parser, ValueEnum};
use index::hnsw::{self, Plan};
use index::{Hit, Hnsw, Index, Metric, SearchParams};
use rayon::prelude::*;
use roaring::RoaringBitmap;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "sift1m")]
    dataset: Dataset,
    /// Fractions of the base ids a filter keeps, comma-separated.
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "0.001,0.005,0.01,0.02,0.05,0.1,0.5"
    )]
    selectivity: Vec<f64>,
    /// HNSW search beam widths, comma-separated.
    #[arg(long, value_delimiter = ',', default_value = "64,128")]
    ef: Vec<usize>,
    /// Plans to run: `auto` is what `search` picks; `graph` and `brute`
    /// force one side of the planner's threshold.
    #[arg(long, value_delimiter = ',', default_value = "auto")]
    plan: Vec<PlanArg>,
    /// Neighbours per query; recall is measured at this k.
    #[arg(long, default_value_t = 10)]
    k: usize,
    /// Run only the first N queries (ground truth costs N full scans of
    /// each filter).
    #[arg(long, default_value_t = 1000)]
    queries: usize,
    #[arg(long, default_value_t = 16)]
    m: usize,
    #[arg(long, default_value_t = 200)]
    ef_construction: usize,
    /// Seeds HNSW node levels and the filters.
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Append one CSV row per (selectivity, ef, plan) to this file.
    #[arg(long)]
    csv: Option<PathBuf>,
    /// Dataset root.
    #[arg(long, env = "NEARPOINT_DATA", default_value = "~/data/Nearpoint")]
    data: String,
}

#[derive(Clone, Copy, ValueEnum)]
enum Dataset {
    Siftsmall,
    Sift1m,
}

#[derive(Clone, Copy, ValueEnum)]
enum PlanArg {
    Auto,
    Graph,
    Brute,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let root = match args.data.strip_prefix("~/") {
        Some(rest) => PathBuf::from(std::env::var("HOME").context("HOME unset")?).join(rest),
        None => PathBuf::from(&args.data),
    };
    let (dir, prefix, name) = match args.dataset {
        Dataset::Siftsmall => ("siftsmall", "siftsmall", "siftsmall"),
        Dataset::Sift1m => ("sift1m", "sift", "sift1m"),
    };
    let file = |kind: &str| root.join(dir).join(format!("{prefix}_{kind}.fvecs"));
    let base = read_xvecs::<f32>(&file("base"), None)?;
    let query = read_xvecs::<f32>(&file("query"), Some(args.queries))?;
    ensure!(base.dim() == query.dim(), "base/query dims differ");
    ensure!(args.k > 0, "k must be at least 1");

    let t = Instant::now();
    let cfg = hnsw::Config {
        m: args.m,
        ef_construction: args.ef_construction,
        seed: args.seed,
    };
    let mut index = Hnsw::new(base.dim(), Metric::L2, cfg);
    index.add(base.as_flat())?;
    println!(
        "{name}: {} base, {} queries; built hnsw m={} efc={} in {:.1}s",
        base.len(),
        query.len(),
        args.m,
        args.ef_construction,
        t.elapsed().as_secs_f64()
    );

    let mut csv = match &args.csv {
        Some(path) => {
            let new = !path.exists();
            let mut f = File::options().create(true).append(true).open(path)?;
            if new {
                writeln!(
                    f,
                    "dataset,m,ef_construction,seed,selectivity,filtered,ef,k,plan,\
                     recall,id_recall,mean_distances,p50_ms,p99_ms,qps"
                )?;
            }
            Some(f)
        }
        None => None,
    };

    for &s in &args.selectivity {
        let filter = random_filter(base.len(), s, args.seed);
        let truth = ground_truth(&base, &query, &filter, args.k);
        for &ef in &args.ef {
            let params = SearchParams {
                ef,
                filter: Some(&filter),
                ..SearchParams::default()
            };
            for &plan in &args.plan {
                let plan = match plan {
                    PlanArg::Auto => index.plan(args.k, &params),
                    PlanArg::Graph => Plan::Graph,
                    PlanArg::Brute => Plan::BruteForce,
                };
                let r = run(&index, &base, &query, &truth, &params, args.k, plan)?;
                let label = match plan {
                    Plan::Graph => "graph",
                    Plan::BruteForce => "brute",
                };
                println!(
                    "sel={s:<6} n={:<7} ef={ef:<4} {label:<5} recall@{k}={:.4} (id {:.4}) \
                     dists={:<8.0} p50={:.3}ms p99={:.3}ms qps={:.0}",
                    filter.len(),
                    r.recall,
                    r.id_recall,
                    r.distances,
                    ms(r.p50),
                    ms(r.p99),
                    r.qps,
                    k = args.k,
                );
                if let Some(f) = &mut csv {
                    writeln!(
                        f,
                        "{name},{},{},{},{s},{},{ef},{},{label},{:.4},{:.4},{:.0},{:.4},{:.4},{:.1}",
                        args.m,
                        args.ef_construction,
                        args.seed,
                        filter.len(),
                        args.k,
                        r.recall,
                        r.id_recall,
                        r.distances,
                        ms(r.p50),
                        ms(r.p99),
                        r.qps,
                    )?;
                }
            }
        }
    }
    Ok(())
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// Each id kept independently with probability `s`, from a hash of
/// (seed, selectivity, id), so a run is reproducible.
fn random_filter(n: usize, s: f64, seed: u64) -> RoaringBitmap {
    let cut = (s.clamp(0.0, 1.0) * u64::MAX as f64) as u64;
    let seed = seed ^ s.to_bits();
    (0..n as u32)
        .filter(|&id| splitmix64(seed ^ u64::from(id).wrapping_mul(0x9E37_79B9)) < cut)
        .collect()
}

fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Exact top-`k` filtered neighbours of every query, nearest first, from
/// the scalar kernel over the base vectors (fewer if the filter is smaller).
fn ground_truth(
    base: &Vectors<f32>,
    query: &Vectors<f32>,
    filter: &RoaringBitmap,
    k: usize,
) -> Vec<Vec<Hit>> {
    (0..query.len())
        .into_par_iter()
        .map(|i| {
            let q = query.row(i);
            let mut all: Vec<Hit> = filter
                .iter()
                .map(|id| Hit {
                    id,
                    dist: kernels::scalar::l2_sq(q, base.row(id as usize)),
                })
                .collect();
            if all.len() > k {
                all.select_nth_unstable(k - 1);
                all.truncate(k);
            }
            all.sort_unstable();
            all
        })
        .collect()
}

struct Run {
    recall: f64,
    id_recall: f64,
    /// Mean distances computed per search (`hnsw::distances`).
    distances: f64,
    p50: Duration,
    p99: Duration,
    qps: f64,
}

fn run(
    index: &Hnsw,
    base: &Vectors<f32>,
    query: &Vectors<f32>,
    truth: &[Vec<Hit>],
    params: &SearchParams,
    k: usize,
    plan: Plan,
) -> Result<Run> {
    let mut latencies = Vec::with_capacity(query.len());
    // Tie-aware and id-match recall as in the main harness (ADR 0002): a
    // result counts if its true distance, recomputed from the base vectors,
    // is <= the k-th true filtered neighbour's.
    let (mut id_hits, mut tie_hits, mut wanted) = (0usize, 0usize, 0usize);
    let distances = hnsw::distances();
    let started = Instant::now();
    for (i, truth) in truth.iter().enumerate() {
        let q = query.row(i);
        let t = Instant::now();
        let result = index.search_with(q, k, params, plan)?;
        latencies.push(t.elapsed());

        let Some(kth) = truth.last() else { continue };
        wanted += truth.len();
        for h in &result {
            id_hits += usize::from(truth.iter().any(|t| t.id == h.id));
            tie_hits += usize::from(kernels::scalar::l2_sq(q, base.row(h.id as usize)) <= kth.dist);
        }
    }
    let total = started.elapsed();
    let distances = (hnsw::distances() - distances) as f64 / query.len() as f64;

    latencies.sort();
    let pct = |p: f64| latencies[((latencies.len() - 1) as f64 * p).round() as usize];
    let n = wanted.max(1) as f64;
    Ok(Run {
        recall: tie_hits as f64 / n,
        id_recall: id_hits as f64 / n,
        distances,
        p50: pct(0.50),
        p99: pct(0.99),
        qps: query.len() as f64 / total.as_secs_f64(),
    })
}
