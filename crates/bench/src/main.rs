//! Recall/latency harness: builds an index over a dataset's base vectors,
//! runs its queries one at a time, and scores them against ground truth.
//! With several `--ef` values it builds once and sweeps the search width,
//! giving a recall-vs-latency curve (`--csv` writes it out).
//!
//! `cargo run --release -p bench -- --dataset sift1m --index hnsw --ef 16,32,64,128`

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use bench::dataset::{Vectors, read_xvecs};
use clap::{Parser, ValueEnum};
use index::{Flat, Hnsw, Index, Metric, SearchParams, hnsw};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    dataset: Dataset,
    #[arg(long, default_value = "hnsw")]
    index: IndexKind,
    /// Neighbours per query; recall is measured at this k.
    #[arg(long, default_value_t = 10)]
    k: usize,
    /// Run only the first N queries.
    #[arg(long)]
    queries: Option<usize>,
    /// HNSW links per node (layer 0 gets 2m).
    #[arg(long, default_value_t = 16)]
    m: usize,
    /// HNSW beam width while building.
    #[arg(long, default_value_t = 200)]
    ef_construction: usize,
    /// HNSW search beam widths to sweep, comma-separated.
    #[arg(long, value_delimiter = ',', default_value = "64")]
    ef: Vec<usize>,
    /// Seeds HNSW node levels.
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Append one CSV row per ef value to this file.
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

#[derive(Clone, Copy, ValueEnum, PartialEq)]
enum IndexKind {
    /// Exact brute force; each query uses every core.
    Flat,
    Hnsw,
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
    let file = |kind: &str, ext: &str| root.join(dir).join(format!("{prefix}_{kind}.{ext}"));

    let t = Instant::now();
    let base = read_xvecs::<f32>(&file("base", "fvecs"), None)?;
    let query = read_xvecs::<f32>(&file("query", "fvecs"), args.queries)?;
    let gt = read_xvecs::<i32>(&file("groundtruth", "ivecs"), args.queries)?;
    ensure!(base.dim() == query.dim(), "base/query dims differ");
    ensure!(
        (1..=gt.dim()).contains(&args.k),
        "k must be in 1..={}",
        gt.dim()
    );
    println!(
        "loaded {} base, {} queries, dim {} in {:.2?}",
        base.len(),
        query.len(),
        base.dim(),
        t.elapsed()
    );

    let t = Instant::now();
    let (mut index, label): (Box<dyn Index>, String) = match args.index {
        IndexKind::Flat => (Box::new(Flat::new(base.dim(), Metric::L2)), "flat".into()),
        IndexKind::Hnsw => {
            let cfg = hnsw::Config {
                m: args.m,
                ef_construction: args.ef_construction,
                seed: args.seed,
            };
            let label = format!("hnsw m={} efc={}", cfg.m, cfg.ef_construction);
            (Box::new(Hnsw::new(base.dim(), Metric::L2, cfg)), label)
        }
    };
    index.add(base.as_flat())?;
    let build = t.elapsed();
    println!(
        "built {label} in {:.1}s ({} threads)",
        build.as_secs_f64(),
        rayon::current_num_threads()
    );

    let mut csv = match &args.csv {
        Some(path) => {
            let new = !path.exists();
            let mut f = File::options().create(true).append(true).open(path)?;
            if new {
                writeln!(
                    f,
                    "dataset,index,m,ef_construction,seed,ef,k,recall,id_recall,\
                     p50_ms,p95_ms,p99_ms,qps,build_s,threads"
                )?;
            }
            Some(f)
        }
        None => None,
    };

    // Flat ignores ef, so sweeping it would just repeat one measurement.
    let efs = if args.index == IndexKind::Flat {
        vec![0]
    } else {
        args.ef.clone()
    };
    for ef in efs {
        let r = run_queries(&*index, &base, &query, &gt, args.k, ef)?;
        println!(
            "ef={ef:<4} recall@{k}={:.4} (id match {:.4}) p50={:.3}ms p95={:.3}ms \
             p99={:.3}ms qps={:.0}",
            r.recall,
            r.id_recall,
            ms(r.p50),
            ms(r.p95),
            ms(r.p99),
            r.qps,
            k = args.k,
        );
        if let Some(f) = &mut csv {
            let hnsw = args.index == IndexKind::Hnsw;
            writeln!(
                f,
                "{name},{kind},{m},{efc},{seed},{ef},{k},{:.4},{:.4},{:.4},{:.4},{:.4},{:.1},{:.1},{}",
                r.recall,
                r.id_recall,
                ms(r.p50),
                ms(r.p95),
                ms(r.p99),
                r.qps,
                build.as_secs_f64(),
                rayon::current_num_threads(),
                kind = if hnsw { "hnsw" } else { "flat" },
                m = if hnsw {
                    args.m.to_string()
                } else {
                    String::new()
                },
                efc = if hnsw {
                    args.ef_construction.to_string()
                } else {
                    String::new()
                },
                seed = if hnsw {
                    args.seed.to_string()
                } else {
                    String::new()
                },
                k = args.k,
            )?;
        }
    }

    let mem = storage::sys::peak_memory()?;
    let mib = |b: u64| b as f64 / (1 << 20) as f64;
    println!(
        "peak_rss={:.0}MiB peak_phys_footprint={:.0}MiB",
        mib(mem.rss),
        mib(mem.phys_footprint)
    );
    Ok(())
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

struct Run {
    recall: f64,
    id_recall: f64,
    p50: Duration,
    p95: Duration,
    p99: Duration,
    /// Queries per second, issued one at a time from one thread.
    qps: f64,
}

fn run_queries(
    index: &dyn Index,
    base: &Vectors<f32>,
    query: &Vectors<f32>,
    gt: &Vectors<i32>,
    k: usize,
    ef: usize,
) -> Result<Run> {
    let params = SearchParams { ef };
    let mut latencies = Vec::with_capacity(query.len());
    // Two recalls. `id_hits`: result ids found in the ground-truth top-k.
    // `tie_hits`: results whose true distance is <= the k-th ground-truth
    // distance (big-ann-benchmarks' definition, ADR 0002), so an equally
    // near id the ground truth happened not to list still counts. True
    // distances are recomputed from the base vectors, never taken from the
    // index.
    let (mut id_hits, mut tie_hits) = (0usize, 0usize);
    let started = Instant::now();
    for i in 0..query.len() {
        let q = query.row(i);
        let t = Instant::now();
        let result = index.search(q, k, &params)?;
        latencies.push(t.elapsed());

        let truth = &gt.row(i)[..k];
        let true_dist = |id: usize| kernels::scalar::l2_sq(q, base.row(id));
        let kth = true_dist(truth[k - 1] as usize);
        for h in &result {
            id_hits += usize::from(truth.contains(&(h.id as i32)));
            tie_hits += usize::from(true_dist(h.id as usize) <= kth);
        }
    }
    let total = started.elapsed();

    latencies.sort();
    let pct = |p: f64| latencies[((latencies.len() - 1) as f64 * p).round() as usize];
    let n = (query.len() * k) as f64;
    Ok(Run {
        recall: tie_hits as f64 / n,
        id_recall: id_hits as f64 / n,
        p50: pct(0.50),
        p95: pct(0.95),
        p99: pct(0.99),
        qps: query.len() as f64 / total.as_secs_f64(),
    })
}
