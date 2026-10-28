//! Recall/latency harness: runs a dataset's queries against an index and
//! scores them against the dataset's ground truth.
//!
//! `cargo run --release -p bench -- --dataset sift1m --index flat`

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use bench::dataset::read_xvecs;
use clap::{Parser, ValueEnum};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    dataset: Dataset,
    #[arg(long, default_value = "flat")]
    index: IndexKind,
    /// Neighbours per query; recall is measured at this k.
    #[arg(long, default_value_t = 10)]
    k: usize,
    /// Run only the first N queries.
    #[arg(long)]
    queries: Option<usize>,
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
enum IndexKind {
    /// Exact brute force (single-threaded).
    Flat,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let root = match args.data.strip_prefix("~/") {
        Some(rest) => PathBuf::from(std::env::var("HOME").context("HOME unset")?).join(rest),
        None => PathBuf::from(&args.data),
    };
    let (dir, prefix) = match args.dataset {
        Dataset::Siftsmall => ("siftsmall", "siftsmall"),
        Dataset::Sift1m => ("sift1m", "sift"),
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

    let mut latencies = Vec::with_capacity(query.len());
    // Two recalls. `id_hits`: result ids found in the ground-truth top-k.
    // `tie_hits`: results whose true distance is <= the k-th ground-truth
    // distance (big-ann-benchmarks' definition), so an equally near id the
    // ground truth happened not to list still counts. True distances are
    // recomputed from the base vectors, never taken from the index.
    let (mut id_hits, mut tie_hits) = (0usize, 0usize);
    let started = Instant::now();
    for i in 0..query.len() {
        let q = query.row(i);
        let t = Instant::now();
        let result = match args.index {
            IndexKind::Flat => index::flat::search(base.as_flat(), base.dim(), q, args.k),
        };
        latencies.push(t.elapsed());

        let truth = &gt.row(i)[..args.k];
        let true_dist = |id: usize| kernels::scalar::l2_sq(q, base.row(id));
        let kth = true_dist(truth[args.k - 1] as usize);
        for h in &result {
            id_hits += usize::from(truth.contains(&(h.id as i32)));
            tie_hits += usize::from(true_dist(h.id as usize) <= kth);
        }
    }
    let total = started.elapsed();

    latencies.sort();
    let pct = |p: f64| latencies[((latencies.len() - 1) as f64 * p).round() as usize];
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    let mem = storage::sys::peak_memory()?;
    let mib = |b: u64| b as f64 / (1 << 20) as f64;
    println!(
        "recall@{k}={recall:.4} (id match {id_recall:.4}) p50={p50:.3}ms p95={p95:.3}ms p99={p99:.3}ms \
         qps={qps:.1} (1 thread) peak_rss={rss:.0}MiB peak_phys_footprint={phys:.0}MiB",
        k = args.k,
        recall = tie_hits as f64 / (query.len() * args.k) as f64,
        id_recall = id_hits as f64 / (query.len() * args.k) as f64,
        p50 = ms(pct(0.50)),
        p95 = ms(pct(0.95)),
        p99 = ms(pct(0.99)),
        qps = query.len() as f64 / total.as_secs_f64(),
        rss = mib(mem.rss),
        phys = mib(mem.phys_footprint),
    );
    Ok(())
}
