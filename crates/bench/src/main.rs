//! Recall/latency harness: builds an index over a dataset's base vectors,
//! runs its queries one at a time, and scores them against ground truth.
//! With several `--ef` (or `--nprobe`, `--rerank`) values it builds once
//! and sweeps every combination, giving a recall-vs-latency curve (`--csv`
//! writes it out).
//!
//! `cargo run --release -p bench -- --dataset sift1m --index hnsw --ef 16,32,64,128`
//! `cargo run --release -p bench -- --dataset sift1m --index ivfpq --nprobe 8,16,32 --rerank 0,100`

use std::borrow::Cow;
use std::fs::File;
use std::io::Write;
use std::ops::Range;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use bench::dataset::{Vectors, read_bin, read_bin_ground_truth, read_xvecs};
use clap::{Parser, ValueEnum};
use index::{Flat, Hnsw, Index, IvfPq, Metric, SearchParams, hnsw, ivfpq};

/// Vectors handed to `add` per call: u8 datasets are converted to f32 one
/// batch at a time, so the harness never holds a full f32 copy.
const ADD_BATCH: usize = 100_000;

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
    /// Seeds HNSW node levels and IVF-PQ training.
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// IVF-PQ coarse cells.
    #[arg(long, default_value_t = 1024)]
    nlist: usize,
    /// IVF-PQ subspaces (bytes per code at 8 bits).
    #[arg(long, default_value_t = 16)]
    pq_m: usize,
    /// IVF-PQ bits per sub-code: 8, or 4 for fast-scan.
    #[arg(long, default_value_t = 8)]
    nbits: u32,
    /// IVF-PQ training sample size.
    #[arg(long, default_value_t = 262_144)]
    max_train: usize,
    /// IVF-PQ lists probed per query, comma-separated.
    #[arg(long, value_delimiter = ',', default_value = "16")]
    nprobe: Vec<usize>,
    /// IVF-PQ candidates re-ranked with int8 vectors (0 = off), comma-separated.
    #[arg(long, value_delimiter = ',', default_value = "0")]
    rerank: Vec<usize>,
    /// Append one CSV row per parameter combination to this file.
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
    /// The first 10M BIGANN vectors (u8), from big-ann-benchmarks.
    Bigann10m,
}

#[derive(Clone, Copy, ValueEnum, PartialEq)]
enum IndexKind {
    /// Exact brute force; each query uses every core.
    Flat,
    Hnsw,
    Ivfpq,
}

/// Base or query vectors as stored on disk; rows come out as f32.
enum Rows {
    F32(Vectors<f32>),
    U8(Vectors<u8>),
}

impl Rows {
    fn len(&self) -> usize {
        match self {
            Rows::F32(v) => v.len(),
            Rows::U8(v) => v.len(),
        }
    }

    fn dim(&self) -> usize {
        match self {
            Rows::F32(v) => v.dim(),
            Rows::U8(v) => v.dim(),
        }
    }

    /// Rows `r`, row-major; borrowed when already f32.
    fn rows(&self, r: Range<usize>) -> Cow<'_, [f32]> {
        let d = self.dim();
        match self {
            Rows::F32(v) => Cow::Borrowed(&v.as_flat()[r.start * d..r.end * d]),
            Rows::U8(v) => Cow::Owned(
                v.as_flat()[r.start * d..r.end * d]
                    .iter()
                    .map(|&x| f32::from(x))
                    .collect(),
            ),
        }
    }

    fn row(&self, i: usize) -> Cow<'_, [f32]> {
        self.rows(i..i + 1)
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let root = match args.data.strip_prefix("~/") {
        Some(rest) => PathBuf::from(std::env::var("HOME").context("HOME unset")?).join(rest),
        None => PathBuf::from(&args.data),
    };
    let name = match args.dataset {
        Dataset::Siftsmall => "siftsmall",
        Dataset::Sift1m => "sift1m",
        Dataset::Bigann10m => "bigann10m",
    };
    let dir = root.join(name);

    let t = Instant::now();
    let (base, query, gt) = match args.dataset {
        Dataset::Siftsmall | Dataset::Sift1m => {
            let p = if name == "sift1m" { "sift" } else { name };
            let file = |kind: &str, ext: &str| dir.join(format!("{p}_{kind}.{ext}"));
            (
                Rows::F32(read_xvecs(&file("base", "fvecs"), None)?),
                Rows::F32(read_xvecs(&file("query", "fvecs"), args.queries)?),
                // Ids are non-negative, so the i32 file reads as u32.
                read_xvecs::<u32>(&file("groundtruth", "ivecs"), args.queries)?,
            )
        }
        // tools/fetch-dataset.sh bigann10m. The base is a byte prefix of
        // the 1B file, so its header still says 1B: read exactly 10M.
        Dataset::Bigann10m => (
            Rows::U8(read_bin(&dir.join("base.10M.u8bin"), Some(10_000_000))?),
            Rows::U8(read_bin(&dir.join("query.public.10K.u8bin"), args.queries)?),
            read_bin_ground_truth(&dir.join("gt.10M.ibin"))?.0,
        ),
    };
    ensure!(base.dim() == query.dim(), "base/query dims differ");
    ensure!(
        gt.len() >= query.len(),
        "fewer ground-truth rows than queries"
    );
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
    // The peak so far is the harness's own data; the build adds the index.
    let loaded = storage::sys::peak_memory()?.phys_footprint;

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
        IndexKind::Ivfpq => {
            let cfg = ivfpq::Config {
                nlist: args.nlist,
                m: args.pq_m,
                nbits: args.nbits,
                max_train: args.max_train,
                seed: args.seed,
            };
            let mut ivf = IvfPq::new(base.dim(), Metric::L2, cfg)?;
            // Train on rows spread over the whole base, not the first batch.
            let (n, t) = (base.len(), base.len().min(cfg.max_train));
            let mut sample = Vec::with_capacity(t * base.dim());
            for i in 0..t {
                sample.extend_from_slice(&base.row(i * n / t));
            }
            ivf.train(&sample)?;
            let label = format!("ivfpq nlist={} m={} nbits={}", cfg.nlist, cfg.m, cfg.nbits);
            (Box::new(ivf), label)
        }
    };
    for start in (0..base.len()).step_by(ADD_BATCH) {
        index.add(&base.rows(start..(start + ADD_BATCH).min(base.len())))?;
    }
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
                    "dataset,index,m,ef_construction,seed,ef,nlist,pq_m,nbits,nprobe,rerank,\
                     k,recall,id_recall,p50_ms,p95_ms,p99_ms,qps,build_s,threads"
                )?;
            }
            Some(f)
        }
        None => None,
    };

    // Sweep only the parameters this index reads; a list for one it
    // ignores would just repeat a measurement.
    let (hnsw, ivf) = (
        args.index == IndexKind::Hnsw,
        args.index == IndexKind::Ivfpq,
    );
    let sweep = |on: bool, v: &[usize]| if on { v.to_vec() } else { vec![0] };
    let mut runs = Vec::new();
    for &ef in &sweep(hnsw, &args.ef) {
        for &nprobe in &sweep(ivf, &args.nprobe) {
            for &rerank in &sweep(ivf, &args.rerank) {
                runs.push(SearchParams {
                    ef,
                    nprobe,
                    rerank,
                    filter: None,
                });
            }
        }
    }
    let opt = |on: bool, v: &dyn std::fmt::Display| {
        if on { v.to_string() } else { String::new() }
    };
    for params in runs {
        let r = run_queries(&*index, &base, &query, &gt, args.k, &params)?;
        let what = match args.index {
            IndexKind::Flat => String::new(),
            IndexKind::Hnsw => format!("ef={:<4} ", params.ef),
            IndexKind::Ivfpq => format!("nprobe={:<4} rerank={:<5} ", params.nprobe, params.rerank),
        };
        println!(
            "{what}recall@{k}={:.4} (id match {:.4}) p50={:.3}ms p95={:.3}ms \
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
            writeln!(
                f,
                "{name},{kind},{m},{efc},{seed},{ef},{nlist},{pq_m},{nbits},{nprobe},{rerank},\
                 {k},{:.4},{:.4},{:.4},{:.4},{:.4},{:.1},{:.1},{}",
                r.recall,
                r.id_recall,
                ms(r.p50),
                ms(r.p95),
                ms(r.p99),
                r.qps,
                build.as_secs_f64(),
                rayon::current_num_threads(),
                kind = args
                    .index
                    .to_possible_value()
                    .map_or(String::new(), |v| v.get_name().to_owned()),
                m = opt(hnsw, &args.m),
                efc = opt(hnsw, &args.ef_construction),
                seed = opt(hnsw || ivf, &args.seed),
                ef = opt(hnsw, &params.ef),
                nlist = opt(ivf, &args.nlist),
                pq_m = opt(ivf, &args.pq_m),
                nbits = opt(ivf, &args.nbits),
                nprobe = opt(ivf, &params.nprobe),
                rerank = opt(ivf, &params.rerank),
                k = args.k,
            )?;
        }
    }

    let mem = storage::sys::peak_memory()?;
    let mib = |b: u64| b as f64 / (1 << 20) as f64;
    println!(
        "peak_rss={:.0}MiB peak_phys_footprint={:.0}MiB (harness data {:.0}MiB, \
         so build + index ~{:.0}MiB)",
        mib(mem.rss),
        mib(mem.phys_footprint),
        mib(loaded),
        mib(mem.phys_footprint.saturating_sub(loaded))
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
    base: &Rows,
    query: &Rows,
    gt: &Vectors<u32>,
    k: usize,
    params: &SearchParams,
) -> Result<Run> {
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
        let result = index.search(&q, k, params)?;
        latencies.push(t.elapsed());

        let truth = &gt.row(i)[..k];
        let true_dist = |id: usize| kernels::scalar::l2_sq(&q, &base.row(id));
        let kth = true_dist(truth[k - 1] as usize);
        for h in &result {
            id_hits += usize::from(truth.contains(&h.id));
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
