//! Loads a prefix of a big-ann-benchmarks `.u8bin` base through the HTTP
//! API and measures the result (Phase 8.3).
//!
//! 1. Load: `--writers` connections stream `--batch`-row bulk upserts of
//!    ids `0..n` with payload `{"tag": "t<id % 10>"}`. A 503 (the server's
//!    `--memory-limit-mib`) is retried after a pause. Progress, with the
//!    server's phys_footprint from `/metrics`, every `--progress` rows.
//! 2. Settle: wait until the segment layout hasn't changed for
//!    `--settle-secs` (background merges done).
//! 3. Measure, one query at a time: `--queries` unfiltered k = 10 searches
//!    scored against the official ground truth (top 100, so deleted ids can
//!    be skipped), and `--filtered` searches with `{"eq": {"tag": "t3"}}`
//!    (10%) scored against an exact top 10 computed here by one streaming
//!    pass over the base file.
//! 4. Delete every id `≡ 0 (mod --delete-every)` and measure again: no
//!    deleted id may come back (counted as violations).
//!
//! ```text
//! cargo run --release -p server --bin bigload -- --n 10000000 \
//!     --base ~/data/Nearpoint/bigann10m/base.10M.u8bin \
//!     --query-file ~/data/Nearpoint/bigann10m/query.public.10K.u8bin \
//!     --gt ~/data/Nearpoint/bigann10m/gt.10M.ibin \
//!     --collection-config '{"exact_vectors": false}'
//! ```

use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, bail, ensure};
use clap::Parser;
use serde_json::{Value, json};

#[path = "common/http.rs"]
mod http;
use http::{Http, RECONNECTS};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "127.0.0.1:7700")]
    addr: String,
    #[arg(long, default_value = "bigann")]
    collection: String,
    /// Base vectors (`.u8bin`; a prefix of a bigger file is fine).
    #[arg(long)]
    base: PathBuf,
    /// Rows to load (and to compute filtered ground truth over).
    #[arg(long)]
    n: u64,
    /// Query vectors (`.u8bin`).
    #[arg(long)]
    query_file: PathBuf,
    /// Ground truth for the first `n` base rows (`.ibin`: ids then dists).
    #[arg(long)]
    gt: PathBuf,
    /// Merged into `{"dim", "metric": "l2", "schema": {"tag": "keyword"}}`.
    #[arg(long, default_value = "{}")]
    collection_config: String,
    /// Measure an existing collection: no create, no load.
    #[arg(long)]
    skip_load: bool,
    #[arg(long, default_value_t = 10_000)]
    batch: usize,
    #[arg(long, default_value_t = 4)]
    writers: usize,
    #[arg(long, default_value_t = 5_000_000)]
    progress: u64,
    #[arg(long, default_value_t = 60)]
    settle_secs: u64,
    /// Unfiltered queries per measurement.
    #[arg(long, default_value_t = 10_000)]
    queries: usize,
    /// Filtered (10%) queries per measurement.
    #[arg(long, default_value_t = 100)]
    filtered: usize,
    /// Search knobs (server defaults if unset).
    #[arg(long)]
    nprobe: Option<usize>,
    #[arg(long)]
    rerank: Option<usize>,
    /// Delete ids `≡ 0 (mod this)` before the second measurement; 0 = skip.
    #[arg(long, default_value_t = 100)]
    delete_every: u64,
}

/// A `.u8bin`/`.ibin` header: count and dim.
fn header(r: &mut impl Read, path: &Path) -> anyhow::Result<(u64, usize)> {
    let mut h = [0u8; 8];
    r.read_exact(&mut h)
        .with_context(|| format!("{}: header", path.display()))?;
    let [a, b] = h.as_chunks::<4>().0 else {
        bail!("short header")
    };
    Ok((
        u64::from(u32::from_le_bytes(*a)),
        u32::from_le_bytes(*b) as usize,
    ))
}

fn open(path: &Path) -> anyhow::Result<BufReader<File>> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    Ok(BufReader::with_capacity(1 << 22, f))
}

/// The first `n` rows of a `.u8bin`.
fn read_u8bin(path: &Path, n: usize) -> anyhow::Result<(Vec<u8>, usize)> {
    let mut r = open(path)?;
    let (count, dim) = header(&mut r, path)?;
    let n = n.min(count as usize);
    let mut v = vec![0u8; n * dim];
    r.read_exact(&mut v)
        .with_context(|| format!("{}: {n} rows", path.display()))?;
    Ok((v, dim))
}

/// Ground-truth ids: `nq` rows of `k`.
fn read_gt(path: &Path) -> anyhow::Result<(Vec<u32>, usize)> {
    let mut r = open(path)?;
    let (nq, k) = header(&mut r, path)?;
    let mut b = vec![0u8; nq as usize * k * 4];
    r.read_exact(&mut b)
        .with_context(|| format!("{}: ids", path.display()))?;
    let ids = b
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    Ok((ids, k))
}

fn tag(id: u64) -> String {
    format!("t{}", id % 10)
}

const FILTER_TAG: u64 = 3;

fn bulk_body(first: u64, rows: &[u8], dim: usize) -> Vec<u8> {
    let n = rows.len() / dim;
    let lines: String = (first..first + n as u64)
        .map(|id| format!("{{\"tag\":\"{}\"}}\n", tag(id)))
        .collect();
    let mut b = Vec::with_capacity(12 + n * (8 + 4 * dim) + lines.len());
    b.extend_from_slice(&(n as u32).to_le_bytes());
    b.extend_from_slice(&(dim as u32).to_le_bytes());
    b.extend_from_slice(&(lines.len() as u32).to_le_bytes());
    for id in first..first + n as u64 {
        b.extend_from_slice(&id.to_le_bytes());
    }
    for &x in rows {
        b.extend_from_slice(&f32::from(x).to_le_bytes());
    }
    b.extend_from_slice(lines.as_bytes());
    b
}

/// `name value` lines of `/metrics` (labels kept in `name`).
fn metric(text: &str, name: &str) -> Option<f64> {
    text.lines()
        .find_map(|l| l.strip_prefix(name)?.strip_prefix(' ')?.parse().ok())
}

fn metrics(http: &mut Http) -> anyhow::Result<String> {
    let (status, body) = http.call("GET", "/metrics", "text/plain", &[])?;
    ensure!(status == 200, "/metrics: {status}");
    Ok(String::from_utf8_lossy(&body).into_owned())
}

fn mib(text: &str, name: &str) -> f64 {
    metric(text, name).unwrap_or(f64::NAN) / f64::from(1 << 20)
}

fn segments(http: &mut Http, coll: &str) -> anyhow::Result<Value> {
    Ok(http.json("GET", &format!("/collections/{coll}"), &Value::Null)?["segments"].clone())
}

fn load(a: &Args, dim_in: usize) -> anyhow::Result<()> {
    let mut r = open(&a.base)?;
    let (count, dim) = header(&mut r, &a.base)?;
    ensure!(dim == dim_in, "base dim {dim}, queries {dim_in}");
    ensure!(a.n <= count, "base header says {count} rows");
    let next = Mutex::new((r, 0u64));
    let start = Instant::now();
    let done = AtomicU64::new(0);
    let report = Mutex::new(a.progress);
    std::thread::scope(|s| {
        let workers: Vec<_> = (0..a.writers)
            .map(|_| {
                s.spawn(|| -> anyhow::Result<()> {
                    let mut http = Http::new(&a.addr);
                    let mut rows = vec![0u8; a.batch * dim];
                    let path = format!("/collections/{}/points/bulk", a.collection);
                    loop {
                        let first = {
                            let mut g = next.lock().map_err(|_| anyhow::anyhow!("poisoned"))?;
                            let (r, at) = &mut *g;
                            let n = (a.n - *at).min(a.batch as u64) as usize;
                            if n == 0 {
                                return Ok(());
                            }
                            rows.truncate(n * dim);
                            rows.resize(n * dim, 0);
                            r.read_exact(&mut rows)
                                .with_context(|| format!("base rows from {at}"))?;
                            *at += n as u64;
                            *at - n as u64
                        };
                        let body = bulk_body(first, &rows, dim);
                        loop {
                            let (status, reply) =
                                http.call("POST", &path, "application/octet-stream", &body)?;
                            match status {
                                200 => break,
                                503 => std::thread::sleep(Duration::from_secs(2)),
                                _ => bail!("bulk {first}: {status} {}", String::from_utf8_lossy(&reply)),
                            }
                        }
                        let n = (rows.len() / dim) as u64;
                        let total = done.fetch_add(n, Ordering::Relaxed) + n;
                        let mut due = report.lock().map_err(|_| anyhow::anyhow!("poisoned"))?;
                        if total >= *due {
                            *due += a.progress;
                            let m = metrics(&mut http)?;
                            let secs = start.elapsed().as_secs_f64();
                            println!(
                                "{:>8.0} s: {total} rows, {:.0} rows/s, phys_footprint {:.0} MiB (peak {:.0}), segments {}",
                                secs,
                                total as f64 / secs,
                                mib(&m, "nearpoint_phys_footprint_bytes"),
                                mib(&m, "nearpoint_phys_footprint_peak_bytes"),
                                segments(&mut http, &a.collection)?
                            );
                        }
                    }
                })
            })
            .collect();
        workers
            .into_iter()
            .try_for_each(|w| w.join().map_err(|_| anyhow::anyhow!("writer panicked"))?)
    })?;
    println!(
        "loaded {} rows in {:.0} s",
        a.n,
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Waits until the segment layout is unchanged for `secs`.
fn settle(http: &mut Http, coll: &str, secs: u64) -> anyhow::Result<()> {
    let start = Instant::now();
    let (mut last, mut since) = (segments(http, coll)?, Instant::now());
    while since.elapsed() < Duration::from_secs(secs) {
        std::thread::sleep(Duration::from_secs(5));
        let now = segments(http, coll)?;
        if now != last {
            (last, since) = (now, Instant::now());
        }
    }
    println!(
        "settled after {:.0} s: {last}",
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Exact filtered top 10 for each query: one pass over the first `n` base
/// rows, queries split across threads. u8 squared L2 in u32: exact.
fn filtered_truth(a: &Args, queries: &[u8], dim: usize) -> anyhow::Result<Vec<Vec<u64>>> {
    let t = Instant::now();
    let nq = queries.len() / dim;
    let mut r = open(&a.base)?;
    header(&mut r, &a.base)?;
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let mut heaps: Vec<BinaryHeap<(u32, u64)>> = vec![BinaryHeap::new(); nq];
    const CHUNK: usize = 1 << 20;
    let mut buf = vec![0u8; CHUNK * dim];
    let mut at = 0u64;
    while at < a.n {
        let rows = (a.n - at).min(CHUNK as u64) as usize;
        buf.truncate(rows * dim);
        r.read_exact(&mut buf)?;
        let per = nq.div_ceil(threads);
        std::thread::scope(|s| {
            for (qs, hs) in queries.chunks(per * dim).zip(heaps.chunks_mut(per)) {
                let buf = &buf;
                s.spawn(move || {
                    for (q, h) in qs.chunks_exact(dim).zip(hs) {
                        // Rows with id % 10 == FILTER_TAG.
                        let first = (FILTER_TAG + 10 - at % 10) % 10;
                        for row in (first as usize..rows).step_by(10) {
                            let v = &buf[row * dim..][..dim];
                            let d: u32 = q
                                .iter()
                                .zip(v)
                                .map(|(&x, &y)| u32::from(x.abs_diff(y)).pow(2))
                                .sum();
                            if h.len() < 10 {
                                h.push((d, at + row as u64));
                            } else if h.peek().is_some_and(|&top| (d, at + row as u64) < top) {
                                h.pop();
                                h.push((d, at + row as u64));
                            }
                        }
                    }
                });
            }
        });
        at += rows as u64;
    }
    println!(
        "filtered ground truth: {nq} queries in {:.0} s",
        t.elapsed().as_secs_f64()
    );
    Ok(heaps
        .into_iter()
        .map(|h| h.into_sorted_vec().into_iter().map(|(_, id)| id).collect())
        .collect())
}

struct Run {
    p50: f64,
    p99: f64,
    mean: f64,
    recall: f64,
    violations: usize,
}

/// Sequential k = 10 searches; recall against `truth[q]` (first 10 ids
/// not deleted), violations = deleted ids returned.
fn run(
    a: &Args,
    http: &mut Http,
    queries: &[u8],
    dim: usize,
    filter: Option<&Value>,
    truth: &dyn Fn(usize) -> Vec<u64>,
    deleted: &dyn Fn(u64) -> bool,
) -> anyhow::Result<Run> {
    let path = format!("/collections/{}/search", a.collection);
    let (mut lat, mut hits, mut violations) = (Vec::new(), 0usize, 0usize);
    let nq = queries.len() / dim;
    for (q, v) in queries.chunks_exact(dim).enumerate() {
        let mut req =
            json!({"vector": v.iter().map(|&x| f32::from(x)).collect::<Vec<_>>(), "k": 10});
        if let Some(f) = filter {
            req["filter"] = f.clone();
        }
        if let Some(n) = a.nprobe {
            req["nprobe"] = n.into();
        }
        if let Some(n) = a.rerank {
            req["rerank"] = n.into();
        }
        let t = Instant::now();
        let reply = http.json("POST", &path, &req)?;
        lat.push(t.elapsed().as_secs_f64() * 1e3);
        let want = truth(q);
        for h in reply["hits"].as_array().context("no hits array")? {
            let id = h["id"].as_u64().context("hit without id")?;
            violations += usize::from(deleted(id));
            hits += usize::from(want.contains(&id));
        }
    }
    lat.sort_by(f64::total_cmp);
    let pct = |p: f64| lat[((lat.len() - 1) as f64 * p).round() as usize];
    Ok(Run {
        p50: pct(0.5),
        p99: pct(0.99),
        mean: lat.iter().sum::<f64>() / lat.len() as f64,
        recall: hits as f64 / (10 * nq) as f64,
        violations,
    })
}

fn main() -> anyhow::Result<()> {
    let a = Args::parse();
    let (queries, dim) = read_u8bin(&a.query_file, a.queries.max(a.filtered))?;
    let (gt, gk) = read_gt(&a.gt)?;
    ensure!(gt.len() / gk >= a.queries, "ground truth has fewer queries");
    let mut http = Http::new(&a.addr);
    if !a.skip_load {
        let mut cfg = json!({"dim": dim, "metric": "l2", "schema": {"tag": "keyword"}});
        let extra: Value = serde_json::from_str(&a.collection_config)?;
        for (k, v) in extra
            .as_object()
            .context("--collection-config: an object")?
        {
            cfg[k] = v.clone();
        }
        http.json("PUT", &format!("/collections/{}", a.collection), &cfg)?;
        println!("created {}: {cfg}", a.collection);
        load(&a, dim)?;
    }
    settle(&mut http, &a.collection, a.settle_secs)?;
    let ftruth = filtered_truth(&a, &queries[..a.filtered * dim], dim)?;
    let filter = json!({"eq": {"tag": tag(FILTER_TAG)}});

    let report = |http: &mut Http,
                  label: &str,
                  deleted: &dyn Fn(u64) -> bool|
     -> anyhow::Result<()> {
        let un = run(
            &a,
            http,
            &queries[..a.queries * dim],
            dim,
            None,
            &|q| {
                gt[q * gk..(q + 1) * gk]
                    .iter()
                    .map(|&i| u64::from(i))
                    .filter(|&i| !deleted(i))
                    .take(10)
                    .collect()
            },
            deleted,
        )?;
        let fi = run(
            &a,
            http,
            &queries[..a.filtered * dim],
            dim,
            Some(&filter),
            &|q| ftruth[q].clone(),
            deleted,
        )?;
        let m = metrics(http)?;
        let c = &a.collection;
        println!(
            "{label}: unfiltered recall@10 {:.4} p50 {:.2} p99 {:.2} mean {:.2} ms ({} queries); \
             filtered 10% recall@10 {:.4} p50 {:.2} p99 {:.2} ms ({} queries); violations {}; \
             phys_footprint {:.0} MiB (peak {:.0}); memory MiB: id_map {:.0}, ids {:.0}, index {:.0}, \
             payloads {:.0}, mutable {:.0}; mapped {:.0} MiB",
            un.recall,
            un.p50,
            un.p99,
            un.mean,
            a.queries,
            fi.recall,
            fi.p50,
            fi.p99,
            a.filtered,
            un.violations + fi.violations,
            mib(&m, "nearpoint_phys_footprint_bytes"),
            mib(&m, "nearpoint_phys_footprint_peak_bytes"),
            mib(
                &m,
                &format!("nearpoint_memory_bytes{{collection=\"{c}\",component=\"id_map\"}}")
            ),
            mib(
                &m,
                &format!("nearpoint_memory_bytes{{collection=\"{c}\",component=\"ids\"}}")
            ),
            mib(
                &m,
                &format!("nearpoint_memory_bytes{{collection=\"{c}\",component=\"index\"}}")
            ),
            mib(
                &m,
                &format!("nearpoint_memory_bytes{{collection=\"{c}\",component=\"payloads\"}}")
            ),
            mib(
                &m,
                &format!("nearpoint_memory_bytes{{collection=\"{c}\",component=\"mutable\"}}")
            ),
            mib(&m, &format!("nearpoint_mapped_bytes{{collection=\"{c}\"}}")),
        );
        ensure!(un.violations + fi.violations == 0, "deleted ids returned");
        Ok(())
    };
    report(&mut http, "before deletes", &|_| false)?;
    if a.delete_every > 0 {
        let k = a.delete_every;
        let t = Instant::now();
        let path = format!("/collections/{}/points/delete", a.collection);
        let ids: Vec<u64> = (0..a.n).step_by(k as usize).collect();
        for chunk in ids.chunks(10_000) {
            http.json("POST", &path, &json!({"ids": chunk}))?;
        }
        println!(
            "deleted {} ids in {:.1} s",
            ids.len(),
            t.elapsed().as_secs_f64()
        );
        report(&mut http, "after deletes", &|id| id % k == 0)?;
    }
    println!("reconnects {}", RECONNECTS.load(Ordering::Relaxed));
    Ok(())
}
