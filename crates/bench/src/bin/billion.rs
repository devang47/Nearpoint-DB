//! Billion mode (Phase 9, ADR 0014): train on a sample, encode a stream of
//! raw vectors into a build directory, sort it into lists, evaluate.
//!
//! ```text
//! billion train --sample ~/data/Nearpoint/bigann10m/base.10M.u8bin --dir D
//! billion status --dir D                      # rows committed so far
//! curl -r <8 + 128 * from>-... URL | billion build --dir D --from <from> --rows 1000000000
//! billion sort --dir D --budget-mib 3072      # then `stream` can go
//! billion eval --dir D --queries Q.u8bin --gt GT.ibin --nprobe 64,128 --rerank 0,1000
//! ```
//!
//! `build` reads raw u8 rows (no header) from stdin, starting at row
//! `--from`, which must equal the committed row count (`status`): a resumed
//! download starts exactly where the last commit ended. Batches of
//! `--batch` rows are encoded in parallel and committed (fsync) one by one;
//! a short read at EOF commits the whole rows read and exits non-zero
//! unless `--rows` were all read.

use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use bench::dataset::{read_bin, read_bin_ground_truth};
use clap::{Parser, Subcommand};
use index::Hit;
use index::billion::{BillionParams, Lists, Quantizers, TrainConfig};
use index::ivfpq::Shared;
use storage::billion::{Billion, QUANT_MAGIC, QUANT_VERSION, StreamWriter};

#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Train the quantizers on the first `--rows` of a `.u8bin` file.
    Train {
        #[arg(long)]
        sample: PathBuf,
        #[arg(long, default_value_t = 10_000_000)]
        rows: usize,
        #[arg(long)]
        dir: PathBuf,
        #[arg(long, default_value_t = 65_536)]
        nlist: usize,
        #[arg(long, default_value_t = 256)]
        k1: usize,
        #[arg(long, default_value_t = 16)]
        m: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Print the committed row count (where `build --from` must start).
    Status {
        #[arg(long)]
        dir: PathBuf,
    },
    /// Encode raw u8 rows from stdin.
    Build {
        #[arg(long)]
        dir: PathBuf,
        #[arg(long)]
        from: u64,
        /// Total rows of the finished build.
        #[arg(long)]
        rows: u64,
        #[arg(long, default_value_t = 1_000_000)]
        batch: usize,
    },
    /// Sort `stream` into `lists` (passes of at most `--budget-mib`).
    Sort {
        #[arg(long)]
        dir: PathBuf,
        #[arg(long, default_value_t = 3072)]
        budget_mib: usize,
        /// Check every section's CRC afterwards.
        #[arg(long)]
        verify: bool,
    },
    /// Recall@k and latency, one query at a time.
    Eval {
        #[arg(long)]
        dir: PathBuf,
        #[arg(long)]
        queries: PathBuf,
        #[arg(long)]
        gt: PathBuf,
        #[arg(long, default_value_t = 10_000)]
        nq: usize,
        #[arg(long, default_value_t = 10)]
        k: usize,
        #[arg(long, value_delimiter = ',', default_value = "64")]
        nprobe: Vec<usize>,
        #[arg(long, value_delimiter = ',', default_value = "1000")]
        rerank: Vec<usize>,
    },
}

fn quant(dir: &Path) -> Result<Quantizers> {
    let b = storage::framed::read(&dir.join("quant"), QUANT_MAGIC, QUANT_VERSION)
        .context("reading quant")?;
    Ok(Quantizers::from_bytes(&b)?)
}

fn mib(b: u64) -> f64 {
    b as f64 / f64::from(1 << 20)
}

fn train(sample: &Path, rows: usize, dir: &Path, cfg: TrainConfig) -> Result<()> {
    let t = Instant::now();
    let v = read_bin::<u8>(sample, Some(rows))?;
    let dim = v.dim();
    let data: Vec<f32> = v.as_flat().iter().map(|&x| f32::from(x)).collect();
    drop(v);
    println!("loaded {rows} rows in {:.1} s", t.elapsed().as_secs_f64());
    let q = Quantizers::train(&data, dim, cfg)?;
    std::fs::create_dir_all(dir)?;
    storage::framed::write(
        &dir.join("quant"),
        QUANT_MAGIC,
        QUANT_VERSION,
        &q.to_bytes(),
    )?;
    println!(
        "trained nlist {} m {} in {:.1} s, peak phys_footprint {:.0} MiB",
        q.nlist(),
        q.m(),
        t.elapsed().as_secs_f64(),
        mib(storage::sys::peak_memory()?.phys_footprint)
    );
    Ok(())
}

fn build(dir: &Path, from: u64, total: u64, batch: usize) -> Result<()> {
    let q = quant(dir)?;
    let (d, m) = (q.dim(), q.m());
    let (mut w, done) = StreamWriter::open(dir, m)?;
    ensure!(from == done, "--from {from}, but {done} rows are committed");
    let (t, mut rows) = (Instant::now(), done);
    let mut stdin = io::stdin().lock();
    let mut raw = vec![0u8; batch * d];
    let (mut lists, mut codes, mut refine) = (Vec::new(), Vec::new(), Vec::new());
    while rows < total {
        let want = (total - rows).min(batch as u64) as usize * d;
        let got = read_full(&mut stdin, &mut raw[..want])?;
        let n = got / d;
        if n > 0 {
            let x: Vec<f32> = raw[..n * d].iter().map(|&b| f32::from(b)).collect();
            lists.clear();
            codes.clear();
            refine.clear();
            q.encode(&x, &mut lists, &mut codes, &mut refine)?;
            w.append(&lists, &codes, &refine)?;
            rows = w.commit()?;
            let secs = t.elapsed().as_secs_f64();
            println!(
                "{rows} rows ({:.1}%), {:.0} rows/s, {:.0} s",
                100.0 * rows as f64 / total as f64,
                (rows - done) as f64 / secs,
                secs
            );
        }
        if got < want {
            bail!("stdin ended at row {rows} of {total}; resume with --from {rows}");
        }
    }
    w.finish()?;
    println!(
        "stream complete: {rows} rows in {:.0} s",
        t.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Reads until `buf` is full or EOF; returns the bytes read.
fn read_full(r: &mut impl Read, buf: &mut [u8]) -> io::Result<usize> {
    let mut at = 0;
    while at < buf.len() {
        match r.read(&mut buf[at..]) {
            Ok(0) => break,
            Ok(n) => at += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(at)
}

/// The mapped refine codes as the index's prefetching byte source.
struct RefineBytes(Arc<Billion>);

impl Shared for RefineBytes {
    fn bytes(&self) -> &[u8] {
        self.0.refine()
    }

    fn will_need(&self, r: std::ops::Range<usize>) {
        self.0.will_need_refine(r);
    }
}

#[allow(clippy::too_many_arguments)]
fn eval(
    dir: &Path,
    queries: &Path,
    gt: &Path,
    nq: usize,
    k: usize,
    nprobe: &[usize],
    rerank: &[usize],
) -> Result<()> {
    let q = quant(dir)?;
    let b = Arc::new(Billion::open(dir)?);
    ensure!(
        b.m() == q.m() && b.nlist() == q.nlist(),
        "lists and quant differ"
    );
    let qs = read_bin::<u8>(queries, Some(nq))?;
    let (gt_ids, _) = read_bin_ground_truth(gt)?;
    ensure!(
        gt_ids.len() >= nq && gt_ids.dim() >= k,
        "ground truth too small"
    );
    let hint = |c: std::ops::Range<usize>, i: std::ops::Range<usize>| b.will_need_list(c, i);
    let lists = Lists {
        offsets: b.offsets(),
        codes: b.codes(),
        ids: b.ids(),
        will_need: Some(&hint),
    };
    let refine: index::billion::Refine = Arc::new(RefineBytes(Arc::clone(&b)));
    println!(
        "{} rows, nlist {}, m {}; {nq} queries, recall@{k} (id match vs the first {k} ground-truth ids)",
        b.len(),
        b.nlist(),
        b.m()
    );
    for &np in nprobe {
        for &rr in rerank {
            let p = BillionParams {
                nprobe: np,
                rerank: rr,
                skip: None,
            };
            let (mut lat, mut hits) = (Vec::with_capacity(nq), 0usize);
            for i in 0..nq {
                let v: Vec<f32> = qs.row(i).iter().map(|&x| f32::from(x)).collect();
                let t = Instant::now();
                let got: Vec<Hit> = q.search(&lists, &refine, &v, k, &p)?;
                lat.push(t.elapsed().as_secs_f64() * 1e3);
                let truth = &gt_ids.row(i)[..k];
                hits += got.iter().filter(|h| truth.contains(&h.id)).count();
            }
            lat.sort_by(f64::total_cmp);
            let pct = |x: f64| lat[((lat.len() - 1) as f64 * x).round() as usize];
            println!(
                "nprobe {np:>4} rerank {rr:>5}: recall@{k} {:.4}, p50 {:.2} ms, p99 {:.2} ms, mean {:.2} ms, \
                 phys_footprint {:.0} MiB (peak {:.0})",
                hits as f64 / (k * nq) as f64,
                pct(0.5),
                pct(0.99),
                lat.iter().sum::<f64>() / nq as f64,
                mib(storage::sys::phys_footprint()?),
                mib(storage::sys::peak_memory()?.phys_footprint)
            );
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    match Args::parse().cmd {
        Cmd::Train {
            sample,
            rows,
            dir,
            nlist,
            k1,
            m,
            seed,
        } => train(&sample, rows, &dir, TrainConfig { nlist, k1, m, seed }),
        Cmd::Status { dir } => {
            println!("{}", storage::billion::committed_rows(&dir)?);
            Ok(())
        }
        Cmd::Build {
            dir,
            from,
            rows,
            batch,
        } => build(&dir, from, rows, batch),
        Cmd::Sort {
            dir,
            budget_mib,
            verify,
        } => {
            let q = quant(&dir)?;
            let t = Instant::now();
            let passes = storage::billion::sort(&dir, q.nlist(), budget_mib << 20)?;
            println!(
                "sorted in {passes} passes, {:.0} s",
                t.elapsed().as_secs_f64()
            );
            if verify {
                let t = Instant::now();
                Billion::open(&dir)?.verify()?;
                println!("verified in {:.0} s", t.elapsed().as_secs_f64());
            }
            Ok(())
        }
        Cmd::Eval {
            dir,
            queries,
            gt,
            nq,
            k,
            nprobe,
            rerank,
        } => eval(&dir, &queries, &gt, nq, k, &nprobe, &rerank),
    }
}
