//! IVF-PQ (Jégou et al., 2011): a k-means coarse quantizer splits the space
//! into `nlist` cells; each vector goes in its cell's inverted list as the
//! PQ code of its residual (vector minus cell centroid). A query scans the
//! `nprobe` nearest lists with an ADC table built for its own residual in
//! each list. Design in docs/adr/0009-ivf-pq.md.
//!
//! Codes are 8-bit (one table lookup per byte) or 4-bit, stored in the
//! fast-scan block layout (`kernels::pq4_set`) and scanned 32 at a time
//! with u8-quantized tables (`kernels::pq4_scan`).
//!
//! Re-ranking: every vector is also kept as int8 SQ codes, by id. With
//! `params.rerank > 0` the best `max(k, rerank)` PQ candidates are re-scored
//! against them (f32 query vs decoded code) and those distances returned.
//!
//! L2 only: `new` rejects other metrics.
//!
//! Building is `train` (k-means and PQ codebooks on a sample) then `add`;
//! `add` on an untrained index trains on that first batch.

use std::collections::BinaryHeap;
use std::ops::Range;

use quant::{Centroids, KMeans, Pq, PqConfig, Sq};
use rayon::prelude::*;

use crate::{Error, Hit, Index, Metric, Result, SearchParams, batch_ids, check_dim};

#[derive(Debug, Clone, Copy)]
pub struct Config {
    /// Coarse cells (inverted lists).
    pub nlist: usize,
    /// PQ subspaces = bytes per code. Must divide `dim`.
    pub m: usize,
    /// Bits per PQ sub-code: 8, or 4 for fast-scan (then `m <= 256`).
    pub nbits: u32,
    /// Train on a random sample of at most this many vectors.
    pub max_train: usize,
    /// Seeds the training sample, k-means and PQ.
    pub seed: u64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            nlist: 1024,
            m: 16,
            nbits: 8,
            max_train: 262_144,
            seed: 42,
        }
    }
}

pub struct IvfPq {
    dim: usize,
    cfg: Config,
    quant: Option<Quantizers>,
    lists: Vec<List>,
    /// SQ int8 codes by id, `dim` bytes each, for re-ranking.
    sq_codes: Vec<i8>,
    len: usize,
}

struct Quantizers {
    coarse: Centroids,
    pq: Pq,
    sq: Sq,
}

#[derive(Default)]
struct List {
    ids: Vec<u32>,
    /// PQ codes of the residuals in `ids` order: `m` bytes each (8-bit),
    /// or 4-bit fast-scan blocks of 32 vectors, the last one zero-padded.
    codes: Vec<u8>,
}

impl IvfPq {
    pub fn new(dim: usize, metric: Metric, cfg: Config) -> Result<Self> {
        if metric != Metric::L2 {
            return Err(Error::InvalidConfig("IVF-PQ supports only the L2 metric"));
        }
        if dim == 0 || cfg.nlist == 0 || cfg.m == 0 || !dim.is_multiple_of(cfg.m) {
            return Err(Error::InvalidConfig(
                "need dim > 0, nlist > 0, m dividing dim",
            ));
        }
        if !(cfg.nbits == 8 || cfg.nbits == 4 && cfg.m <= 256) {
            return Err(Error::InvalidConfig("nbits must be 8, or 4 with m <= 256"));
        }
        Ok(IvfPq {
            dim,
            cfg,
            quant: None,
            lists: Vec::new(),
            sq_codes: Vec::new(),
            len: 0,
        })
    }

    pub fn config(&self) -> Config {
        self.cfg
    }

    /// Trains the coarse quantizer, PQ codebooks and SQ ranges on (a sample
    /// of at most `max_train` rows of) `data`. Only on an empty index: codes
    /// already stored would no longer match.
    pub fn train(&mut self, data: &[f32]) -> Result<()> {
        if self.len > 0 {
            return Err(Error::InvalidConfig("train needs an empty index"));
        }
        batch_ids(self.dim, 0, data)?;
        let (dim, cfg) = (self.dim, self.cfg);
        let x = quant::sample(data, dim, cfg.max_train, cfg.seed);
        let coarse = KMeans {
            max_train: usize::MAX, // already sampled
            seed: cfg.seed,
            ..KMeans::new(cfg.nlist, dim)
        }
        .train(&x)?;
        let sq = Sq::train(&x, dim, usize::MAX, cfg.seed)?;
        let mut residuals = x.into_owned();
        residuals
            .par_chunks_exact_mut(dim)
            .for_each(|v| subtract(v, coarse.get(coarse.assign(v).0 as usize)));
        let pq = PqConfig {
            nbits: cfg.nbits,
            seed: cfg.seed,
            ..PqConfig::new(dim, cfg.m)
        }
        .train(&residuals)?;
        self.quant = Some(Quantizers { coarse, pq, sq });
        self.lists = (0..cfg.nlist).map(|_| List::default()).collect();
        Ok(())
    }

    /// Encodes `vectors` with trained quantizers and appends them.
    fn append(&mut self, q: &Quantizers, vectors: &[f32]) {
        let (dim, m) = (self.dim, self.cfg.m);
        let n = vectors.len() / dim;
        let mut cells = vec![0u32; n];
        let mut codes = vec![0u8; n * m];
        vectors
            .par_chunks_exact(dim)
            .zip(cells.par_iter_mut())
            .zip(codes.par_chunks_exact_mut(m))
            .for_each_init(
                || vec![0f32; dim],
                |r, ((v, cell), code)| {
                    *cell = q.coarse.assign(v).0;
                    r.copy_from_slice(v);
                    subtract(r, q.coarse.get(*cell as usize));
                    q.pq.encode_into(r, code);
                },
            );

        // Exact reservations: lists only ever grow by `add`, and amortized
        // doubling would leave up to 2x slack on the biggest structure.
        let mut grow = vec![0usize; self.lists.len()];
        for &c in &cells {
            grow[c as usize] += 1;
        }
        let four = self.cfg.nbits == 4;
        for (list, g) in self.lists.iter_mut().zip(grow) {
            list.ids.reserve_exact(g);
            if four {
                let blocks = (list.ids.len() + g).div_ceil(kernels::PQ4_BLOCK);
                list.codes.reserve_exact(blocks * m * 16 - list.codes.len());
            } else {
                list.codes.reserve_exact(g * m);
            }
        }
        for (i, (&c, code)) in cells.iter().zip(codes.chunks_exact(m)).enumerate() {
            let list = &mut self.lists[c as usize];
            let at = list.ids.len();
            list.ids.push((self.len + i) as u32);
            if !four {
                list.codes.extend_from_slice(code);
                continue;
            }
            if at.is_multiple_of(kernels::PQ4_BLOCK) {
                list.codes.resize(list.codes.len() + m * 16, 0);
            }
            kernels::pq4_set(&mut list.codes, at, code);
        }
        let start = self.sq_codes.len();
        self.sq_codes.resize(start + n * dim, 0);
        self.sq_codes[start..]
            .par_chunks_exact_mut(dim)
            .zip(vectors.par_chunks_exact(dim))
            .for_each(|(code, v)| q.sq.encode_into(v, code));
        self.len += n;
    }
}

/// Quantizes a float ADC table (`m` rows of 16) to u8 for `pq4_scan`:
/// each row minus its minimum, all scaled by one factor so the widest row
/// spans 0..=255. Returns `(bias, step)`: a u16 sum `s` of quantized
/// entries approximates the float sum as `bias + s * step`. With `m <= 256`
/// the sum can't overflow u16.
fn quantize_lut(table: &[f32], lut: &mut Vec<u8>) -> (f32, f32) {
    let rows = table.as_chunks::<16>().0;
    let lo = |r: &[f32; 16]| r.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = |r: &[f32; 16]| r.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let bias: f32 = rows.iter().map(lo).sum();
    let span = rows.iter().map(|r| hi(r) - lo(r)).fold(0f32, f32::max);
    let scale = if span > 0.0 { 255.0 / span } else { 1.0 };
    lut.clear();
    for r in rows {
        let min = lo(r);
        lut.extend(r.iter().map(|&x| ((x - min) * scale).round() as u8));
    }
    (bias, 1.0 / scale)
}

/// `v -= c`, component-wise.
fn subtract(v: &mut [f32], c: &[f32]) {
    for (a, b) in v.iter_mut().zip(c) {
        *a -= b;
    }
}

impl Index for IvfPq {
    fn dim(&self) -> usize {
        self.dim
    }

    fn metric(&self) -> Metric {
        Metric::L2
    }

    fn len(&self) -> usize {
        self.len
    }

    fn add(&mut self, vectors: &[f32]) -> Result<Range<u32>> {
        let ids = batch_ids(self.dim, self.len, vectors)?;
        if ids.is_empty() {
            return Ok(ids);
        }
        if self.quant.is_none() {
            self.train(vectors)?;
        }
        // Taken out so `append` can borrow the lists mutably; it can't fail.
        if let Some(q) = self.quant.take() {
            self.append(&q, vectors);
            self.quant = Some(q);
        }
        Ok(ids)
    }

    /// Scans the `params.nprobe` nearest lists for the `max(k,
    /// params.rerank)` nearest by ADC distance (the squared L2 to the PQ
    /// reconstruction). Ids not in `params.filter` are skipped while
    /// scanning. With `params.rerank > 0` those candidates are re-scored
    /// with their SQ codes (squared L2 to the int8 reconstruction); the `k`
    /// nearest are returned either way.
    fn search(&self, query: &[f32], k: usize, params: &SearchParams) -> Result<Vec<Hit>> {
        check_dim(self.dim, query.len())?;
        let Some(q) = &self.quant else {
            return Ok(Vec::new());
        };
        if k == 0 {
            return Ok(Vec::new());
        }

        let mut cells: Vec<Hit> = q
            .coarse
            .as_slice()
            .chunks_exact(self.dim)
            .enumerate()
            .map(|(i, c)| Hit {
                id: i as u32,
                dist: kernels::l2_sq(query, c),
            })
            .collect();
        let nprobe = params.nprobe.clamp(1, cells.len());
        if nprobe < cells.len() {
            cells.select_nth_unstable(nprobe - 1);
            cells.truncate(nprobe);
        }

        let short = k.max(params.rerank);
        let mut best = BinaryHeap::with_capacity(short + 1);
        let mut offer = |id: u32, dist: f32| {
            if best.len() == short && best.peek().is_some_and(|w: &Hit| dist >= w.dist) {
                return;
            }
            // After the distance test: most ids never get this far, and a
            // bitmap lookup costs more than a few table lookups.
            if params.filter.is_some_and(|f| !f.contains(id)) {
                return;
            }
            best.push(Hit { id, dist });
            if best.len() > short {
                best.pop();
            }
        };
        let mut residual = vec![0f32; self.dim];
        let (mut lut, mut sums) = (Vec::new(), Vec::new());
        for cell in &cells {
            let list = &self.lists[cell.id as usize];
            if list.ids.is_empty() {
                continue;
            }
            residual.copy_from_slice(query);
            subtract(&mut residual, q.coarse.get(cell.id as usize));
            let table = q.pq.distance_table(&residual);
            if self.cfg.nbits == 8 {
                for (&id, code) in list.ids.iter().zip(list.codes.chunks_exact(self.cfg.m)) {
                    offer(id, quant::adc_distance(&table, code));
                }
                continue;
            }
            let (bias, step) = quantize_lut(&table, &mut lut);
            sums.resize(list.ids.len().next_multiple_of(kernels::PQ4_BLOCK), 0);
            kernels::pq4_scan(&list.codes, &lut, &mut sums);
            for (&id, &s) in list.ids.iter().zip(&sums) {
                offer(id, bias + f32::from(s) * step);
            }
        }

        let mut hits = best.into_vec();
        if params.rerank > 0 {
            // |q - decode(c)|² = scale² · |(q - min) / scale - 128 - c|²: the
            // query goes into code units once, then each candidate is one
            // i8 -> f32 widening and an f32 L2 kernel call.
            let scale = q.sq.scale();
            let qs: Vec<f32> = query
                .iter()
                .zip(q.sq.min())
                .map(|(&x, &lo)| (x - lo) / scale - 128.0)
                .collect();
            let mut code = vec![0f32; self.dim];
            for h in &mut hits {
                let c = &self.sq_codes[h.id as usize * self.dim..][..self.dim];
                for (f, &b) in code.iter_mut().zip(c) {
                    *f = f32::from(b);
                }
                h.dist = kernels::l2_sq(&qs, &code) * scale * scale;
            }
        }
        hits.sort_unstable();
        hits.truncate(k);
        Ok(hits)
    }
}

#[cfg(test)]
mod tests {
    use roaring::RoaringBitmap;

    use super::*;
    use crate::{Flat, test_data};

    const DIM: usize = 16;

    fn cfg() -> Config {
        Config {
            nlist: 16,
            m: 8,
            ..Config::default()
        }
    }

    /// Mean fraction of the exact top-k that `index` returns.
    fn recall(index: &dyn Index, data: &[f32], queries: &[f32], k: usize, p: &SearchParams) -> f64 {
        let mut exact = Flat::new(DIM, Metric::L2);
        exact.add(data).unwrap();
        let mut hits = 0;
        for q in queries.as_chunks::<DIM>().0 {
            let truth = exact.search(q, k, &SearchParams::default()).unwrap();
            let got = index.search(q, k, p).unwrap();
            assert!(got.len() == k && got.is_sorted());
            hits += got
                .iter()
                .filter(|h| truth.iter().any(|t| t.id == h.id))
                .count();
        }
        hits as f64 / (k * queries.len() / DIM) as f64
    }

    #[test]
    fn probing_more_lists_finds_more() {
        let data = test_data::vectors(3000, DIM, 1);
        let queries = test_data::vectors(50, DIM, 2);
        // 4-bit with m = 16 is the same 8 bytes per vector as 8-bit m = 8.
        for cfg in [
            cfg(),
            Config {
                nbits: 4,
                m: 16,
                ..cfg()
            },
        ] {
            let mut ivf = IvfPq::new(DIM, Metric::L2, cfg).unwrap();
            // Batches that end mid-block, so 4-bit appends into a
            // partly filled block.
            assert_eq!(ivf.add(&data[..2001 * DIM]).unwrap(), 0..2001);
            assert_eq!(ivf.add(&data[2001 * DIM..]).unwrap(), 2001..3000);
            assert_eq!(ivf.lists.iter().map(|l| l.ids.len()).sum::<usize>(), 3000);

            let at = |nprobe| {
                let p = SearchParams {
                    nprobe,
                    ..SearchParams::default()
                };
                recall(&ivf, &data, &queries, 10, &p)
            };
            let (one, all) = (at(1), at(16));
            assert!(one < all, "nbits {}: nprobe 1: {one}, 16: {all}", cfg.nbits);
            assert!(all > 0.6, "nbits {}: {all}", cfg.nbits);
        }
    }

    #[test]
    fn four_bit_distances_track_the_float_tables() {
        let data = test_data::vectors(2000, DIM, 7);
        let mut ivf = IvfPq::new(DIM, Metric::L2, Config { nbits: 4, ..cfg() }).unwrap();
        ivf.add(&data).unwrap();
        let q = &data[5 * DIM..][..DIM];
        let p = SearchParams {
            nprobe: 16,
            ..SearchParams::default()
        };
        let quant = ivf.quant.as_ref().unwrap();
        for h in ivf.search(q, 20, &p).unwrap() {
            let v = &data[h.id as usize * DIM..][..DIM];
            let c = quant.coarse.get(quant.coarse.assign(v).0 as usize);
            let (mut r, mut rq) = (v.to_vec(), q.to_vec());
            subtract(&mut r, c);
            subtract(&mut rq, c);
            let table = quant.pq.distance_table(&rq);
            let want = quant::adc_distance(&table, &quant.pq.encode(&r));
            let (_, step) = quantize_lut(&table, &mut Vec::new());
            // Each of the m entries is off by at most half a step.
            let bound = ivf.cfg.m as f32 * step / 2.0 + 1e-4 * want;
            assert!((h.dist - want).abs() <= bound, "{h:?} vs {want} ± {bound}");
        }
    }

    #[test]
    fn adc_distance_is_to_the_reconstruction() {
        let data = test_data::vectors(1000, DIM, 3);
        let mut ivf = IvfPq::new(DIM, Metric::L2, cfg()).unwrap();
        ivf.add(&data).unwrap();
        let q = &data[..DIM];
        let p = SearchParams {
            nprobe: 16,
            ..SearchParams::default()
        };
        let quant = ivf.quant.as_ref().unwrap();
        for h in ivf.search(q, 5, &p).unwrap() {
            let v = &data[h.id as usize * DIM..][..DIM];
            let (cell, _) = quant.coarse.assign(v);
            let mut r = v.to_vec();
            subtract(&mut r, quant.coarse.get(cell as usize));
            let mut recon = quant.pq.decode(&quant.pq.encode(&r));
            for (a, c) in recon.iter_mut().zip(quant.coarse.get(cell as usize)) {
                *a += c;
            }
            let want = kernels::l2_sq(q, &recon);
            assert!(
                (h.dist - want).abs() <= 1e-4 * want.max(1.0),
                "{h:?} vs {want}"
            );
        }
    }

    #[test]
    fn rerank_returns_sq_distances_and_lifts_recall() {
        let data = test_data::vectors(3000, DIM, 8);
        let queries = test_data::vectors(50, DIM, 9);
        for nbits in [8, 4] {
            let mut ivf = IvfPq::new(DIM, Metric::L2, Config { nbits, ..cfg() }).unwrap();
            ivf.add(&data).unwrap();
            let at = |rerank| {
                let p = SearchParams {
                    nprobe: 16,
                    rerank,
                    ..SearchParams::default()
                };
                recall(&ivf, &data, &queries, 10, &p)
            };
            let (plain, reranked) = (at(0), at(200));
            assert!(
                plain < reranked && reranked > 0.9,
                "nbits {nbits}: {plain} -> {reranked}"
            );

            let p = SearchParams {
                nprobe: 16,
                rerank: 50,
                ..SearchParams::default()
            };
            let q = &queries[..DIM];
            let sq = &ivf.quant.as_ref().unwrap().sq;
            for h in ivf.search(q, 10, &p).unwrap() {
                let code = &ivf.sq_codes[h.id as usize * DIM..][..DIM];
                let want = kernels::l2_sq(q, &sq.decode(code));
                assert!(
                    (h.dist - want).abs() <= 1e-3 * want.max(1.0),
                    "{h:?} vs {want}"
                );
            }
        }
    }

    #[test]
    fn filter_skips_ids_while_scanning() {
        let data = test_data::vectors(2000, DIM, 4);
        let mut ivf = IvfPq::new(DIM, Metric::L2, cfg()).unwrap();
        ivf.add(&data).unwrap();
        let odd: RoaringBitmap = (1..2000).step_by(2).collect();
        let p = SearchParams {
            nprobe: 16,
            filter: Some(&odd),
            ..SearchParams::default()
        };
        let hits = ivf.search(&data[..DIM], 10, &p).unwrap();
        assert_eq!(hits.len(), 10);
        assert!(hits.iter().all(|h| h.id % 2 == 1));
        // A filter matching nothing returns nothing.
        let none = RoaringBitmap::new();
        let p = SearchParams {
            filter: Some(&none),
            ..p
        };
        assert!(ivf.search(&data[..DIM], 10, &p).unwrap().is_empty());
    }

    #[test]
    fn config_and_input_errors() {
        assert!(IvfPq::new(DIM, Metric::Cosine, cfg()).is_err());
        assert!(IvfPq::new(DIM, Metric::L2, Config { m: 5, ..cfg() }).is_err());
        assert!(IvfPq::new(DIM, Metric::L2, Config { nbits: 3, ..cfg() }).is_err());
        let wide = Config {
            nbits: 4,
            m: 512,
            ..cfg()
        };
        assert!(IvfPq::new(512, Metric::L2, wide).is_err()); // u16 sums could wrap

        let mut ivf = IvfPq::new(DIM, Metric::L2, cfg()).unwrap();
        let p = SearchParams::default();
        assert!(ivf.search(&[0.0; DIM], 10, &p).unwrap().is_empty()); // untrained
        // Fewer vectors than lists: k-means can't train.
        assert!(matches!(
            ivf.add(&test_data::vectors(10, DIM, 5)),
            Err(Error::Quant(_))
        ));
        ivf.add(&test_data::vectors(500, DIM, 5)).unwrap();
        assert!(ivf.search(&[0.0; 3], 10, &p).is_err());
        assert!(ivf.add(&[0.0; DIM + 1]).is_err());
        assert!(ivf.train(&test_data::vectors(500, DIM, 6)).is_err()); // not empty
        assert!(ivf.search(&[0.0; DIM], 0, &p).unwrap().is_empty());
    }
}
