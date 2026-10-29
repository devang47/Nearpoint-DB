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
//! Metrics. Cells, residuals and codes are always L2 (k-means, PQ).
//! Cosine normalizes vectors on insert and queries on search, then runs as
//! L2: on unit vectors `|q - x|² = 2 (1 - cos)`, so distances are halved on
//! the way out. Inner product probes the lists with the largest `<q, c>`
//! and scores `<q, c + r> = <q, c> + <q, r>`: one table of sub-vector dot
//! products per query, plus the list's `<q, c>` (negated: nearer = smaller).
//!
//! Building is `train` (k-means and PQ codebooks on a sample) then `add`;
//! `add` on an untrained index trains on that first batch.

use std::borrow::Cow;
use std::collections::BinaryHeap;
use std::ops::Range;

use quant::{Centroids, KMeans, Pq, PqConfig, Sq};
use rayon::prelude::*;

use crate::{Error, Hit, Index, Metric, Result, SearchParams, batch_ids, blob, check_dim};

const MAGIC: &[u8; 4] = b"NPIV";

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
    metric: Metric,
    cfg: Config,
    quant: Option<Quantizers>,
    lists: Vec<List>,
    /// SQ int8 codes by id, `dim` bytes each, for re-ranking.
    sq_codes: Vec<i8>,
    len: usize,
}

#[derive(Clone)]
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
            metric,
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

    /// An empty index with this one's config and trained quantizers:
    /// `add` then only encodes. For rebuilding a segment without
    /// retraining (compaction).
    pub fn empty_clone(&self) -> IvfPq {
        IvfPq {
            lists: if self.quant.is_some() {
                (0..self.cfg.nlist).map(|_| List::default()).collect()
            } else {
                Vec::new()
            },
            quant: self.quant.clone(),
            sq_codes: Vec::new(),
            len: 0,
            ..*self
        }
    }

    /// The index blob: config, quantizers, every list and the SQ codes. It
    /// is the whole index: no f32 vectors are needed to load or search it,
    /// so its segment can be written without them. Layout in
    /// docs/formats/segment.md.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        blob::put_header(&mut out, MAGIC, self.dim, self.metric);
        let c = &self.cfg;
        for x in [
            c.nlist,
            c.m,
            c.nbits as usize,
            usize::from(self.quant.is_some()),
        ] {
            blob::put_u32(&mut out, x as u32);
        }
        blob::put_u64(&mut out, c.max_train as u64);
        blob::put_u64(&mut out, c.seed);
        blob::put_u64(&mut out, self.len as u64);
        let Some(q) = &self.quant else {
            return out;
        };
        let floats = q.coarse.as_slice().len()
            + q.pq
                .codebooks()
                .iter()
                .map(|b| b.as_slice().len())
                .sum::<usize>()
            + self.dim
            + 1;
        let lists: usize = self
            .lists
            .iter()
            .map(|l| 4 + 4 * l.ids.len() + l.codes.len())
            .sum();
        out.reserve_exact(4 * floats + lists + self.sq_codes.len());
        blob::put_f32s(&mut out, q.coarse.as_slice());
        for b in q.pq.codebooks() {
            blob::put_f32s(&mut out, b.as_slice());
        }
        blob::put_f32s(&mut out, q.sq.min());
        blob::put_f32s(&mut out, &[q.sq.scale()]);
        for l in &self.lists {
            blob::put_u32(&mut out, l.ids.len() as u32);
        }
        for l in &self.lists {
            for &id in &l.ids {
                blob::put_u32(&mut out, id);
            }
            out.extend_from_slice(&l.codes);
        }
        out.extend_from_slice(bytemuck::cast_slice(&self.sq_codes));
        out
    }

    /// Rebuilds an index from `to_bytes` output. Everything is validated
    /// (config, quantizer shapes, list sizes, every id in `0..len` exactly
    /// once), so a corrupt blob is an error here, never a panic or an
    /// out-of-bounds read in a later search. Allocations are bounded by the
    /// bytes actually present.
    pub fn from_bytes(bytes: &[u8]) -> Result<IvfPq> {
        let mut r = blob::Reader::new(bytes);
        let (dim, metric) = r.header(MAGIC)?;
        let (nlist, m, nbits, trained) = (r.u32()?, r.u32()?, r.u32()?, r.u32()?);
        let (max_train, seed, n) = (r.u64()?, r.u64()?, r.u64()?);
        let cfg = Config {
            nlist: nlist as usize,
            m: m as usize,
            nbits,
            max_train: usize::try_from(max_train).map_err(|_| Error::Corrupt("bad config"))?,
            seed,
        };
        let mut ivf = IvfPq::new(dim, metric, cfg).map_err(|_| Error::Corrupt("bad config"))?;
        if n > u64::from(u32::MAX) {
            return Err(Error::Corrupt("more vectors than u32 ids"));
        }
        let n = n as usize;
        match (trained, n) {
            (0, 0) => {
                r.finish()?;
                return Ok(ivf);
            }
            (1, _) => {}
            _ => return Err(Error::Corrupt("bad trained flag")),
        }

        let (nlist, m) = (cfg.nlist, cfg.m);
        let (ksub, dsub) = (1usize << nbits, dim / m);
        let coarse = Centroids::from_vec(dim, r.f32s(nlist * dim)?)?;
        let books = (0..m)
            .map(|_| Ok(Centroids::from_vec(dsub, r.f32s(ksub * dsub)?)?))
            .collect::<Result<Vec<_>>>()?;
        let pq = Pq::from_codebooks(books)?;
        let min = r.f32s(dim)?;
        let sq = Sq::from_parts(min, f32::from_bits(r.u32()?))
            .map_err(|_| Error::Corrupt("bad SQ parameters"))?;
        let counts: Vec<usize> = r
            .take_n(nlist, 4)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| u32::from_le_bytes(c) as usize)
            .collect();
        if counts.iter().sum::<usize>() != n {
            return Err(Error::Corrupt(
                "list sizes do not add up to the vector count",
            ));
        }
        for count in counts {
            let ids = r.take_n(count, 4)?.as_chunks::<4>().0;
            let ids = ids.iter().map(|&c| u32::from_le_bytes(c)).collect();
            let code_bytes = if nbits == 8 {
                count * m
            } else {
                count.div_ceil(kernels::PQ4_BLOCK) * m * 16
            };
            let codes = r.take(code_bytes)?.to_vec();
            ivf.lists.push(List { ids, codes });
        }
        ivf.sq_codes = bytemuck::cast_slice(r.take_n(n, dim)?).to_vec();
        r.finish()?;

        // Search indexes the SQ codes by id, so every id must be < n.
        let mut seen = vec![false; n];
        for &id in ivf.lists.iter().flat_map(|l| &l.ids) {
            match seen.get_mut(id as usize) {
                Some(s) if !*s => *s = true,
                _ => return Err(Error::Corrupt("list ids are not each of 0..len once")),
            }
        }
        ivf.quant = Some(Quantizers { coarse, pq, sq });
        ivf.len = n;
        Ok(ivf)
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
        let mut x = quant::sample(data, dim, cfg.max_train, cfg.seed).into_owned();
        if self.metric == Metric::Cosine {
            x.par_chunks_exact_mut(dim).for_each(|v| {
                let unit = self.metric.prepare(v).into_owned();
                v.copy_from_slice(&unit);
            });
        }
        let coarse = KMeans {
            max_train: usize::MAX, // already sampled
            seed: cfg.seed,
            ..KMeans::new(cfg.nlist, dim)
        }
        .train(&x)?;
        let sq = Sq::train(&x, dim, usize::MAX, cfg.seed)?;
        let mut residuals = x;
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
        let start = self.sq_codes.len();
        self.sq_codes.resize(start + n * dim, 0);
        let metric = self.metric;
        vectors
            .par_chunks_exact(dim)
            .zip(cells.par_iter_mut())
            .zip(codes.par_chunks_exact_mut(m))
            .zip(self.sq_codes[start..].par_chunks_exact_mut(dim))
            .for_each_init(
                || vec![0f32; dim],
                |r, (((v, cell), code), sq)| {
                    let v = metric.prepare(v);
                    *cell = q.coarse.assign(&v).0;
                    q.sq.encode_into(&v, sq);
                    r.copy_from_slice(&v);
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
        self.metric
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
    /// params.rerank)` nearest by ADC distance (the metric's distance to
    /// the PQ reconstruction). Ids not in `params.filter` are skipped while
    /// scanning. With `params.rerank > 0` those candidates are re-scored
    /// with their SQ codes (distance to the int8 reconstruction); the `k`
    /// nearest are returned either way.
    fn search(&self, query: &[f32], k: usize, params: &SearchParams) -> Result<Vec<Hit>> {
        check_dim(self.dim, query.len())?;
        let Some(q) = &self.quant else {
            return Ok(Vec::new());
        };
        if k == 0 {
            return Ok(Vec::new());
        }
        let query = &*self.metric.prepare(query);
        let ip = self.metric == Metric::InnerProduct;

        // A cell's distance is the query's to its centroid: L2, or for
        // inner product `-<q, c>`, which is also that list's ADC offset.
        let mut cells: Vec<Hit> = q
            .coarse
            .as_slice()
            .chunks_exact(self.dim)
            .enumerate()
            .map(|(i, c)| Hit {
                id: i as u32,
                dist: if ip {
                    -kernels::dot(query, c)
                } else {
                    kernels::l2_sq(query, c)
                },
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
        // Inner product: `-<q_j, centroid>` per subspace, the same for
        // every list.
        let ip_table: Option<Vec<f32>> =
            ip.then(|| q.pq.dot_table(query).iter().map(|x| -x).collect());
        let mut residual = vec![0f32; self.dim];
        let (mut lut, mut sums) = (Vec::new(), Vec::new());
        for cell in &cells {
            let list = &self.lists[cell.id as usize];
            if list.ids.is_empty() {
                continue;
            }
            let (base, table) = match &ip_table {
                Some(t) => (cell.dist, Cow::Borrowed(t.as_slice())),
                None => {
                    residual.copy_from_slice(query);
                    subtract(&mut residual, q.coarse.get(cell.id as usize));
                    (0.0, Cow::Owned(q.pq.distance_table(&residual)))
                }
            };
            if self.cfg.nbits == 8 {
                for (&id, code) in list.ids.iter().zip(list.codes.chunks_exact(self.cfg.m)) {
                    offer(id, base + quant::adc_distance(&table, code));
                }
                continue;
            }
            let (bias, step) = quantize_lut(&table, &mut lut);
            sums.resize(list.ids.len().next_multiple_of(kernels::PQ4_BLOCK), 0);
            kernels::pq4_scan(&list.codes, &lut, &mut sums);
            for (&id, &s) in list.ids.iter().zip(&sums) {
                offer(id, base + bias + f32::from(s) * step);
            }
        }

        let mut hits = best.into_vec();
        if params.rerank > 0 {
            // decode(c) = (c + 128) · scale + min, so
            //   |q - decode(c)|² = scale² · |(q - min) / scale - 128 - c|²
            //   <q, decode(c)>   = scale · <q, c> + 128 · scale · Σq + <q, min>.
            // The query-only parts are computed once; each candidate is one
            // i8 -> f32 widening and one f32 kernel call.
            let (scale, min) = (q.sq.scale(), q.sq.min());
            let qs: Vec<f32> = query
                .iter()
                .zip(min)
                .map(|(&x, &lo)| (x - lo) / scale - 128.0)
                .collect();
            let offset = 128.0 * scale * query.iter().sum::<f32>() + kernels::dot(query, min);
            let mut code = vec![0f32; self.dim];
            for h in &mut hits {
                let c = &self.sq_codes[h.id as usize * self.dim..][..self.dim];
                for (f, &b) in code.iter_mut().zip(c) {
                    *f = f32::from(b);
                }
                h.dist = if ip {
                    -(kernels::dot(query, &code) * scale + offset)
                } else {
                    kernels::l2_sq(&qs, &code) * scale * scale
                };
            }
        }
        hits.sort_unstable();
        hits.truncate(k);
        if self.metric == Metric::Cosine {
            for h in &mut hits {
                h.dist *= 0.5; // |q - x|² / 2 = 1 - cos for unit q and x
            }
        }
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

    /// Mean fraction of the exact top-k (under the index's metric) that
    /// `index` returns.
    fn recall(index: &dyn Index, data: &[f32], queries: &[f32], k: usize, p: &SearchParams) -> f64 {
        let mut exact = Flat::new(DIM, index.metric());
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

    /// `empty_clone` keeps the quantizers: re-adding a subset encodes it
    /// exactly as the original did (same lists, codes and re-rank codes).
    #[test]
    fn empty_clone_reuses_quantizers() {
        let data = test_data::vectors(3000, DIM, 1);
        let mut ivf = IvfPq::new(DIM, Metric::L2, cfg()).unwrap();
        ivf.add(&data).unwrap();
        let mut copy = ivf.empty_clone();
        assert_eq!((copy.len, copy.sq_codes.len()), (0, 0));
        // Rows 1000.. again, in order: like a compaction dropping 0..1000.
        copy.add(&data[1000 * DIM..]).unwrap();
        assert_eq!(copy.sq_codes, ivf.sq_codes[1000 * DIM..]);
        for (a, b) in ivf.lists.iter().zip(&copy.lists) {
            let kept: Vec<u32> = a
                .ids
                .iter()
                .filter(|&&i| i >= 1000)
                .map(|i| i - 1000)
                .collect();
            assert_eq!(kept, b.ids);
        }
        assert!(
            IvfPq::new(DIM, Metric::L2, cfg())
                .unwrap()
                .empty_clone()
                .quant
                .is_none()
        );
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
    fn cosine_and_inner_product() {
        let data = test_data::vectors(1500, DIM, 10);
        let queries = test_data::vectors(50, DIM, 11);
        for metric in [Metric::Cosine, Metric::InnerProduct] {
            for cfg in [
                cfg(),
                Config {
                    nbits: 4,
                    m: 16,
                    ..cfg()
                },
            ] {
                let mut ivf = IvfPq::new(DIM, metric, cfg).unwrap();
                ivf.add(&data).unwrap();
                let p = |nprobe, rerank| SearchParams {
                    nprobe,
                    rerank,
                    ..SearchParams::default()
                };
                let (few, plain, reranked) = (
                    recall(&ivf, &data, &queries, 10, &p(2, 0)),
                    recall(&ivf, &data, &queries, 10, &p(16, 0)),
                    recall(&ivf, &data, &queries, 10, &p(16, 200)),
                );
                let what = format!("{metric:?} nbits {}", cfg.nbits);
                assert!(few < plain && plain > 0.5, "{what}: {few} {plain}");
                assert!(reranked > 0.9, "{what}: {plain} -> {reranked}");

                // Re-ranked distances are the metric's, up to the SQ error
                // e = |decode(code) - v| <= scale / 2 per component.
                let e = ivf.quant.as_ref().unwrap().sq.scale() / 2.0 * (DIM as f32).sqrt();
                for q in queries.as_chunks::<DIM>().0.iter().take(5) {
                    let qn = metric.prepare(q);
                    let bound = match metric {
                        Metric::InnerProduct => kernels::dot(q, q).sqrt() * e,
                        _ => e + e * e / 2.0, // (2|q - v| e + e²) / 2, |q - v| <= 1
                    } + 1e-4;
                    for h in ivf.search(q, 10, &p(16, 50)).unwrap() {
                        let v = &data[h.id as usize * DIM..][..DIM];
                        let want = metric.distance(&qn, &metric.prepare(v));
                        assert!((h.dist - want).abs() <= bound, "{what}: {h:?} vs {want}");
                    }
                }
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

    /// Results for each query with and without re-rank and a filter.
    fn search_all(ivf: &IvfPq, queries: &[f32]) -> Vec<Vec<Hit>> {
        let odd: RoaringBitmap = (1..10_000).step_by(2).collect();
        let mut out = Vec::new();
        for q in queries.as_chunks::<DIM>().0 {
            for (rerank, filter) in [(0, None), (50, None), (50, Some(&odd))] {
                let p = SearchParams {
                    nprobe: 4,
                    rerank,
                    filter,
                    ..SearchParams::default()
                };
                out.push(ivf.search(q, 10, &p).unwrap());
            }
        }
        out
    }

    /// 8-bit m = 8, or 4-bit m = 16 (the same 8 bytes per vector).
    fn cfg_bits(nbits: u32) -> Config {
        let m = if nbits == 4 { 16 } else { 8 };
        Config { nbits, m, ..cfg() }
    }

    #[test]
    fn blob_round_trip_searches_identically() {
        let data = test_data::vectors(601, DIM, 12);
        let queries = test_data::vectors(20, DIM, 13);
        let more = test_data::vectors(100, DIM, 14);
        for (metric, nbits) in [
            (Metric::L2, 8),
            (Metric::InnerProduct, 4),
            (Metric::Cosine, 4),
        ] {
            let mut ivf = IvfPq::new(DIM, metric, cfg_bits(nbits)).unwrap();
            ivf.add(&data).unwrap();
            let bytes = ivf.to_bytes();
            assert_eq!(bytes.len(), bytes.capacity());
            let mut back = IvfPq::from_bytes(&bytes).unwrap();
            assert_eq!(back.to_bytes(), bytes);
            assert_eq!((back.len(), back.dim(), back.metric()), (601, DIM, metric));
            assert_eq!(search_all(&back, &queries), search_all(&ivf, &queries));
            // A loaded index keeps accepting inserts (4-bit: into a partly
            // filled block).
            ivf.add(&more).unwrap();
            back.add(&more).unwrap();
            assert_eq!(back.to_bytes(), ivf.to_bytes());
        }
        // Untrained, and trained but empty.
        let mut ivf = IvfPq::new(DIM, Metric::L2, cfg()).unwrap();
        let back = IvfPq::from_bytes(&ivf.to_bytes()).unwrap();
        assert!(back.is_empty() && back.quant.is_none());
        ivf.train(&data).unwrap();
        let back = IvfPq::from_bytes(&ivf.to_bytes()).unwrap();
        assert!(back.is_empty() && back.quant.is_some());
        assert_eq!(back.to_bytes(), ivf.to_bytes());
        assert!(
            back.search(&data[..DIM], 5, &SearchParams::default())
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn end_to_end_through_a_segment_file_without_vectors() {
        use storage::segment::{self, BlobKind, Segment, SegmentData};
        let data = test_data::vectors(1500, DIM, 15);
        let queries = test_data::vectors(20, DIM, 16);
        let mut ivf = IvfPq::new(DIM, Metric::Cosine, cfg_bits(4)).unwrap();
        ivf.add(&data).unwrap();
        let ids: Vec<u64> = (0..ivf.len() as u64).map(|i| 5_000_000 + i).collect();
        let dir = std::env::temp_dir().join(format!("nearpoint-ivf-e2e-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ivf.seg");
        segment::write(
            &path,
            &SegmentData {
                dim: DIM,
                metric: ivf.metric().code(),
                vectors: &[], // the blob is the whole index
                ids: &ids,
                blob_kind: BlobKind::IvfPq,
                blob: &ivf.to_bytes(),
            },
        )
        .unwrap();

        let seg = Segment::open(&path).unwrap();
        assert_eq!(seg.blob_kind(), BlobKind::IvfPq);
        assert!(seg.vectors().is_empty());
        assert_eq!(Metric::from_code(seg.metric()), Some(Metric::Cosine));
        let back = IvfPq::from_bytes(seg.index_blob()).unwrap();
        assert_eq!(search_all(&back, &queries), search_all(&ivf, &queries));
        let p = SearchParams {
            rerank: 10,
            ..SearchParams::default()
        };
        let hit = back.search(&data[42 * DIM..][..DIM], 1, &p).unwrap()[0];
        assert_eq!(seg.ids()[hit.id as usize], 5_000_042);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    fn small_blob(nbits: u32) -> Vec<u8> {
        let cfg = Config {
            nlist: 4,
            ..cfg_bits(nbits)
        };
        let mut ivf = IvfPq::new(DIM, Metric::L2, cfg).unwrap();
        ivf.add(&test_data::vectors(300, DIM, 17)).unwrap(); // 8-bit PQ needs 256
        ivf.to_bytes()
    }

    #[test]
    fn corrupt_blobs_are_errors() {
        let bytes = small_blob(8);
        let load = |b: &[u8]| IvfPq::from_bytes(b).err();
        for len in 0..bytes.len() {
            assert!(load(&bytes[..len]).is_some(), "truncated to {len}");
        }
        let mut longer = bytes.clone();
        longer.push(0);
        assert!(load(&longer).is_some());

        let ivf = IvfPq::from_bytes(&bytes).unwrap();
        let (nlist, m, n) = (4, 8, 300);
        let scale_at = 56 + 4 * (nlist * DIM + m * 256 * (DIM / m) + DIM);
        let ids_at = scale_at + 4 + 4 * nlist; // list 0's first id
        assert!(!ivf.lists[0].ids.is_empty());
        let second = ivf.lists[0].ids[1];
        let at = |off: usize, v: u32| {
            let mut b = bytes.clone();
            b[off..off + 4].copy_from_slice(&v.to_le_bytes());
            b
        };
        for (b, what) in [
            (at(20, 5), "bad config"), // m doesn't divide dim
            (at(24, 3), "bad config"), // nbits
            (at(28, 2), "bad trained flag"),
            (at(28, 0), "bad trained flag"), // untrained but not empty
            (
                at(48, n + 1),
                "list sizes do not add up to the vector count",
            ),
            (at(scale_at, 0), "bad SQ parameters"),
            (at(scale_at, f32::NAN.to_bits()), "bad SQ parameters"),
            (at(ids_at, n), "list ids are not each of 0..len once"),
            (at(ids_at, second), "list ids are not each of 0..len once"),
        ] {
            assert_eq!(load(&b), Some(Error::Corrupt(what)));
        }
        assert_eq!(load(&at(4, 2)), Some(Error::UnsupportedBlobVersion(2)));
    }

    proptest::proptest! {
        #[test]
        fn random_corruption_never_panics(
            four in proptest::bool::ANY,
            flips in proptest::collection::vec((0usize..1 << 20, 1u8..=255), 1..8),
            cut in 0usize..1 << 20,
        ) {
            static BLOBS: std::sync::LazyLock<[Vec<u8>; 2]> =
                std::sync::LazyLock::new(|| [small_blob(8), small_blob(4)]);
            let mut b = BLOBS[usize::from(four)].clone();
            for (pos, x) in flips {
                let i = pos % b.len();
                b[i] ^= x;
            }
            if cut % 4 == 0 {
                b.truncate(cut % b.len());
            }
            if let Ok(mut ivf) = IvfPq::from_bytes(&b) {
                // Whatever loads must be searchable and extendable.
                let dim = ivf.dim();
                let qs = test_data::vectors(3, dim, 3);
                for q in qs.chunks_exact(dim) {
                    for rerank in [0, 20] {
                        let p = SearchParams { nprobe: 2, rerank, ..SearchParams::default() };
                        let _ = ivf.search(q, 10, &p);
                    }
                }
                let _ = ivf.add(&qs);
            }
        }
    }

    #[test]
    fn config_and_input_errors() {
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
