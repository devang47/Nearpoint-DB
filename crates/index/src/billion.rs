//! Billion mode (Phase 9, ADR 0014): an IVF-PQ whose lists and re-rank
//! codes live in files, built by streaming the raw vectors once.
//!
//! - **Coarse quantizer:** `nlist` (up to 65,536) centroids from two-level
//!   k-means (`k1` cells, then `nlist / k1` centroids inside each), with an
//!   HNSW graph over them for assignment and probing: exact-enough nearest
//!   centroids at ~1k distance computations instead of `nlist`.
//! - **Codes:** each vector's residual `x - c` as a PQ code (`m` bytes),
//!   plus a refinement code (`m` bytes): PQ of what the first code missed,
//!   `x - c - pq(x - c)`. The scan ranks by ADC on the first code; the best
//!   `rerank` are re-scored against `c + pq + refine` (two codes' worth of
//!   precision, read by id from the refine file).
//!
//! L2 only: BIGANN and the other billion-scale sets are L2. This module
//! holds no files: callers pass byte slices (mapped by `storage`).

use std::collections::BinaryHeap;

use quant::{Centroids, KMeans, Pq, PqConfig};
use rayon::prelude::*;

use crate::ivfpq::SharedBytes;
use crate::{Error, Hit, Index, Metric, Result, SearchParams, blob, check_dim};

const MAGIC: &[u8; 4] = b"NPBQ";

/// HNSW over the centroids: assignment and probing.
const COARSE_HNSW: crate::hnsw::Config = crate::hnsw::Config {
    m: 32,
    ef_construction: 200,
    seed: 42,
};

#[derive(Debug, Clone, Copy)]
pub struct TrainConfig {
    /// Total coarse centroids (≤ 65,536: list ids are u16 on disk).
    pub nlist: usize,
    /// First-level cells; `nlist / k1` centroids are trained in each.
    pub k1: usize,
    /// Bytes per PQ code and per refinement code.
    pub m: usize,
    pub seed: u64,
}

/// The trained quantizers: everything needed to encode and search.
pub struct Quantizers {
    dim: usize,
    coarse: Centroids,
    graph: crate::Hnsw,
    pq: Pq,
    refine: Pq,
}

fn sub(a: &[f32], b: &[f32], out: &mut [f32]) {
    for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
        *o = x - y;
    }
}

impl Quantizers {
    /// Trains on `data` (row-major, `dim` floats per row; a 10M sample in
    /// Phase 9). Deterministic for a given `seed`, except the HNSW graph
    /// (parallel insertion), which only affects which near-equal centroid
    /// wins an assignment.
    pub fn train(data: &[f32], dim: usize, cfg: TrainConfig) -> Result<Quantizers> {
        crate::batch_ids(dim, 0, data)?;
        if cfg.nlist == 0 || cfg.nlist > 1 << 16 || cfg.k1 == 0 || !cfg.nlist.is_multiple_of(cfg.k1)
        {
            return Err(Error::InvalidConfig(
                "need 0 < nlist <= 65536 and k1 dividing nlist",
            ));
        }
        if cfg.m == 0 || !dim.is_multiple_of(cfg.m) {
            return Err(Error::InvalidConfig("m must divide dim"));
        }
        let k2 = cfg.nlist / cfg.k1;
        // Level 1, then each cell's members (a bounded sample of them).
        let top = KMeans {
            seed: cfg.seed,
            ..KMeans::new(cfg.k1, dim)
        }
        .train(data)?;
        let assign = top.assign_all(data)?;
        let mut members: Vec<Vec<u32>> = vec![Vec::new(); cfg.k1];
        for (i, &(c, _)) in assign.iter().enumerate() {
            members[c as usize].push(i as u32);
        }
        let per_cell: Vec<Vec<f32>> = members
            .par_iter()
            .enumerate()
            .map(|(cell, rows)| -> Result<Vec<f32>> {
                let x: Vec<f32> = rows
                    .iter()
                    .flat_map(|&r| data[r as usize * dim..][..dim].iter().copied())
                    .collect();
                if rows.len() < k2 {
                    return Err(Error::InvalidConfig(
                        "a first-level cell has fewer rows than nlist / k1: train on more data",
                    ));
                }
                let c = KMeans {
                    seed: cfg.seed ^ cell as u64,
                    ..KMeans::new(k2, dim)
                }
                .train(&x)?;
                Ok(c.as_slice().to_vec())
            })
            .collect::<Result<_>>()?;
        let coarse = Centroids::from_vec(dim, per_cell.concat())?;
        let mut graph = crate::Hnsw::new(dim, Metric::L2, COARSE_HNSW);
        graph.reserve(coarse.k());
        graph.add(coarse.as_slice())?;

        // PQ on residuals, then the refinement PQ on what PQ left, both on
        // a sample (PqConfig's max_train).
        let sample = quant::sample(data, dim, 262_144, cfg.seed).into_owned();
        let s = sample.len() / dim;
        let mut residuals = vec![0f32; s * dim];
        residuals
            .par_chunks_exact_mut(dim)
            .zip(sample.par_chunks_exact(dim))
            .for_each(|(r, x)| sub(x, coarse.get(coarse.assign(x).0 as usize), r));
        let pq_cfg = PqConfig {
            seed: cfg.seed,
            ..PqConfig::new(dim, cfg.m)
        };
        let pq = pq_cfg.train(&residuals)?;
        residuals.par_chunks_exact_mut(dim).for_each(|r| {
            let p = pq.decode(&pq.encode(r));
            for (x, y) in r.iter_mut().zip(p) {
                *x -= y;
            }
        });
        let refine = PqConfig {
            seed: cfg.seed ^ 1,
            ..pq_cfg
        }
        .train(&residuals)?;
        Ok(Quantizers {
            dim,
            coarse,
            graph,
            pq,
            refine,
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn nlist(&self) -> usize {
        self.coarse.k()
    }

    /// Bytes per PQ code (and per refinement code).
    pub fn m(&self) -> usize {
        self.pq.m()
    }

    /// The nearest centroids to `v` (HNSW, beam `ef`).
    fn nearest(&self, v: &[f32], n: usize, ef: usize) -> Result<Vec<Hit>> {
        self.graph.search(
            v,
            n,
            &SearchParams {
                ef: ef.max(n),
                ..SearchParams::default()
            },
        )
    }

    /// Encodes `rows` (in parallel): per row its list id, PQ code and
    /// refinement code, appended to the three outputs.
    pub fn encode(
        &self,
        rows: &[f32],
        lists: &mut Vec<u16>,
        codes: &mut Vec<u8>,
        refine: &mut Vec<u8>,
    ) -> Result<()> {
        let (d, m) = (self.dim, self.m());
        let n = crate::batch_ids(d, 0, rows)?.len();
        let (l0, c0, r0) = (lists.len(), codes.len(), refine.len());
        lists.resize(l0 + n, 0);
        codes.resize(c0 + n * m, 0);
        refine.resize(r0 + n * m, 0);
        lists[l0..]
            .par_iter_mut()
            .zip(codes[c0..].par_chunks_exact_mut(m))
            .zip(refine[r0..].par_chunks_exact_mut(m))
            .zip(rows.par_chunks_exact(d))
            .try_for_each_init(
                || vec![0f32; d],
                |r, (((list, code), rcode), x)| -> Result<()> {
                    let c = self.nearest(x, 1, 32)?.first().map_or(0, |h| h.id);
                    *list = c as u16;
                    sub(x, self.coarse.get(c as usize), r);
                    self.pq.encode_into(r, code);
                    for (x, y) in r.iter_mut().zip(self.pq.decode(code)) {
                        *x -= y;
                    }
                    self.refine.encode_into(r, rcode);
                    Ok(())
                },
            )
    }

    /// The `k` nearest encoded vectors to `query`: `lists` is the sorted
    /// layout, `refine` the refinement codes by id. Probed lists are
    /// scanned in parallel; each candidate carries its list (for the
    /// centroid) and row (for its PQ code) to the re-rank.
    pub fn search(
        &self,
        lists: &Lists<'_>,
        refine: &Refine,
        query: &[f32],
        k: usize,
        params: &BillionParams,
    ) -> Result<Vec<Hit>> {
        check_dim(self.dim, query.len())?;
        let (d, m) = (self.dim, self.m());
        let probes = self.nearest(query, params.nprobe.max(1), params.nprobe * 2 + 32)?;
        // Mapped lists: request every probed list first, so the reads
        // overlap instead of faulting in one list at a time.
        if let Some(hint) = lists.will_need {
            for cell in &probes {
                let l = cell.id as usize;
                let (a, b) = (lists.offsets[l] as usize, lists.offsets[l + 1] as usize);
                hint(a * m..b * m, a..b);
            }
        }
        let short = k.max(params.rerank);
        // One worker per probed list; each keeps its own top `short`.
        let per: Vec<Vec<Cand>> = probes
            .par_iter()
            .map_init(
                || vec![0f32; d],
                |r, cell| {
                    let l = cell.id as usize;
                    let (a, b) = (lists.offsets[l] as usize, lists.offsets[l + 1] as usize);
                    if a == b {
                        return Vec::new();
                    }
                    sub(query, self.coarse.get(l), r);
                    let table = self.pq.distance_table(r);
                    let mut best = BinaryHeap::with_capacity(short + 1);
                    for (row, code) in (a..b).zip(lists.codes[a * m..b * m].chunks_exact(m)) {
                        let dist = quant::adc_distance(&table, code);
                        if best.len() == short
                            && best.peek().is_some_and(|w: &Cand| dist >= w.0.dist)
                        {
                            continue;
                        }
                        let id = lists.ids[row];
                        if params.skip.is_some_and(|s| s(id)) {
                            continue;
                        }
                        best.push(Cand(Hit { id, dist }, cell.id, row));
                        if best.len() > short {
                            best.pop();
                        }
                    }
                    best.into_vec()
                },
            )
            .collect();
        let mut cand: Vec<Cand> = per.into_iter().flatten().collect();
        cand.sort_unstable();
        cand.truncate(short);
        if params.rerank == 0 {
            cand.truncate(k);
            return Ok(cand.into_iter().map(|c| c.0).collect());
        }
        // Re-score `c + pq + refine` exactly; the PQ code is re-read from
        // its list row (just scanned, cached), the refinement code from the
        // refine buffer (random reads, prefetched first so they overlap).
        for Cand(h, ..) in &cand {
            refine.will_need(h.id as usize * m..(h.id as usize + 1) * m);
        }
        let bytes = refine.bytes();
        let mut out = Vec::with_capacity(cand.len());
        let mut x = vec![0f32; d];
        for Cand(h, list, row) in &cand {
            let rc = bytes
                .get(h.id as usize * m..(h.id as usize + 1) * m)
                .ok_or(Error::Corrupt("refine code past the end"))?;
            let pc = &lists.codes[row * m..(row + 1) * m];
            x.copy_from_slice(self.coarse.get(*list as usize));
            for ((v, p), r) in x
                .iter_mut()
                .zip(self.pq.decode(pc))
                .zip(self.refine.decode(rc))
            {
                *v += p + r;
            }
            out.push(Hit {
                id: h.id,
                dist: kernels::l2_sq(query, &x),
            });
        }
        out.sort_unstable();
        out.truncate(k);
        Ok(out)
    }

    /// Id `id`'s vector as stored, `c + pq + refine` (`None` past the end):
    /// its list from `list_of`, its row by binary search (ids ascend
    /// within a list).
    pub fn reconstruct(
        &self,
        lists: &Lists<'_>,
        refine: &[u8],
        list_of: &[u16],
        id: u32,
    ) -> Option<Vec<f32>> {
        let m = self.m();
        let l = usize::from(*list_of.get(id as usize)?);
        let (a, b) = (
            *lists.offsets.get(l)? as usize,
            *lists.offsets.get(l + 1)? as usize,
        );
        let row = a + lists.ids.get(a..b)?.binary_search(&id).ok()?;
        let pc = lists.codes.get(row * m..(row + 1) * m)?;
        let rc = refine.get(id as usize * m..(id as usize + 1) * m)?;
        let mut x = self.coarse.get(l).to_vec();
        for ((v, p), r) in x
            .iter_mut()
            .zip(self.pq.decode(pc))
            .zip(self.refine.decode(rc))
        {
            *v += p + r;
        }
        Some(x)
    }

    /// Bytes held in RAM: centroids, codebooks and the centroid graph.
    pub fn heap_bytes(&self) -> usize {
        let books: usize = [&self.pq, &self.refine]
            .iter()
            .flat_map(|p| p.codebooks())
            .map(|b| 4 * b.as_slice().len())
            .sum();
        4 * self.coarse.as_slice().len() + books + self.graph.heap_bytes()
    }

    /// Config, centroids, both codebooks and the HNSW graph (so a resumed
    /// build assigns with the same graph). Layout in docs/formats/billion.md.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        blob::put_header(&mut out, MAGIC, self.dim, Metric::L2);
        for x in [self.nlist(), self.m(), self.pq.ksub(), self.refine.m()] {
            blob::put_u32(&mut out, x as u32);
        }
        blob::put_f32s(&mut out, self.coarse.as_slice());
        for pq in [&self.pq, &self.refine] {
            for b in pq.codebooks() {
                blob::put_f32s(&mut out, b.as_slice());
            }
        }
        let g = self.graph.to_bytes();
        blob::put_u64(&mut out, g.len() as u64);
        out.extend_from_slice(&g);
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Quantizers> {
        let mut r = blob::Reader::new(bytes);
        let (dim, metric) = r.header(MAGIC)?;
        if metric != Metric::L2 {
            return Err(Error::Corrupt("billion mode is L2 only"));
        }
        let (nlist, m, ksub, rm) = (
            r.u32()? as usize,
            r.u32()? as usize,
            r.u32()? as usize,
            r.u32()? as usize,
        );
        if nlist == 0
            || nlist > 1 << 16
            || m == 0
            || !dim.is_multiple_of(m)
            || rm != m
            || ksub != 256
        {
            return Err(Error::Corrupt("bad billion config"));
        }
        let coarse = Centroids::from_vec(dim, r.f32s(nlist * dim)?)?;
        let dsub = dim / m;
        let mut books = || -> Result<Pq> {
            let b = (0..m)
                .map(|_| Ok(Centroids::from_vec(dsub, r.f32s(ksub * dsub)?)?))
                .collect::<Result<Vec<_>>>()?;
            Ok(Pq::from_codebooks(b)?)
        };
        let (pq, refine) = (books()?, books()?);
        let glen = usize::try_from(r.u64()?).map_err(|_| Error::Corrupt("bad graph length"))?;
        let graph = crate::Hnsw::from_bytes(r.take(glen)?, coarse.as_slice().to_vec())?;
        r.finish()?;
        if graph.len() != nlist {
            return Err(Error::Corrupt("graph and centroids differ"));
        }
        Ok(Quantizers {
            dim,
            coarse,
            graph,
            pq,
            refine,
        })
    }
}

#[derive(Clone, Copy)]
pub struct BillionParams<'a> {
    pub nprobe: usize,
    pub rerank: usize,
    /// Ids to leave out (tombstoned, or outside a filter); tested only for
    /// rows whose ADC distance would enter the candidate list.
    pub skip: Option<&'a Skip<'a>>,
}

pub type Skip<'a> = dyn Fn(u32) -> bool + Sync + 'a;

/// The sorted list layout, borrowed from a mapped file.
pub struct Lists<'a> {
    /// `nlist + 1` row offsets.
    pub offsets: &'a [u64],
    /// `m` bytes per row, list by list.
    pub codes: &'a [u8],
    /// The id of each row, list by list.
    pub ids: &'a [u32],
    /// Prefetch hint for a list's code bytes and id rows, called for every
    /// probed list before the scan (`None`: in RAM).
    pub will_need: Option<&'a WillNeed<'a>>,
}

/// A prefetch hint: code byte range, id row range.
pub type WillNeed<'a> = dyn Fn(std::ops::Range<usize>, std::ops::Range<usize>) + Sync + 'a;

/// A scan candidate: hit (ADC distance), its list, its row in the layout.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct Cand(Hit, u32, usize);

/// Refinement codes by id (`m` bytes each), usually a mapped file.
pub type Refine = SharedBytes;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{Flat, test_data};

    const D: usize = 16;

    fn cfg() -> TrainConfig {
        TrainConfig {
            nlist: 64,
            k1: 8,
            m: 8,
            seed: 7,
        }
    }

    fn encode(q: &Quantizers, data: &[f32]) -> (Vec<u16>, Vec<u8>, Vec<u8>) {
        let (mut l, mut c, mut r) = (Vec::new(), Vec::new(), Vec::new());
        q.encode(data, &mut l, &mut c, &mut r).unwrap();
        (l, c, r)
    }

    /// The sorted layout, as the on-disk sort produces it.
    fn sort(nlist: usize, m: usize, lists: &[u16], codes: &[u8]) -> (Vec<u64>, Vec<u8>, Vec<u32>) {
        let mut off = vec![0u64; nlist + 1];
        for &l in lists {
            off[l as usize + 1] += 1;
        }
        for i in 0..nlist {
            off[i + 1] += off[i];
        }
        let mut at = off.clone();
        let (mut out, mut ids) = (vec![0u8; codes.len()], vec![0u32; lists.len()]);
        for (id, &l) in lists.iter().enumerate() {
            let p = at[l as usize] as usize;
            at[l as usize] += 1;
            ids[p] = id as u32;
            out[p * m..(p + 1) * m].copy_from_slice(&codes[id * m..(id + 1) * m]);
        }
        (off, out, ids)
    }

    /// Recall rises with the refinement re-rank and with probes; a blob
    /// round trip encodes identically.
    #[test]
    fn train_encode_search() {
        let data = test_data::vectors(20_000, D, 1);
        let q = Quantizers::train(&data, D, cfg()).unwrap();
        assert_eq!((q.nlist(), q.m(), q.dim()), (64, 8, D));
        let (l, c, r) = encode(&q, &data);
        let (off, codes, ids) = sort(q.nlist(), q.m(), &l, &c);
        let seen = std::sync::Mutex::new(0usize);
        let hint = |c: std::ops::Range<usize>, i: std::ops::Range<usize>| {
            assert_eq!(c.len(), i.len() * 8);
            *seen.lock().unwrap() += 1;
        };
        let lists = Lists {
            offsets: &off,
            codes: &codes,
            ids: &ids,
            will_need: Some(&hint),
        };
        let refine: Refine = Arc::new(r.clone());
        let mut flat = Flat::new(D, Metric::L2);
        flat.add(&data).unwrap();
        let queries = test_data::vectors(50, D, 2);
        let recall = |nprobe, rerank| {
            let p = BillionParams {
                nprobe,
                rerank,
                skip: None,
            };
            let mut hits = 0;
            for v in queries.as_chunks::<D>().0 {
                let truth = flat.search(v, 10, &SearchParams::default()).unwrap();
                let got = q.search(&lists, &refine, v, 10, &p).unwrap();
                assert_eq!(got.len(), 10);
                hits += got
                    .iter()
                    .filter(|g| truth.iter().any(|t| t.id == g.id))
                    .count();
            }
            hits as f64 / 500.0
        };
        let (pq_only, refined, few) = (recall(64, 0), recall(64, 200), recall(2, 200));
        assert!(
            refined >= 0.9 && refined > pq_only + 0.1,
            "{pq_only} -> {refined}"
        );
        assert!(few < refined, "nprobe 2: {few}, 64: {refined}");
        assert_eq!(*seen.lock().unwrap(), 50 * (64 + 64 + 2)); // every probe hinted

        // Skipped ids never come back; reconstruct is what re-ranking scores.
        let odd = |id: u32| id % 2 == 1;
        let p = BillionParams {
            nprobe: 64,
            rerank: 200,
            skip: Some(&odd),
        };
        let v = &queries[..D];
        let got = q.search(&lists, &refine, v, 10, &p).unwrap();
        assert!(got.len() == 10 && got.iter().all(|h| h.id % 2 == 0));
        let list_of = &l;
        for h in &got {
            let x = q.reconstruct(&lists, &r, list_of, h.id).unwrap();
            assert!((kernels::l2_sq(v, &x) - h.dist).abs() <= 1e-4 * h.dist.max(1.0));
        }
        assert!(q.reconstruct(&lists, &r, list_of, 20_000).is_none());

        let back = Quantizers::from_bytes(&q.to_bytes()).unwrap();
        assert_eq!(back.to_bytes(), q.to_bytes());
        let (l2, c2, r2) = encode(&back, &data[..1000 * D]);
        assert_eq!(
            (&l2[..], &c2[..], &r2[..]),
            (&l[..1000], &c[..8000], &r[..8000])
        );
    }

    #[test]
    fn bad_configs_and_blobs_are_errors() {
        let data = test_data::vectors(2000, D, 1);
        for bad in [
            TrainConfig {
                nlist: 1 << 17,
                k1: 2,
                ..cfg()
            },
            TrainConfig {
                nlist: 64,
                k1: 5,
                ..cfg()
            },
            TrainConfig { m: 5, ..cfg() },
            TrainConfig {
                nlist: 4096,
                k1: 2,
                ..cfg()
            }, // cells too small
        ] {
            assert!(Quantizers::train(&data, D, bad).is_err(), "{bad:?}");
        }
        let q = Quantizers::train(
            &data,
            D,
            TrainConfig {
                nlist: 16,
                k1: 4,
                ..cfg()
            },
        )
        .unwrap();
        let bytes = q.to_bytes();
        for n in (0..bytes.len()).step_by(97) {
            assert!(Quantizers::from_bytes(&bytes[..n]).is_err());
        }
        let mut flipped = bytes.clone();
        flipped[17] ^= 0xff; // nlist
        assert!(Quantizers::from_bytes(&flipped).is_err());
    }
}
