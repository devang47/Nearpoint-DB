//! A collection (tasks 3.3–3.4, 4.2–4.3): a WAL, one mutable in-RAM HNSW
//! segment, sealed segments, an external-id map and tombstones, all in one
//! directory. Layout, LSN coverage and the seal protocol are in
//! docs/adr/0008-collection-recovery.md; file formats in
//! docs/formats/manifest.md.
//!
//! A write is logged to the WAL first and applied in memory only once the
//! WAL acknowledged it. `seal` writes the mutable segment to files, then
//! publishes a new manifest (the commit point), then deletes the WAL files
//! the manifest covers. `open` loads the manifest's segments and replays
//! only WAL records above the manifest's LSN, so a crash at any step
//! neither loses nor double-applies a write.
//!
//! Concurrency (ADR 0011): one writer at a time (`writer` mutex, held
//! through the WAL sync), any number of readers. Readers clone the current
//! `Arc<Snapshot>` and search it without blocking on the writer's fsync.
//! The writer appends rows to the mutable segment under its `RwLock` (in
//! chunks, so readers wait at most one chunk), then publishes a new
//! snapshot whose row count covers them and whose tombstone bitmaps are
//! copy-on-write clones. Rows past a snapshot's count are invisible to it.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

use index::{Hnsw, Index, Metric, SearchParams, hnsw};
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use storage::framed;
use storage::segment::{self, BlobKind, Segment, SegmentData};
use storage::wal::{Lsn, Record, SyncPolicy, Wal, WalError};

use crate::{Filter, PayloadStore, Schema, evaluate};

const MANIFEST: &str = "MANIFEST";
const MANIFEST_MAGIC: [u8; 8] = *b"NPMANIFS";
const MANIFEST_VERSION: u32 = 1;
const PAYLOADS_MAGIC: [u8; 8] = *b"NPPAYLDS";
const PAYLOADS_VERSION: u32 = 1;
/// Rows the writer adds to the mutable segment per write-lock hold.
const APPLY_CHUNK: usize = 1024;
/// WAL records replayed per batch on open (bounds the extra copy).
const REPLAY_CHUNK: usize = 65_536;

/// A collection's settings. Everything but `sync` is persisted by
/// [`crate::Db`] (docs/formats/manifest.md, `CONFIG`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub dim: usize,
    #[serde(with = "MetricDef")]
    pub metric: Metric,
    /// For new mutable segments; sealed ones keep the config they were
    /// built with (it's in their index blob).
    #[serde(default, with = "HnswDef")]
    pub hnsw: hnsw::Config,
    /// Declared payload field types (see `payload.rs`).
    #[serde(default)]
    pub schema: Schema,
    /// Not persisted: the server picks it at startup.
    #[serde(skip)]
    pub sync: SyncPolicy,
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        if self.dim == 0 || self.hnsw.m < 2 || self.hnsw.ef_construction == 0 {
            return Err(Error::Invalid("need dim > 0, m >= 2, ef_construction >= 1"));
        }
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "Metric", rename_all = "snake_case")]
enum MetricDef {
    L2,
    InnerProduct,
    Cosine,
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "hnsw::Config", deny_unknown_fields)]
struct HnswDef {
    #[serde(default = "default_m")]
    m: usize,
    #[serde(default = "default_ef_construction")]
    ef_construction: usize,
    #[serde(default = "default_seed")]
    seed: u64,
}

fn default_m() -> usize {
    hnsw::Config::default().m
}

fn default_ef_construction() -> usize {
    hnsw::Config::default().ef_construction
}

fn default_seed() -> u64 {
    hnsw::Config::default().seed
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The WAL failed with this write in flight (ADR 0007): it may or may
    /// not be durable, and it was not applied in memory. Every later write
    /// fails the same way; reopen the collection to recover and find out.
    #[error("write outcome unknown, WAL failed ({0}); reopen the collection")]
    Unknown(WalError),
    #[error("wal: {0}")]
    Wal(#[from] WalError),
    #[error("{0}: {1}")]
    Segment(PathBuf, segment::Error),
    #[error("{0}: {1}")]
    File(PathBuf, framed::Error),
    #[error("index: {0}")]
    Index(#[from] index::Error),
    #[error(transparent)]
    Payload(#[from] crate::Error),
    #[error("i/o: {0}")]
    Io(#[from] io::Error),
    #[error("invalid: {0}")]
    Invalid(&'static str),
    #[error("corrupt collection: {0}")]
    Corrupt(String),
    #[error("collection was dropped")]
    Closed,
    #[error("collection `{0}` not found")]
    NotFound(String),
    #[error("collection `{0}` already exists")]
    Exists(String),
    #[error("{0} is locked by another process")]
    Locked(PathBuf),
    #[error("collection `{0}`: {1}")]
    Open(String, Box<Error>),
}

pub type Result<T> = std::result::Result<T, Error>;

/// What `open` found and did.
#[derive(Debug, Default)]
pub struct Recovery {
    /// Sealed segments loaded.
    pub segments: usize,
    /// WAL records applied on top of them.
    pub replayed: usize,
    /// Torn or corrupt WAL bytes and files dropped (see `wal::Recovered`).
    pub dropped_bytes: u64,
    pub dropped_files: usize,
}

/// Search knobs beyond the query and `k`.
#[derive(Debug, Clone, Copy, Default)]
pub struct SearchOptions<'a> {
    /// HNSW beam width (`None`: the index default, 64); raised to `k`.
    pub ef: Option<usize>,
    pub filter: Option<&'a Filter>,
    pub with_payload: bool,
    pub with_vector: bool,
}

/// A search result. `payload` (null if it has none) and `vector` (stored
/// form) are filled only when asked for.
#[derive(Debug, Clone, PartialEq)]
pub struct Match {
    pub id: u64,
    pub dist: f32,
    pub payload: Option<Value>,
    pub vector: Option<Vec<f32>>,
}

/// One segment's data. A sealed one never changes; the mutable one only
/// grows, under its write lock, which only the writer takes.
struct Seg {
    index: Hnsw,
    /// External id of each row.
    ids: Vec<u64>,
    payloads: PayloadStore,
}

/// A segment as one snapshot sees it.
#[derive(Clone)]
struct View {
    /// The LSN it was sealed at (its file name); 0 while mutable.
    lsn: Lsn,
    seg: Arc<RwLock<Seg>>,
    /// Rows `0..rows` are visible. The mutable segment can hold more:
    /// applied by the writer but not yet published.
    rows: u32,
    /// Rows deleted or replaced by a later upsert. Copy-on-write: the
    /// writer clones it (`Arc::make_mut`) into the next snapshot.
    dead: Arc<RoaringBitmap>,
}

/// What readers search: sealed segments in LSN order, then the mutable one
/// (always last). Never changed once published; the writer (and, from
/// Phase 6, a background builder holding the writer lock) swaps in a new
/// one.
#[derive(Clone)]
struct Snapshot {
    segs: Vec<View>,
}

/// State only the writer touches.
struct Writer {
    /// Every write with LSN <= this is in the manifest.
    covered: Lsn,
    /// Set by `close`: later writes fail with `Closed`.
    closed: bool,
}

pub struct Collection {
    dir: PathBuf,
    cfg: Config,
    wal: Wal,
    /// Held by each write and seal for its whole run, WAL sync included.
    // ponytail: one sync per request, so concurrent single-point writers
    // don't share a group commit; batch (or bulk) to go faster. Collect
    // pending requests into one WAL batch if that becomes the bottleneck.
    writer: Mutex<Writer>,
    /// External id → (segment, row) of its one live row. Changed only by
    /// the writer, together with `snap`, under this lock (so a reader
    /// holding it sees the two agree). Lock order: live, snap, a segment.
    live: RwLock<HashMap<u64, (u32, u32)>>,
    /// Held only to clone or swap the `Arc`.
    snap: RwLock<Arc<Snapshot>>,
}

impl Collection {
    /// Opens (or creates) the collection in `dir`: loads the manifest's
    /// segments, deletes files a crashed seal left behind, and replays WAL
    /// records the manifest doesn't cover. One process at a time.
    pub fn open(dir: impl AsRef<Path>, cfg: Config) -> Result<(Collection, Recovery)> {
        cfg.validate()?;
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;
        let mpath = dir.join(MANIFEST);
        let manifest = match framed::read(&mpath, MANIFEST_MAGIC, MANIFEST_VERSION) {
            Ok(body) => decode_manifest(&body).ok_or_else(|| corrupt(&mpath, "bad manifest"))?,
            Err(framed::Error::Io(e)) if e.kind() == io::ErrorKind::NotFound => Manifest::default(),
            Err(e) => return Err(Error::File(mpath, e)),
        };
        let keep: Vec<Lsn> = manifest.segs.iter().map(|s| s.0).collect();
        remove_garbage(&dir, &keep)?;

        let mut segs = Vec::new();
        let mut live = HashMap::new();
        for (lsn, dead) in manifest.segs {
            let seg = load_segment(&dir, &cfg, lsn, &dead)?;
            for (row, &id) in seg.ids.iter().enumerate() {
                if !dead.contains(row as u32)
                    && live.insert(id, (segs.len() as u32, row as u32)).is_some()
                {
                    return Err(Error::Corrupt(format!("id {id} is live in two rows")));
                }
            }
            segs.push(View {
                lsn,
                rows: seg.ids.len() as u32,
                seg: Arc::new(RwLock::new(seg)),
                dead: Arc::new(dead),
            });
        }
        segs.push(new_mutable(&cfg));

        let (wal, rec) = Wal::open(dir.join("wal"), cfg.sync)?;
        let c = Collection {
            dir,
            cfg,
            wal,
            writer: Mutex::new(Writer {
                covered: manifest.covered,
                closed: false,
            }),
            live: RwLock::new(live),
            snap: RwLock::new(Arc::new(Snapshot { segs })),
        };
        let covered = manifest.covered;
        let start = rec.records.partition_point(|r| r.0 <= covered);
        let resume = rec.records.get(start).map_or(c.wal.next_lsn(), |r| r.0);
        if resume != covered + 1 {
            return Err(Error::Corrupt(format!(
                "WAL resumes at LSN {resume}, manifest covers up to {covered}"
            )));
        }
        let is_upsert = |r: &(Lsn, Record)| matches!(r.1, Record::Upsert { .. });
        for run in rec.records[start..].chunk_by(|a, b| is_upsert(a) == is_upsert(b)) {
            for batch in run.chunks(REPLAY_CHUNK) {
                c.replay(batch)?;
            }
        }
        // Finishes a seal that crashed after publishing its manifest.
        c.wal.remove_before(covered + 1)?;
        let report = Recovery {
            segments: c.snapshot().segs.len() - 1,
            replayed: rec.records.len() - start,
            dropped_bytes: rec.dropped_bytes,
            dropped_files: rec.dropped_files,
        };
        Ok((c, report))
    }

    /// Applies WAL records that are all upserts or all deletes.
    fn replay(&self, batch: &[(Lsn, Record)]) -> Result<()> {
        let (mut ids, mut vectors, mut payloads) = (Vec::new(), Vec::new(), Vec::new());
        for (lsn, r) in batch {
            match r {
                Record::Upsert {
                    id,
                    vector,
                    payload,
                } => {
                    self.check_dim(vector)?;
                    ids.push(*id);
                    vectors.extend_from_slice(vector);
                    payloads.push(match payload.as_slice() {
                        [] => Value::Null,
                        b => serde_json::from_slice(b)
                            .map_err(|e| Error::Corrupt(format!("WAL LSN {lsn} payload: {e}")))?,
                    });
                }
                Record::Delete { id } => ids.push(*id),
            }
        }
        if payloads.is_empty() {
            self.apply_deletes(&ids);
            Ok(())
        } else {
            self.apply_upserts(&ids, &vectors, &payloads)
        }
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Inserts or replaces `id`. `payload` is a JSON object or null. Durable
    /// on return per the sync policy; `Error::Unknown` means it may or may
    /// not be (ADR 0007).
    pub fn upsert(&self, id: u64, vector: &[f32], payload: &Value) -> Result<()> {
        self.check_dim(vector)?;
        self.upsert_batch(&[id], vector, std::slice::from_ref(payload))
    }

    /// Upserts `ids[i]` with vector `vectors[i * dim..][..dim]` and payload
    /// `payloads[i]` (`payloads` may be empty: no payloads), in order, so a
    /// repeated id ends with its last version. The whole batch is checked
    /// before anything is logged, then logged with one WAL sync. On
    /// `Error::Unknown` any prefix of it may be durable.
    pub fn upsert_batch(&self, ids: &[u64], vectors: &[f32], payloads: &[Value]) -> Result<()> {
        let n = ids.len();
        if vectors.len() != n * self.cfg.dim {
            return Err(Error::Invalid("need ids.len() * dim vector components"));
        }
        if !payloads.is_empty() && payloads.len() != n {
            return Err(Error::Invalid("need one payload per id, or none"));
        }
        if !vectors.iter().all(|x| x.is_finite()) {
            return Err(Error::Invalid("vector has a NaN or infinite component"));
        }
        let nulls;
        let payloads = if payloads.is_empty() {
            nulls = vec![Value::Null; n];
            &nulls
        } else {
            payloads
        };
        let _w = self.writer()?;
        {
            let snap = self.snapshot();
            let seg = read(&snap.segs[snap.segs.len() - 1].seg);
            if seg.ids.len() + n > u32::MAX as usize {
                return Err(index::Error::Full.into());
            }
            seg.payloads.check(payloads)?;
        }
        let records: Vec<Record> = ids
            .iter()
            .zip(vectors.chunks_exact(self.cfg.dim))
            .zip(payloads)
            .map(|((&id, v), p)| Record::Upsert {
                id,
                vector: v.to_vec(),
                payload: match p {
                    Value::Null => Vec::new(),
                    p => p.to_string().into_bytes(),
                },
            })
            .collect();
        self.log(&records)?;
        self.apply_upserts(ids, vectors, payloads)
    }

    /// Deletes `id`; returns whether it existed. Durable on return like
    /// `upsert` (a missing id writes nothing).
    pub fn delete(&self, id: u64) -> Result<bool> {
        Ok(self.delete_batch(&[id])? == 1)
    }

    /// Deletes every live id in `ids` with one WAL sync; returns how many
    /// existed.
    pub fn delete_batch(&self, ids: &[u64]) -> Result<usize> {
        let _w = self.writer()?;
        let doomed: Vec<u64> = {
            let live = read(&self.live);
            let mut seen = HashSet::new();
            ids.iter()
                .copied()
                .filter(|id| live.contains_key(id) && seen.insert(*id))
                .collect()
        };
        if doomed.is_empty() {
            return Ok(0);
        }
        let records: Vec<_> = doomed.iter().map(|&id| Record::Delete { id }).collect();
        self.log(&records)?;
        self.apply_deletes(&doomed);
        Ok(doomed.len())
    }

    /// The vector (in stored form: unit length for cosine) and payload of
    /// `id`. A payload with no fields reads back as null.
    pub fn get(&self, id: u64) -> Option<(Vec<f32>, Value)> {
        let live = read(&self.live);
        let &(s, row) = live.get(&id)?;
        let snap = self.snapshot();
        let seg = read(&snap.segs[s as usize].seg);
        Some((self.vector(&seg, row)?, payload(&seg, row)))
    }

    /// Live ids.
    pub fn len(&self) -> usize {
        read(&self.live).len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The `k` nearest live ids matching `opts.filter`, nearest first.
    pub fn search(&self, query: &[f32], k: usize, opts: &SearchOptions) -> Result<Vec<Match>> {
        self.check_dim(query)?;
        let snap = self.snapshot();
        let mut hits = Vec::new(); // (dist, segment, row)
        for (s, view) in snap.segs.iter().enumerate() {
            let n = view.rows;
            if n == 0 {
                continue;
            }
            let seg = read(&view.seg);
            let allowed = match opts.filter {
                Some(f) => Some(evaluate(f, &seg.payloads, n)? - &*view.dead),
                None if !view.dead.is_empty() || seg.ids.len() > n as usize => {
                    let mut all = RoaringBitmap::new();
                    all.insert_range(0..n);
                    Some(all - &*view.dead)
                }
                None => None,
            };
            if allowed.as_ref().is_some_and(RoaringBitmap::is_empty) {
                continue;
            }
            let mut params = SearchParams {
                filter: allowed.as_ref(),
                ..SearchParams::default()
            };
            params.ef = opts.ef.unwrap_or(params.ef);
            // The re-check guards against an index that ignores the filter:
            // it may then return fewer than k, but never a dead row.
            for h in seg.index.search(query, k, &params)? {
                if allowed.as_ref().is_none_or(|a| a.contains(h.id)) {
                    hits.push((h.dist, s, h.id));
                }
            }
        }
        // ponytail: sequential fan-out; Phase 6.4 parallelizes it.
        hits.sort_by(|a, b| a.0.total_cmp(&b.0));
        hits.truncate(k);
        Ok(hits
            .into_iter()
            .map(|(dist, s, row)| {
                let seg = read(&snap.segs[s].seg);
                Match {
                    id: seg.ids[row as usize],
                    dist,
                    payload: opts.with_payload.then(|| payload(&seg, row)),
                    vector: opts.with_vector.then(|| self.vector(&seg, row)).flatten(),
                }
            })
            .collect())
    }

    /// Writes the mutable segment to disk and truncates the WAL behind it.
    /// Crash-safe at every step: segment files are written first, the new
    /// manifest is the commit point, and covered WAL files go last. Writes
    /// wait for it; searches don't.
    pub fn seal(&self) -> Result<()> {
        let mut w = self.writer()?;
        let last = self.wal.next_lsn() - 1;
        if last == w.covered {
            return Ok(());
        }
        // Syncs the old file and starts LSN last + 1 in a new one, so the
        // files to drop later hold exactly the LSNs <= last.
        self.wal.rotate()?;
        crash_point(1)?;
        let snap = self.snapshot();
        let m = snap.segs.len() - 1;
        let has_rows = snap.segs[m].rows > 0;
        if has_rows {
            self.write_segment(&read(&snap.segs[m].seg), last)?;
        }
        crash_point(2)?;
        let entries: Vec<_> = snap.segs[..m + usize::from(has_rows)]
            .iter()
            .map(|v| (if v.lsn == 0 { last } else { v.lsn }, &*v.dead))
            .collect();
        let path = self.dir.join(MANIFEST);
        framed::write(
            &path,
            MANIFEST_MAGIC,
            MANIFEST_VERSION,
            &encode_manifest(last, &entries)?,
        )
        .map_err(|e| Error::File(path, e))?;
        crash_point(3)?;
        w.covered = last;
        if has_rows {
            let mut next = (*snap).clone();
            next.segs[m].lsn = last;
            next.segs.push(new_mutable(&self.cfg));
            *write(&self.snap) = Arc::new(next);
        }
        self.wal.remove_before(last + 1)?;
        Ok(())
    }

    /// Waits for the write in flight, then fails every later one with
    /// `Closed` (`Db` calls it before deleting the directory).
    pub fn close(&self) {
        lock(&self.writer).closed = true;
    }

    fn writer(&self) -> Result<MutexGuard<'_, Writer>> {
        let w = lock(&self.writer);
        if w.closed {
            return Err(Error::Closed);
        }
        Ok(w)
    }

    fn snapshot(&self) -> Arc<Snapshot> {
        Arc::clone(&read(&self.snap))
    }

    fn vector(&self, seg: &Seg, row: u32) -> Option<Vec<f32>> {
        let (d, r) = (self.cfg.dim, row as usize);
        Some(seg.index.vectors().get(r * d..(r + 1) * d)?.to_vec())
    }

    fn check_dim(&self, v: &[f32]) -> Result<()> {
        if v.len() != self.cfg.dim {
            return Err(index::Error::DimMismatch {
                expected: self.cfg.dim,
                got: v.len(),
            }
            .into());
        }
        Ok(())
    }

    /// Appends to the WAL with one sync. Only a batch rejected up front is
    /// a plain failure; anything else may have reached the disk.
    fn log(&self, records: &[Record]) -> Result<()> {
        match self.wal.append_batch(records) {
            Ok(_) => Ok(()),
            Err(e @ WalError::RecordTooLarge(_)) => Err(e.into()),
            Err(e) => Err(Error::Unknown(e)),
        }
    }

    /// Adds the rows to the mutable segment, then publishes them. The
    /// caller is the writer (or `open`) and has validated the batch, so
    /// nothing here fails in practice.
    fn apply_upserts(&self, ids: &[u64], vectors: &[f32], payloads: &[Value]) -> Result<()> {
        let snap = self.snapshot();
        let m = snap.segs.len() - 1;
        let start = snap.segs[m].rows;
        let d = self.cfg.dim;
        for (i, chunk) in ids.chunks(APPLY_CHUNK).enumerate() {
            let first = i * APPLY_CHUNK;
            let mut seg = write(&snap.segs[m].seg);
            let rows = seg
                .index
                .add(&vectors[first * d..(first + chunk.len()) * d])?;
            for (row, p) in rows.zip(&payloads[first..]) {
                seg.payloads.insert(row, p)?;
            }
            seg.ids.extend_from_slice(chunk);
        }
        let mut next = (*snap).clone();
        let mut live = write(&self.live);
        for (row, &id) in (start..).zip(ids) {
            if let Some((s, r)) = live.insert(id, (m as u32, row)) {
                Arc::make_mut(&mut next.segs[s as usize].dead).insert(r);
            }
        }
        next.segs[m].rows = start + ids.len() as u32;
        *write(&self.snap) = Arc::new(next);
        Ok(())
    }

    fn apply_deletes(&self, ids: &[u64]) {
        let mut next = (*self.snapshot()).clone();
        let mut live = write(&self.live);
        for id in ids {
            if let Some((s, r)) = live.remove(id) {
                Arc::make_mut(&mut next.segs[s as usize].dead).insert(r);
            }
        }
        *write(&self.snap) = Arc::new(next);
    }

    fn write_segment(&self, seg: &Seg, lsn: Lsn) -> Result<()> {
        let path = file_path(&self.dir, lsn, "seg");
        let data = SegmentData {
            dim: self.cfg.dim,
            metric: self.cfg.metric.code(),
            vectors: seg.index.vectors(),
            ids: &seg.ids,
            blob_kind: BlobKind::Hnsw,
            blob: &seg.index.to_bytes(),
        };
        segment::write(&path, &data).map_err(|e| Error::Segment(path, e))?;
        let path = file_path(&self.dir, lsn, "payloads");
        let body = encode_payloads(&seg.payloads, seg.ids.len() as u32);
        framed::write(&path, PAYLOADS_MAGIC, PAYLOADS_VERSION, &body)
            .map_err(|e| Error::File(path, e))
    }
}

fn new_mutable(cfg: &Config) -> View {
    let seg = Seg {
        index: Hnsw::new(cfg.dim, cfg.metric, cfg.hnsw),
        ids: Vec::new(),
        payloads: PayloadStore::with_schema(&cfg.schema),
    };
    View {
        lsn: 0,
        seg: Arc::new(RwLock::new(seg)),
        rows: 0,
        dead: Arc::default(),
    }
}

fn load_segment(dir: &Path, cfg: &Config, lsn: Lsn, dead: &RoaringBitmap) -> Result<Seg> {
    let path = file_path(dir, lsn, "seg");
    let s = Segment::open(&path).map_err(|e| Error::Segment(path.clone(), e))?;
    if s.dim() != cfg.dim || s.metric() != cfg.metric.code() {
        return Err(Error::Invalid(
            "config dim/metric differ from the collection's",
        ));
    }
    if s.blob_kind() != BlobKind::Hnsw {
        return Err(corrupt(&path, "not an HNSW segment"));
    }
    let index = Hnsw::from_bytes(s.index_blob(), s.vectors().to_vec())?;
    let ids = s.ids().to_vec();
    if dead.max().is_some_and(|r| r as usize >= ids.len()) {
        return Err(corrupt(&path, "manifest tombstones a row past the end"));
    }
    let path = file_path(dir, lsn, "payloads");
    let body = framed::read(&path, PAYLOADS_MAGIC, PAYLOADS_VERSION)
        .map_err(|e| Error::File(path.clone(), e))?;
    let payloads = decode_payloads(&body, ids.len() as u32, &cfg.schema)
        .ok_or_else(|| corrupt(&path, "bad payloads"))?;
    Ok(Seg {
        index,
        ids,
        payloads,
    })
}

fn payload(seg: &Seg, row: u32) -> Value {
    seg.payloads.get(row).map_or(Value::Null, Value::Object)
}

// Lock holders don't panic in practice; if one did, its state is no worse
// than the panic left it, so keep serving rather than cascade.
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

fn read<T>(l: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    l.read().unwrap_or_else(PoisonError::into_inner)
}

fn write<T>(l: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    l.write().unwrap_or_else(PoisonError::into_inner)
}

fn corrupt(path: &Path, what: &str) -> Error {
    Error::Corrupt(format!("{}: {what}", path.display()))
}

/// `<dir>/<lsn, 20 digits>.<ext>`, like WAL file names.
fn file_path(dir: &Path, lsn: Lsn, ext: &str) -> PathBuf {
    dir.join(format!("{lsn:020}.{ext}"))
}

/// Deletes `*.tmp` files and segment files the manifest doesn't list:
/// leftovers of a crashed write or seal.
fn remove_garbage(dir: &Path, keep: &[Lsn]) -> Result<()> {
    let mut removed = false;
    for entry in fs::read_dir(dir)? {
        let name = entry?.file_name();
        let Some(name) = name.to_str() else { continue };
        let lsn = name
            .strip_suffix(".seg")
            .or_else(|| name.strip_suffix(".payloads"))
            .filter(|d| d.len() == 20 && d.bytes().all(|b| b.is_ascii_digit()))
            .and_then(|d| d.parse::<Lsn>().ok());
        if name.ends_with(".tmp") || lsn.is_some_and(|l| !keep.contains(&l)) {
            fs::remove_file(dir.join(name))?;
            removed = true;
        }
    }
    if removed {
        File::open(dir)?.sync_all()?;
    }
    Ok(())
}

#[derive(Debug, Default, PartialEq)]
struct Manifest {
    covered: Lsn,
    /// Each sealed segment's LSN and tombstones, ascending.
    segs: Vec<(Lsn, RoaringBitmap)>,
}

// ponytail: every seal rewrites every segment's tombstones; fine while
// they're small, split into per-segment files if seals get slow.
fn encode_manifest(covered: Lsn, segs: &[(Lsn, &RoaringBitmap)]) -> io::Result<Vec<u8>> {
    let mut out = Vec::new();
    out.extend_from_slice(&covered.to_le_bytes());
    out.extend_from_slice(&(segs.len() as u32).to_le_bytes());
    for (lsn, dead) in segs {
        out.extend_from_slice(&lsn.to_le_bytes());
        out.extend_from_slice(&(dead.serialized_size() as u32).to_le_bytes());
        dead.serialize_into(&mut out)?;
    }
    Ok(out)
}

fn decode_manifest(mut b: &[u8]) -> Option<Manifest> {
    let covered = u64_at(&mut b)?;
    let n = u32_at(&mut b)?;
    let mut segs = Vec::new();
    let mut prev = 0;
    for _ in 0..n {
        let lsn = u64_at(&mut b)?;
        if lsn <= prev || lsn > covered {
            return None;
        }
        prev = lsn;
        let len = u32_at(&mut b)? as usize;
        let bytes = take(&mut b, len)?;
        let dead = RoaringBitmap::deserialize_from(bytes).ok()?;
        if dead.serialized_size() != len {
            return None;
        }
        segs.push((lsn, dead));
    }
    b.is_empty().then_some(Manifest { covered, segs })
}

// ponytail: payloads round-trip as JSON text per row, re-parsed on open;
// a columnar section would load faster if segment opens show up in profiles.
fn encode_payloads(store: &PayloadStore, rows: u32) -> Vec<u8> {
    let mut out = rows.to_le_bytes().to_vec();
    for row in 0..rows {
        let json = store
            .get(row)
            .map_or_else(String::new, |m| Value::Object(m).to_string());
        out.extend_from_slice(&(json.len() as u32).to_le_bytes());
        out.extend_from_slice(json.as_bytes());
    }
    out
}

fn decode_payloads(mut b: &[u8], rows: u32, schema: &Schema) -> Option<PayloadStore> {
    if u32_at(&mut b)? != rows {
        return None;
    }
    let mut store = PayloadStore::with_schema(schema);
    for row in 0..rows {
        let len = u32_at(&mut b)? as usize;
        let json = take(&mut b, len)?;
        if !json.is_empty() {
            let v: Value = serde_json::from_slice(json).ok()?;
            store.insert(row, &v).ok()?;
        }
    }
    b.is_empty().then_some(store)
}

fn take<'a>(b: &mut &'a [u8], n: usize) -> Option<&'a [u8]> {
    let (head, rest) = b.split_at_checked(n)?;
    *b = rest;
    Some(head)
}

fn u32_at(b: &mut &[u8]) -> Option<u32> {
    Some(u32::from_le_bytes(take(b, 4)?.try_into().ok()?))
}

fn u64_at(b: &mut &[u8]) -> Option<u64> {
    Some(u64::from_le_bytes(take(b, 8)?.try_into().ok()?))
}

/// Test fail point: `seal` stops with an error right after `step` (1: WAL
/// rotated, 2: segment files written, 3: manifest published), leaving the
/// directory as a crash there would.
fn crash_point(step: u8) -> Result<()> {
    #[cfg(test)]
    if tests::CRASH_AFTER.get() == step {
        return Err(Error::Io(io::Error::other("fail point")));
    }
    let _ = step;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::cell::Cell;
    use std::sync::atomic::{AtomicUsize, Ordering};

    thread_local! {
        pub(super) static CRASH_AFTER: Cell<u8> = const { Cell::new(0) };
    }

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            static N: AtomicUsize = AtomicUsize::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            let p = std::env::temp_dir()
                .join(format!("nearpoint-coll-{tag}-{}-{n}", std::process::id()));
            let _ = fs::remove_dir_all(&p);
            TempDir(p)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    const DIM: usize = 4;

    fn cfg() -> Config {
        Config {
            dim: DIM,
            metric: Metric::L2,
            hnsw: hnsw::Config {
                m: 4,
                ef_construction: 16,
                seed: 1,
            },
            schema: Schema::new(),
            sync: SyncPolicy::Batch,
        }
    }

    fn opts(filter: Option<&Filter>) -> SearchOptions<'_> {
        SearchOptions {
            filter,
            ..SearchOptions::default()
        }
    }

    fn open(d: &TempDir) -> Collection {
        Collection::open(&d.0, cfg()).unwrap().0
    }

    fn vec_for(id: u64, ver: u64) -> Vec<f32> {
        (0..DIM as u64)
            .map(|i| (id * 3 + ver * 7 + i) as f32)
            .collect()
    }

    fn payload(id: u64, ver: u64) -> Value {
        if id.is_multiple_of(4) {
            return Value::Null;
        }
        let color = if id.is_multiple_of(2) { "red" } else { "blue" };
        json!({"ver": ver, "color": color})
    }

    /// Expected live state: id → version.
    type Model = std::collections::BTreeMap<u64, u64>;

    fn check(c: &Collection, model: &Model) {
        assert_eq!(c.len(), model.len());
        for id in 0..40 {
            match model.get(&id) {
                Some(&ver) => {
                    let (v, p) = c.get(id).unwrap_or_else(|| panic!("id {id} missing"));
                    assert_eq!(v, vec_for(id, ver), "id {id}");
                    assert_eq!(p, payload(id, ver), "id {id}");
                }
                None => assert!(c.get(id).is_none(), "id {id} should be gone"),
            }
        }
    }

    /// Rows across all segments, dead or alive: replaying a covered
    /// record twice would add one.
    fn rows(c: &Collection) -> usize {
        let snap = c.snapshot();
        snap.segs.iter().map(|v| read(&v.seg).ids.len()).sum()
    }

    /// Ops 0..n: mostly upserts over ids 0..40, every 5th a delete.
    fn run(c: &mut Collection, model: &mut Model, ops: std::ops::Range<u64>) {
        for i in ops {
            let id = (i * 7) % 40;
            if i % 5 == 4 {
                assert_eq!(c.delete(id).unwrap(), model.remove(&id).is_some());
            } else {
                c.upsert(id, &vec_for(id, i), &payload(id, i)).unwrap();
                model.insert(id, i);
            }
        }
    }

    #[test]
    fn crud_search_and_reopen() {
        let d = TempDir::new("crud");
        let mut c = open(&d);
        let mut model = Model::new();
        run(&mut c, &mut model, 0..60);
        check(&c, &model);
        assert!(c.upsert(1, &[1.0], &Value::Null).is_err()); // wrong dim
        assert!(c.upsert(1, &[f32::NAN; DIM], &Value::Null).is_err());
        assert!(matches!(
            c.upsert(1, &vec_for(1, 0), &json!({"color": 3})),
            Err(Error::Payload(_))
        ));
        assert!(!c.delete(1_000).unwrap());

        let (&id, &ver) = model.iter().next().unwrap();
        let hits = c.search(&vec_for(id, ver), 5, &opts(None)).unwrap();
        assert_eq!((hits[0].id, hits[0].dist), (id, 0.0));
        let all = c.search(&vec_for(0, 0), 100, &opts(None)).unwrap();
        assert!(all.iter().all(|m| model.contains_key(&m.id)));
        let red: Filter = serde_json::from_value(json!({"eq": {"color": "red"}})).unwrap();
        let hits = c.search(&vec_for(0, 0), 100, &opts(Some(&red))).unwrap();
        assert!(!hits.is_empty());
        assert!(
            hits.iter()
                .all(|m| model.contains_key(&m.id) && m.id % 2 == 0)
        );

        let (before, logged) = (rows(&c), c.wal.next_lsn() - 1);
        drop(c);
        let (c, rep) = Collection::open(&d.0, cfg()).unwrap();
        assert_eq!((rep.segments, rep.replayed as u64), (0, logged));
        assert_eq!(rows(&c), before);
        check(&c, &model);
    }

    #[test]
    fn seal_keeps_payloads_and_tombstones_across_restart() {
        let d = TempDir::new("seal");
        let mut c = open(&d);
        let mut model = Model::new();
        run(&mut c, &mut model, 0..50);
        c.seal().unwrap();
        c.seal().unwrap(); // nothing new: no-op
        run(&mut c, &mut model, 50..90); // overwrites and deletes sealed rows
        c.seal().unwrap();
        for _ in 0..2 {
            // Deletes only: the seal persists tombstones, no new segment.
            let (id, _) = model.pop_first().unwrap();
            assert!(c.delete(id).unwrap());
        }
        c.seal().unwrap();
        run(&mut c, &mut model, 90..100); // left in the WAL
        check(&c, &model);
        let (before, unsealed) = (rows(&c), c.wal.next_lsn() - 1 - lock(&c.writer).covered);
        let wal_files = fs::read_dir(d.0.join("wal")).unwrap().count();
        assert_eq!(wal_files, 1, "sealed WAL files are dropped");
        drop(c);

        let (c, rep) = Collection::open(&d.0, cfg()).unwrap();
        assert_eq!(rep.segments, 2);
        assert_eq!(rep.replayed as u64, unsealed);
        assert_eq!(rows(&c), before);
        check(&c, &model);
    }

    /// A crash after each seal step: reopening must give the same state
    /// with no row applied twice, and a later seal must still work.
    #[test]
    fn crash_between_seal_steps() {
        for step in 1..=3 {
            let d = TempDir::new("crash-seal");
            let mut c = open(&d);
            let mut model = Model::new();
            run(&mut c, &mut model, 0..30);
            c.seal().unwrap();
            run(&mut c, &mut model, 30..55);
            let rows_before = rows(&c);
            CRASH_AFTER.set(step);
            assert!(c.seal().is_err(), "step {step}");
            CRASH_AFTER.set(0);
            drop(c);

            let (mut c, rep) = Collection::open(&d.0, cfg()).unwrap();
            check(&c, &model);
            assert_eq!(rows(&c), rows_before, "step {step}: double-applied");
            assert_eq!(rep.segments, if step == 3 { 2 } else { 1 }, "step {step}");
            let names: Vec<_> = fs::read_dir(&d.0)
                .unwrap()
                .map(|e| e.unwrap().file_name().into_string().unwrap())
                .collect();
            assert_eq!(names.len(), 2 * rep.segments + 2, "step {step}: {names:?}");
            run(&mut c, &mut model, 55..70);
            c.seal().unwrap();
            drop(c);
            check(&open(&d), &model);
        }
    }

    #[test]
    fn leftover_tmp_and_orphan_files_are_removed() {
        let d = TempDir::new("garbage");
        let mut c = open(&d);
        let mut model = Model::new();
        run(&mut c, &mut model, 0..10);
        c.seal().unwrap();
        let lsn = format!("{:020}", lock(&c.writer).covered);
        drop(c);
        for name in [
            "MANIFEST.tmp",
            "00000000000000000099.seg",
            "00000000000000000099.payloads.tmp",
            "notes.txt",
        ] {
            fs::write(d.0.join(name), b"junk").unwrap();
        }
        let c = open(&d);
        check(&c, &model);
        let mut names: Vec<_> = fs::read_dir(&d.0)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        assert_eq!(
            names,
            [
                format!("{lsn}.payloads"),
                format!("{lsn}.seg"),
                "MANIFEST".into(),
                "notes.txt".into(),
                "wal".into()
            ]
        );
    }

    #[test]
    fn reopen_with_other_dim_is_an_error() {
        let d = TempDir::new("dim");
        let mut c = open(&d);
        run(&mut c, &mut Model::new(), 0..3);
        c.seal().unwrap();
        drop(c);
        let other = Config { dim: 3, ..cfg() };
        assert!(matches!(
            Collection::open(&d.0, other),
            Err(Error::Invalid(_))
        ));
    }

    #[test]
    fn wal_behind_manifest_is_corrupt() {
        let d = TempDir::new("behind");
        let mut c = open(&d);
        run(&mut c, &mut Model::new(), 0..3);
        c.seal().unwrap();
        drop(c);
        fs::remove_dir_all(d.0.join("wal")).unwrap();
        assert!(matches!(open_err(&d), Error::Corrupt(_)));
    }

    fn open_err(d: &TempDir) -> Error {
        Collection::open(&d.0, cfg()).err().unwrap()
    }

    #[test]
    fn batches_are_checked_whole_and_replayed_in_order() {
        let d = TempDir::new("batch");
        let c = open(&d);
        let vecs: Vec<f32> = [1, 2, 1].iter().flat_map(|&i| vec_for(i, i)).collect();
        // The third payload conflicts with the first: nothing is written.
        let bad = [json!({"n": 1}), Value::Null, json!({"n": "x"})];
        let e = c.upsert_batch(&[1, 2, 1], &vecs, &bad);
        assert!(matches!(e, Err(Error::Payload(_))));
        assert!(c.upsert_batch(&[1, 2], &vecs, &[]).is_err()); // short ids
        assert_eq!((c.len(), c.wal.next_lsn()), (0, 1));
        let good = [json!({"n": 1}), Value::Null, json!({"n": 3})];
        c.upsert_batch(&[1, 2, 1], &vecs, &good).unwrap();
        assert_eq!(c.wal.next_lsn(), 4);
        assert_eq!(c.get(1).unwrap(), (vec_for(1, 1), json!({"n": 3})));
        assert_eq!(c.delete_batch(&[1, 1, 9]).unwrap(), 1);
        assert_eq!(c.upsert_batch(&[], &[], &[]).ok(), Some(()));
        drop(c);
        let (c, rep) = Collection::open(&d.0, cfg()).unwrap();
        assert_eq!(rep.replayed, 4);
        assert_eq!(rows(&c), 3);
        assert!(c.get(1).is_none());
        assert_eq!(c.get(2).unwrap(), (vec_for(2, 2), Value::Null));
    }

    /// One writer upserting, replacing, deleting and sealing while readers
    /// search: a delete acknowledged before a search started is never in
    /// its results, and no id appears twice.
    #[test]
    fn readers_never_see_acknowledged_deletes() {
        use std::sync::atomic::{AtomicBool, AtomicU64};
        let d = TempDir::new("concurrent");
        let c = open(&d);
        // id → sequence number of its delete's ack (ids never come back).
        let deleted = RwLock::new(HashMap::<u64, u64>::new());
        let (acks, done) = (AtomicU64::new(0), AtomicBool::new(false));
        std::thread::scope(|s| {
            for r in 0..3u64 {
                let (c, deleted, acks, done) = (&c, &deleted, &acks, &done);
                s.spawn(move || {
                    let mut searches = 0;
                    while !done.load(Ordering::SeqCst) || searches < 10 {
                        let seen = acks.load(Ordering::SeqCst);
                        let q = vec_for(r * 11 + searches, 0);
                        let hits = c.search(&q, 20, &opts(None)).unwrap();
                        let dead = read(deleted);
                        let mut ids = HashSet::new();
                        for m in &hits {
                            assert!(ids.insert(m.id), "id {} twice", m.id);
                            if let Some(&at) = dead.get(&m.id) {
                                assert!(at > seen, "id {} deleted at {at}, search at {seen}", m.id);
                            }
                        }
                        searches += 1;
                    }
                });
            }
            for round in 0..60u64 {
                let ids: Vec<u64> = (round * 50..round * 50 + 50).collect();
                let vecs: Vec<f32> = ids.iter().flat_map(|&i| vec_for(i, round)).collect();
                c.upsert_batch(&ids, &vecs, &[]).unwrap();
                // Replace some earlier live ids (odd), delete others (even).
                let old: Vec<u64> = (0..round * 50).filter(|i| i % 7 == round % 7).collect();
                let (gone, keep): (Vec<u64>, Vec<u64>) = old.iter().partition(|&&i| i % 2 == 0);
                let vecs: Vec<f32> = keep.iter().flat_map(|&i| vec_for(i, round + 1)).collect();
                c.upsert_batch(&keep, &vecs, &[]).unwrap();
                c.delete_batch(&gone).unwrap();
                let at = acks.fetch_add(1, Ordering::SeqCst) + 1;
                write(&deleted).extend(gone.iter().map(|&i| (i, at)));
                if round % 20 == 19 {
                    c.seal().unwrap();
                }
            }
            done.store(true, Ordering::SeqCst);
        });
        let live = c.len();
        drop(c);
        assert_eq!(open(&d).len(), live);
    }

    #[test]
    fn torn_wal_append_is_unknown_and_recovers() {
        use storage::wal::failpoint::tear_next_append;
        for cut in [0, 1, 17, 40] {
            let d = TempDir::new("torn");
            let mut c = open(&d);
            let mut model = Model::new();
            run(&mut c, &mut model, 0..10);
            tear_next_append(cut);
            let e = c.upsert(39, &vec_for(39, 99), &payload(39, 99));
            assert!(matches!(e, Err(Error::Unknown(WalError::FailPoint))));
            // Poisoned: every later write is unknown too, reads still work.
            assert!(matches!(
                c.delete(*model.keys().next().unwrap()),
                Err(Error::Unknown(_))
            ));
            assert!(c.seal().is_err());
            check(&c, &model);
            drop(c);
            let (mut c, rep) = Collection::open(&d.0, cfg()).unwrap();
            assert_eq!(rep.dropped_bytes, cut as u64);
            check(&c, &model);
            run(&mut c, &mut model, 10..20);
            drop(c);
            check(&open(&d), &model);
        }
    }

    /// Committed v1 files: the formats must keep reading them.
    #[test]
    fn v1_fixtures_still_load_and_encode_identically() {
        let dir = TempDir::new("fixture");
        fs::create_dir_all(&dir.0).unwrap();
        let mut dead = RoaringBitmap::new();
        dead.insert(1);
        dead.insert(5);
        let m = encode_manifest(9, &[(4, &dead), (9, &RoaringBitmap::new())]).unwrap();
        let p = dir.0.join("MANIFEST");
        framed::write(&p, MANIFEST_MAGIC, MANIFEST_VERSION, &m).unwrap();
        let fixture: &[u8] = include_bytes!("../testdata/manifest-v1");
        assert_eq!(fs::read(&p).unwrap(), fixture);
        let want = Manifest {
            covered: 9,
            segs: vec![(4, dead), (9, RoaringBitmap::new())],
        };
        let body = framed::read(&p, MANIFEST_MAGIC, MANIFEST_VERSION).unwrap();
        assert_eq!(decode_manifest(&body), Some(want));

        let mut store = PayloadStore::new();
        let rows = [
            json!({"n": 1, "tag": ["a", "b"]}),
            Value::Null,
            json!({"f": 2.5}),
        ];
        for (r, v) in rows.iter().enumerate() {
            store.insert(r as u32, v).unwrap();
        }
        let p = dir.0.join("x.payloads");
        let body = encode_payloads(&store, 3);
        framed::write(&p, PAYLOADS_MAGIC, PAYLOADS_VERSION, &body).unwrap();
        let fixture: &[u8] = include_bytes!("../testdata/payloads-v1");
        assert_eq!(fs::read(&p).unwrap(), fixture);
        let body = framed::read(&p, PAYLOADS_MAGIC, PAYLOADS_VERSION).unwrap();
        let back = decode_payloads(&body, 3, &Schema::new()).unwrap();
        for (r, v) in rows.iter().enumerate() {
            assert_eq!(back.get(r as u32), v.as_object().cloned());
        }
        assert!(decode_payloads(&body, 4, &Schema::new()).is_none());
    }
}
