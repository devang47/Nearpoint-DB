//! A collection (tasks 3.3–3.4, 4.2–4.3, 6.1–6.4): a WAL, one mutable
//! in-RAM HNSW segment, sealed segments, an external-id map and tombstones,
//! all in one directory. Layout, LSN coverage and the seal protocol are in
//! docs/adr/0008-collection-recovery.md; sealing at a size, the builder,
//! compaction and fan-out in docs/adr/0012-segments-and-compaction.md; file
//! formats in docs/formats/manifest.md.
//!
//! A write is logged to the WAL first and applied in memory only once the
//! WAL acknowledged it. A seal freezes the mutable segment (WAL rotated, a
//! fresh mutable segment swapped in, so writes go on), writes it to files
//! with no lock held, then publishes a new manifest (the commit point) and
//! deletes the WAL files the manifest covers. `open` loads the manifest's
//! segments and replays only WAL records above the manifest's LSN, so a
//! crash at any step neither loses nor double-applies a write.
//!
//! Concurrency (ADR 0011): one writer at a time (`writer` mutex, held
//! through the WAL sync), any number of readers. Readers clone the current
//! `Arc<Snapshot>` and search it without blocking on the writer's fsync.
//! The writer appends rows to the mutable segment under its `RwLock` (in
//! chunks, so readers wait at most one chunk), then publishes a new
//! snapshot whose row count covers them and whose tombstone bitmaps are
//! copy-on-write clones. Rows past a snapshot's count are invisible to it.
//! Background work (`maintain`: seal, build, compact) runs one step at a
//! time under the `maint` mutex and takes the writer lock only to publish.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{
    Arc, Mutex, MutexGuard, OnceLock, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard,
};
use std::time::Instant;

use index::{Hnsw, Index, IvfPq, Metric, SearchParams, hnsw, ivfpq};
use rayon::prelude::*;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use storage::framed;
use storage::segment::{self, BlobKind, Segment, SegmentData};
use storage::wal::{Lsn, Record, SyncPolicy, Wal, WalError};

use crate::{Filter, PayloadStore, Schema, evaluate};

const MANIFEST: &str = "MANIFEST";
const MANIFEST_MAGIC: [u8; 8] = *b"NPMANIFS";
/// v2: segments are named by a file id from a counter (v1: by LSN).
const MANIFEST_VERSION: u32 = 2;
const MANIFEST_V1: u32 = 1;
const PAYLOADS_MAGIC: [u8; 8] = *b"NPPAYLDS";
/// v2: columnar binary (v1: one JSON text per row, read-only).
const PAYLOADS_VERSION: u32 = 2;
const PAYLOADS_V1: u32 = 1;
/// Rows the writer adds to the mutable segment per write-lock hold.
const APPLY_CHUNK: usize = 1024;
/// WAL records replayed per batch on open (bounds the extra copy).
const REPLAY_CHUNK: usize = 65_536;
/// Compaction (ADR 0012): a sealed segment with at least this fraction of
/// dead rows is rewritten without them.
const DEAD_FRACTION: f64 = 0.25;
/// Compaction: this many segments of one size tier merge into one.
const MERGE_FACTOR: u64 = 4;
/// IVF-PQ search defaults: 0.92 recall@10 on BIGANN 10M (RESULTS.md).
pub const DEFAULT_NPROBE: usize = 32;
pub const DEFAULT_RERANK: usize = 100;
/// Smallest `ivf_rows`: 8-bit PQ trains 256 centroids per subspace.
const MIN_IVF_ROWS: u32 = 1024;
/// Largest accepted vector component: keeps squared distances finite.
const MAX_COMPONENT: f32 = 1e15;
/// A merge's f32 copy at most this big (4M rows at dim 128).
const MERGE_BYTES: u64 = 2 << 30;

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
    /// `maintain` seals the mutable segment once it has this many rows.
    #[serde(default = "default_seal_rows")]
    pub seal_rows: u32,
    /// Sealed segments with at least this many rows are rebuilt as IVF-PQ
    /// (L2 and cosine; inner product stays HNSW and merges stay below it).
    #[serde(default = "default_ivf_rows")]
    pub ivf_rows: u32,
    /// Compaction never merges segments into one bigger than this.
    #[serde(default = "default_max_merge_rows")]
    pub max_merge_rows: u32,
    /// Not persisted: the server picks it at startup.
    #[serde(skip)]
    pub sync: SyncPolicy,
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        if self.dim == 0 || self.hnsw.m < 2 || self.hnsw.ef_construction == 0 {
            return Err(Error::Invalid("need dim > 0, m >= 2, ef_construction >= 1"));
        }
        if self.seal_rows == 0 || self.max_merge_rows == 0 || self.ivf_rows < MIN_IVF_ROWS {
            return Err(Error::Invalid(
                "need seal_rows >= 1, max_merge_rows >= 1, ivf_rows >= 1024",
            ));
        }
        // Else fresh seals start at the merge cap and never merge, so the
        // segment count (and search fan-out) grows without bound.
        let ip_cap = if self.metric == Metric::InnerProduct {
            self.ivf_rows
        } else {
            u32::MAX
        };
        if u64::from(self.seal_rows) * 2 > u64::from(self.max_merge_rows.min(ip_cap)) {
            return Err(Error::Invalid(
                "need seal_rows <= max_merge_rows / 2 (and ivf_rows / 2 for inner product)",
            ));
        }
        Ok(())
    }
}

/// ADR 0012: bounds WAL replay after a crash and the mutable HNSW's RAM,
/// while a seal still writes few enough files for merges to keep up.
fn default_seal_rows() -> u32 {
    100_000
}

/// Below ~1M rows HNSW is faster and more accurate and fits in RAM; IVF-PQ
/// pays off above it (SIFT1M IVF-PQ reached 0.92, RESULTS.md).
fn default_ivf_rows() -> u32 {
    1_000_000
}

/// A merge holds the rows' f32 vectors plus the new index (twice while it
/// is written): ~2.6 GB at 4M x 128 dims, inside the 10 GB budget.
fn default_max_merge_rows() -> u32 {
    4_000_000
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

/// Threads for background index builds; 0 (the default) means half the
/// cores, so builds don't starve searches. Takes effect only before the
/// first build.
pub fn set_build_threads(n: usize) {
    BUILD_THREADS.store(n, Ordering::Relaxed);
}

static BUILD_THREADS: AtomicUsize = AtomicUsize::new(0);
static BUILD_POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

fn build_pool() -> Result<&'static rayon::ThreadPool> {
    if let Some(p) = BUILD_POOL.get() {
        return Ok(p);
    }
    let n = match BUILD_THREADS.load(Ordering::Relaxed) {
        0 => std::thread::available_parallelism().map_or(1, |n| (n.get() / 2).max(1)),
        n => n,
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .thread_name(|i| format!("nearpoint-build-{i}"))
        .build()
        .map_err(io::Error::other)?;
    Ok(BUILD_POOL.get_or_init(|| pool))
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
    /// IVF-PQ lists probed in a segment of `ivf_rows` rows (`None`:
    /// [`DEFAULT_NPROBE`]); bigger segments probe the same share of their
    /// lists ([`segment_nprobe`]).
    pub nprobe: Option<usize>,
    /// IVF-PQ candidates re-ranked with int8 vectors (`None`:
    /// [`DEFAULT_RERANK`]; at least `k` either way).
    pub rerank: Option<usize>,
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

/// One segment as `Collection::segments` reports it.
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentInfo {
    /// `"mutable"`, `"hnsw"` or `"ivf_pq"`.
    pub index: &'static str,
    pub rows: u32,
    pub dead: u64,
}

/// A segment's index. IVF-PQ keeps its f32 vectors in the mapped file
/// (not in RAM): `get` stays exact and Phase 8 can re-rank from them.
enum SegIndex {
    Hnsw(Hnsw),
    IvfPq(IvfPq, Segment),
}

/// One segment's data. A sealed one never changes; the mutable one only
/// grows, under its write lock, which only the writer takes.
struct Seg {
    index: SegIndex,
    /// External id of each row.
    ids: Vec<u64>,
    payloads: PayloadStore,
}

impl Seg {
    fn index(&self) -> &dyn Index {
        match &self.index {
            SegIndex::Hnsw(h) => h,
            SegIndex::IvfPq(i, _) => i,
        }
    }

    /// Row `row`'s vector in stored form.
    fn vector(&self, dim: usize, row: u32) -> Option<&[f32]> {
        let all = match &self.index {
            SegIndex::Hnsw(h) => h.vectors(),
            SegIndex::IvfPq(_, file) => file.vectors(),
        };
        all.get(row as usize * dim..(row as usize + 1) * dim)
    }
}

/// A segment as one snapshot sees it.
#[derive(Clone)]
struct View {
    /// Never reused within a process: what `live` points at, so removing
    /// or merging segments only touches the ids that moved.
    key: u32,
    /// Its file id in the manifest (`<file>.seg`); 0 while it isn't in
    /// one (the mutable segment, or a frozen one whose seal is pending).
    file: u64,
    seg: Arc<RwLock<Seg>>,
    /// Rows `0..rows` are visible. The mutable segment can hold more:
    /// applied by the writer but not yet published.
    rows: u32,
    /// Rows deleted or replaced by a later upsert. Copy-on-write: the
    /// writer clones it (`Arc::make_mut`) into the next snapshot.
    dead: Arc<RoaringBitmap>,
}

/// What readers search: sealed segments, then the mutable one (always
/// last). Never changed once published; the writer and the background
/// work (holding the writer lock) swap in a new one.
#[derive(Clone)]
struct Snapshot {
    segs: Vec<View>,
}

impl Snapshot {
    fn pos(&self, key: u32) -> Option<usize> {
        self.segs.iter().position(|v| v.key == key)
    }

    fn mutable(&self) -> &View {
        &self.segs[self.segs.len() - 1]
    }
}

/// State only the writer touches.
struct Writer {
    /// Every write with LSN <= this is in the manifest.
    covered: Lsn,
    /// Set by `close`: later writes fail with `Closed`.
    closed: bool,
}

/// State of the background work, one step at a time.
struct Maint {
    /// The next segment file id.
    next_file: u64,
    /// A frozen segment whose seal hasn't published yet: it is finished
    /// before anything else (so `covered` only grows).
    pending: Option<Pending>,
    /// Set by `close`: no more background work.
    closed: bool,
}

#[derive(Clone, Copy)]
struct Pending {
    key: u32,
    file: u64,
    /// The WAL's last LSN at the freeze: the manifest's `covered` once
    /// this segment is in it.
    covered: Lsn,
}

pub struct Collection {
    dir: PathBuf,
    cfg: Config,
    wal: Wal,
    /// Held by each write for its whole run, WAL sync included, and by
    /// background work to freeze or publish. Lock order: maint, writer,
    /// live, snap, a segment.
    // ponytail: one sync per request, so concurrent single-point writers
    // don't share a group commit; batch (or bulk) to go faster. Collect
    // pending requests into one WAL batch if that becomes the bottleneck.
    writer: Mutex<Writer>,
    /// Held for a whole background step (seal, build or compaction).
    maint: Mutex<Maint>,
    /// External id → (segment key, row) of its one live row. Changed only
    /// by the writer (or a publish holding the writer lock), together with
    /// `snap`, under this lock, so a reader holding it sees the two agree.
    live: RwLock<HashMap<u64, (u32, u32)>>,
    /// Held only to clone or swap the `Arc`.
    snap: RwLock<Arc<Snapshot>>,
    next_key: AtomicU32,
}

impl Collection {
    /// Opens (or creates) the collection in `dir`: loads the manifest's
    /// segments, deletes files a crashed seal left behind, and replays WAL
    /// records the manifest doesn't cover. One process at a time.
    pub fn open(dir: impl AsRef<Path>, cfg: Config) -> Result<(Collection, Recovery)> {
        cfg.validate()?;
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;
        let manifest = read_manifest(&dir.join(MANIFEST))?;
        let keep: Vec<u64> = manifest.segs.iter().map(|s| s.0).collect();
        remove_garbage(&dir, &keep)?;

        let mut segs = Vec::new();
        let mut live = HashMap::new();
        for (key, (file, dead)) in (0u32..).zip(manifest.segs) {
            let seg = load_segment(&dir, &cfg, file, &dead)?;
            for (row, &id) in (0u32..).zip(&seg.ids) {
                if !dead.contains(row) && live.insert(id, (key, row)).is_some() {
                    return Err(Error::Corrupt(format!("id {id} is live in two rows")));
                }
            }
            segs.push(View {
                key,
                file,
                rows: seg.ids.len() as u32,
                seg: Arc::new(RwLock::new(seg)),
                dead: Arc::new(dead),
            });
        }
        let key = segs.len() as u32;
        segs.push(new_mutable(&cfg, key));

        let (wal, rec) = Wal::open(dir.join("wal"), cfg.sync)?;
        let c = Collection {
            dir,
            cfg,
            wal,
            writer: Mutex::new(Writer {
                covered: manifest.covered,
                closed: false,
            }),
            maint: Mutex::new(Maint {
                next_file: keep.iter().max().map_or(1, |m| m + 1),
                pending: None,
                closed: false,
            }),
            live: RwLock::new(live),
            snap: RwLock::new(Arc::new(Snapshot { segs })),
            next_key: AtomicU32::new(key + 1),
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
        self.check_vectors(vectors)?;
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
            let seg = read(&snap.mutable().seg);
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
        let &(key, row) = live.get(&id)?;
        let snap = self.snapshot();
        let seg = read(&snap.segs[snap.pos(key)?].seg);
        Some((seg.vector(self.cfg.dim, row)?.to_vec(), payload(&seg, row)))
    }

    /// Live ids.
    pub fn len(&self) -> usize {
        read(&self.live).len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The current segments, mutable one last.
    pub fn segments(&self) -> Vec<SegmentInfo> {
        let snap = self.snapshot();
        let last = snap.segs.len() - 1;
        snap.segs
            .iter()
            .enumerate()
            .map(|(i, v)| SegmentInfo {
                // The mutable segment's lock is skipped: the writer holds it.
                index: if i == last {
                    "mutable"
                } else {
                    match read(&v.seg).index {
                        SegIndex::Hnsw(_) => "hnsw",
                        SegIndex::IvfPq(..) => "ivf_pq",
                    }
                },
                rows: v.rows,
                dead: v.dead.len(),
            })
            .collect()
    }

    /// The `k` nearest live ids matching `opts.filter`, nearest first.
    /// Segments are searched in parallel (rayon) and their hits merged.
    pub fn search(&self, query: &[f32], k: usize, opts: &SearchOptions) -> Result<Vec<Match>> {
        self.check_dim(query)?;
        self.check_vectors(query)?;
        let snap = self.snapshot();
        let one = |(s, view): (usize, &View)| -> Result<Vec<(f32, usize, u32)>> {
            let n = view.rows;
            if n == 0 {
                return Ok(Vec::new());
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
                return Ok(Vec::new());
            }
            let d = SearchParams::default();
            let nprobe = opts.nprobe.unwrap_or(DEFAULT_NPROBE);
            let params = SearchParams {
                ef: opts.ef.unwrap_or(d.ef),
                nprobe: match &seg.index {
                    SegIndex::IvfPq(i, _) => {
                        segment_nprobe(nprobe, i.config().nlist, self.cfg.ivf_rows)
                    }
                    SegIndex::Hnsw(_) => nprobe,
                },
                rerank: opts.rerank.unwrap_or(DEFAULT_RERANK),
                filter: allowed.as_ref(),
            };
            // The re-check guards against an index that ignores the filter:
            // it may then return fewer than k, but never a dead row.
            Ok(seg
                .index()
                .search(query, k, &params)?
                .into_iter()
                .filter(|h| allowed.as_ref().is_none_or(|a| a.contains(h.id)))
                .map(|h| (h.dist, s, h.id))
                .collect())
        };
        // Sealed segments in parallel. The mutable one (last) runs here,
        // after them, never on a rayon worker: the writer holds its write
        // lock while `Hnsw::add` waits on rayon jobs, so a worker blocked
        // on that lock could deadlock the pool.
        let last = snap.segs.len() - 1;
        let sealed = &snap.segs[..last];
        let mut per: Vec<Vec<_>> = if sealed.iter().filter(|v| v.rows > 0).count() > 1 {
            sealed
                .par_iter()
                .enumerate()
                .map(one)
                .collect::<Result<_>>()?
        } else {
            sealed.iter().enumerate().map(one).collect::<Result<_>>()?
        };
        per.push(one((last, &snap.segs[last]))?);
        let mut hits: Vec<_> = per.into_iter().flatten().collect();
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
                    vector: opts
                        .with_vector
                        .then(|| seg.vector(self.cfg.dim, row).map(<[f32]>::to_vec))
                        .flatten(),
                }
            })
            .collect())
    }

    /// Seals everything written so far: freezes the mutable segment, writes
    /// it to files (writes go on meanwhile), publishes the manifest and
    /// truncates the WAL behind it. Crash-safe at every step (ADR 0008,
    /// 0012). Waits for a background step in flight.
    pub fn seal(&self) -> Result<()> {
        let mut m = lock(&self.maint);
        if m.closed {
            return Err(Error::Closed);
        }
        self.seal_locked(&mut m)
    }

    /// Does one unit of background work and returns whether it did any:
    /// seals the mutable segment once it has `seal_rows` rows, else runs
    /// one build or compaction step (policy in ADR 0012). `Db` calls it
    /// from its background thread; tests call it directly.
    pub fn maintain(&self) -> Result<bool> {
        let mut m = lock(&self.maint);
        if m.closed {
            return Ok(false);
        }
        let snap = self.snapshot();
        if m.pending.is_some() || snap.mutable().rows >= self.cfg.seal_rows {
            self.seal_locked(&mut m)?;
            return Ok(true);
        }
        let Some(inputs) = self.plan(&snap) else {
            return Ok(false);
        };
        self.rewrite(&mut m, &snap, &inputs)?;
        Ok(true)
    }

    /// Waits for the write and background step in flight, then fails every
    /// later write with `Closed` and stops background work (`Db` calls it
    /// before deleting the directory).
    pub fn close(&self) {
        lock(&self.writer).closed = true;
        lock(&self.maint).closed = true;
    }

    fn seal_locked(&self, m: &mut Maint) -> Result<()> {
        self.finish_seal(m)?;
        self.freeze(m)?;
        self.finish_seal(m)
    }

    /// Seal, part 1 (writer lock): rotate the WAL at the last LSN and swap
    /// in a fresh mutable segment, leaving the old one pending.
    fn freeze(&self, m: &mut Maint) -> Result<()> {
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
        let frozen = snap.mutable();
        if frozen.rows == 0 {
            // Only deletes since the last seal: persist their tombstones.
            self.publish(&mut w, last, &snap, 3)?;
            drop(w);
            self.wal.remove_before(last + 1)?;
            return Ok(());
        }
        let mut next = (*snap).clone();
        next.segs.push(new_mutable(&self.cfg, self.new_key()));
        *write(&self.snap) = Arc::new(next);
        m.pending = Some(Pending {
            key: frozen.key,
            file: m.next_file,
            covered: last,
        });
        m.next_file += 1;
        Ok(())
    }

    /// Seal, part 2 (no lock) and 3 (writer lock): write the frozen
    /// segment's files, then publish it.
    fn finish_seal(&self, m: &mut Maint) -> Result<()> {
        let Some(p) = m.pending else { return Ok(()) };
        let t = Instant::now();
        let snap = self.snapshot();
        let pos = snap
            .pos(p.key)
            .ok_or(Error::Invalid("pending segment is gone"))?;
        {
            let seg = read(&snap.segs[pos].seg);
            let SegIndex::Hnsw(h) = &seg.index else {
                return Err(Error::Invalid("pending segment is not HNSW"));
            };
            let blob = h.to_bytes();
            self.write_files(
                p.file,
                h.vectors(),
                &seg.ids,
                &seg.payloads,
                BlobKind::Hnsw,
                &blob,
            )?;
        }
        crash_point(2)?;
        let mut w = self.writer()?;
        let mut next = (*self.snapshot()).clone();
        if let Some(pos) = next.pos(p.key) {
            next.segs[pos].file = p.file;
        }
        self.publish(&mut w, p.covered, &next, 3)?;
        *write(&self.snap) = Arc::new(next);
        m.pending = None;
        drop(w);
        self.wal.remove_before(p.covered + 1)?;
        tracing::info!(ms = t.elapsed().as_millis(), "seal");
        Ok(())
    }

    /// Writes the manifest for `next` (the commit point). Caller holds the
    /// writer lock and swaps `next` in afterwards.
    fn publish(&self, w: &mut Writer, covered: Lsn, next: &Snapshot, step: u8) -> Result<()> {
        // The tombstones include rows killed by WAL records above
        // `covered`. Those records must be durable before the manifest
        // says so: a crash then replays them (idempotently) instead of
        // losing the delete or upsert while its tombstone survives.
        self.wal.sync()?;
        let mut entries: Vec<_> = next
            .segs
            .iter()
            .filter(|v| v.file != 0)
            .map(|v| (v.file, &*v.dead))
            .collect();
        entries.sort_by_key(|e| e.0);
        let path = self.dir.join(MANIFEST);
        let body = encode_manifest(covered, &entries)?;
        framed::write(&path, MANIFEST_MAGIC, MANIFEST_VERSION, &body)
            .map_err(|e| Error::File(path, e))?;
        crash_point(step)?;
        w.covered = covered;
        Ok(())
    }

    /// The compaction policy (ADR 0012); `None` if nothing is due. Returns
    /// positions in `snap` of sealed segments to rewrite into one.
    fn plan(&self, snap: &Snapshot) -> Option<Vec<usize>> {
        let sealed: Vec<usize> = (0..snap.segs.len())
            .filter(|&i| snap.segs[i].file != 0)
            .collect();
        let live = |i: usize| u64::from(snap.segs[i].rows) - snap.segs[i].dead.len();
        let dead_share = |i: usize| snap.segs[i].dead.len() as f64 / f64::from(snap.segs[i].rows);
        let is_ivf = |i: usize| matches!(read(&snap.segs[i].seg).index, SegIndex::IvfPq(..));
        // 0. Absorb: fold small sealed HNSW segments into the IVF-PQ segment
        // with the most dead rows that has room. Its live codes are kept and
        // only the new rows are encoded, so this costs ~a seal, and it drops
        // that segment's dead rows on the way. Without it, small segments
        // merged into an HNSW tier (7-8 s per 330k graph) and then into a
        // freshly trained IVF-PQ (14-17 s): the soak's p99 windows.
        let cap = self.merge_cap();
        if self.ivf_allowed() {
            let small: Vec<usize> = sealed
                .iter()
                .copied()
                .filter(|&i| !is_ivf(i) && live(i) < u64::from(self.cfg.ivf_rows))
                .collect();
            let target = sealed
                .iter()
                .copied()
                .filter(|&i| is_ivf(i) && live(i) < cap)
                .max_by(|&a, &b| dead_share(a).total_cmp(&dead_share(b)));
            // Waits for MERGE_FACTOR - 1 of them: recent rows stay in small
            // HNSW segments a while (faster and more accurate than IVF-PQ at
            // 100k; absorbing each seal at once cost ~0.4 ms of p99 and ~3%
            // recall in a soak), and the tier merge (MERGE_FACTOR) never wins.
            if let Some(t) = target
                && small.len() as u64 >= MERGE_FACTOR - 1
            {
                let mut kinds = self.cfg.schema.clone();
                merge_kinds(&mut kinds, &read(&snap.segs[t].seg).payloads);
                let (mut pick, mut sum) = (vec![t], live(t));
                for i in small {
                    if sum + live(i) <= cap
                        && merge_kinds(&mut kinds, &read(&snap.segs[i].seg).payloads)
                    {
                        sum += live(i);
                        pick.push(i);
                    }
                }
                // Only once it changes >= 1/MERGE_FACTOR of the result (new
                // plus dropped rows): bounds the rewrite's write amplification
                // (the f32 section is rewritten whole) for big, wide segments.
                let changed = sum - live(t) + snap.segs[t].dead.len();
                if pick.len() > 1 && changed * MERGE_FACTOR >= sum {
                    return Some(pick);
                }
            }
        }
        // 1. Drop dead rows from the segment that has the most, by share.
        if let Some(&i) = sealed
            .iter()
            .filter(|&&i| dead_share(i) >= DEAD_FRACTION)
            .max_by(|&&a, &&b| dead_share(a).total_cmp(&dead_share(b)))
        {
            return Some(vec![i]);
        }
        // 2. Build: a big HNSW segment becomes IVF-PQ.
        let ivf_rows = u64::from(self.cfg.ivf_rows);
        if self.ivf_allowed()
            && let Some(&i) = sealed.iter().find(|&&i| {
                live(i) >= ivf_rows && matches!(read(&snap.segs[i].seg).index, SegIndex::Hnsw(_))
            })
        {
            return Some(vec![i]);
        }
        // 3. Merge MERGE_FACTOR segments of the lowest size tier that has
        // that many, smallest first, up to the cap and only with payload
        // columns of matching types.
        let mut cap = cap;
        if !self.ivf_allowed() {
            cap = cap.min(ivf_rows - 1);
        }
        let seal = u64::from(self.cfg.seal_rows);
        let mut tiers: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for &i in &sealed {
            if live(i) < cap {
                tiers.entry(tier(live(i), seal)).or_default().push(i);
            }
        }
        for mut group in tiers.into_values() {
            if (group.len() as u64) < MERGE_FACTOR {
                continue;
            }
            group.sort_by_key(|&i| live(i));
            let (mut pick, mut sum, mut kinds) = (Vec::new(), 0, self.cfg.schema.clone());
            for i in group {
                if pick.len() as u64 == MERGE_FACTOR {
                    break;
                }
                if sum + live(i) > cap
                    || !merge_kinds(&mut kinds, &read(&snap.segs[i].seg).payloads)
                {
                    continue;
                }
                sum += live(i);
                pick.push(i);
            }
            if pick.len() >= 2 {
                return Some(pick);
            }
        }
        None
    }

    /// IVF-PQ only for L2 and cosine: inner product tops out at ~0.84
    /// recall with the int8 re-rank (ADR 0012).
    /// Also not when no PQ split of `dim` has subspaces of <= 16 dims
    /// (prime dims): one-byte codes would make the re-rank shortlist noise.
    fn ivf_allowed(&self) -> bool {
        self.cfg.metric != Metric::InnerProduct && self.cfg.dim / pq_m(self.cfg.dim) <= 16
    }

    /// `max_merge_rows`, lowered so a merge's f32 copy stays within
    /// `MERGE_BYTES` for wide vectors.
    fn merge_cap(&self) -> u64 {
        let by_bytes = MERGE_BYTES / (self.cfg.dim as u64 * 4);
        u64::from(self.cfg.max_merge_rows).min(by_bytes.max(1))
    }

    /// Build or compaction step: copies the live rows of `inputs` (as of
    /// `snap`) into a new segment, written with no lock held, then
    /// publishes it in their place. Rows that died meanwhile are dead in
    /// the new segment; ids that moved are remapped.
    fn rewrite(&self, m: &mut Maint, snap: &Snapshot, inputs: &[usize]) -> Result<()> {
        // A poisoned WAL fails the publish anyway: don't build (and leave
        // an orphan file pair behind) on every retry.
        self.wal.sync()?;
        let t = Instant::now();
        let d = self.cfg.dim;
        let (mut ids, mut vectors, mut src) = (Vec::new(), Vec::new(), Vec::new());
        // The biggest IVF-PQ input goes first: its live rows keep their
        // codes (`IvfPq::retain`), so only the other inputs' rows are
        // encoded, with its quantizers (no retraining).
        let mut inputs = inputs.to_vec();
        let big = |i: &usize| {
            let seg = read(&snap.segs[*i].seg);
            matches!(seg.index, SegIndex::IvfPq(..)).then_some(seg.ids.len())
        };
        if let Some(at) = (0..inputs.len()).max_by_key(|&j| big(&inputs[j]))
            && big(&inputs[at]).is_some()
        {
            inputs.swap(0, at);
        }
        let inputs = &inputs[..];
        let segs: Vec<_> = inputs.iter().map(|&i| read(&snap.segs[i].seg)).collect();
        let template = match &segs[0].index {
            SegIndex::IvfPq(ivf, _) => {
                let v = &snap.segs[inputs[0]];
                let rows: Vec<u32> = (0..v.rows).filter(|&r| !v.dead.contains(r)).collect();
                Some(match ivf.retain(&rows) {
                    Some(kept) => (kept, rows.len()),
                    None => (ivf.empty_clone(), 0),
                })
            }
            SegIndex::Hnsw(_) => None,
        };
        let mut kinds = self.cfg.schema.clone();
        for seg in &segs {
            merge_kinds(&mut kinds, &seg.payloads);
        }
        let mut payloads = PayloadStore::with_schema(&kinds);
        for (&i, seg) in inputs.iter().zip(&segs) {
            let v = &snap.segs[i];
            for row in (0..v.rows).filter(|&r| !v.dead.contains(r)) {
                let vector = seg
                    .vector(d, row)
                    .ok_or(Error::Invalid("segment row has no vector"))?;
                if let Some(p) = seg.payloads.get(row) {
                    payloads.insert(ids.len() as u32, &Value::Object(p))?;
                }
                ids.push(seg.ids[row as usize]);
                vectors.extend_from_slice(vector);
                src.push((v.key, row));
            }
        }
        drop(segs);
        let (rows, copy) = (ids.len(), t.elapsed());
        let file = m.next_file;
        m.next_file += 1;
        let built = if ids.is_empty() {
            None
        } else {
            Some(self.build(file, ids, vectors, payloads, template)?)
        };
        crash_point(4)?;
        let built_at = t.elapsed();

        let mut w = self.writer()?;
        let cur = self.snapshot();
        let gone: Vec<u32> = inputs.iter().map(|&i| snap.segs[i].key).collect();
        let mut next = Snapshot {
            segs: cur
                .segs
                .iter()
                .filter(|v| !gone.contains(&v.key))
                .cloned()
                .collect(),
        };
        let mut moved = Vec::new();
        if let Some(seg) = built {
            let key = self.new_key();
            let mut dead = RoaringBitmap::new();
            let live = read(&self.live);
            for (j, (&(k, row), &id)) in (0u32..).zip(src.iter().zip(&seg.ids)) {
                if live.get(&id) == Some(&(k, row)) {
                    moved.push((id, (key, j)));
                } else {
                    dead.insert(j); // deleted or replaced since `snap`
                }
            }
            drop(live);
            let at = next.segs.len() - 1; // before the mutable segment
            next.segs.insert(
                at,
                View {
                    key,
                    file,
                    rows: seg.ids.len() as u32,
                    seg: Arc::new(RwLock::new(seg)),
                    dead: Arc::new(dead),
                },
            );
        }
        let covered = w.covered;
        self.publish(&mut w, covered, &next, 5)?;
        {
            let mut live = write(&self.live);
            live.extend(moved);
            *write(&self.snap) = Arc::new(next);
        }
        drop(w);
        tracing::info!(
            inputs = inputs.len(),
            rows,
            copy_ms = copy.as_millis(),
            build_ms = (built_at - copy).as_millis(),
            publish_ms = (t.elapsed() - built_at).as_millis(),
            "rewrite"
        );
        // Readers may still hold the old segments: HNSW ones are in RAM and
        // a mapped IVF-PQ file stays readable after it is unlinked.
        for &i in inputs {
            for ext in ["seg", "payloads"] {
                match fs::remove_file(file_path(&self.dir, snap.segs[i].file, ext)) {
                    Err(e) if e.kind() != io::ErrorKind::NotFound => return Err(e.into()),
                    _ => {}
                }
            }
        }
        Ok(())
    }

    /// Builds a segment's index on the build pool and writes its files.
    /// IVF-PQ from `ivf_rows` rows up (L2 and cosine), HNSW below.
    fn build(
        &self,
        file: u64,
        ids: Vec<u64>,
        vectors: Vec<f32>,
        payloads: PayloadStore,
        template: Option<(IvfPq, usize)>,
    ) -> Result<Seg> {
        let (d, n) = (self.cfg.dim, ids.len());
        let pool = build_pool()?;
        if self.ivf_allowed() && n >= self.cfg.ivf_rows as usize {
            let fresh = ivf_config(d, n, self.cfg.hnsw.seed);
            // Retraining (k-means + PQ) dominates a rebuild; under steady
            // deletes every big segment is rewritten every few minutes, and
            // the builds were the soak's p99 spikes. Reuse the quantizers,
            // and the first `done` rows' codes, unless the segment outgrew
            // them (nlist < half of a fresh one's).
            // ponytail: no retrain on distribution drift; retrain every Nth
            // rewrite, or on a recall probe, if real data drifts.
            let (mut index, done) = match template {
                Some((t, done)) if t.config().nlist * 2 >= fresh.nlist => (t, done),
                _ => (IvfPq::new(d, self.cfg.metric, fresh)?, 0),
            };
            let t = Instant::now();
            match pool.install(|| index.add(&vectors[done * d..])) {
                Ok(_) => {
                    let encoded = t.elapsed();
                    // The f32 vectors go in the file (element type 0), not in RAM.
                    let blob = index.to_bytes();
                    self.write_files(file, &vectors, &ids, &payloads, BlobKind::IvfPq, &blob)?;
                    drop((vectors, blob));
                    let written = t.elapsed();
                    let path = file_path(&self.dir, file, "seg");
                    let mapped = Segment::open(&path).map_err(|e| Error::Segment(path, e))?;
                    tracing::info!(
                        rows = n,
                        encoded = n - done,
                        encode_ms = encoded.as_millis(),
                        write_ms = (written - encoded).as_millis(),
                        open_ms = (t.elapsed() - written).as_millis(),
                        "ivf_pq build"
                    );
                    return Ok(Seg {
                        index: SegIndex::IvfPq(index, mapped),
                        ids,
                        payloads,
                    });
                }
                // Data IVF-PQ can't train on stays searchable as HNSW.
                Err(e) => tracing::warn!("IVF-PQ build failed ({e}); building HNSW instead"),
            }
        }
        let t = Instant::now();
        let mut index = Hnsw::new(d, self.cfg.metric, self.cfg.hnsw);
        index.reserve(n);
        pool.install(|| index.add(&vectors))?;
        drop(vectors);
        let encoded = t.elapsed();
        let blob = index.to_bytes();
        self.write_files(
            file,
            index.vectors(),
            &ids,
            &payloads,
            BlobKind::Hnsw,
            &blob,
        )?;
        tracing::info!(
            rows = n,
            graph_ms = encoded.as_millis(),
            write_ms = (t.elapsed() - encoded).as_millis(),
            "hnsw build"
        );
        Ok(Seg {
            index: SegIndex::Hnsw(index),
            ids,
            payloads,
        })
    }

    fn new_key(&self) -> u32 {
        self.next_key.fetch_add(1, Ordering::Relaxed)
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

    /// Bounded components keep f32 distances and IVF-PQ's int8 scale
    /// finite; cosine has no direction for a zero vector (HNSW and IVF-PQ
    /// would score it differently).
    fn check_vectors(&self, vectors: &[f32]) -> Result<()> {
        if !vectors.iter().all(|x| x.abs() <= MAX_COMPONENT) {
            return Err(Error::Invalid(
                "vector component is NaN, infinite or beyond ±1e15",
            ));
        }
        if self.cfg.metric == Metric::Cosine
            && vectors
                .chunks_exact(self.cfg.dim)
                .any(|v| v.iter().all(|&x| x == 0.0))
        {
            return Err(Error::Invalid("cosine needs a nonzero vector"));
        }
        Ok(())
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
            let SegIndex::Hnsw(index) = &mut seg.index else {
                return Err(Error::Invalid("mutable segment is not HNSW"));
            };
            let rows = index.add(&vectors[first * d..(first + chunk.len()) * d])?;
            for (row, p) in rows.zip(&payloads[first..]) {
                seg.payloads.insert(row, p)?;
            }
            seg.ids.extend_from_slice(chunk);
        }
        let mut next = (*snap).clone();
        let key = next.segs[m].key;
        let mut live = write(&self.live);
        for (row, &id) in (start..).zip(ids) {
            if let Some((k, r)) = live.insert(id, (key, row)) {
                kill(&mut next, k, r);
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
            if let Some((k, r)) = live.remove(id) {
                kill(&mut next, k, r);
            }
        }
        *write(&self.snap) = Arc::new(next);
    }

    /// Writes `<file>.seg` and `<file>.payloads`.
    fn write_files(
        &self,
        file: u64,
        vectors: &[f32],
        ids: &[u64],
        payloads: &PayloadStore,
        kind: BlobKind,
        blob: &[u8],
    ) -> Result<()> {
        let path = file_path(&self.dir, file, "seg");
        let data = SegmentData {
            dim: self.cfg.dim,
            metric: self.cfg.metric.code(),
            vectors,
            ids,
            blob_kind: kind,
            blob,
        };
        segment::write(&path, &data).map_err(|e| Error::Segment(path, e))?;
        let path = file_path(&self.dir, file, "payloads");
        let body = payloads.encode(ids.len() as u32)?;
        framed::write(&path, PAYLOADS_MAGIC, PAYLOADS_VERSION, &body)
            .map_err(|e| Error::File(path, e))
    }
}

/// Tombstones row `r` of the segment with key `k` in `next`.
fn kill(next: &mut Snapshot, k: u32, r: u32) {
    if let Some(pos) = next.pos(k) {
        Arc::make_mut(&mut next.segs[pos].dead).insert(r);
    }
}

/// Size tier: 0 below `seal * MERGE_FACTOR` live rows, then one per
/// further factor.
fn tier(live: u64, seal: u64) -> u32 {
    let (mut t, mut x) = (0, live);
    while x >= seal.saturating_mul(MERGE_FACTOR) {
        x /= MERGE_FACTOR;
        t += 1;
    }
    t
}

/// Adds `store`'s column types to `kinds`; false (and `kinds` unchanged)
/// if a field has another type there: those segments can't merge.
fn merge_kinds(kinds: &mut Schema, store: &PayloadStore) -> bool {
    if store
        .kinds()
        .any(|(f, k)| kinds.get(f).is_some_and(|&have| have != k))
    {
        return false;
    }
    kinds.extend(store.kinds().map(|(f, k)| (f.clone(), k)));
    true
}

/// IVF-PQ for `n` rows: nlist = sqrt(n) (the measured SIFT1M and 10M
/// points used ~1x and 1.3x sqrt(n) with nprobe 16 and 32), 8-bit PQ with
/// `pq_m` subspaces.
fn ivf_config(dim: usize, n: usize, seed: u64) -> ivfpq::Config {
    let nlist = nlist_for(n);
    ivfpq::Config {
        nlist,
        m: pq_m(dim),
        nbits: 8,
        max_train: (nlist * 64).max(ivfpq::Config::default().max_train),
        seed,
    }
}

fn nlist_for(n: usize) -> usize {
    ((n as f64).sqrt().round() as usize).max(1)
}

/// `nprobe` scaled from a fresh `ivf_rows`-row segment's nlist to this
/// segment's: every IVF-PQ segment probes the same share of its lists, so
/// recall and work don't depend on how compaction split the rows (one 1M
/// segment probed 32 of ~1000 lists where two 500k ones probed 32 of ~700
/// each: −1.6% recall).
fn segment_nprobe(nprobe: usize, nlist: usize, ivf_rows: u32) -> usize {
    nprobe
        .saturating_mul(nlist)
        .div_ceil(nlist_for(ivf_rows as usize))
        .max(1)
}

/// The largest m <= max(16, dim / 8) dividing `dim`: 16 bytes per code
/// (SIFT's measured setting) up to dim 128, subspaces of ~8 dims beyond.
fn pq_m(dim: usize) -> usize {
    (1..=16.max(dim / 8))
        .rev()
        .find(|m| dim.is_multiple_of(*m))
        .unwrap_or(1)
}

fn new_mutable(cfg: &Config, key: u32) -> View {
    let seg = Seg {
        index: SegIndex::Hnsw(Hnsw::new(cfg.dim, cfg.metric, cfg.hnsw)),
        ids: Vec::new(),
        payloads: PayloadStore::with_schema(&cfg.schema),
    };
    View {
        key,
        file: 0,
        seg: Arc::new(RwLock::new(seg)),
        rows: 0,
        dead: Arc::default(),
    }
}

fn load_segment(dir: &Path, cfg: &Config, file: u64, dead: &RoaringBitmap) -> Result<Seg> {
    let path = file_path(dir, file, "seg");
    let s = Segment::open(&path).map_err(|e| Error::Segment(path.clone(), e))?;
    if s.dim() != cfg.dim || s.metric() != cfg.metric.code() {
        return Err(Error::Invalid(
            "config dim/metric differ from the collection's",
        ));
    }
    let ids = s.ids().to_vec();
    if s.vectors().len() != ids.len() * cfg.dim {
        return Err(corrupt(&path, "segment has no f32 vectors"));
    }
    let index = match s.blob_kind() {
        BlobKind::Hnsw => SegIndex::Hnsw(Hnsw::from_bytes(s.index_blob(), s.vectors().to_vec())?),
        BlobKind::IvfPq => SegIndex::IvfPq(IvfPq::from_bytes(s.index_blob())?, s),
        BlobKind::Flat => return Err(corrupt(&path, "flat segments are not used")),
    };
    let seg = Seg {
        index,
        ids,
        payloads: PayloadStore::new(),
    };
    if seg.index().len() != seg.ids.len() {
        return Err(corrupt(&path, "index and ids differ in length"));
    }
    if dead.max().is_some_and(|r| r as usize >= seg.ids.len()) {
        return Err(corrupt(&path, "manifest tombstones a row past the end"));
    }
    let path = file_path(dir, file, "payloads");
    let payloads = read_payloads(&path, seg.ids.len() as u32, &cfg.schema)?;
    Ok(Seg { payloads, ..seg })
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

/// `<dir>/<file id, 20 digits>.<ext>`, like WAL file names.
fn file_path(dir: &Path, file: u64, ext: &str) -> PathBuf {
    dir.join(format!("{file:020}.{ext}"))
}

/// Deletes `*.tmp` files and segment files the manifest doesn't list:
/// leftovers of a crashed write, seal or compaction.
fn remove_garbage(dir: &Path, keep: &[u64]) -> Result<()> {
    let mut removed = false;
    for entry in fs::read_dir(dir)? {
        let name = entry?.file_name();
        let Some(name) = name.to_str() else { continue };
        let file = name
            .strip_suffix(".seg")
            .or_else(|| name.strip_suffix(".payloads"))
            .filter(|d| d.len() == 20 && d.bytes().all(|b| b.is_ascii_digit()))
            .and_then(|d| d.parse::<u64>().ok());
        if name.ends_with(".tmp") || file.is_some_and(|f| !keep.contains(&f)) {
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
    /// Each sealed segment's file id and tombstones, ascending by id.
    segs: Vec<(u64, RoaringBitmap)>,
}

/// Reads a v2 or v1 manifest; none means no sealed segments.
fn read_manifest(path: &Path) -> Result<Manifest> {
    let read = |v| framed::read(path, MANIFEST_MAGIC, v);
    let (body, version) = match read(MANIFEST_VERSION) {
        Ok(body) => (body, MANIFEST_VERSION),
        Err(framed::Error::UnsupportedVersion {
            found: MANIFEST_V1, ..
        }) => (
            read(MANIFEST_V1).map_err(|e| Error::File(path.to_path_buf(), e))?,
            MANIFEST_V1,
        ),
        Err(framed::Error::Io(e)) if e.kind() == io::ErrorKind::NotFound => {
            return Ok(Manifest::default());
        }
        Err(e) => return Err(Error::File(path.to_path_buf(), e)),
    };
    decode_manifest(&body, version).ok_or_else(|| corrupt(path, "bad manifest"))
}

// ponytail: every publish rewrites every segment's tombstones; fine while
// they're small, split into per-segment files if seals get slow.
fn encode_manifest(covered: Lsn, segs: &[(u64, &RoaringBitmap)]) -> io::Result<Vec<u8>> {
    let mut out = Vec::new();
    out.extend_from_slice(&covered.to_le_bytes());
    out.extend_from_slice(&(segs.len() as u32).to_le_bytes());
    for (file, dead) in segs {
        out.extend_from_slice(&file.to_le_bytes());
        out.extend_from_slice(&(dead.serialized_size() as u32).to_le_bytes());
        dead.serialize_into(&mut out)?;
    }
    Ok(out)
}

/// v1 and v2 share the layout; v1 ids are LSNs, so also <= covered.
fn decode_manifest(mut b: &[u8], version: u32) -> Option<Manifest> {
    let covered = u64_at(&mut b)?;
    let n = u32_at(&mut b)?;
    let mut segs = Vec::new();
    let mut prev = 0;
    for _ in 0..n {
        let file = u64_at(&mut b)?;
        if file <= prev || version == MANIFEST_V1 && file > covered {
            return None;
        }
        prev = file;
        let len = u32_at(&mut b)? as usize;
        let bytes = take(&mut b, len)?;
        let dead = RoaringBitmap::deserialize_from(bytes).ok()?;
        if dead.serialized_size() != len {
            return None;
        }
        segs.push((file, dead));
    }
    b.is_empty().then_some(Manifest { covered, segs })
}

/// Reads a v2 (columnar) or v1 (JSON per row) payload sidecar.
fn read_payloads(path: &Path, rows: u32, schema: &Schema) -> Result<PayloadStore> {
    let read =
        |v| framed::read(path, PAYLOADS_MAGIC, v).map_err(|e| Error::File(path.to_path_buf(), e));
    let store = match framed::read(path, PAYLOADS_MAGIC, PAYLOADS_VERSION) {
        Ok(body) => PayloadStore::decode(&body, rows, schema),
        Err(framed::Error::UnsupportedVersion {
            found: PAYLOADS_V1, ..
        }) => decode_payloads_v1(&read(PAYLOADS_V1)?, rows, schema),
        Err(e) => return Err(Error::File(path.to_path_buf(), e)),
    };
    store.ok_or_else(|| corrupt(path, "bad payloads"))
}

/// v1: each row's JSON re-inserted into a fresh store.
fn decode_payloads_v1(mut b: &[u8], rows: u32, schema: &Schema) -> Option<PayloadStore> {
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

pub(crate) fn take<'a>(b: &mut &'a [u8], n: usize) -> Option<&'a [u8]> {
    let (head, rest) = b.split_at_checked(n)?;
    *b = rest;
    Some(head)
}

pub(crate) fn u32_at(b: &mut &[u8]) -> Option<u32> {
    Some(u32::from_le_bytes(take(b, 4)?.try_into().ok()?))
}

fn u64_at(b: &mut &[u8]) -> Option<u64> {
    Some(u64::from_le_bytes(take(b, 8)?.try_into().ok()?))
}

/// Test fail point: the seal or rewrite stops with an error right after
/// `step`, leaving the directory as a crash there would. Seal: 1 WAL
/// rotated, 2 segment files written, 3 manifest published. Build or
/// compaction: 4 files written, 5 manifest published. Tests can also run
/// a hook at each step (`tests::HOOK`), e.g. to write while no lock is held.
fn crash_point(step: u8) -> Result<()> {
    #[cfg(test)]
    {
        tests::HOOK.with_borrow_mut(|h| {
            if let Some(f) = h {
                f(step)
            }
        });
        if tests::CRASH_AFTER.get() == step {
            return Err(Error::Io(io::Error::other("fail point")));
        }
    }
    let _ = step;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::cell::{Cell, RefCell};
    use std::sync::atomic::AtomicUsize;

    type Hook = Box<dyn FnMut(u8)>;

    thread_local! {
        pub(super) static CRASH_AFTER: Cell<u8> = const { Cell::new(0) };
        /// Runs at each `crash_point` step on this thread.
        pub(super) static HOOK: RefCell<Option<Hook>> = RefCell::new(None);
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
            seal_rows: 100_000, // tests seal explicitly unless they lower it
            ivf_rows: MIN_IVF_ROWS,
            max_merge_rows: 1_000_000,
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
        assert!(c.upsert(1, &[2e38; DIM], &Value::Null).is_err()); // overflows L2
        assert!(c.search(&[f32::INFINITY; DIM], 1, &opts(None)).is_err());
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
        let lsn = format!("{:020}", 1); // the first file id
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
    /// search and background steps (seal, merge, rewrite) run: a delete
    /// acknowledged before a search started is never in its results, no id
    /// appears twice, and an id that is never deleted never goes missing.
    #[test]
    fn readers_never_see_acknowledged_deletes() {
        use std::sync::atomic::{AtomicBool, AtomicU64};
        let d = TempDir::new("concurrent");
        let cfg = Config {
            seal_rows: 200,
            ..cfg()
        };
        let c = Collection::open(&d.0, cfg.clone()).unwrap().0;
        // id → sequence number of its delete's ack (ids never come back).
        let deleted = RwLock::new(HashMap::<u64, u64>::new());
        let (acks, done) = (AtomicU64::new(0), AtomicBool::new(false));
        // Ids below this were inserted; odd ones are never deleted.
        let inserted = AtomicU64::new(0);
        let steps = AtomicU64::new(0);
        std::thread::scope(|s| {
            let (c2, done2, steps2) = (&c, &done, &steps);
            s.spawn(move || {
                while !done2.load(Ordering::SeqCst) {
                    if c2.maintain().unwrap() {
                        steps2.fetch_add(1, Ordering::SeqCst);
                    } else {
                        std::thread::yield_now();
                    }
                }
            });
            for r in 0..3u64 {
                let (c, deleted, acks, done, inserted) = (&c, &deleted, &acks, &done, &inserted);
                s.spawn(move || {
                    let mut searches = 0;
                    while !done.load(Ordering::SeqCst) || searches < 10 {
                        let n = inserted.load(Ordering::SeqCst);
                        if n > 1 {
                            let id = ((r * 7 + searches * 13) % n) | 1;
                            if id < n {
                                assert!(c.get(id).is_some(), "live id {id} missing");
                            }
                        }
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
                inserted.store(round * 50 + 50, Ordering::SeqCst);
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
        assert!(steps.load(Ordering::SeqCst) > 0, "no background step ran");
        while c.maintain().unwrap() {}
        let live = c.len();
        let n: usize = c.segments().iter().map(|s| s.rows as usize).sum();
        drop(c);
        let c = Collection::open(&d.0, cfg).unwrap().0;
        assert_eq!((c.len(), rows(&c)), (live, n));
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
        framed::write(&p, MANIFEST_MAGIC, MANIFEST_V1, &m).unwrap();
        let fixture: &[u8] = include_bytes!("../testdata/manifest-v1");
        assert_eq!(fs::read(&p).unwrap(), fixture);
        let want = Manifest {
            covered: 9,
            segs: vec![(4, dead.clone()), (9, RoaringBitmap::new())],
        };
        assert_eq!(read_manifest(&p).unwrap(), want);

        // v2: same layout, file ids from a counter (may exceed covered).
        let m = encode_manifest(9, &[(4, &dead), (12, &RoaringBitmap::new())]).unwrap();
        framed::write(&p, MANIFEST_MAGIC, MANIFEST_VERSION, &m).unwrap();
        let fixture: &[u8] = include_bytes!("../testdata/manifest-v2");
        assert_eq!(fs::read(&p).unwrap(), fixture);
        let want = Manifest {
            covered: 9,
            segs: vec![(4, dead), (12, RoaringBitmap::new())],
        };
        assert_eq!(read_manifest(&p).unwrap(), want);
        assert_eq!(
            decode_manifest(&m, MANIFEST_V1),
            None,
            "v1 ids are LSNs <= covered"
        );
        let dup = encode_manifest(9, &[(4, &RoaringBitmap::new()), (4, &RoaringBitmap::new())]);
        assert_eq!(decode_manifest(&dup.unwrap(), MANIFEST_VERSION), None);

        let mut store = PayloadStore::new();
        let rows = [
            json!({"n": 1, "tag": ["a", "b"]}),
            Value::Null,
            json!({"f": 2.5}),
        ];
        for (r, v) in rows.iter().enumerate() {
            store.insert(r as u32, v).unwrap();
        }
        // Payloads: v1 (JSON per row) still reads; v2 (columnar) is what's
        // written now, byte for byte.
        let p = dir.0.join("x.payloads");
        let schema = Schema::new();
        let check = |back: PayloadStore| {
            for (r, v) in rows.iter().enumerate() {
                assert_eq!(back.get(r as u32), v.as_object().cloned());
            }
        };
        fs::write(&p, include_bytes!("../testdata/payloads-v1")).unwrap();
        check(read_payloads(&p, 3, &schema).unwrap());
        assert!(read_payloads(&p, 4, &schema).is_err());
        let body = store.encode(3).unwrap();
        framed::write(&p, PAYLOADS_MAGIC, PAYLOADS_VERSION, &body).unwrap();
        let fixture: &[u8] = include_bytes!("../testdata/payloads-v2");
        assert_eq!(fs::read(&p).unwrap(), fixture);
        check(read_payloads(&p, 3, &schema).unwrap());
        assert!(read_payloads(&p, 4, &schema).is_err());
        // A flipped bit or a cut file is an error, not a panic.
        let mut bad = fixture.to_vec();
        bad[30] ^= 1;
        fs::write(&p, &bad).unwrap();
        assert!(read_payloads(&p, 3, &schema).is_err());
        fs::write(&p, &fixture[..fixture.len() - 1]).unwrap();
        assert!(read_payloads(&p, 3, &schema).is_err());
    }

    /// Review fixes: configs that could never merge are rejected, cosine
    /// refuses zero vectors, prime dims stay HNSW, wide dims cap merges.
    #[test]
    fn config_and_input_guards() {
        let ip = |seal_rows| Config {
            metric: Metric::InnerProduct,
            seal_rows,
            ..cfg()
        };
        assert!(ip(512).validate().is_ok());
        assert!(ip(513).validate().is_err()); // ivf_rows 1024 caps IP merges
        assert!(
            Config {
                max_merge_rows: 150_000,
                ..cfg()
            }
            .validate()
            .is_err()
        );
        assert_eq!(
            (pq_m(4), pq_m(32), pq_m(128), pq_m(768), pq_m(97)),
            (4, 16, 16, 96, 1)
        );

        let d = TempDir::new("guards");
        let cos = Config {
            metric: Metric::Cosine,
            ..cfg()
        };
        let c = open_with(&d, cos);
        assert!(c.upsert(1, &[0.0; DIM], &Value::Null).is_err());
        assert!(c.search(&[0.0; DIM], 1, &opts(None)).is_err());
        assert!(c.ivf_allowed());
        drop(c);
        let d = TempDir::new("guards-prime");
        let c = open_with(&d, Config { dim: 97, ..cfg() });
        assert!(!c.ivf_allowed());
        drop(c);
        let d = TempDir::new("guards-wide");
        let c = open_with(
            &d,
            Config {
                dim: 1024,
                max_merge_rows: 4_000_000,
                ..cfg()
            },
        );
        assert_eq!(c.merge_cap(), (2 << 30) / 4096);
    }

    /// Any IVF-PQ segment probes the share of its lists a fresh `ivf_rows`
    /// one does.
    #[test]
    fn nprobe_scales_with_nlist() {
        assert_eq!(segment_nprobe(32, 1000, 1_000_000), 32);
        assert_eq!(segment_nprobe(32, 2000, 1_000_000), 64); // a 4M merge
        assert_eq!(segment_nprobe(32, 775, 400_000), 40); // soak: 32 * 775 / 632
        assert_eq!(segment_nprobe(32, 100, 1_000_000), 4); // never 0
        assert_eq!(segment_nprobe(1, 1, 1_000_000), 1);
        assert_eq!(segment_nprobe(usize::MAX, 4, 1024), usize::MAX.div_ceil(32));
    }

    fn open_with(d: &TempDir, cfg: Config) -> Arc<Collection> {
        Arc::new(Collection::open(&d.0, cfg).unwrap().0)
    }

    fn file_names(d: &TempDir) -> Vec<String> {
        let mut names: Vec<_> = fs::read_dir(&d.0)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        names
    }

    /// (index, rows, dead) of each segment, mutable last.
    fn shape(c: &Collection) -> Vec<(&'static str, u32, u64)> {
        c.segments()
            .into_iter()
            .map(|s| (s.index, s.rows, s.dead))
            .collect()
    }

    /// 6.1: `maintain` seals at `seal_rows`, and the segment files are
    /// written with the writer lock free: writes made then (replacing and
    /// deleting frozen rows) land in the new mutable segment and survive.
    #[test]
    fn seals_at_size_with_writes_during_the_file_write() {
        let d = TempDir::new("autoseal");
        let cfg = Config {
            seal_rows: 20,
            ..cfg()
        };
        let c = open_with(&d, cfg.clone());
        let mut model = Model::new();
        for id in 0..19 {
            c.upsert(id, &vec_for(id, 1), &payload(id, 1)).unwrap();
            model.insert(id, 1);
        }
        assert!(!c.maintain().unwrap(), "19 rows: nothing to do");
        c.upsert(19, &vec_for(19, 1), &payload(19, 1)).unwrap();
        model.insert(19, 1);
        let c2 = Arc::clone(&c);
        HOOK.set(Some(Box::new(move |step| {
            if step == 2 {
                assert!(c2.writer.try_lock().is_ok(), "seal holds the writer lock");
                c2.upsert(3, &vec_for(3, 9), &payload(3, 9)).unwrap();
                c2.delete(4).unwrap();
                c2.upsert(30, &vec_for(30, 9), &payload(30, 9)).unwrap();
                c2.upsert(5, &vec_for(5, 9), &payload(5, 9)).unwrap(); // then deleted
                c2.delete(5).unwrap();
            }
        })));
        assert!(c.maintain().unwrap());
        HOOK.set(None);
        model.insert(3, 9);
        model.remove(&4);
        model.insert(30, 9);
        model.remove(&5);
        assert_eq!(shape(&c), [("hnsw", 20, 3), ("mutable", 3, 1)]);
        check(&c, &model);
        assert_eq!(fs::read_dir(d.0.join("wal")).unwrap().count(), 1);
        drop(c);
        let (c, rep) = Collection::open(&d.0, cfg).unwrap();
        assert_eq!((rep.segments, rep.replayed), (1, 5));
        check(&c, &model);
    }

    /// Four sealed segments of rows 0..9, 10..19, 20..29, 30..39 (ids
    /// 9, 19, 29, 39 left out), 2 deletes in the first.
    fn four_segments(c: &Collection, model: &mut Model) {
        for s in 0..4 {
            for id in s * 10..s * 10 + 9 {
                c.upsert(id, &vec_for(id, 1), &payload(id, 1)).unwrap();
                model.insert(id, 1);
            }
            c.seal().unwrap();
        }
        for id in [0, 1] {
            assert!(c.delete(id).unwrap());
            model.remove(&id);
        }
    }

    /// 6.3: four small segments merge into one without their dead rows;
    /// rows that die while the merge runs (outside every lock) are dead in
    /// the new segment, and ids upserted meanwhile are not resurrected.
    /// Then a segment that is >= 25% dead is rewritten.
    #[test]
    fn compaction_merges_and_drops_dead_rows() {
        let d = TempDir::new("compact");
        let c = open_with(&d, cfg());
        let mut model = Model::new();
        four_segments(&c, &mut model);
        assert_eq!(c.segments().len(), 5);
        let c2 = Arc::clone(&c);
        let mut fired = false;
        HOOK.set(Some(Box::new(move |step| {
            if step == 4 && !fired {
                fired = true;
                assert!(
                    c2.writer.try_lock().is_ok(),
                    "rewrite holds the writer lock"
                );
                c2.upsert(15, &vec_for(15, 7), &payload(15, 7)).unwrap();
                c2.delete(16).unwrap();
                c2.upsert(39, &vec_for(39, 7), &payload(39, 7)).unwrap();
            }
        })));
        let mut steps = 0;
        while c.maintain().unwrap() {
            steps += 1;
        }
        HOOK.set(None);
        model.insert(15, 7);
        model.remove(&16);
        model.insert(39, 7);
        assert_eq!(steps, 1);
        assert_eq!(shape(&c), [("hnsw", 34, 2), ("mutable", 2, 0)]);
        check(&c, &model);
        let hits = c.search(&vec_for(16, 1), 40, &opts(None)).unwrap();
        assert!(hits.iter().all(|m| model.contains_key(&m.id)));
        assert_eq!(hits.len(), model.len());

        // 10 more deletes: 12 of 34 dead, so the segment is rewritten.
        for id in 20..29 {
            c.delete(id).unwrap();
            model.remove(&id);
        }
        c.delete(2).unwrap();
        model.remove(&2);
        assert!(c.maintain().unwrap());
        assert!(!c.maintain().unwrap());
        assert_eq!(shape(&c), [("hnsw", 22, 0), ("mutable", 2, 0)]);
        assert_eq!(rows(&c), 24, "dead rows are gone from RAM");
        check(&c, &model);
        assert_eq!(file_names(&d).len(), 2 + 2, "old segment files deleted");
        drop(c);
        let (c, rep) = Collection::open(&d.0, cfg()).unwrap();
        assert_eq!(rep.segments, 1);
        check(&c, &model);
    }

    /// A crash after each rewrite step: reopening gives the same state,
    /// with exactly the manifest's files.
    #[test]
    fn crash_between_rewrite_steps() {
        for step in [4, 5] {
            let d = TempDir::new("crash-rewrite");
            let c = open_with(&d, cfg());
            let mut model = Model::new();
            four_segments(&c, &mut model);
            CRASH_AFTER.set(step);
            assert!(c.maintain().is_err(), "step {step}");
            CRASH_AFTER.set(0);
            drop(c);

            let (c, rep) = Collection::open(&d.0, cfg()).unwrap();
            check(&c, &model);
            let segs = if step == 5 { 1 } else { 4 };
            assert_eq!(rep.segments, segs, "step {step}");
            assert_eq!(file_names(&d).len(), 2 * segs + 2, "step {step}");
            assert_eq!(rows(&c), if step == 5 { 34 } else { 36 });
            while c.maintain().unwrap() {}
            drop(c);
            check(&open(&d), &model);
        }
    }

    /// Deterministic pseudo-random vectors in [0, 1).
    fn rand_vec(id: u64, dim: usize) -> Vec<f32> {
        (0..dim as u64)
            .map(|i| {
                let mut z = (id * 131 + i).wrapping_add(0x9E37_79B9_7F4A_7C15);
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32
            })
            .collect()
    }

    /// 6.2: a segment with >= ivf_rows rows is rebuilt as IVF-PQ for L2 and
    /// cosine (vectors stay exact: served from the file), never for inner
    /// product; it searches, filters dead rows and reopens.
    #[test]
    fn big_segments_become_ivf_pq() {
        for metric in [Metric::L2, Metric::Cosine, Metric::InnerProduct] {
            let d = TempDir::new("ivf");
            // seal_rows <= ivf_rows / 2: inner product must still merge.
            let cfg = Config {
                metric,
                seal_rows: 512,
                ..cfg()
            };
            let mut c = open_with(&d, cfg.clone());
            let n = 1500u64;
            let ids: Vec<u64> = (0..n).collect();
            let vecs: Vec<f32> = ids.iter().flat_map(|&i| rand_vec(i, DIM)).collect();
            let pays: Vec<Value> = ids.iter().map(|&i| json!({"odd": i % 2 == 1})).collect();
            c.upsert_batch(&ids, &vecs, &pays).unwrap();
            c.seal().unwrap();
            let before: Vec<_> = (0..n).map(|i| c.get(i).unwrap()).collect();
            while c.maintain().unwrap() {}
            let kind = if metric == Metric::InnerProduct {
                "hnsw"
            } else {
                "ivf_pq"
            };
            assert_eq!(
                shape(&c),
                [(kind, 1500, 0), ("mutable", 0, 0)],
                "{metric:?}"
            );
            for i in (0..n).filter(|i| i % 10 == 0) {
                assert!(c.delete(i).unwrap());
            }
            // Round 2 runs after 30% are dead: a rewrite that reuses the
            // (reopened) segment's quantizers.
            let dead = |i: u64, round| i.is_multiple_of(10) || (round == 2 && i % 10 <= 2);
            for round in 0..3 {
                for i in 0..n {
                    let want = (!dead(i, round)).then(|| before[i as usize].clone());
                    assert_eq!(c.get(i), want, "{metric:?} id {i}");
                }
                let odd: Filter = serde_json::from_value(json!({"eq": {"odd": true}})).unwrap();
                let (mut found, mut asked) = (0, 0);
                for i in (1..n).step_by(37) {
                    let o = SearchOptions {
                        nprobe: Some(64),
                        rerank: Some(200),
                        ..opts(None)
                    };
                    let hits = c.search(&rand_vec(i, DIM), 10, &o).unwrap();
                    assert_eq!(hits.len(), 10);
                    assert!(hits.iter().all(|m| !dead(m.id, round)), "dead row returned");
                    found += usize::from(!dead(i, round) && hits.iter().any(|m| m.id == i));
                    asked += usize::from(!dead(i, round));
                    let hits = c.search(&rand_vec(i, DIM), 10, &opts(Some(&odd))).unwrap();
                    assert!(hits.iter().all(|m| m.id % 2 == 1 && !dead(m.id, round)));
                }
                // Under inner product a vector need not be its own nearest.
                let ip = metric == Metric::InnerProduct;
                assert!(
                    ip || found * 10 >= asked * 8,
                    "{metric:?}: {found}/{asked} found"
                );
                if round == 0 {
                    drop(c);
                    c = open_with(&d, cfg.clone());
                    assert_eq!(c.segments()[0].index, kind);
                }
                if round == 1 {
                    for i in (0..n).filter(|i| matches!(i % 10, 1 | 2)) {
                        assert!(c.delete(i).unwrap());
                    }
                    while c.maintain().unwrap() {}
                    // 1050 live rows >= ivf_rows (1024): rewritten as IVF-PQ.
                    assert_eq!(
                        shape(&c),
                        [(kind, 1050, 0), ("mutable", 0, 0)],
                        "{metric:?}"
                    );
                }
            }
        }
    }

    /// A small sealed HNSW segment is folded into the IVF-PQ segment
    /// (codes kept, only new rows encoded) instead of merging into an HNSW
    /// tier; dead rows go on the way; reads stay exact across a reopen.
    #[test]
    fn small_segments_are_absorbed_into_ivf_pq() {
        let d = TempDir::new("absorb");
        let cfg = Config {
            seal_rows: 512,
            ..cfg()
        };
        let mut c = open_with(&d, cfg.clone());
        let vecs =
            |ids: &[u64]| -> Vec<f32> { ids.iter().flat_map(|&i| rand_vec(i, DIM)).collect() };
        let first: Vec<u64> = (0..1500).collect();
        c.upsert_batch(&first, &vecs(&first), &[]).unwrap();
        c.seal().unwrap();
        while c.maintain().unwrap() {}
        assert_eq!(shape(&c), [("ivf_pq", 1500, 0), ("mutable", 0, 0)]);
        // Two small segments: not yet (absorb waits for MERGE_FACTOR - 1).
        let few: Vec<u64> = (2000..2040).collect();
        c.upsert_batch(&few, &vecs(&few), &[]).unwrap();
        c.seal().unwrap();
        let more: Vec<u64> = (3000..3200).collect();
        c.upsert_batch(&more, &vecs(&more), &[]).unwrap();
        c.seal().unwrap();
        while c.maintain().unwrap() {}
        assert_eq!(shape(&c).len(), 4);
        let last: Vec<u64> = (3200..3400).collect();
        c.upsert_batch(&last, &vecs(&last), &[]).unwrap();
        for i in (0..1500).step_by(15) {
            assert!(c.delete(i).unwrap());
        }
        c.seal().unwrap();
        while c.maintain().unwrap() {}
        assert_eq!(shape(&c), [("ivf_pq", 1840, 0), ("mutable", 0, 0)]);
        let more: Vec<u64> = (3000..3400).collect();
        for round in 0..2 {
            for &i in first.iter().chain(&few).chain(&more) {
                let want = (i >= 1500 || i % 15 != 0).then(|| rand_vec(i, DIM));
                assert_eq!(c.get(i).map(|g| g.0), want, "id {i}");
            }
            let o = SearchOptions {
                nprobe: Some(64),
                rerank: Some(200),
                ..opts(None)
            };
            for i in [1, 2001, 3001, 3399] {
                let hits = c.search(&rand_vec(i, DIM), 5, &o).unwrap();
                assert_eq!(hits[0].id, i);
            }
            if round == 0 {
                drop(c);
                c = open_with(&d, cfg.clone());
            }
        }
    }

    /// A v1 directory (segments named by LSN, manifest v1) still opens,
    /// and later seals and merges work on it.
    #[test]
    fn v1_data_dir_still_opens() {
        let d = TempDir::new("v1dir");
        let mut c = open(&d);
        let mut model = Model::new();
        run(&mut c, &mut model, 0..30);
        c.seal().unwrap();
        let covered = lock(&c.writer).covered;
        drop(c);
        for ext in ["seg", "payloads"] {
            let from = file_path(&d.0, 1, ext);
            fs::rename(from, file_path(&d.0, covered, ext)).unwrap();
        }
        let p = d.0.join(MANIFEST);
        let mut m = read_manifest(&p).unwrap();
        m.segs[0].0 = covered;
        let body = encode_manifest(m.covered, &[(covered, &m.segs[0].1)]).unwrap();
        framed::write(&p, MANIFEST_MAGIC, MANIFEST_V1, &body).unwrap();

        let c = open_with(&d, cfg());
        check(&c, &model);
        for i in 30..33 {
            c.upsert(i, &vec_for(i, 1), &payload(i, 1)).unwrap();
            model.insert(i, 1);
            c.seal().unwrap();
        }
        while c.maintain().unwrap() {}
        assert_eq!(c.segments().len(), 2, "merged");
        drop(c);
        check(&open(&d), &model);
    }
}
