//! A collection (tasks 3.3–3.4): a WAL, one mutable in-RAM HNSW segment,
//! sealed segments, an external-id map and tombstones, all in one
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

use std::collections::HashMap;
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};

use index::{Hnsw, Index, Metric, SearchParams, hnsw};
use roaring::RoaringBitmap;
use serde_json::Value;
use storage::framed;
use storage::segment::{self, BlobKind, Segment, SegmentData};
use storage::wal::{Lsn, Record, SyncPolicy, Wal, WalError};

use crate::{Filter, PayloadStore, evaluate};

const MANIFEST: &str = "MANIFEST";
const MANIFEST_MAGIC: [u8; 8] = *b"NPMANIFS";
const MANIFEST_VERSION: u32 = 1;
const PAYLOADS_MAGIC: [u8; 8] = *b"NPPAYLDS";
const PAYLOADS_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub dim: usize,
    pub metric: Metric,
    /// For new mutable segments; sealed ones keep the config they were
    /// built with (it's in their index blob).
    pub hnsw: hnsw::Config,
    pub sync: SyncPolicy,
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

/// One segment in memory. A sealed one only ever changes in `dead`.
struct Seg {
    /// The LSN it was sealed at (its file name); 0 while mutable.
    lsn: Lsn,
    index: Hnsw,
    /// External id of each row.
    ids: Vec<u64>,
    payloads: PayloadStore,
    /// Rows deleted or replaced by a later upsert.
    dead: RoaringBitmap,
}

// ponytail: single owner, `&mut self` writes. Phase 4.3 puts the segment
// list behind an `Arc<Snapshot>`; `dead` is the only part of a sealed
// segment that changes, so it would go copy-on-write there.
pub struct Collection {
    dir: PathBuf,
    cfg: Config,
    wal: Wal,
    /// Sealed segments in LSN order, then the mutable one (always last).
    segs: Vec<Seg>,
    /// External id → (segment, row) of its one live row.
    live: HashMap<u64, (u32, u32)>,
    /// Every write with LSN <= this is in the manifest.
    covered: Lsn,
}

impl Collection {
    /// Opens (or creates) the collection in `dir`: loads the manifest's
    /// segments, deletes files a crashed seal left behind, and replays WAL
    /// records the manifest doesn't cover. One process at a time.
    pub fn open(dir: impl AsRef<Path>, cfg: Config) -> Result<(Collection, Recovery)> {
        if cfg.dim == 0 || cfg.hnsw.m < 2 || cfg.hnsw.ef_construction == 0 {
            return Err(Error::Invalid("need dim > 0, m >= 2, ef_construction >= 1"));
        }
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

        let (wal, rec) = Wal::open(dir.join("wal"), cfg.sync)?;
        let mut c = Collection {
            dir,
            cfg,
            wal,
            segs: Vec::new(),
            live: HashMap::new(),
            covered: manifest.covered,
        };
        for (lsn, dead) in manifest.segs {
            let seg = c.load_segment(lsn, dead)?;
            c.segs.push(seg);
        }
        for (s, seg) in c.segs.iter().enumerate() {
            for (row, &id) in seg.ids.iter().enumerate() {
                if !seg.dead.contains(row as u32)
                    && c.live.insert(id, (s as u32, row as u32)).is_some()
                {
                    return Err(Error::Corrupt(format!("id {id} is live in two rows")));
                }
            }
        }
        c.segs.push(c.new_mutable());

        let start = rec.records.partition_point(|r| r.0 <= c.covered);
        let resume = rec.records.get(start).map_or(c.wal.next_lsn(), |r| r.0);
        if resume != c.covered + 1 {
            return Err(Error::Corrupt(format!(
                "WAL resumes at LSN {resume}, manifest covers up to {}",
                c.covered
            )));
        }
        for (lsn, r) in &rec.records[start..] {
            match r {
                Record::Upsert {
                    id,
                    vector,
                    payload,
                } => {
                    let p = match payload.as_slice() {
                        [] => Value::Null,
                        b => serde_json::from_slice(b)
                            .map_err(|e| Error::Corrupt(format!("WAL LSN {lsn} payload: {e}")))?,
                    };
                    c.apply_upsert(*id, vector, &p)?;
                }
                Record::Delete { id } => c.apply_delete(*id),
            }
        }
        // Finishes a seal that crashed after publishing its manifest.
        c.wal.remove_before(c.covered + 1)?;
        let report = Recovery {
            segments: c.segs.len() - 1,
            replayed: rec.records.len() - start,
            dropped_bytes: rec.dropped_bytes,
            dropped_files: rec.dropped_files,
        };
        Ok((c, report))
    }

    /// Inserts or replaces `id`. `payload` is a JSON object or null. Durable
    /// on return per the sync policy; `Error::Unknown` means it may or may
    /// not be (ADR 0007).
    pub fn upsert(&mut self, id: u64, vector: &[f32], payload: &Value) -> Result<()> {
        self.check_dim(vector)?;
        if !vector.iter().all(|x| x.is_finite()) {
            return Err(Error::Invalid("vector has a NaN or infinite component"));
        }
        self.mutable().payloads.check(payload)?;
        let bytes = match payload {
            Value::Null => Vec::new(),
            p => p.to_string().into_bytes(),
        };
        self.log(&Record::Upsert {
            id,
            vector: vector.to_vec(),
            payload: bytes,
        })?;
        self.apply_upsert(id, vector, payload)
    }

    /// Deletes `id`; returns whether it existed. Durable on return like
    /// `upsert` (a missing id writes nothing).
    pub fn delete(&mut self, id: u64) -> Result<bool> {
        if !self.live.contains_key(&id) {
            return Ok(false);
        }
        self.log(&Record::Delete { id })?;
        self.apply_delete(id);
        Ok(true)
    }

    /// The vector (in stored form: unit length for cosine) and payload of
    /// `id`. A payload with no fields reads back as null.
    pub fn get(&self, id: u64) -> Option<(&[f32], Value)> {
        let &(s, row) = self.live.get(&id)?;
        let seg = &self.segs[s as usize];
        let d = self.cfg.dim;
        let r = row as usize;
        let v = seg.index.vectors().get(r * d..(r + 1) * d)?;
        Some((v, seg.payloads.get(row).map_or(Value::Null, Value::Object)))
    }

    /// Live ids.
    pub fn len(&self) -> usize {
        self.live.len()
    }

    pub fn is_empty(&self) -> bool {
        self.live.is_empty()
    }

    /// The `k` nearest live ids matching `filter`, nearest first, with
    /// their distances.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<(u64, f32)>> {
        self.check_dim(query)?;
        let mut hits = Vec::new();
        for seg in &self.segs {
            let n = seg.ids.len() as u32;
            let allowed = match filter {
                Some(f) => Some(evaluate(f, &seg.payloads, n)? - &seg.dead),
                None if !seg.dead.is_empty() => {
                    let mut all = RoaringBitmap::new();
                    all.insert_range(0..n);
                    Some(all - &seg.dead)
                }
                None => None,
            };
            if n == 0 || allowed.as_ref().is_some_and(RoaringBitmap::is_empty) {
                continue;
            }
            let params = SearchParams {
                filter: allowed.as_ref(),
                ..SearchParams::default()
            };
            // The re-check guards against an index that ignores the filter:
            // it may then return fewer than k, but never a dead row.
            for h in seg.index.search(query, k, &params)? {
                if allowed.as_ref().is_none_or(|a| a.contains(h.id)) {
                    hits.push((seg.ids[h.id as usize], h.dist));
                }
            }
        }
        // ponytail: sequential fan-out; Phase 6.4 parallelizes it.
        hits.sort_by(|a, b| a.1.total_cmp(&b.1));
        hits.truncate(k);
        Ok(hits)
    }

    /// Writes the mutable segment to disk and truncates the WAL behind it.
    /// Crash-safe at every step: segment files are written first, the new
    /// manifest is the commit point, and covered WAL files go last.
    pub fn seal(&mut self) -> Result<()> {
        let last = self.wal.next_lsn() - 1;
        if last == self.covered {
            return Ok(());
        }
        // Syncs the old file and starts LSN last + 1 in a new one, so the
        // files to drop later hold exactly the LSNs <= last.
        self.wal.rotate()?;
        crash_point(1)?;
        let m = self.segs.len() - 1;
        let has_rows = !self.segs[m].ids.is_empty();
        if has_rows {
            self.write_segment(&self.segs[m], last)?;
        }
        crash_point(2)?;
        let sealed = &self.segs[..m + usize::from(has_rows)];
        let entries: Vec<_> = sealed
            .iter()
            .map(|s| (if s.lsn == 0 { last } else { s.lsn }, &s.dead))
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
        self.covered = last;
        if has_rows {
            self.segs[m].lsn = last;
            let fresh = self.new_mutable();
            self.segs.push(fresh);
        }
        self.wal.remove_before(last + 1)?;
        Ok(())
    }

    fn mutable(&self) -> &Seg {
        &self.segs[self.segs.len() - 1]
    }

    fn new_mutable(&self) -> Seg {
        Seg {
            lsn: 0,
            index: Hnsw::new(self.cfg.dim, self.cfg.metric, self.cfg.hnsw),
            ids: Vec::new(),
            payloads: PayloadStore::new(),
            dead: RoaringBitmap::new(),
        }
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

    /// Appends to the WAL. Only a record rejected up front is a plain
    /// failure; anything else may have reached the disk.
    fn log(&self, r: &Record) -> Result<()> {
        match self.wal.append(r) {
            Ok(_) => Ok(()),
            Err(e @ WalError::RecordTooLarge(_)) => Err(e.into()),
            Err(e) => Err(Error::Unknown(e)),
        }
    }

    fn apply_upsert(&mut self, id: u64, vector: &[f32], payload: &Value) -> Result<()> {
        let m = self.segs.len() - 1;
        let seg = &mut self.segs[m];
        let row = seg.ids.len() as u32;
        seg.payloads.insert(row, payload)?;
        seg.index.add(vector)?;
        seg.ids.push(id);
        if let Some((s, r)) = self.live.insert(id, (m as u32, row)) {
            self.segs[s as usize].dead.insert(r);
        }
        Ok(())
    }

    fn apply_delete(&mut self, id: u64) {
        if let Some((s, r)) = self.live.remove(&id) {
            self.segs[s as usize].dead.insert(r);
        }
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

    fn load_segment(&self, lsn: Lsn, dead: RoaringBitmap) -> Result<Seg> {
        let path = file_path(&self.dir, lsn, "seg");
        let s = Segment::open(&path).map_err(|e| Error::Segment(path.clone(), e))?;
        if s.dim() != self.cfg.dim || s.metric() != self.cfg.metric.code() {
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
        let path = file_path(&self.dir, lsn, "payloads");
        let body = framed::read(&path, PAYLOADS_MAGIC, PAYLOADS_VERSION)
            .map_err(|e| Error::File(path.clone(), e))?;
        let payloads = decode_payloads(&body, ids.len() as u32)
            .ok_or_else(|| corrupt(&path, "bad payloads"))?;
        Ok(Seg {
            lsn,
            index,
            ids,
            payloads,
            dead,
        })
    }
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

fn decode_payloads(mut b: &[u8], rows: u32) -> Option<PayloadStore> {
    if u32_at(&mut b)? != rows {
        return None;
    }
    let mut store = PayloadStore::new();
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
            sync: SyncPolicy::Batch,
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
        c.segs.iter().map(|s| s.ids.len()).sum()
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
        let hits = c.search(&vec_for(id, ver), 5, None).unwrap();
        assert_eq!(hits[0], (id, 0.0));
        let all = c.search(&vec_for(0, 0), 100, None).unwrap();
        assert!(all.iter().all(|(id, _)| model.contains_key(id)));
        let red: Filter = serde_json::from_value(json!({"eq": {"color": "red"}})).unwrap();
        let hits = c.search(&vec_for(0, 0), 100, Some(&red)).unwrap();
        assert!(!hits.is_empty());
        assert!(
            hits.iter()
                .all(|(id, _)| model.contains_key(id) && id % 2 == 0)
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
        let (before, unsealed) = (rows(&c), c.wal.next_lsn() - 1 - c.covered);
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
        let lsn = format!("{:020}", c.covered);
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
        let back = decode_payloads(&body, 3).unwrap();
        for (r, v) in rows.iter().enumerate() {
            assert_eq!(back.get(r as u32), v.as_object().cloned());
        }
        assert!(decode_payloads(&body, 4).is_none());
    }
}
