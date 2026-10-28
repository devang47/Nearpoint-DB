//! Write-ahead log (task 3.1). See docs/formats/wal.md.
//!
//! A WAL is a directory of segment files named by the LSN of their first
//! record (`00000000000000000001.wal`). Every record carries a CRC32; on
//! open, all files are replayed and validated, and a torn or corrupt tail
//! is truncated away.
//!
//! Durability: `File::sync_data` on Apple targets is `fcntl(F_FULLFSYNC)`
//! in Rust 1.98 std (checked in `library/std/src/sys/fs/unix.rs`), so it
//! flushes the drive cache; no extra libc call is needed.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::Duration;

/// Log sequence number. The first record ever written has LSN 1; every
/// append gets the previous LSN + 1, with no gaps, across reopens.
pub type Lsn = u64;

pub const MAGIC: [u8; 8] = *b"NPNT_WAL";
pub const VERSION: u32 = 1;
const FILE_HEADER_LEN: usize = 16; // magic 8 | version u32 | crc u32
const RECORD_HEADER_LEN: usize = 17; // crc u32 | len u32 | kind u8 | lsn u64
/// Largest record body accepted on write and on read. A longer length
/// field is "impossible" and treated as corruption.
pub const MAX_BODY_LEN: u32 = 64 << 20;
const KIND_UPSERT: u8 = 1;
const KIND_DELETE: u8 = 2;

#[derive(Debug, thiserror::Error)]
pub enum WalError {
    #[error("wal i/o: {0}")]
    Io(#[from] io::Error),
    #[error("{0}: not a WAL file (bad magic)")]
    BadMagic(PathBuf),
    #[error("{path}: WAL format version {found} is not supported (this build reads {VERSION})")]
    UnsupportedVersion { path: PathBuf, found: u32 },
    #[error("{0}: WAL file header is corrupt")]
    CorruptHeader(PathBuf),
    #[error("record body is {0} bytes, limit is {MAX_BODY_LEN}")]
    RecordTooLarge(usize),
    #[error("WAL is poisoned by an earlier failed write or sync; reopen to recover")]
    Poisoned,
    #[cfg(feature = "failpoints")]
    #[error("failpoint: torn write")]
    FailPoint,
}

pub type Result<T> = std::result::Result<T, WalError>;

#[derive(Debug, Clone, PartialEq)]
pub enum Record {
    /// `payload` is opaque bytes (metadata from Phase 7).
    Upsert {
        id: u64,
        vector: Vec<f32>,
        payload: Vec<u8>,
    },
    Delete {
        id: u64,
    },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SyncPolicy {
    /// Each append is written and fsynced under the log lock before it
    /// returns: one F_FULLFSYNC per append, appends fully serialized.
    Always,
    /// Group commit: each append is durable before it returns, but
    /// concurrent appenders wait for one shared fsync instead of one each.
    #[default]
    Batch,
    /// Appends return once written to the page cache; a background thread
    /// fsyncs every interval. A power loss can lose the last interval of
    /// acknowledged appends (a process crash loses nothing).
    Interval(Duration),
}

/// What `Wal::open` found on disk.
#[derive(Debug, Default)]
pub struct Recovered {
    /// Every valid record, in LSN order.
    // ponytail: whole WAL in RAM; it mirrors the in-RAM mutable segment, so
    // it's bounded by that. Switch to a streaming iterator if that changes.
    pub records: Vec<(Lsn, Record)>,
    /// Bytes discarded: the torn/corrupt tail plus every later file.
    pub dropped_bytes: u64,
    /// Whole files deleted because they followed a torn/corrupt record.
    pub dropped_files: usize,
}

/// The write-ahead log. All methods take `&self` and lock internally, so a
/// `Wal` (or `Arc<Wal>`) can be shared by every writer thread; group commit
/// in `SyncPolicy::Batch` only helps if writers can append concurrently.
pub struct Wal {
    shared: Arc<Shared>,
    ticker: Option<(mpsc::Sender<()>, JoinHandle<()>)>,
}

struct Shared {
    dir: PathBuf,
    policy: SyncPolicy,
    /// Set by a failed write or fsync. After that the file may hold a
    /// partial record or lost pages, so no further appends are accepted.
    poisoned: AtomicBool,
    inner: Mutex<Inner>,
    sync: Mutex<SyncState>,
    synced: Condvar,
}

struct Inner {
    file: Arc<File>,
    /// First LSN of each file, ascending; the last one is being appended to.
    files: Vec<Lsn>,
    next_lsn: Lsn,
}

struct SyncState {
    /// Every LSN <= this is durable.
    durable: Lsn,
    /// One thread is inside `sync_data`; others wait on `synced`.
    syncing: bool,
}

impl Wal {
    /// Opens (or creates) the WAL in `dir`, replays and validates every
    /// file, and truncates at the first torn or corrupt record (dropping
    /// all later files). A bad file header is an error, not a truncation:
    /// headers are published atomically, so a bad one is not a torn write.
    pub fn open(dir: impl AsRef<Path>, policy: SyncPolicy) -> Result<(Wal, Recovered)> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;
        let mut files = Vec::new();
        for entry in fs::read_dir(&dir)? {
            let name = entry?.file_name();
            let Some(name) = name.to_str() else { continue };
            if name.ends_with(".wal.tmp") {
                fs::remove_file(dir.join(name))?; // crashed mid-create
            } else if let Some(lsn) = parse_file_name(name) {
                files.push(lsn);
            }
        }
        files.sort_unstable();

        let mut rec = Recovered::default();
        let mut expected = files.first().copied().unwrap_or(1);
        let mut keep = files.len();
        for (k, &first) in files.iter().enumerate() {
            if first != expected {
                keep = k; // gap or overlap between files
                break;
            }
            let path = file_path(&dir, first);
            let bytes = fs::read(&path)?;
            check_header(&path, &bytes)?;
            let mut at = FILE_HEADER_LEN;
            while let Some((record, len)) = decode(&bytes[at..], expected) {
                rec.records.push((expected, record));
                expected += 1;
                at += len;
            }
            if at < bytes.len() {
                rec.dropped_bytes += (bytes.len() - at) as u64;
                let f = OpenOptions::new().write(true).open(&path)?;
                f.set_len(at as u64)?;
                f.sync_all()?;
                keep = k + 1;
                break;
            }
        }
        for &lsn in &files[keep..] {
            let path = file_path(&dir, lsn);
            rec.dropped_bytes += fs::metadata(&path)?.len();
            fs::remove_file(path)?;
            rec.dropped_files += 1;
        }
        if rec.dropped_files > 0 {
            sync_dir(&dir)?;
        }
        files.truncate(keep);

        let file = match files.last() {
            Some(&lsn) => {
                let f = OpenOptions::new().append(true).open(file_path(&dir, lsn))?;
                // The previous process may have crashed before syncing.
                f.sync_data()?;
                f
            }
            None => {
                files.push(expected);
                create_file(&dir, expected)?
            }
        };

        let shared = Arc::new(Shared {
            dir,
            policy,
            poisoned: AtomicBool::new(false),
            inner: Mutex::new(Inner {
                file: Arc::new(file),
                files,
                next_lsn: expected,
            }),
            sync: Mutex::new(SyncState {
                durable: expected - 1,
                syncing: false,
            }),
            synced: Condvar::new(),
        });
        let ticker = match policy {
            SyncPolicy::Interval(every) => {
                let (tx, rx) = mpsc::channel::<()>();
                let s = Arc::clone(&shared);
                let handle = std::thread::spawn(move || {
                    while let Err(RecvTimeoutError::Timeout) = rx.recv_timeout(every) {
                        let _ = s.sync(); // a failure poisons; appends report it
                    }
                });
                Some((tx, handle))
            }
            _ => None,
        };
        Ok((Wal { shared, ticker }, rec))
    }

    /// Appends one record and returns its LSN. Durable on return under
    /// `Always` and `Batch`; see `SyncPolicy::Interval` for the exception.
    pub fn append(&self, record: &Record) -> Result<Lsn> {
        self.append_batch(std::slice::from_ref(record))
    }

    /// Appends `records` with consecutive LSNs in one write and one sync
    /// (per the policy) and returns the last LSN (`next_lsn() - 1` if
    /// `records` is empty). Every record is encoded before anything is
    /// written, so `RecordTooLarge` writes nothing; after any other error
    /// a prefix of the batch may be durable.
    pub fn append_batch(&self, records: &[Record]) -> Result<Lsn> {
        let s = &*self.shared;
        let mut inner = lock(&s.inner);
        s.check_poisoned()?;
        let first = inner.next_lsn;
        let mut buf = Vec::new();
        for (lsn, r) in (first..).zip(records) {
            buf.extend_from_slice(&encode(lsn, r)?);
        }
        let lsn = first + records.len() as u64 - 1;
        if records.is_empty() {
            return Ok(lsn);
        }
        #[cfg(feature = "failpoints")]
        if let Some(n) = failpoint::take() {
            let _ = (&*inner.file).write_all(&buf[..n.min(buf.len())]);
            s.poisoned.store(true, Ordering::SeqCst);
            return Err(WalError::FailPoint);
        }
        if let Err(e) = (&*inner.file).write_all(&buf) {
            s.poisoned.store(true, Ordering::SeqCst);
            return Err(e.into());
        }
        inner.next_lsn = lsn + 1;
        match s.policy {
            SyncPolicy::Always => {
                if let Err(e) = inner.file.sync_data() {
                    s.poisoned.store(true, Ordering::SeqCst);
                    return Err(e.into());
                }
                drop(inner);
                s.mark_durable(lsn);
            }
            SyncPolicy::Batch => {
                drop(inner);
                s.wait_durable(lsn)?;
            }
            SyncPolicy::Interval(_) => {}
        }
        Ok(lsn)
    }

    /// Makes every record appended so far durable.
    pub fn sync(&self) -> Result<()> {
        self.shared.sync()
    }

    /// LSN the next append will get.
    pub fn next_lsn(&self) -> Lsn {
        lock(&self.shared.inner).next_lsn
    }

    /// Syncs the current file and starts a new one named by the next LSN.
    /// A no-op if the current file holds no records yet.
    pub fn rotate(&self) -> Result<()> {
        let s = &*self.shared;
        let mut inner = lock(&s.inner);
        s.check_poisoned()?;
        let next = inner.next_lsn;
        if inner.files.last() == Some(&next) {
            return Ok(());
        }
        if let Err(e) = inner.file.sync_data() {
            s.poisoned.store(true, Ordering::SeqCst);
            return Err(e.into());
        }
        s.mark_durable(next - 1);
        inner.file = Arc::new(create_file(&s.dir, next)?);
        inner.files.push(next);
        Ok(())
    }

    /// Deletes every file whose records all have LSN < `lsn`. The current
    /// file is never deleted. Returns how many files were removed.
    pub fn remove_before(&self, lsn: Lsn) -> Result<usize> {
        let s = &*self.shared;
        let doomed: Vec<Lsn> = {
            let mut inner = lock(&s.inner);
            // File i holds [files[i], files[i+1]); removable if files[i+1] <= lsn.
            let n = inner.files.windows(2).take_while(|w| w[1] <= lsn).count();
            inner.files.drain(..n).collect()
        };
        for &first in &doomed {
            fs::remove_file(file_path(&s.dir, first))?;
        }
        if !doomed.is_empty() {
            sync_dir(&s.dir)?;
        }
        Ok(doomed.len())
    }
}

impl Drop for Wal {
    fn drop(&mut self) {
        if let Some((tx, handle)) = self.ticker.take() {
            drop(tx);
            let _ = handle.join();
        }
        let _ = self.shared.sync();
    }
}

impl Shared {
    fn check_poisoned(&self) -> Result<()> {
        if self.poisoned.load(Ordering::SeqCst) {
            return Err(WalError::Poisoned);
        }
        Ok(())
    }

    fn sync(&self) -> Result<()> {
        let last = lock(&self.inner).next_lsn - 1;
        self.wait_durable(last)
    }

    fn mark_durable(&self, lsn: Lsn) {
        let mut st = lock(&self.sync);
        st.durable = st.durable.max(lsn);
        self.synced.notify_all();
    }

    /// Group commit: returns once `lsn` is durable. If no sync is in flight
    /// this thread runs one covering everything written so far; otherwise
    /// it waits for the in-flight one and re-checks.
    fn wait_durable(&self, lsn: Lsn) -> Result<()> {
        let mut st = lock(&self.sync);
        loop {
            self.check_poisoned()?;
            if st.durable >= lsn {
                return Ok(());
            }
            if !st.syncing {
                break;
            }
            st = self.synced.wait(st).unwrap_or_else(PoisonError::into_inner);
        }
        st.syncing = true;
        drop(st);
        // Lock order is always inner -> sync, never the reverse: `st` is
        // released above. Records < `target` in older files were synced by
        // `rotate` before it switched files.
        let (file, target) = {
            let inner = lock(&self.inner);
            (Arc::clone(&inner.file), inner.next_lsn - 1)
        };
        let res = file.sync_data();
        let mut st = lock(&self.sync);
        st.syncing = false;
        match &res {
            Ok(()) => st.durable = st.durable.max(target),
            Err(_) => self.poisoned.store(true, Ordering::SeqCst),
        }
        self.synced.notify_all();
        Ok(res?)
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

fn file_path(dir: &Path, first_lsn: Lsn) -> PathBuf {
    dir.join(format!("{first_lsn:020}.wal"))
}

fn parse_file_name(name: &str) -> Option<Lsn> {
    let digits = name.strip_suffix(".wal")?;
    if digits.len() != 20 || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

fn sync_dir(dir: &Path) -> io::Result<()> {
    File::open(dir)?.sync_all()
}

fn file_header() -> [u8; FILE_HEADER_LEN] {
    let mut h = [0u8; FILE_HEADER_LEN];
    h[..8].copy_from_slice(&MAGIC);
    h[8..12].copy_from_slice(&VERSION.to_le_bytes());
    let crc = crc32fast::hash(&h[..12]);
    h[12..].copy_from_slice(&crc.to_le_bytes());
    h
}

fn check_header(path: &Path, bytes: &[u8]) -> Result<()> {
    if bytes.get(..8) != Some(&MAGIC[..]) {
        return Err(WalError::BadMagic(path.into()));
    }
    let h = bytes
        .get(..FILE_HEADER_LEN)
        .ok_or_else(|| WalError::CorruptHeader(path.into()))?;
    let version = le_u32(&h[8..12]);
    if version != VERSION {
        return Err(WalError::UnsupportedVersion {
            path: path.into(),
            found: version,
        });
    }
    if crc32fast::hash(&h[..12]) != le_u32(&h[12..16]) {
        return Err(WalError::CorruptHeader(path.into()));
    }
    Ok(())
}

/// Atomic publish: header goes to `<name>.tmp`, is synced, renamed, and
/// the directory synced, so a `.wal` file always has a complete header.
fn create_file(dir: &Path, first_lsn: Lsn) -> Result<File> {
    let tmp = dir.join(format!("{first_lsn:020}.wal.tmp"));
    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&tmp)?;
    f.write_all(&file_header())?;
    f.sync_data()?;
    fs::rename(&tmp, file_path(dir, first_lsn))?;
    sync_dir(dir)?;
    Ok(f) // cursor is at the end of the header
}

fn le_u32(b: &[u8]) -> u32 {
    let mut a = [0u8; 4];
    a.copy_from_slice(b);
    u32::from_le_bytes(a)
}

fn le_u64(b: &[u8]) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(b);
    u64::from_le_bytes(a)
}

fn encode(lsn: Lsn, record: &Record) -> Result<Vec<u8>> {
    let (kind, body_len) = match record {
        Record::Upsert {
            vector, payload, ..
        } => (KIND_UPSERT, 8 + 4 + 4 * vector.len() + 4 + payload.len()),
        Record::Delete { .. } => (KIND_DELETE, 8),
    };
    if body_len > MAX_BODY_LEN as usize {
        return Err(WalError::RecordTooLarge(body_len));
    }
    let mut buf = Vec::with_capacity(RECORD_HEADER_LEN + body_len);
    buf.extend_from_slice(&[0; 4]); // crc, filled below
    buf.extend_from_slice(&(body_len as u32).to_le_bytes());
    buf.push(kind);
    buf.extend_from_slice(&lsn.to_le_bytes());
    match record {
        Record::Upsert {
            id,
            vector,
            payload,
        } => {
            buf.extend_from_slice(&id.to_le_bytes());
            buf.extend_from_slice(&(vector.len() as u32).to_le_bytes());
            for x in vector {
                buf.extend_from_slice(&x.to_le_bytes());
            }
            buf.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            buf.extend_from_slice(payload);
        }
        Record::Delete { id } => buf.extend_from_slice(&id.to_le_bytes()),
    }
    let crc = crc32fast::hash(&buf[4..]);
    buf[..4].copy_from_slice(&crc.to_le_bytes());
    Ok(buf)
}

/// Decodes the record at the start of `buf`; returns it and its encoded
/// length. `None` for anything torn, corrupt or out of sequence.
fn decode(buf: &[u8], expected_lsn: Lsn) -> Option<(Record, usize)> {
    let h = buf.get(..RECORD_HEADER_LEN)?;
    let len = le_u32(&h[4..8]);
    if len > MAX_BODY_LEN {
        return None;
    }
    let total = RECORD_HEADER_LEN + len as usize;
    let rec = buf.get(..total)?;
    if crc32fast::hash(&rec[4..]) != le_u32(&rec[..4]) || le_u64(&rec[9..17]) != expected_lsn {
        return None;
    }
    let mut body = &rec[RECORD_HEADER_LEN..];
    let mut take = |n: usize| -> Option<&[u8]> {
        let (head, rest) = body.split_at_checked(n)?;
        body = rest;
        Some(head)
    };
    let record = match rec[8] {
        KIND_UPSERT => {
            let id = le_u64(take(8)?);
            let dim = le_u32(take(4)?) as usize;
            let vector = take(dim.checked_mul(4)?)?
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&c| f32::from_le_bytes(c))
                .collect();
            let plen = le_u32(take(4)?) as usize;
            let payload = take(plen)?.to_vec();
            Record::Upsert {
                id,
                vector,
                payload,
            }
        }
        KIND_DELETE => Record::Delete {
            id: le_u64(take(8)?),
        },
        _ => return None,
    };
    body.is_empty().then_some((record, total))
}

/// Crash simulation for task 3.4a. Compiled only with `--features failpoints`.
#[cfg(feature = "failpoints")]
pub mod failpoint {
    use std::cell::Cell;

    thread_local! {
        static TEAR: Cell<Option<usize>> = const { Cell::new(None) };
    }

    /// The next `append` on this thread writes only the first `bytes`
    /// bytes of its record, poisons the WAL and returns `FailPoint`.
    pub fn tear_next_append(bytes: usize) {
        TEAR.with(|t| t.set(Some(bytes)));
    }

    pub(super) fn take() -> Option<usize> {
        TEAR.with(Cell::take)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::HashSet;
    use std::sync::atomic::AtomicUsize;
    use std::time::Instant;

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            static N: AtomicUsize = AtomicUsize::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            let p = std::env::temp_dir()
                .join(format!("nearpoint-wal-{tag}-{}-{n}", std::process::id()));
            let _ = fs::remove_dir_all(&p);
            TempDir(p)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn upsert(id: u64) -> Record {
        Record::Upsert {
            id,
            vector: (0..4).map(|i| (id * 10 + i) as f32 * 0.5).collect(),
            payload: id.to_le_bytes()[..(id % 5) as usize].to_vec(),
        }
    }

    fn sample(i: u64) -> Record {
        if i % 4 == 3 {
            Record::Delete { id: i }
        } else {
            upsert(i)
        }
    }

    fn wal_files(dir: &Path) -> Vec<String> {
        let mut v: Vec<String> = fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        v.sort();
        v
    }

    fn arb_record() -> impl Strategy<Value = Record> {
        prop_oneof![
            (
                any::<u64>(),
                prop::collection::vec(-1e6f32..1e6, 0..64),
                prop::collection::vec(any::<u8>(), 0..64)
            )
                .prop_map(|(id, vector, payload)| Record::Upsert {
                    id,
                    vector,
                    payload
                }),
            any::<u64>().prop_map(|id| Record::Delete { id }),
        ]
    }

    #[test]
    fn round_trip_and_reopen_continues_lsns() {
        let d = TempDir::new("roundtrip");
        let (wal, rec) = Wal::open(&d.0, SyncPolicy::Batch).unwrap();
        assert!(rec.records.is_empty());
        for i in 1..=500 {
            assert_eq!(wal.append(&sample(i)).unwrap(), i);
        }
        drop(wal);
        let (wal, rec) = Wal::open(&d.0, SyncPolicy::Batch).unwrap();
        assert_eq!(rec.dropped_bytes, 0);
        let want: Vec<_> = (1..=500).map(|i| (i, sample(i))).collect();
        assert_eq!(rec.records, want);
        assert_eq!(wal.append(&sample(501)).unwrap(), 501);
        drop(wal);
        let (wal, rec) = Wal::open(&d.0, SyncPolicy::Always).unwrap();
        assert_eq!(rec.records.len(), 501);
        assert_eq!(wal.next_lsn(), 502);
    }

    #[test]
    fn append_batch_is_consecutive_and_all_or_nothing_on_too_large() {
        let d = TempDir::new("batch");
        let (wal, _) = Wal::open(&d.0, SyncPolicy::Batch).unwrap();
        assert_eq!(wal.append_batch(&[]).unwrap(), 0);
        let batch: Vec<_> = (1..=10).map(sample).collect();
        assert_eq!(wal.append_batch(&batch).unwrap(), 10);
        let huge = Record::Upsert {
            id: 0,
            vector: Vec::new(),
            payload: vec![0; MAX_BODY_LEN as usize],
        };
        let e = wal.append_batch(&[sample(11), huge]);
        assert!(matches!(e, Err(WalError::RecordTooLarge(_))));
        assert_eq!(wal.append(&sample(11)).unwrap(), 11);
        drop(wal);
        let (_, rec) = Wal::open(&d.0, SyncPolicy::Batch).unwrap();
        let want: Vec<_> = (1..=11).map(|i| (i, sample(i))).collect();
        assert_eq!(rec.records, want);
    }

    #[test]
    fn rotate_and_remove_before() {
        let d = TempDir::new("rotate");
        let (wal, _) = Wal::open(&d.0, SyncPolicy::Batch).unwrap();
        for i in 1..=25 {
            wal.append(&sample(i)).unwrap();
            if i == 10 || i == 20 {
                wal.rotate().unwrap();
                wal.rotate().unwrap(); // empty current file: no-op
            }
        }
        assert_eq!(wal_files(&d.0).len(), 3);
        assert_eq!(wal.remove_before(15).unwrap(), 1); // file 11 holds 15..
        assert_eq!(wal.remove_before(21).unwrap(), 1);
        assert_eq!(wal.remove_before(u64::MAX).unwrap(), 0); // never current
        assert_eq!(wal_files(&d.0), ["00000000000000000021.wal"]);
        drop(wal);
        let (wal, rec) = Wal::open(&d.0, SyncPolicy::Batch).unwrap();
        let want: Vec<_> = (21..=25).map(|i| (i, sample(i))).collect();
        assert_eq!(rec.records, want);
        // Rotate, reopen with only an empty file: LSNs still continue.
        wal.rotate().unwrap();
        wal.remove_before(26).unwrap();
        drop(wal);
        let (wal, rec) = Wal::open(&d.0, SyncPolicy::Batch).unwrap();
        assert!(rec.records.is_empty());
        assert_eq!(wal.append(&sample(26)).unwrap(), 26);
    }

    /// Writes records 1..=n into a fresh WAL and returns the file bytes and
    /// the offset where record `n` starts.
    fn single_file(n: u64) -> (Vec<u8>, usize) {
        let d = TempDir::new("src");
        let (wal, _) = Wal::open(&d.0, SyncPolicy::Interval(Duration::from_secs(60))).unwrap();
        for i in 1..n {
            wal.append(&sample(i)).unwrap();
        }
        drop(wal);
        let path = file_path(&d.0, 1);
        let last_start = fs::metadata(&path).unwrap().len() as usize;
        let (wal, _) = Wal::open(&d.0, SyncPolicy::Batch).unwrap();
        wal.append(&sample(n)).unwrap();
        drop(wal);
        (fs::read(path).unwrap(), last_start)
    }

    #[test]
    fn torn_tail_at_every_offset_of_last_record() {
        let (bytes, last_start) = single_file(4);
        let earlier: Vec<_> = (1..4).map(|i| (i, sample(i))).collect();
        for cut in last_start..bytes.len() {
            let d = TempDir::new("torn");
            fs::create_dir_all(&d.0).unwrap();
            fs::write(file_path(&d.0, 1), &bytes[..cut]).unwrap();
            let (wal, rec) = Wal::open(&d.0, SyncPolicy::Batch).unwrap();
            assert_eq!(rec.records, earlier, "cut at {cut}");
            assert_eq!(rec.dropped_bytes, (cut - last_start) as u64);
            assert_eq!(wal.append(&sample(4)).unwrap(), 4);
            drop(wal);
            let (_, rec) = Wal::open(&d.0, SyncPolicy::Batch).unwrap();
            assert_eq!(rec.records.len(), 4);
            assert_eq!(rec.dropped_bytes, 0);
        }
    }

    #[test]
    fn corruption_drops_later_files() {
        let d = TempDir::new("later");
        let (wal, _) = Wal::open(&d.0, SyncPolicy::Batch).unwrap();
        for i in 1..=6 {
            wal.append(&sample(i)).unwrap();
            if i % 2 == 0 {
                wal.rotate().unwrap();
            }
        }
        drop(wal);
        // Flip the last byte of file 3 (record 4).
        let p = file_path(&d.0, 3);
        let mut b = fs::read(&p).unwrap();
        *b.last_mut().unwrap() ^= 1;
        fs::write(&p, b).unwrap();
        let (wal, rec) = Wal::open(&d.0, SyncPolicy::Batch).unwrap();
        assert_eq!(rec.records.len(), 3);
        assert_eq!(rec.dropped_files, 2); // files 5 and 7 (empty)
        assert_eq!(wal.next_lsn(), 4);
        assert_eq!(wal_files(&d.0).len(), 2);
    }

    #[test]
    fn bad_file_header_is_typed_error() {
        let d = TempDir::new("header");
        drop(Wal::open(&d.0, SyncPolicy::Batch).unwrap());
        let p = file_path(&d.0, 1);
        let good = fs::read(&p).unwrap();

        let mut b = good.clone();
        b[0] = b'X';
        fs::write(&p, &b).unwrap();
        assert!(matches!(
            Wal::open(&d.0, SyncPolicy::Batch),
            Err(WalError::BadMagic(_))
        ));

        let mut b = good.clone();
        b[8..12].copy_from_slice(&2u32.to_le_bytes());
        fs::write(&p, &b).unwrap();
        assert!(matches!(
            Wal::open(&d.0, SyncPolicy::Batch),
            Err(WalError::UnsupportedVersion { found: 2, .. })
        ));

        let mut b = good.clone();
        b[13] ^= 0x40;
        fs::write(&p, &b).unwrap();
        assert!(matches!(
            Wal::open(&d.0, SyncPolicy::Batch),
            Err(WalError::CorruptHeader(_))
        ));

        fs::write(&p, &good[..10]).unwrap();
        assert!(matches!(
            Wal::open(&d.0, SyncPolicy::Batch),
            Err(WalError::CorruptHeader(_))
        ));
    }

    #[test]
    fn too_large_record_is_rejected() {
        let d = TempDir::new("large");
        let (wal, _) = Wal::open(&d.0, SyncPolicy::Batch).unwrap();
        let big = Record::Upsert {
            id: 1,
            vector: Vec::new(),
            payload: vec![0; MAX_BODY_LEN as usize],
        };
        assert!(matches!(wal.append(&big), Err(WalError::RecordTooLarge(_))));
        assert_eq!(wal.append(&sample(1)).unwrap(), 1); // not poisoned
    }

    #[test]
    fn batch_concurrent_appenders() {
        const THREADS: u64 = 8;
        const PER: u64 = 200;
        let d = TempDir::new("batch");
        let (wal, _) = Wal::open(&d.0, SyncPolicy::Batch).unwrap();
        let lsns: Vec<Lsn> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..THREADS)
                .map(|t| {
                    let wal = &wal;
                    s.spawn(move || {
                        (0..PER)
                            .map(|i| wal.append(&upsert(t * PER + i)).unwrap())
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
        let unique: HashSet<_> = lsns.iter().copied().collect();
        assert_eq!(unique.len(), (THREADS * PER) as usize);
        assert_eq!(unique, (1..=THREADS * PER).collect());
        drop(wal);
        let (_, rec) = Wal::open(&d.0, SyncPolicy::Batch).unwrap();
        let ids: HashSet<u64> = rec
            .records
            .iter()
            .map(|(_, r)| match r {
                Record::Upsert { id, .. } | Record::Delete { id } => *id,
            })
            .collect();
        assert_eq!(ids, (0..THREADS * PER).collect());
    }

    #[test]
    fn interval_policy_syncs_in_background() {
        let d = TempDir::new("interval");
        let (wal, _) = Wal::open(&d.0, SyncPolicy::Interval(Duration::from_millis(5))).unwrap();
        wal.append(&sample(1)).unwrap();
        let t = Instant::now();
        while lock(&wal.shared.sync).durable < 1 {
            assert!(t.elapsed() < Duration::from_secs(10), "never synced");
            std::thread::yield_now();
        }
        drop(wal); // stops the ticker without hanging
        let (_, rec) = Wal::open(&d.0, SyncPolicy::Batch).unwrap();
        assert_eq!(rec.records.len(), 1);
    }

    #[cfg(feature = "failpoints")]
    #[test]
    fn failpoint_tears_append() {
        let len = encode(4, &sample(4)).unwrap().len();
        for n in [0, 1, RECORD_HEADER_LEN - 1, RECORD_HEADER_LEN, len - 1] {
            let d = TempDir::new("fail");
            let (wal, _) = Wal::open(&d.0, SyncPolicy::Batch).unwrap();
            for i in 1..4 {
                wal.append(&sample(i)).unwrap();
            }
            failpoint::tear_next_append(n);
            assert!(matches!(wal.append(&sample(4)), Err(WalError::FailPoint)));
            assert!(matches!(wal.append(&sample(4)), Err(WalError::Poisoned)));
            drop(wal);
            let (wal, rec) = Wal::open(&d.0, SyncPolicy::Batch).unwrap();
            assert_eq!(rec.records.len(), 3);
            assert_eq!(rec.dropped_bytes, n as u64);
            assert_eq!(wal.append(&sample(4)).unwrap(), 4);
        }
    }

    /// Run on a quiet machine:
    /// `cargo test --release -p storage -- --ignored --nocapture sync_latency`
    #[test]
    #[ignore]
    fn sync_latency_always_200_appends() {
        let d = TempDir::new("latency");
        let (wal, _) = Wal::open(&d.0, SyncPolicy::Always).unwrap();
        let mut us: Vec<f64> = (0..200)
            .map(|i| {
                let t = Instant::now();
                wal.append(&upsert(i)).unwrap();
                t.elapsed().as_secs_f64() * 1e6
            })
            .collect();
        us.sort_by(f64::total_cmp);
        let mean = us.iter().sum::<f64>() / us.len() as f64;
        println!(
            "SyncPolicy::Always append+F_FULLFSYNC: mean {mean:.0} us, p50 {:.0} us, p99 {:.0} us",
            us[us.len() / 2],
            us[us.len() * 99 / 100]
        );
    }

    /// Durable appends/s under `Batch` as writer threads grow: group commit
    /// should let N writers share each F_FULLFSYNC.
    /// `cargo test --release -p storage -- --ignored --nocapture sync_throughput`
    #[test]
    #[ignore]
    fn sync_throughput_batch_by_threads() {
        for threads in [1u64, 2, 4, 8, 16, 32] {
            let d = TempDir::new("throughput");
            let (wal, _) = Wal::open(&d.0, SyncPolicy::Batch).unwrap();
            let per = 1600 / threads;
            let t = Instant::now();
            std::thread::scope(|s| {
                for w in 0..threads {
                    let wal = &wal;
                    s.spawn(move || {
                        for i in 0..per {
                            wal.append(&upsert(w * per + i)).unwrap();
                        }
                    });
                }
            });
            let secs = t.elapsed().as_secs_f64();
            println!(
                "SyncPolicy::Batch {threads:>2} writers: {:.0} durable appends/s",
                (threads * per) as f64 / secs
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn encode_decode_round_trip(lsn in 1u64..u64::MAX, r in arb_record()) {
            let buf = encode(lsn, &r).unwrap();
            prop_assert_eq!(decode(&buf, lsn), Some((r, buf.len())));
            prop_assert_eq!(decode(&buf, lsn + 1), None);
            for cut in 0..buf.len() {
                prop_assert!(decode(&buf[..cut], lsn).is_none());
            }
        }

        #[test]
        fn any_flipped_bit_is_detected(r in arb_record(), pos in any::<prop::sample::Index>(), bit in 0u8..8) {
            let mut buf = encode(7, &r).unwrap();
            let i = pos.index(buf.len());
            buf[i] ^= 1 << bit;
            prop_assert!(decode(&buf, 7).is_none());
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(24))]

        #[test]
        fn flipped_bit_in_file_recovers_prefix(
            recs in prop::collection::vec(arb_record(), 1..8),
            pos in any::<prop::sample::Index>(),
            bit in 0u8..8,
        ) {
            let d = TempDir::new("flip");
            let (wal, _) = Wal::open(&d.0, SyncPolicy::Interval(Duration::from_secs(60))).unwrap();
            let mut ends = Vec::new();
            for r in &recs {
                wal.append(r).unwrap();
                ends.push(fs::metadata(file_path(&d.0, 1)).unwrap().len() as usize);
            }
            drop(wal);
            let p = file_path(&d.0, 1);
            let mut b = fs::read(&p).unwrap();
            let i = FILE_HEADER_LEN + pos.index(b.len() - FILE_HEADER_LEN);
            b[i] ^= 1 << bit;
            fs::write(&p, b).unwrap();
            let bad = ends.iter().position(|&e| i < e).unwrap();
            let (_, rec) = Wal::open(&d.0, SyncPolicy::Batch).unwrap();
            let want: Vec<_> = recs[..bad].iter().cloned().zip(1..).map(|(r, l)| (l, r)).collect();
            prop_assert_eq!(rec.records, want);
            prop_assert!(rec.dropped_bytes > 0);
        }
    }
}
