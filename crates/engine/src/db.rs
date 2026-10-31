//! The collection registry (task 4.1): every collection under one data
//! directory, reopened on start without flags. Layout and the `CONFIG`
//! file: docs/formats/manifest.md; design: ADR 0011.
//!
//! ```text
//! <data>/LOCK                         flock'd while a Db has it open
//! <data>/collections/<name>/CONFIG    the collection's Config (JSON, framed)
//! <data>/collections/<name>/...       the collection (MANIFEST, wal/, segments)
//! <data>/collections/.<anything>      a create or drop in progress; deleted on open
//! ```
//!
//! Create writes `CONFIG` into `.new-<name>`, then renames it into place
//! (the commit point). Drop renames `<name>` to `.drop-<name>` (the commit
//! point), then deletes it. A crash at any step leaves either the whole
//! collection or a dot-directory the next open removes.
//!
//! Each collection has its own background thread running
//! `Collection::maintain` (seal, build, compaction; ADR 0012) one step at
//! a time, sleeping when it had no work, so one collection's long build
//! never delays another's seal (heavy work runs on the shared build pool).
//! A collection's thread stops when it is dropped or the Db is.

use std::collections::BTreeMap;
use std::fs::{self, File, TryLockError};
use std::io;
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, PoisonError, RwLock};
use std::thread::JoinHandle;
use std::time::Duration;

use storage::framed;
use storage::wal::SyncPolicy;

use crate::collection::{Collection, Config, Error, Result};

const CONFIG: &str = "CONFIG";
const CONFIG_MAGIC: [u8; 8] = *b"NPCOLCFG";
const CONFIG_VERSION: u32 = 1;
/// How long a background thread sleeps after a step with no work.
const IDLE: Duration = Duration::from_millis(500);

/// Set (and notified) to stop a background thread.
type Stop = (Mutex<bool>, Condvar);

pub struct Db {
    dir: PathBuf,
    sync: SyncPolicy,
    colls: RwLock<BTreeMap<String, Entry>>,
    /// Serializes create and drop, so the slow file work stays out of
    /// `colls`, which every request reads.
    admin: Mutex<()>,
    /// Holds the exclusive lock on `<data>/LOCK` while the Db lives
    /// (declared after `colls`, so it is released after every join).
    _lock: File,
}

impl Drop for Db {
    /// Stops every background thread at once; each is then joined after
    /// its step in flight as `colls` drops.
    fn drop(&mut self) {
        let colls = self.colls.get_mut().unwrap_or_else(PoisonError::into_inner);
        for e in colls.values() {
            e.signal();
        }
    }
}

/// A collection and its background thread. The thread holds only its
/// collection, never `colls` or `admin`, so joining it under either
/// can't deadlock.
// ponytail: one OS thread per collection, idle in a condvar wait between
// steps; fine to hundreds of collections. A few workers pulling from a
// queue if collections reach the thousands.
struct Entry {
    coll: Arc<Collection>,
    stop: Arc<Stop>,
    thread: Option<JoinHandle<()>>,
}

impl Entry {
    fn spawn(name: &str, coll: Collection) -> io::Result<Entry> {
        let coll = Arc::new(coll);
        let stop = Arc::new((Mutex::new(false), Condvar::new()));
        let thread = {
            let (n, c, s) = (name.to_owned(), Arc::clone(&coll), Arc::clone(&stop));
            std::thread::Builder::new()
                .name(format!("np-maintain-{name}"))
                .spawn(move || maintain_loop(&n, &c, &s))?
        };
        Ok(Entry {
            coll,
            stop,
            thread: Some(thread),
        })
    }

    fn signal(&self) {
        *self.stop.0.lock().unwrap_or_else(PoisonError::into_inner) = true;
        self.stop.1.notify_all();
    }
}

impl Drop for Entry {
    /// Stops and joins the thread after its step in flight.
    fn drop(&mut self) {
        self.signal();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// A collection's background thread: `maintain` steps until stopped;
/// errors and panics are logged and retried after the idle wait.
fn maintain_loop(name: &str, c: &Collection, stop: &Stop) {
    loop {
        // A panic must not end this collection's maintenance.
        let busy = match panic::catch_unwind(AssertUnwindSafe(|| c.maintain())) {
            Ok(Ok(did)) => did,
            Ok(Err(Error::Closed)) => false,
            Ok(Err(e)) => {
                tracing::error!(collection = %name, "background step failed: {e}");
                false
            }
            Err(_) => {
                tracing::error!(collection = %name, "background step panicked");
                false
            }
        };
        let mut stopped = stop.0.lock().unwrap_or_else(PoisonError::into_inner);
        if !busy && !*stopped {
            stopped = stop
                .1
                .wait_timeout(stopped, IDLE)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
        if *stopped {
            return;
        }
    }
}

impl Db {
    /// Opens (or creates) the data directory and every collection in it.
    /// Fails if another process has it open.
    pub fn open(dir: impl AsRef<Path>, sync: SyncPolicy) -> Result<Db> {
        let root = dir.as_ref();
        let dir = root.join("collections");
        fs::create_dir_all(&dir)?;
        let path = root.join("LOCK");
        let lock = File::options()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&path)?;
        lock.try_lock().map_err(|e| match e {
            TryLockError::WouldBlock => Error::Locked(path),
            TryLockError::Error(e) => Error::Io(e),
        })?;
        let mut colls = BTreeMap::new();
        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let Ok(name) = entry.file_name().into_string() else {
                continue;
            };
            if name.starts_with('.') {
                fs::remove_dir_all(entry.path())?;
            } else if check_name(&name).is_ok() {
                let c = open_collection(&entry.path(), sync)
                    .map_err(|e| Error::Open(name.clone(), Box::new(e)))?;
                let e = Entry::spawn(&name, c)?;
                colls.insert(name, e);
            }
        }
        sync_dir(&dir)?;
        Ok(Db {
            dir,
            sync,
            colls: RwLock::new(colls),
            admin: Mutex::new(()),
            _lock: lock,
        })
    }

    /// Creates collection `name`; `cfg.sync` is ignored (the Db's applies).
    pub fn create(&self, name: &str, mut cfg: Config) -> Result<Arc<Collection>> {
        check_name(name)?;
        cfg.validate()?;
        cfg.sync = self.sync;
        let _admin = self.admin.lock().unwrap_or_else(PoisonError::into_inner);
        if self.get(name).is_some() {
            return Err(Error::Exists(name.to_owned()));
        }
        let body = serde_json::to_vec(&cfg).map_err(io::Error::other)?;
        let staging = self.dir.join(format!(".new-{name}"));
        remove_if_exists(&staging)?;
        fs::create_dir(&staging)?;
        let path = staging.join(CONFIG);
        framed::write(&path, CONFIG_MAGIC, CONFIG_VERSION, &body)
            .map_err(|e| Error::File(path, e))?;
        let path = self.dir.join(name);
        fs::rename(&staging, &path)?;
        sync_dir(&self.dir)?;
        let (c, _) = Collection::open(&path, cfg)?;
        let e = Entry::spawn(name, c)?;
        let c = Arc::clone(&e.coll);
        let mut colls = self.colls.write().unwrap_or_else(PoisonError::into_inner);
        colls.insert(name.to_owned(), e);
        Ok(c)
    }

    pub fn get(&self, name: &str) -> Option<Arc<Collection>> {
        let colls = self.colls.read().unwrap_or_else(PoisonError::into_inner);
        colls.get(name).map(|e| Arc::clone(&e.coll))
    }

    /// Every collection, by name.
    pub fn list(&self) -> Vec<(String, Arc<Collection>)> {
        let colls = self.colls.read().unwrap_or_else(PoisonError::into_inner);
        colls
            .iter()
            .map(|(n, e)| (n.clone(), Arc::clone(&e.coll)))
            .collect()
    }

    /// Drops collection `name` and deletes its files. Writes and the
    /// background step still in flight on it finish first (its thread is
    /// joined); later writes fail with `Closed`.
    pub fn drop_collection(&self, name: &str) -> Result<()> {
        let _admin = self.admin.lock().unwrap_or_else(PoisonError::into_inner);
        let e = {
            let mut colls = self.colls.write().unwrap_or_else(PoisonError::into_inner);
            colls.remove(name)
        }
        .ok_or_else(|| Error::NotFound(name.to_owned()))?;
        // Joined outside the `colls` lock, so requests never wait on it.
        let c = Arc::clone(&e.coll);
        drop(e);
        c.close();
        let tomb = self.dir.join(format!(".drop-{name}"));
        remove_if_exists(&tomb)?;
        fs::rename(self.dir.join(name), &tomb)?;
        sync_dir(&self.dir)?;
        fs::remove_dir_all(&tomb)?;
        Ok(())
    }
}

fn open_collection(dir: &Path, sync: SyncPolicy) -> Result<Collection> {
    let path = dir.join(CONFIG);
    let body =
        framed::read(&path, CONFIG_MAGIC, CONFIG_VERSION).map_err(|e| Error::File(path, e))?;
    let mut cfg: Config =
        serde_json::from_slice(&body).map_err(|e| Error::Corrupt(format!("{CONFIG}: {e}")))?;
    cfg.sync = sync;
    Ok(Collection::open(dir, cfg)?.0)
}

/// 1–64 of `[A-Za-z0-9_-]`: safe as a directory name and in a URL path,
/// and never starts with the `.` that marks work in progress.
fn check_name(name: &str) -> Result<()> {
    let ok = (1..=64).contains(&name.len())
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');
    if !ok {
        return Err(Error::Invalid(
            "collection names are 1-64 characters of A-Z a-z 0-9 _ -",
        ));
    }
    Ok(())
}

fn remove_if_exists(dir: &Path) -> io::Result<()> {
    match fs::remove_dir_all(dir) {
        Err(e) if e.kind() != io::ErrorKind::NotFound => Err(e),
        _ => Ok(()),
    }
}

fn sync_dir(dir: &Path) -> io::Result<()> {
    File::open(dir)?.sync_all()
}

#[cfg(test)]
mod tests {
    use super::*;
    use index::{Metric, hnsw};
    use serde_json::{Value, json};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            static N: AtomicUsize = AtomicUsize::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            let p =
                std::env::temp_dir().join(format!("nearpoint-db-{tag}-{}-{n}", std::process::id()));
            let _ = fs::remove_dir_all(&p);
            TempDir(p)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn cfg(v: Value) -> Config {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn config_json_defaults_and_rejects_unknown_fields() {
        let c = cfg(json!({"dim": 3, "metric": "cosine"}));
        assert_eq!((c.dim, c.metric), (3, Metric::Cosine));
        assert_eq!(c.hnsw.m, hnsw::Config::default().m);
        let c = cfg(
            json!({"dim": 3, "metric": "inner_product", "hnsw": {"m": 8},
                           "schema": {"price": "float"}}),
        );
        assert_eq!((c.hnsw.m, c.hnsw.seed), (8, hnsw::Config::default().seed));
        assert_eq!(c.schema["price"], crate::Kind::Float);
        for bad in [
            json!({"dim": 3, "metric": "l1"}),
            json!({"dim": 3, "metric": "l2", "sync": "always"}),
            json!({"dim": 3, "metric": "l2", "hnsw": {"M": 8}}),
            json!({"dim": 3, "metric": "l2", "schema": {"x": "date"}}),
        ] {
            assert!(serde_json::from_value::<Config>(bad).is_err());
        }
    }

    #[test]
    fn create_reopen_drop_and_lock() {
        let d = TempDir::new("reg");
        let db = Db::open(&d.0, SyncPolicy::Batch).unwrap();
        assert!(matches!(
            Db::open(&d.0, SyncPolicy::Batch),
            Err(Error::Locked(_))
        ));
        let spec = json!({"dim": 2, "metric": "l2", "hnsw": {"m": 4, "ef_construction": 8},
                          "schema": {"n": "float"}});
        let a = db.create("a", cfg(spec.clone())).unwrap();
        a.upsert(7, &[1.0, 2.0], &json!({"n": 1})).unwrap();
        db.create("b-2_", cfg(spec.clone())).unwrap();
        assert!(matches!(
            db.create("a", cfg(spec.clone())),
            Err(Error::Exists(_))
        ));
        for bad in ["", ".a", "a/b", "..", "é", &"x".repeat(65)] {
            assert!(matches!(
                db.create(bad, cfg(spec.clone())),
                Err(Error::Invalid(_))
            ));
        }
        assert!(matches!(
            db.create("z", cfg(json!({"dim": 0, "metric": "l2"}))),
            Err(Error::Invalid(_))
        ));
        assert!(!d.0.join("collections/z").exists());
        db.drop_collection("b-2_").unwrap();
        assert!(matches!(a.upsert(8, &[0.0; 2], &Value::Null), Ok(())));
        assert!(matches!(
            db.drop_collection("b-2_"),
            Err(Error::NotFound(_))
        ));
        drop((a, db));

        // Leftovers of a crashed create and drop are removed on open.
        fs::create_dir_all(d.0.join("collections/.new-x")).unwrap();
        fs::create_dir_all(d.0.join("collections/.drop-y/wal")).unwrap();
        let db = Db::open(&d.0, SyncPolicy::Always).unwrap();
        let names: Vec<_> = db.list().into_iter().map(|(n, _)| n).collect();
        assert_eq!(names, ["a"]);
        let a = db.get("a").unwrap();
        assert_eq!(a.config().schema["n"], crate::Kind::Float);
        assert_eq!(a.config().hnsw.m, 4);
        assert_eq!(a.get(7).unwrap().1, json!({"n": 1.0}));
        assert_eq!(a.len(), 2);
        let mut left: Vec<_> = fs::read_dir(d.0.join("collections"))
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        left.sort();
        assert_eq!(left, ["a"]);
        db.drop_collection("a").unwrap();
        assert!(matches!(a.delete(7), Err(Error::Closed)));
        assert!(!d.0.join("collections/a").exists());
    }

    /// Committed v1 `CONFIG` (written before the segment keys existed):
    /// must keep loading, with the segment keys defaulted.
    #[test]
    fn v1_config_fixture_still_loads() {
        let d = TempDir::new("cfg");
        fs::create_dir_all(&d.0).unwrap();
        let fixture: &[u8] = include_bytes!("../testdata/config-v1");
        fs::write(d.0.join(CONFIG), fixture).unwrap();
        let c = open_collection(&d.0, SyncPolicy::Batch).unwrap();
        let cfg = c.config();
        assert_eq!((cfg.dim, cfg.hnsw.seed), (4, 7));
        assert_eq!(cfg.schema["tag"], crate::Kind::Keyword);
        assert_eq!(
            (cfg.seal_rows, cfg.ivf_rows, cfg.max_merge_rows),
            (100_000, 1_000_000, 4_000_000)
        );
        // New collections write the segment keys (still CONFIG v1: JSON
        // keys are optional, so this is a compatible addition).
        let v = serde_json::to_value(cfg).unwrap();
        assert_eq!(v["seal_rows"], 100_000);
        assert!(
            serde_json::from_value::<Config>(json!({"dim": 4, "metric": "l2", "ivf_rows": 10}))
                .unwrap()
                .validate()
                .is_err()
        );
    }

    /// The background thread seals a collection past `seal_rows`, and a
    /// drop waits for its step in flight before deleting the directory.
    #[test]
    fn background_thread_seals_and_drop_waits() {
        let d = TempDir::new("bg");
        let db = Db::open(&d.0, SyncPolicy::Batch).unwrap();
        let c = db
            .create(
                "bg",
                cfg(json!({"dim": 2, "metric": "l2", "seal_rows": 10})),
            )
            .unwrap();
        for i in 0..25u64 {
            c.upsert(i, &[i as f32, 1.0], &Value::Null).unwrap();
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        while c.segments().len() < 2 {
            assert!(std::time::Instant::now() < deadline, "never sealed");
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert_eq!(c.len(), 25);
        db.drop_collection("bg").unwrap();
        assert!(!d.0.join("collections/bg").exists());
        assert!(matches!(c.maintain(), Ok(false)));
        drop(db); // joins the thread
    }

    fn wait_for(what: &str, f: impl Fn() -> bool) {
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        while !f() {
            assert!(std::time::Instant::now() < deadline, "never {what}");
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// Each collection has its own thread: one stuck mid-step (as in a
    /// long build) doesn't keep another from sealing.
    #[test]
    fn blocked_collection_does_not_delay_another() {
        let d = TempDir::new("indep");
        let db = Db::open(&d.0, SyncPolicy::Batch).unwrap();
        let spec = json!({"dim": 2, "metric": "l2", "seal_rows": 10});
        let a = db.create("a", cfg(spec.clone())).unwrap();
        let b = db.create("b", cfg(spec)).unwrap();
        let hold = a.hold_maint();
        for i in 0..25u64 {
            a.upsert(i, &[i as f32, 1.0], &Value::Null).unwrap();
            b.upsert(i, &[i as f32, 1.0], &Value::Null).unwrap();
        }
        wait_for("sealed b", || b.segments().len() >= 2);
        assert_eq!(a.segments().len(), 1);
        drop(hold);
        wait_for("sealed a", || a.segments().len() >= 2);
    }

    /// Dropping a collection, and dropping the Db, wait for a step in
    /// flight: the files stay and LOCK stays held until it ends.
    #[test]
    fn drops_wait_for_a_step_in_flight() {
        let d = TempDir::new("join");
        let db = Db::open(&d.0, SyncPolicy::Batch).unwrap();
        let spec = json!({"dim": 2, "metric": "l2"});
        let a = db.create("a", cfg(spec.clone())).unwrap();
        let b = db.create("b", cfg(spec)).unwrap();
        let (hold_a, hold_b) = (a.hold_maint(), b.hold_maint());
        // Past the idle wait, both threads are blocked inside `maintain`.
        std::thread::sleep(IDLE * 3);
        std::thread::scope(|s| {
            let t = s.spawn(|| db.drop_collection("a"));
            std::thread::sleep(Duration::from_millis(200));
            assert!(!t.is_finished());
            assert!(d.0.join("collections/a").exists());
            assert!(db.get("a").is_none() && db.get("b").is_some());
            drop(hold_a);
            t.join().unwrap().unwrap();
        });
        assert!(!d.0.join("collections/a").exists());

        let t = std::thread::spawn(move || drop(db));
        std::thread::sleep(Duration::from_millis(200));
        assert!(!t.is_finished());
        assert!(matches!(
            Db::open(&d.0, SyncPolicy::Batch),
            Err(Error::Locked(_))
        ));
        drop(hold_b);
        t.join().unwrap();
        Db::open(&d.0, SyncPolicy::Batch).unwrap();
    }
}
