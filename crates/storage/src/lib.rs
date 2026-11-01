//! WAL, segment file format, mmap and checksums.

#![allow(unsafe_code)]

pub mod billion;
pub mod framed;
pub mod segment;
pub mod sys;
pub mod wal;

use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};

fn le_u32(b: &[u8], at: usize) -> u32 {
    let mut a = [0; 4];
    a.copy_from_slice(&b[at..at + 4]);
    u32::from_le_bytes(a)
}

fn le_u64(b: &[u8], at: usize) -> u64 {
    let mut a = [0; 8];
    a.copy_from_slice(&b[at..at + 8]);
    u64::from_le_bytes(a)
}

fn sync_dir(dir: &Path) -> io::Result<()> {
    File::open(dir)?.sync_all()
}

/// `<path>.tmp`: where a file is built before it's renamed into place.
fn tmp_path(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".tmp");
    PathBuf::from(s)
}

/// Replaces `path` atomically: `write` fills `<path>.tmp` (and syncs it),
/// which is renamed to `path`, then the parent directory is fsynced.
/// Readers never see a partial file; on error the tmp file is removed.
fn replace_atomically<E: From<io::Error>>(
    path: &Path,
    write: impl FnOnce(&Path) -> Result<(), E>,
) -> Result<(), E> {
    let tmp = tmp_path(path);
    let res = write(&tmp).and_then(|()| Ok(fs::rename(&tmp, path)?));
    if res.is_err() {
        let _ = fs::remove_file(&tmp); // best effort; the error is what matters
    }
    res?;
    let dir = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    Ok(sync_dir(dir)?)
}
