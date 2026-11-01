//! Recall/latency harness and dataset loaders.

pub mod dataset;

use std::fs::File;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// `path` with a leading `~/` replaced by `$HOME`.
pub fn expand_home(path: &str) -> io::Result<PathBuf> {
    match path.strip_prefix("~/") {
        Some(rest) => std::env::var_os("HOME")
            .map(|h| PathBuf::from(h).join(rest))
            .ok_or_else(|| io::Error::other("HOME unset")),
        None => Ok(PathBuf::from(path)),
    }
}

/// Opens `path` for appending CSV rows, writing `header` first if the
/// file is new.
pub fn open_csv(path: &Path, header: &str) -> io::Result<File> {
    let new = !path.exists();
    let mut f = File::options().create(true).append(true).open(path)?;
    if new {
        writeln!(f, "{header}")?;
    }
    Ok(f)
}

/// The `p` quantile (0..=1, nearest rank) of a sorted, non-empty slice.
pub fn percentile<T: Copy>(sorted: &[T], p: f64) -> T {
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

pub fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

pub fn mib(bytes: u64) -> f64 {
    bytes as f64 / f64::from(1 << 20)
}
