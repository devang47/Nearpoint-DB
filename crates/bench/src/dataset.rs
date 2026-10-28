//! Dataset loaders for the texmex formats (`.fvecs`, `.ivecs`, `.bvecs`)
//! and the big-ann-benchmarks formats (`.fbin`, `.u8bin`, `.i8bin`,
//! `.ibin`, and their ground-truth `.ibin` variant).
//!
//! Both read straight into a flat row-major `Vec<T>` with no per-element
//! parsing: the bytes are read in place through `bytemuck`. Every length is
//! checked against the file size *before* allocating, so a corrupt header
//! can't trigger a huge allocation.

use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};

use bytemuck::Pod;

// Files are little-endian and read in place, so the host must be too.
const _: () = assert!(cfg!(target_endian = "little"));

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{path}: {source}")]
    Io { path: PathBuf, source: io::Error },
    #[error("{path}: {msg}")]
    Format { path: PathBuf, msg: String },
}

pub type Result<T> = std::result::Result<T, Error>;

/// `len()` vectors of `dim()` elements each, stored row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct Vectors<T> {
    dim: usize,
    data: Vec<T>,
}

impl<T> Vectors<T> {
    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.data.len() / self.dim
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Vector `i`. Panics if `i >= len()`, like slice indexing.
    pub fn row(&self, i: usize) -> &[T] {
        &self.data[i * self.dim..][..self.dim]
    }

    pub fn as_flat(&self) -> &[T] {
        &self.data
    }
}

/// Reads `.fvecs` (`T = f32`), `.ivecs` (`i32`) or `.bvecs` (`u8`): each
/// vector is an `i32` dim followed by `dim` elements. Reads the first
/// `limit` vectors, or all of them.
pub fn read_xvecs<T: Pod>(path: &Path, limit: Option<usize>) -> Result<Vectors<T>> {
    let (mut r, len) = open(path)?;
    let dim = read_u32(&mut r, path)? as usize;
    check_dim(dim, path)?;
    let record = 4 + (dim * size_of::<T>()) as u64;
    if len % record != 0 {
        return Err(format_err(
            path,
            format!(
                "size {len} is not a multiple of the {record}-byte record \
                 (wrong element type?)"
            ),
        ));
    }
    let n = take(len / record, limit, path)?;

    let mut data = vec![T::zeroed(); n * dim];
    for (i, row) in data.chunks_exact_mut(dim).enumerate() {
        if i > 0 {
            let d = read_u32(&mut r, path)? as usize;
            if d != dim {
                return Err(format_err(
                    path,
                    format!("vector {i} has dim {d}, expected {dim}"),
                ));
            }
        }
        read_into(&mut r, row, path)?;
    }
    Ok(Vectors { dim, data })
}

/// Reads `.fbin` (`T = f32`), `.u8bin` (`u8`), `.i8bin` (`i8`) or `.ibin`
/// (`u32`): a `u32` count, a `u32` dim, then `count * dim` elements.
///
/// With `limit`, only the first `limit` vectors need to be present, so a
/// byte-range prefix of a 1B file (whose header still says 1B) loads fine.
/// Without it the file size must match the header exactly.
pub fn read_bin<T: Pod>(path: &Path, limit: Option<usize>) -> Result<Vectors<T>> {
    let (mut r, len) = open(path)?;
    let count = u64::from(read_u32(&mut r, path)?);
    let dim = read_u32(&mut r, path)? as usize;
    check_dim(dim, path)?;
    let n = take(count, limit, path)?;
    let row_bytes = (dim * size_of::<T>()) as u64;

    let need = 8 + n as u64 * row_bytes;
    if limit.is_none() && len != need {
        return Err(format_err(
            path,
            format!(
                "size {len}, header implies {need} \
                 (wrong element type, or a truncated download?)"
            ),
        ));
    }
    if len < need {
        return Err(format_err(
            path,
            format!(
                "holds {} vectors, {n} requested (truncated?)",
                (len.saturating_sub(8)) / row_bytes
            ),
        ));
    }

    let mut data = vec![T::zeroed(); n * dim];
    read_into(&mut r, &mut data, path)?;
    Ok(Vectors { dim, data })
}

/// Reads a big-ann-benchmarks ground-truth file: a `u32` query count `nq`,
/// a `u32` `k`, then `nq * k` `u32` ids, then `nq * k` `f32` distances.
pub fn read_bin_ground_truth(path: &Path) -> Result<(Vectors<u32>, Vectors<f32>)> {
    let (mut r, len) = open(path)?;
    let nq = read_u32(&mut r, path)? as usize;
    let k = read_u32(&mut r, path)? as usize;
    check_dim(k, path)?;
    let need = 8 + (nq * k * 8) as u64;
    if len != need {
        return Err(format_err(
            path,
            format!("size {len}, header ({nq} queries, k={k}) implies {need}"),
        ));
    }
    let mut ids = vec![0u32; nq * k];
    let mut dists = vec![0f32; nq * k];
    read_into(&mut r, &mut ids, path)?;
    read_into(&mut r, &mut dists, path)?;
    Ok((
        Vectors { dim: k, data: ids },
        Vectors {
            dim: k,
            data: dists,
        },
    ))
}

fn open(path: &Path) -> Result<(BufReader<File>, u64)> {
    let file = File::open(path).map_err(|e| io_err(path, e))?;
    let len = file.metadata().map_err(|e| io_err(path, e))?.len();
    Ok((BufReader::new(file), len))
}

fn read_u32(r: &mut impl Read, path: &Path) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(|e| io_err(path, e))?;
    Ok(u32::from_le_bytes(b))
}

fn read_into<T: Pod>(r: &mut impl Read, out: &mut [T], path: &Path) -> Result<()> {
    r.read_exact(bytemuck::cast_slice_mut(out))
        .map_err(|e| io_err(path, e))
}

fn check_dim(dim: usize, path: &Path) -> Result<()> {
    // Sanity bound: real datasets top out in the low thousands.
    if dim == 0 || dim > 65_536 {
        return Err(format_err(path, format!("implausible dim {dim}")));
    }
    Ok(())
}

/// How many vectors to read: all `available`, or `limit` if it fits.
fn take(available: u64, limit: Option<usize>, path: &Path) -> Result<usize> {
    match limit {
        None => Ok(available as usize),
        Some(n) if n as u64 <= available => Ok(n),
        Some(n) => Err(format_err(
            path,
            format!("{n} vectors requested, file has {available}"),
        )),
    }
}

fn io_err(path: &Path, source: io::Error) -> Error {
    Error::Io {
        path: path.to_owned(),
        source,
    }
}

fn format_err(path: &Path, msg: String) -> Error {
    Error::Format {
        path: path.to_owned(),
        msg,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A file in the temp dir, deleted on drop.
    struct Tmp(PathBuf);

    impl Tmp {
        fn new(name: &str, bytes: &[u8]) -> Self {
            let p = std::env::temp_dir().join(format!("nearpoint-{}-{name}", std::process::id()));
            std::fs::write(&p, bytes).unwrap();
            Tmp(p)
        }
    }

    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn xvecs<T: Pod>(rows: &[&[T]]) -> Vec<u8> {
        let mut out = Vec::new();
        for row in rows {
            out.extend((row.len() as u32).to_le_bytes());
            out.extend(bytemuck::cast_slice(row));
        }
        out
    }

    fn bin<T: Pod>(count: u32, dim: u32, data: &[T]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(count.to_le_bytes());
        out.extend(dim.to_le_bytes());
        out.extend(bytemuck::cast_slice(data));
        out
    }

    #[test]
    fn fvecs_round_trip_and_limit() {
        let rows: [&[f32]; 3] = [&[1.0, 2.0], &[3.0, 4.0], &[5.0, -6.5]];
        let f = Tmp::new("a.fvecs", &xvecs(&rows));

        let all = read_xvecs::<f32>(&f.0, None).unwrap();
        assert_eq!((all.len(), all.dim()), (3, 2));
        assert_eq!(all.row(2), &[5.0, -6.5]);

        let two = read_xvecs::<f32>(&f.0, Some(2)).unwrap();
        assert_eq!(two.as_flat(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn bvecs_and_ivecs() {
        let f = Tmp::new("b.bvecs", &xvecs::<u8>(&[&[1, 2, 3], &[255, 0, 7]]));
        assert_eq!(
            read_xvecs::<u8>(&f.0, None).unwrap().as_flat(),
            &[1, 2, 3, 255, 0, 7]
        );
        let f = Tmp::new("c.ivecs", &xvecs::<i32>(&[&[-1, 9]]));
        assert_eq!(read_xvecs::<i32>(&f.0, None).unwrap().row(0), &[-1, 9]);
    }

    #[test]
    fn xvecs_rejects_bad_files() {
        let f = Tmp::new("d.fvecs", &xvecs::<f32>(&[&[1.0, 2.0], &[3.0, 4.0]]));
        // Read as the wrong element type: record size doesn't divide the file.
        assert!(read_xvecs::<u8>(&f.0, None).is_err());
        // Asking for more than exists.
        assert!(read_xvecs::<f32>(&f.0, Some(3)).is_err());

        // Same total size as two dim-2 records, but the second claims dim 3.
        let mut bytes = xvecs::<f32>(&[&[1.0, 2.0]]);
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(bytemuck::cast_slice(&[0f32, 0.0]));
        let f = Tmp::new("e.fvecs", &bytes);
        assert!(read_xvecs::<f32>(&f.0, None).is_err());

        let f = Tmp::new("f.fvecs", &[]);
        assert!(read_xvecs::<f32>(&f.0, None).is_err());
        let f = Tmp::new("g.fvecs", &0u32.to_le_bytes());
        assert!(read_xvecs::<f32>(&f.0, None).is_err());
    }

    #[test]
    fn fbin_round_trip() {
        let data = [0.5f32, 1.5, 2.5, 3.5, 4.5, 5.5];
        let f = Tmp::new("h.fbin", &bin(3, 2, &data));
        let v = read_bin::<f32>(&f.0, None).unwrap();
        assert_eq!((v.len(), v.dim()), (3, 2));
        assert_eq!(v.as_flat(), &data);
        assert_eq!(read_bin::<f32>(&f.0, Some(1)).unwrap().row(0), &[0.5, 1.5]);
    }

    #[test]
    fn bin_prefix_of_a_bigger_file() {
        // Header says 1B vectors; only 2 are present, like a range download.
        let f = Tmp::new(
            "i.u8bin",
            &bin::<u8>(1_000_000_000, 4, &[1, 2, 3, 4, 5, 6, 7, 8]),
        );
        let v = read_bin::<u8>(&f.0, Some(2)).unwrap();
        assert_eq!(v.row(1), &[5, 6, 7, 8]);
        // Without a limit the size mismatch is an error, not a huge allocation.
        assert!(read_bin::<u8>(&f.0, None).is_err());
        // More than the prefix holds.
        assert!(read_bin::<u8>(&f.0, Some(3)).is_err());
    }

    #[test]
    fn bin_rejects_wrong_type_and_zero_dim() {
        let f = Tmp::new("j.fbin", &bin(2, 2, &[1f32, 2.0, 3.0, 4.0]));
        assert!(read_bin::<u8>(&f.0, None).is_err());
        let f = Tmp::new("k.fbin", &bin::<f32>(5, 0, &[]));
        assert!(read_bin::<f32>(&f.0, None).is_err());
    }

    #[test]
    fn ground_truth_round_trip() {
        let mut bytes = bin::<u32>(2, 2, &[7, 3, 1, 9]);
        bytes.extend(bytemuck::cast_slice(&[0.1f32, 0.2, 0.3, 0.4]));
        let f = Tmp::new("l.ibin", &bytes);
        let (ids, dists) = read_bin_ground_truth(&f.0).unwrap();
        assert_eq!(ids.row(1), &[1, 9]);
        assert_eq!(dists.row(0), &[0.1, 0.2]);
        // Plain ids-only reader sees the trailing distances as a size mismatch.
        assert!(read_bin::<u32>(&f.0, None).is_err());
    }
}
