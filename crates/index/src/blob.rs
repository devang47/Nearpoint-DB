//! Little-endian encoding shared by the index blobs that go into a
//! segment's index section (layouts in docs/formats/segment.md).
//! Every blob starts with `magic[4], version u32, dim u32, metric u32`.

use crate::{Error, Metric, Result};

/// Version of every blob kind this build writes and the newest it reads.
pub(crate) const VERSION: u32 = 1;

pub(crate) fn put_u32(out: &mut Vec<u8>, x: u32) {
    out.extend_from_slice(&x.to_le_bytes());
}

pub(crate) fn put_u64(out: &mut Vec<u8>, x: u64) {
    out.extend_from_slice(&x.to_le_bytes());
}

pub(crate) fn put_header(out: &mut Vec<u8>, magic: &[u8; 4], dim: usize, metric: Metric) {
    out.extend_from_slice(magic);
    put_u32(out, VERSION);
    put_u32(out, dim as u32);
    put_u32(out, metric.code());
}

/// Reads a blob front to back; running out of bytes is `Corrupt`.
pub(crate) struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Reader(bytes)
    }

    pub(crate) fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let (head, tail) = self
            .0
            .split_at_checked(n)
            .ok_or(Error::Corrupt("blob truncated"))?;
        self.0 = tail;
        Ok(head)
    }

    /// `count` items of `width` bytes, with the size checked for overflow.
    pub(crate) fn take_n(&mut self, count: usize, width: usize) -> Result<&'a [u8]> {
        let n = count
            .checked_mul(width)
            .ok_or(Error::Corrupt("blob size overflows"))?;
        self.take(n)
    }

    pub(crate) fn u32(&mut self) -> Result<u32> {
        let mut a = [0; 4];
        a.copy_from_slice(self.take(4)?);
        Ok(u32::from_le_bytes(a))
    }

    pub(crate) fn u64(&mut self) -> Result<u64> {
        let mut a = [0; 8];
        a.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(a))
    }

    /// Checks magic and version; returns `(dim, metric)`.
    pub(crate) fn header(&mut self, magic: &[u8; 4]) -> Result<(usize, Metric)> {
        if self.take(4)? != magic {
            return Err(Error::Corrupt("bad blob magic"));
        }
        let version = self.u32()?;
        if version != VERSION {
            return Err(Error::UnsupportedBlobVersion(version));
        }
        let dim = self.u32()? as usize;
        let metric = Metric::from_code(self.u32()?).ok_or(Error::Corrupt("unknown metric"))?;
        if dim == 0 {
            return Err(Error::Corrupt("dim is zero"));
        }
        Ok((dim, metric))
    }

    pub(crate) fn finish(self) -> Result<()> {
        if !self.0.is_empty() {
            return Err(Error::Corrupt("trailing bytes after blob"));
        }
        Ok(())
    }
}

/// Checks that the vectors handed to `from_bytes` match the blob's count.
pub(crate) fn check_vectors(vectors: &[f32], n: u64, dim: usize) -> Result<()> {
    if u64::try_from(vectors.len()).ok() != n.checked_mul(dim as u64) {
        return Err(Error::Corrupt("vector count does not match the blob"));
    }
    Ok(())
}
