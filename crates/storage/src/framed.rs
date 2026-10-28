//! Small checksummed files written in one piece: `magic | version | len |
//! body | crc32`. The collection manifest and payload sidecars use it
//! (docs/formats/manifest.md); the body layout belongs to the caller.

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;

const HEADER_LEN: usize = 20; // magic 8 | version u32 | body_len u64
const CRC_LEN: usize = 4;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("i/o: {0}")]
    Io(#[from] io::Error),
    #[error("bad magic (expected {0:?})")]
    BadMagic(String),
    #[error("format version {found} is not supported (this build reads {expected})")]
    UnsupportedVersion { found: u32, expected: u32 },
    #[error("file is truncated or has a bad length")]
    Truncated,
    #[error("checksum mismatch")]
    Checksum,
}

pub type Result<T> = std::result::Result<T, Error>;

/// Writes `path` atomically: `<path>.tmp`, `sync_data`, rename, then fsync
/// the parent directory. Replaces any existing file; on error the tmp file
/// is removed.
pub fn write(path: &Path, magic: [u8; 8], version: u32, body: &[u8]) -> Result<()> {
    let mut buf = Vec::with_capacity(HEADER_LEN + body.len() + CRC_LEN);
    buf.extend_from_slice(&magic);
    buf.extend_from_slice(&version.to_le_bytes());
    buf.extend_from_slice(&(body.len() as u64).to_le_bytes());
    buf.extend_from_slice(body);
    let crc = crc32fast::hash(&buf);
    buf.extend_from_slice(&crc.to_le_bytes());

    let tmp = crate::segment::tmp_path(path);
    let res = (|| {
        let mut f = File::create(&tmp)?;
        f.write_all(&buf)?;
        f.sync_data()?;
        fs::rename(&tmp, path)
    })();
    if res.is_err() {
        let _ = fs::remove_file(&tmp); // best effort; the error is what matters
    }
    res?;
    let dir = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    File::open(dir)?.sync_all()?;
    Ok(())
}

/// Reads and verifies a file written by [`write`] and returns its body.
pub fn read(path: &Path, magic: [u8; 8], version: u32) -> Result<Vec<u8>> {
    let mut bytes = fs::read(path)?;
    let body = check(&bytes, magic, version)?;
    bytes.truncate(HEADER_LEN + body);
    bytes.drain(..HEADER_LEN);
    Ok(bytes)
}

/// Validates `b` and returns the body length. The version is checked
/// before any length or CRC, so a newer file reports its version.
fn check(b: &[u8], magic: [u8; 8], version: u32) -> Result<usize> {
    if b.get(..8).is_some_and(|m| m != magic) {
        return Err(Error::BadMagic(
            String::from_utf8_lossy(&magic).into_owned(),
        ));
    }
    if b.len() < HEADER_LEN + CRC_LEN {
        return Err(Error::Truncated);
    }
    let found = u32::from_le_bytes([b[8], b[9], b[10], b[11]]);
    if found != version {
        return Err(Error::UnsupportedVersion {
            found,
            expected: version,
        });
    }
    let mut len = [0u8; 8];
    len.copy_from_slice(&b[12..20]);
    let body = b.len() - HEADER_LEN - CRC_LEN;
    if u64::from_le_bytes(len) != body as u64 {
        return Err(Error::Truncated);
    }
    let (data, crc) = b.split_at(HEADER_LEN + body);
    if crc32fast::hash(data).to_le_bytes() != crc {
        return Err(Error::Checksum);
    }
    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: [u8; 8] = *b"NPTEST\0\0";

    fn tmp_dir(name: &str) -> std::path::PathBuf {
        let d =
            std::env::temp_dir().join(format!("nearpoint-framed-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn round_trip_flips_and_truncation() {
        let path = tmp_dir("rt").join("f");
        for body in [&b""[..], b"hello framed file"] {
            write(&path, M, 3, body).unwrap();
            assert!(!crate::segment::tmp_path(&path).exists());
            assert_eq!(read(&path, M, 3).unwrap(), body);
        }
        let img = fs::read(&path).unwrap();
        let mut bad = img.clone();
        for byte in 0..img.len() {
            for bit in 0..8 {
                bad[byte] ^= 1 << bit;
                assert!(check(&bad, M, 3).is_err(), "flip at {byte}.{bit}");
                bad[byte] ^= 1 << bit;
            }
        }
        for len in 0..img.len() {
            assert!(check(&img[..len], M, 3).is_err(), "truncated to {len}");
        }
        assert!(matches!(
            read(&path, *b"NPOTHER\0", 3),
            Err(Error::BadMagic(_))
        ));
        assert!(matches!(
            read(&path, M, 2),
            Err(Error::UnsupportedVersion { found: 3, .. })
        ));
    }
}
