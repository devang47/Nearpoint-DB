//! Segment file format v2 (v1: task 3.2). See docs/formats/segment.md.
//! v2 adds element type "none": a segment without f32 vectors (an IVF-PQ
//! segment whose index keeps its own codes). v1 files still load.
//!
//! A sealed segment is immutable: it is written once to `<name>.tmp`,
//! synced, renamed into place, and never modified after that (only
//! replaced or deleted). That is what makes mapping it safe.
//!
//! The index blob is opaque here: the `index` crate produces and parses it
//! (`storage` must not depend on `index`). The metric is likewise stored
//! as the raw code from `index::Metric::code`.

use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};

use memmap2::Mmap;

#[cfg(target_endian = "big")]
compile_error!("segment files are little-endian and their sections are mapped in place");

/// First and last 8 bytes of every segment file.
pub const MAGIC: [u8; 8] = *b"NPSEGMT\0";
/// The format version this build writes and the newest it reads (it
/// also reads v1).
pub const VERSION: u32 = 2;

const HEADER_LEN: usize = 64;
const FOOTER_LEN: usize = 96;
/// Every section starts on a multiple of this (from the start of the file).
const ALIGN: u64 = 64;
/// Element type codes: f32 vectors (v1, v2), or no vector section (v2).
const ELEM_F32: u32 = 0;
const ELEM_NONE: u32 = 1;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("segment i/o: {0}")]
    Io(#[from] io::Error),
    #[error("not a segment file (bad magic)")]
    BadMagic,
    #[error("segment format version {0} is not supported (this build reads 1..={VERSION})")]
    UnsupportedVersion(u32),
    #[error("segment element type {0} is not supported")]
    UnsupportedElementType(u32),
    #[error("segment is truncated (or its footer is damaged)")]
    Truncated,
    #[error("segment checksum mismatch in the {0}")]
    Checksum(&'static str),
    #[error("corrupt segment: {0}")]
    Corrupt(&'static str),
    #[error("invalid segment contents: {0}")]
    InvalidInput(&'static str),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Which index produced the blob.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlobKind {
    Flat,
    Hnsw,
    IvfPq,
}

impl BlobKind {
    fn code(self) -> u32 {
        match self {
            BlobKind::Flat => 0,
            BlobKind::Hnsw => 1,
            BlobKind::IvfPq => 2,
        }
    }

    fn from_code(code: u32) -> Option<Self> {
        match code {
            0 => Some(BlobKind::Flat),
            1 => Some(BlobKind::Hnsw),
            2 => Some(BlobKind::IvfPq),
            _ => None,
        }
    }
}

/// The contents of a segment to write.
#[derive(Debug, Clone, Copy)]
pub struct SegmentData<'a> {
    pub dim: usize,
    /// `index::Metric::code()`; not interpreted by storage.
    pub metric: u32,
    /// Row-major, `ids.len() * dim` long, in the form the index stores
    /// them (e.g. normalized for cosine). Empty with a non-empty `ids`:
    /// the segment has no vector section (the index blob is all there is).
    pub vectors: &'a [f32],
    /// External id of each vector; dense index id `i` is `ids[i]`.
    pub ids: &'a [u64],
    pub blob_kind: BlobKind,
    pub blob: &'a [u8],
}

/// Byte ranges of the three sections, plus the header fields.
#[derive(Debug, Clone, PartialEq)]
struct Layout {
    dim: usize,
    metric: u32,
    blob_kind: BlobKind,
    vectors: Range<usize>,
    ids: Range<usize>,
    blob: Range<usize>,
}

/// Where the sections go for these sizes; `None` on overflow. Without
/// vectors the vector section is empty.
fn sections(dim: u64, count: u64, vectors: bool, blob_len: u64) -> Option<[Range<u64>; 3]> {
    let v_start = HEADER_LEN as u64;
    let v_len = if vectors {
        count.checked_mul(dim)?.checked_mul(4)?
    } else {
        0
    };
    let v_end = v_start.checked_add(v_len)?;
    let i_start = v_end.checked_next_multiple_of(ALIGN)?;
    let i_end = i_start.checked_add(count.checked_mul(8)?)?;
    let b_start = i_end.checked_next_multiple_of(ALIGN)?;
    let b_end = b_start.checked_add(blob_len)?;
    b_end.checked_add(FOOTER_LEN as u64)?; // the file length must fit too
    Some([v_start..v_end, i_start..i_end, b_start..b_end])
}

fn usize_range(r: &Range<u64>) -> Option<Range<usize>> {
    Some(usize::try_from(r.start).ok()?..usize::try_from(r.end).ok()?)
}

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

const SECTION_NAMES: [&str; 3] = ["vector section", "id section", "index blob"];

/// Validates a whole segment image and returns its layout. Checks magic,
/// version, footer and header CRCs, that the section table matches the
/// header exactly, that padding is zero, and every section CRC.
fn parse(b: &[u8]) -> Result<Layout> {
    if b.len() >= MAGIC.len() && b[..MAGIC.len()] != MAGIC {
        return Err(Error::BadMagic);
    }
    if b.len() < HEADER_LEN + FOOTER_LEN {
        return Err(Error::Truncated);
    }
    let version = le_u32(b, 8);
    if !(1..=VERSION).contains(&version) {
        return Err(Error::UnsupportedVersion(version));
    }
    let footer = &b[b.len() - FOOTER_LEN..];
    if footer[88..] != MAGIC {
        return Err(Error::Truncated);
    }
    if crc32fast::hash(&footer[..84]) != le_u32(footer, 84) {
        return Err(Error::Checksum("footer"));
    }
    let header = &b[..HEADER_LEN];
    if crc32fast::hash(header) != le_u32(footer, 0) {
        return Err(Error::Checksum("header"));
    }

    let elem = le_u32(header, 12);
    if !(elem == ELEM_F32 || elem == ELEM_NONE && version >= 2) {
        return Err(Error::UnsupportedElementType(elem));
    }
    let dim = le_u32(header, 16);
    let metric = le_u32(header, 20);
    let count = le_u64(header, 24);
    let blob_kind =
        BlobKind::from_code(le_u32(header, 32)).ok_or(Error::Corrupt("unknown blob kind"))?;
    if dim == 0 {
        return Err(Error::Corrupt("dim is zero"));
    }

    // The section table is redundant with the header (one layout per
    // size and element type), so it must match exactly, and the footer
    // must follow the blob directly.
    let blob_len = le_u64(footer, 56 + 8);
    let expected = sections(u64::from(dim), count, elem == ELEM_F32, blob_len)
        .ok_or(Error::Corrupt("section sizes"))?;
    for (i, want) in expected.iter().enumerate() {
        let at = 8 + 24 * i;
        let off = le_u64(footer, at);
        let len = le_u64(footer, at + 8);
        if off != want.start || off.checked_add(len) != Some(want.end) {
            return Err(Error::Corrupt("section table does not match the header"));
        }
    }
    if expected[2].end + FOOTER_LEN as u64 != b.len() as u64 {
        return Err(Error::Corrupt(
            "file length does not match the section table",
        ));
    }
    let [vectors, ids, blob] = expected.map(|r| usize_range(&r));
    let (Some(vectors), Some(ids), Some(blob)) = (vectors, ids, blob) else {
        return Err(Error::Corrupt("section too large for this platform"));
    };

    // Padding isn't under any CRC, so require it to be zero instead.
    if b[vectors.end..ids.start]
        .iter()
        .chain(&b[ids.end..blob.start])
        .any(|&x| x != 0)
    {
        return Err(Error::Corrupt("nonzero padding"));
    }
    for (i, r) in [&vectors, &ids, &blob].into_iter().enumerate() {
        if crc32fast::hash(&b[r.clone()]) != le_u32(footer, 8 + 24 * i + 16) {
            return Err(Error::Checksum(SECTION_NAMES[i]));
        }
    }

    Ok(Layout {
        dim: dim as usize,
        metric,
        blob_kind,
        vectors,
        ids,
        blob,
    })
}

/// `<path>.tmp`: where a segment is built before it's renamed into place.
pub fn tmp_path(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".tmp");
    PathBuf::from(s)
}

/// Writes a segment atomically: to `<path>.tmp`, `sync_data`, rename to
/// `path`, then fsync the parent directory. Readers never see a partial
/// file under `path`; on error the tmp file is removed. Replaces any
/// existing file at `path`.
pub fn write(path: &Path, seg: &SegmentData<'_>) -> Result<()> {
    let count = seg.ids.len();
    if seg.dim == 0 || u32::try_from(seg.dim).is_err() {
        return Err(Error::InvalidInput("dim must be in 1..=u32::MAX"));
    }
    let vectors = !(seg.vectors.is_empty() && count > 0);
    if vectors && Some(seg.vectors.len()) != count.checked_mul(seg.dim) {
        return Err(Error::InvalidInput("vectors.len() != ids.len() * dim"));
    }
    let secs = sections(seg.dim as u64, count as u64, vectors, seg.blob.len() as u64)
        .ok_or(Error::InvalidInput("segment too large"))?;

    let tmp = tmp_path(path);
    let elem = if vectors { ELEM_F32 } else { ELEM_NONE };
    let res = write_file(&tmp, seg, &secs, elem).and_then(|()| Ok(fs::rename(&tmp, path)?));
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

fn write_file(tmp: &Path, seg: &SegmentData<'_>, secs: &[Range<u64>; 3], elem: u32) -> Result<()> {
    let mut header = [0u8; HEADER_LEN];
    header[..8].copy_from_slice(&MAGIC);
    header[8..12].copy_from_slice(&VERSION.to_le_bytes());
    header[12..16].copy_from_slice(&elem.to_le_bytes());
    header[16..20].copy_from_slice(&(seg.dim as u32).to_le_bytes());
    header[20..24].copy_from_slice(&seg.metric.to_le_bytes());
    header[24..32].copy_from_slice(&(seg.ids.len() as u64).to_le_bytes());
    header[32..36].copy_from_slice(&seg.blob_kind.code().to_le_bytes());

    let bodies: [&[u8]; 3] = [
        bytemuck::cast_slice(seg.vectors),
        bytemuck::cast_slice(seg.ids),
        seg.blob,
    ];
    let mut footer = [0u8; FOOTER_LEN];
    footer[..4].copy_from_slice(&crc32fast::hash(&header).to_le_bytes());

    let mut w = BufWriter::with_capacity(1 << 20, File::create(tmp)?);
    w.write_all(&header)?;
    let mut pos = HEADER_LEN as u64;
    for (i, (body, r)) in bodies.iter().zip(secs).enumerate() {
        w.write_all(&[0u8; ALIGN as usize][..(r.start - pos) as usize])?;
        w.write_all(body)?;
        pos = r.end;
        let at = 8 + 24 * i;
        footer[at..at + 8].copy_from_slice(&r.start.to_le_bytes());
        footer[at + 8..at + 16].copy_from_slice(&(body.len() as u64).to_le_bytes());
        footer[at + 16..at + 20].copy_from_slice(&crc32fast::hash(body).to_le_bytes());
    }
    let footer_crc = crc32fast::hash(&footer[..84]);
    footer[84..88].copy_from_slice(&footer_crc.to_le_bytes());
    footer[88..].copy_from_slice(&MAGIC);
    w.write_all(&footer)?;
    w.into_inner()
        .map_err(io::IntoInnerError::into_error)?
        .sync_data()?;
    Ok(())
}

/// An open, fully verified segment, mapped read-only.
pub struct Segment {
    map: Mmap,
    layout: Layout,
}

impl Segment {
    /// Maps `path` and verifies all of it (every CRC), so opening reads the
    /// whole file once. See docs/formats/segment.md for the cost.
    pub fn open(path: &Path) -> Result<Segment> {
        let file = File::open(path)?;
        // SAFETY: segments are immutable once published: they are written
        // under a `.tmp` name and renamed into place, then never modified,
        // only replaced by another rename or deleted (which leaves this
        // mapping's pages intact). So the bytes can't change under the map.
        let map = unsafe { Mmap::map(&file)? };
        let layout = parse(&map)?;
        // The map is page-aligned and sections start on 64-byte offsets, so
        // these can't fail; checking here keeps the accessors panic-free.
        bytemuck::try_cast_slice::<u8, f32>(&map[layout.vectors.clone()])
            .map_err(|_| Error::Corrupt("vector section misaligned"))?;
        bytemuck::try_cast_slice::<u8, u64>(&map[layout.ids.clone()])
            .map_err(|_| Error::Corrupt("id section misaligned"))?;
        Ok(Segment { map, layout })
    }

    pub fn dim(&self) -> usize {
        self.layout.dim
    }

    /// The raw metric code (`index::Metric::from_code` decodes it).
    pub fn metric(&self) -> u32 {
        self.layout.metric
    }

    /// Number of vectors.
    pub fn len(&self) -> usize {
        (self.layout.ids.end - self.layout.ids.start) / 8
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Row-major vectors, viewed in place in the map (no copy). Empty if
    /// the segment was written without them (and isn't empty).
    pub fn vectors(&self) -> &[f32] {
        bytemuck::cast_slice(&self.map[self.layout.vectors.clone()]) // alignment checked in open
    }

    /// External id of each vector, viewed in place.
    pub fn ids(&self) -> &[u64] {
        bytemuck::cast_slice(&self.map[self.layout.ids.clone()]) // alignment checked in open
    }

    pub fn blob_kind(&self) -> BlobKind {
        self.layout.blob_kind
    }

    pub fn index_blob(&self) -> &[u8] {
        &self.map[self.layout.blob.clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh, empty directory under the system temp dir.
    fn tmp_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("nearpoint-seg-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    struct Owned {
        vectors: Vec<f32>,
        ids: Vec<u64>,
        blob: Vec<u8>,
    }

    impl Owned {
        fn new(count: usize, dim: usize, blob_len: usize) -> Owned {
            Owned {
                vectors: (0..count * dim).map(|i| i as f32 * 0.5 - 3.0).collect(),
                ids: (0..count as u64).map(|i| 1_000 + i * 7).collect(),
                blob: (0..blob_len).map(|i| (i * 31 % 251) as u8).collect(),
            }
        }

        fn data(&self, dim: usize) -> SegmentData<'_> {
            SegmentData {
                dim,
                metric: 2,
                vectors: &self.vectors,
                ids: &self.ids,
                blob_kind: BlobKind::Hnsw,
                blob: &self.blob,
            }
        }
    }

    /// Writes a small segment and returns its bytes.
    fn image(name: &str, o: &Owned, dim: usize) -> Vec<u8> {
        let path = tmp_dir(name).join("s.seg");
        write(&path, &o.data(dim)).unwrap();
        fs::read(path).unwrap()
    }

    fn check_contents(s: &Segment, o: &Owned, dim: usize) {
        assert_eq!(s.dim(), dim);
        assert_eq!(s.metric(), 2);
        assert_eq!(s.len(), o.ids.len());
        assert_eq!(s.blob_kind(), BlobKind::Hnsw);
        assert_eq!(s.vectors(), &o.vectors[..]);
        assert_eq!(s.ids(), &o.ids[..]);
        assert_eq!(s.index_blob(), &o.blob[..]);
    }

    #[test]
    fn round_trip_and_alignment() {
        let dir = tmp_dir("round-trip");
        for (count, dim, blob_len) in [(10, 3, 37), (0, 5, 0), (1, 1, 1), (257, 17, 1000)] {
            let o = Owned::new(count, dim, blob_len);
            let path = dir.join(format!("{count}.seg"));
            write(&path, &o.data(dim)).unwrap();
            assert!(!tmp_path(&path).exists());
            let s = Segment::open(&path).unwrap();
            check_contents(&s, &o, dim);
            assert_eq!(s.vectors().as_ptr() as usize % ALIGN as usize, 0);
            assert_eq!(s.ids().as_ptr() as usize % ALIGN as usize, 0);
            assert_eq!(s.index_blob().as_ptr() as usize % ALIGN as usize, 0);
        }
    }

    #[test]
    fn every_flipped_bit_is_detected() {
        let o = Owned::new(10, 3, 37);
        let img = image("flip", &o, 3);
        assert_every_flip_detected(&img);
        // Each section's CRC names the section.
        let lay = parse(&img).unwrap();
        for (r, name) in [lay.vectors, lay.ids, lay.blob].iter().zip(SECTION_NAMES) {
            let mut bad = img.clone();
            bad[r.start + 1] ^= 0x10;
            assert!(matches!(parse(&bad), Err(Error::Checksum(n)) if n == name));
        }
    }

    fn assert_every_flip_detected(img: &[u8]) {
        assert!(parse(img).is_ok());
        let mut bad = img.to_vec();
        for byte in 0..img.len() {
            for bit in 0..8 {
                bad[byte] ^= 1 << bit;
                assert!(parse(&bad).is_err(), "flip at byte {byte} bit {bit}");
                bad[byte] ^= 1 << bit;
            }
        }
    }

    #[test]
    fn flipped_bit_in_a_file_fails_open() {
        let dir = tmp_dir("flip-file");
        let path = dir.join("s.seg");
        let o = Owned::new(10, 3, 37);
        write(&path, &o.data(3)).unwrap();
        let mut img = fs::read(&path).unwrap();
        img[HEADER_LEN + 5] ^= 4;
        fs::write(&path, &img).unwrap();
        assert!(matches!(
            Segment::open(&path),
            Err(Error::Checksum("vector section"))
        ));
    }

    #[test]
    fn every_truncation_is_an_error() {
        let img = image("truncate", &Owned::new(10, 3, 37), 3);
        for len in 0..img.len() {
            assert!(parse(&img[..len]).is_err(), "truncated to {len}");
        }
        assert!(matches!(
            parse(&img[..img.len() - 1]),
            Err(Error::Truncated)
        ));
        assert!(matches!(parse(&img[..100]), Err(Error::Truncated)));
        let empty = tmp_dir("empty").join("e.seg");
        fs::write(&empty, b"").unwrap();
        assert!(matches!(Segment::open(&empty), Err(Error::Truncated)));
    }

    #[test]
    fn bad_magic_and_newer_version_are_typed_errors() {
        let img = image("magic", &Owned::new(4, 2, 8), 2);
        let mut bad = img.clone();
        bad[0] = b'X';
        assert!(matches!(parse(&bad), Err(Error::BadMagic)));
        for v in [0, VERSION + 1] {
            let mut bad = img.clone();
            bad[8..12].copy_from_slice(&v.to_le_bytes());
            assert!(matches!(parse(&bad), Err(Error::UnsupportedVersion(x)) if x == v));
        }
    }

    #[test]
    fn tmp_file_never_visible_and_cleaned_up_on_error() {
        let dir = tmp_dir("tmp");
        let path = dir.join("s.seg");
        let o = Owned::new(5, 2, 3);
        // A stale tmp from a crashed write is just overwritten.
        fs::write(tmp_path(&path), b"half a segment").unwrap();
        assert!(Segment::open(&path).is_err()); // not published yet
        write(&path, &o.data(2)).unwrap();
        assert!(!tmp_path(&path).exists());
        check_contents(&Segment::open(&path).unwrap(), &o, 2);

        // Rename fails (a directory is in the way): no tmp left behind.
        let blocked = dir.join("blocked.seg");
        fs::create_dir(&blocked).unwrap();
        fs::write(blocked.join("x"), b"x").unwrap();
        assert!(matches!(write(&blocked, &o.data(2)), Err(Error::Io(_))));
        assert!(!tmp_path(&blocked).exists());
    }

    #[test]
    fn rejects_inconsistent_input() {
        let path = tmp_dir("invalid").join("s.seg");
        let o = Owned::new(5, 2, 3);
        let mut d = o.data(2);
        d.dim = 3;
        assert!(matches!(write(&path, &d), Err(Error::InvalidInput(_))));
        d.dim = 0;
        assert!(matches!(write(&path, &d), Err(Error::InvalidInput(_))));
        assert!(!path.exists() && !tmp_path(&path).exists());
    }

    /// A v1 file committed to the repo: the format must keep reading it.
    /// Built from `Owned::new(3, 2, 5)` with dim 2. This build writes the
    /// same bytes except the version and the two CRCs it changes.
    #[test]
    fn v1_fixture_still_loads() {
        let fixture: &[u8] = include_bytes!("../testdata/segment-v1.seg");
        let o = Owned::new(3, 2, 5);
        let mut v1 = image("fixture", &o, 2);
        v1[8..12].copy_from_slice(&1u32.to_le_bytes());
        recrc(&mut v1);
        assert_eq!(v1, fixture);
        let path = tmp_dir("fixture-open").join("f.seg");
        fs::write(&path, fixture).unwrap();
        check_contents(&Segment::open(&path).unwrap(), &o, 2);
    }

    /// Recomputes the header and footer CRCs after editing an image.
    fn recrc(img: &mut [u8]) {
        let f = img.len() - FOOTER_LEN;
        let h = crc32fast::hash(&img[..HEADER_LEN]);
        img[f..f + 4].copy_from_slice(&h.to_le_bytes());
        let c = crc32fast::hash(&img[f..f + 84]);
        img[f + 84..f + 88].copy_from_slice(&c.to_le_bytes());
    }

    /// v2: a segment without vectors. Committed as a fixture, built from
    /// `Owned::new(3, 2, 5)` with dim 2 and no vectors.
    #[test]
    fn segment_without_vectors() {
        let dir = tmp_dir("bare");
        let o = Owned::new(10, 3, 37);
        let path = dir.join("s.seg");
        write(
            &path,
            &SegmentData {
                vectors: &[],
                ..o.data(3)
            },
        )
        .unwrap();
        let s = Segment::open(&path).unwrap();
        assert!(s.vectors().is_empty());
        assert_eq!((s.len(), s.ids()), (10, &o.ids[..]));
        assert_eq!(s.index_blob(), &o.blob[..]);
        let img = fs::read(&path).unwrap();
        assert_eq!(le_u32(&img, 12), ELEM_NONE);
        assert_every_flip_detected(&img);
        // v1 has no "no vectors" element type.
        let mut v1 = img.clone();
        v1[8..12].copy_from_slice(&1u32.to_le_bytes());
        recrc(&mut v1);
        assert!(matches!(parse(&v1), Err(Error::UnsupportedElementType(1))));
        // Vectors of the wrong length are still an error.
        let bad = SegmentData {
            vectors: &o.vectors[3..],
            ..o.data(3)
        };
        assert!(matches!(write(&path, &bad), Err(Error::InvalidInput(_))));

        let o = Owned::new(3, 2, 5);
        let path = dir.join("fixture.seg");
        write(
            &path,
            &SegmentData {
                vectors: &[],
                ..o.data(2)
            },
        )
        .unwrap();
        let fixture: &[u8] = include_bytes!("../testdata/segment-v2.seg");
        assert_eq!(fs::read(&path).unwrap(), fixture);
        let s = Segment::open(&path).unwrap();
        assert_eq!(
            (s.len(), s.ids(), s.index_blob()),
            (3, &o.ids[..], &o.blob[..])
        );
        assert!(s.vectors().is_empty());
    }
}
