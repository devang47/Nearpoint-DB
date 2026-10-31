//! Billion-mode files (Phase 9; layouts in docs/formats/billion.md).
//!
//! A build directory holds, in order of creation:
//! - `quant`: the trained quantizers (framed, opaque to storage);
//! - `stream` and `refine.tmp`: append-only, one record per input row
//!   (`list u16 | code`, and the refinement code), with `progress` (framed:
//!   rows committed and each file's CRC so far) so an interrupted build
//!   resumes at the last commit;
//! - `refine`: `refine.tmp` plus a footer, once the stream is complete;
//! - `lists`: the stream sorted by list (offsets, codes, u32 ids), written
//!   by [`sort`] in passes over `stream` under a RAM budget; then `stream`
//!   can go.
//!
//! Section CRCs are checked by [`Billion::verify`], not on open: the files
//! are tens of GB.

use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use crate::framed;

pub const QUANT_MAGIC: [u8; 8] = *b"NPBQUANT";
pub const QUANT_VERSION: u32 = 1;
const PROGRESS_MAGIC: [u8; 8] = *b"NPBPROGR";
const PROGRESS_VERSION: u32 = 1;
const REFINE_MAGIC: [u8; 8] = *b"NPBREFIN";
const REFINE_VERSION: u32 = 1;
const LISTS_MAGIC: [u8; 8] = *b"NPBLISTS";
const LISTS_VERSION: u32 = 1;
const REFINE_FOOTER: usize = 32;
const LISTS_FOOTER: usize = 48;
/// Most lists a `u16` list id can name.
pub const MAX_NLIST: usize = 1 << 16;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("i/o: {0}")]
    Io(#[from] io::Error),
    #[error(transparent)]
    Framed(#[from] framed::Error),
    #[error("{0}: {1}")]
    Corrupt(PathBuf, &'static str),
    #[error("{0}: checksum mismatch in {1}")]
    Checksum(PathBuf, &'static str),
    #[error("{0}")]
    Invalid(&'static str),
}

pub type Result<T> = std::result::Result<T, Error>;

fn le_u32(b: &[u8], at: usize) -> u32 {
    let mut x = [0u8; 4];
    x.copy_from_slice(&b[at..at + 4]);
    u32::from_le_bytes(x)
}

fn le_u64(b: &[u8], at: usize) -> u64 {
    let mut x = [0u8; 8];
    x.copy_from_slice(&b[at..at + 8]);
    u64::from_le_bytes(x)
}

fn sync_dir(dir: &Path) -> io::Result<()> {
    File::open(dir)?.sync_all()
}

/// Rows committed to `stream` / `refine.tmp`, and each file's CRC32 over
/// exactly those rows.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
struct Progress {
    rows: u64,
    m: u32,
    stream_crc: u32,
    refine_crc: u32,
}

impl Progress {
    fn read(dir: &Path) -> Result<Option<Progress>> {
        let path = dir.join("progress");
        let b = match framed::read(&path, PROGRESS_MAGIC, PROGRESS_VERSION) {
            Ok(b) => b,
            Err(framed::Error::Io(e)) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e.into()),
        };
        if b.len() != 20 {
            return Err(Error::Corrupt(path, "bad progress body"));
        }
        Ok(Some(Progress {
            rows: le_u64(&b, 0),
            m: le_u32(&b, 8),
            stream_crc: le_u32(&b, 12),
            refine_crc: le_u32(&b, 16),
        }))
    }

    fn write(&self, dir: &Path) -> Result<()> {
        let mut b = self.rows.to_le_bytes().to_vec();
        for x in [self.m, self.stream_crc, self.refine_crc] {
            b.extend_from_slice(&x.to_le_bytes());
        }
        framed::write(&dir.join("progress"), PROGRESS_MAGIC, PROGRESS_VERSION, &b)?;
        Ok(())
    }
}

/// Rows committed so far (0 before the first commit). Read-only: safe
/// while a build runs.
pub fn committed_rows(dir: &Path) -> Result<u64> {
    Ok(Progress::read(dir)?.map_or(0, |p| p.rows))
}

/// Appends encoded rows to `stream` and `refine.tmp`; [`commit`] makes them
/// durable and resumable.
///
/// [`commit`]: StreamWriter::commit
pub struct StreamWriter {
    dir: PathBuf,
    stream: BufWriter<File>,
    refine: BufWriter<File>,
    /// Hashers continue from the committed CRCs (crc32fast's
    /// `new_with_initial`), so a resumed build's CRCs cover every row.
    stream_crc: crc32fast::Hasher,
    refine_crc: crc32fast::Hasher,
    done: Progress,
    rows: u64,
}

impl StreamWriter {
    /// Opens (or starts) the build in `dir` for `m`-byte codes; returns the
    /// rows already committed. Bytes past the last commit (a crash mid-
    /// batch) are cut off.
    pub fn open(dir: &Path, m: usize) -> Result<(StreamWriter, u64)> {
        let m32 = u32::try_from(m).map_err(|_| Error::Invalid("m too big"))?;
        let done = Progress::read(dir)?.unwrap_or(Progress {
            m: m32,
            ..Progress::default()
        });
        if done.m != m32 {
            return Err(Error::Invalid("progress was written with another m"));
        }
        let open = |name: &str, len: u64| -> Result<BufWriter<File>> {
            let path = dir.join(name);
            let f = OpenOptions::new()
                .create(true)
                .read(true)
                .write(true)
                .truncate(false)
                .open(&path)?;
            if f.metadata()?.len() < len {
                return Err(Error::Corrupt(path, "shorter than its committed rows"));
            }
            f.set_len(len)?;
            let mut f = BufWriter::with_capacity(1 << 22, f);
            io::Seek::seek(f.get_mut(), io::SeekFrom::End(0))?;
            Ok(f)
        };
        let stream = open("stream", done.rows * (2 + m as u64))?;
        let refine = open("refine.tmp", done.rows * m as u64)?;
        Ok((
            StreamWriter {
                dir: dir.to_owned(),
                stream,
                refine,
                stream_crc: crc32fast::Hasher::new_with_initial(done.stream_crc),
                refine_crc: crc32fast::Hasher::new_with_initial(done.refine_crc),
                done,
                rows: done.rows,
            },
            done.rows,
        ))
    }

    /// Appends rows (not yet durable).
    pub fn append(&mut self, lists: &[u16], codes: &[u8], refine: &[u8]) -> Result<()> {
        let m = self.done.m as usize;
        if codes.len() != lists.len() * m || refine.len() != codes.len() {
            return Err(Error::Invalid("codes and lists differ in length"));
        }
        let mut rec = Vec::with_capacity(lists.len() * (2 + m));
        for (&l, c) in lists.iter().zip(codes.chunks_exact(m)) {
            rec.extend_from_slice(&l.to_le_bytes());
            rec.extend_from_slice(c);
        }
        self.stream.write_all(&rec)?;
        self.refine.write_all(refine)?;
        self.stream_crc.update(&rec);
        self.refine_crc.update(refine);
        self.rows += lists.len() as u64;
        Ok(())
    }

    /// Flushes and syncs both files, then records the row count.
    pub fn commit(&mut self) -> Result<u64> {
        for f in [&mut self.stream, &mut self.refine] {
            f.flush()?;
            f.get_ref().sync_data()?;
        }
        self.done = Progress {
            rows: self.rows,
            m: self.done.m,
            stream_crc: self.stream_crc.clone().finalize(),
            refine_crc: self.refine_crc.clone().finalize(),
        };
        self.done.write(&self.dir)?;
        Ok(self.rows)
    }

    /// Commits, then publishes `refine` (footer, sync, rename).
    pub fn finish(mut self) -> Result<u64> {
        let rows = self.commit()?;
        let mut foot = Vec::with_capacity(REFINE_FOOTER);
        foot.extend_from_slice(&REFINE_MAGIC);
        foot.extend_from_slice(&REFINE_VERSION.to_le_bytes());
        foot.extend_from_slice(&self.done.m.to_le_bytes());
        foot.extend_from_slice(&rows.to_le_bytes());
        foot.extend_from_slice(&self.done.refine_crc.to_le_bytes());
        let c = crc32fast::hash(&foot);
        foot.extend_from_slice(&c.to_le_bytes());
        self.refine.write_all(&foot)?;
        self.refine.flush()?;
        self.refine.get_ref().sync_data()?;
        fs::rename(self.dir.join("refine.tmp"), self.dir.join("refine"))?;
        sync_dir(&self.dir)?;
        Ok(rows)
    }
}

/// Called per stream row: id, list, code.
type Visit<'a> = dyn FnMut(u64, u16, &[u8]) -> Result<()> + 'a;

/// Sorts `stream` into `lists`: one counting pass, then passes that each
/// gather a range of lists (at most `budget` bytes of codes + ids) in RAM
/// and write it in place. Checks the stream's CRC against `progress` on
/// every pass. Returns the number of passes after the count.
pub fn sort(dir: &Path, nlist: usize, budget: usize) -> Result<usize> {
    if nlist == 0 || nlist > MAX_NLIST {
        return Err(Error::Invalid("nlist must be 1..=65536"));
    }
    let p = Progress::read(dir)?.ok_or(Error::Invalid("no progress: nothing streamed"))?;
    let (m, n) = (p.m as usize, p.rows);
    let stride = 2 + m;
    let spath = dir.join("stream");
    // Every pass reads the whole stream and checks it, so a bad sector
    // can't end up in the sorted file.
    let pass = |f: &mut Visit<'_>| -> Result<()> {
        let file = File::open(&spath)?;
        if file.metadata()?.len() != n * stride as u64 {
            return Err(Error::Corrupt(
                spath.clone(),
                "length differs from progress",
            ));
        }
        let mut r = BufReader::with_capacity(1 << 24, file);
        let mut crc = crc32fast::Hasher::new();
        let mut buf = vec![0u8; stride * 65_536];
        let mut id = 0u64;
        while id < n {
            let rows = (n - id).min(65_536) as usize;
            let b = &mut buf[..rows * stride];
            r.read_exact(b)?;
            crc.update(b);
            for rec in b.chunks_exact(stride) {
                let l = u16::from_le_bytes([rec[0], rec[1]]);
                if usize::from(l) >= nlist {
                    return Err(Error::Corrupt(spath.clone(), "list id >= nlist"));
                }
                f(id, l, &rec[2..])?;
                id += 1;
            }
        }
        if crc.finalize() != p.stream_crc {
            return Err(Error::Checksum(spath.clone(), "stream"));
        }
        Ok(())
    };
    let off_bytes = 8 * (nlist + 1);
    let codes_at = off_bytes as u64;
    let codes_len = n * m as u64;
    let ids_at = (codes_at + codes_len).next_multiple_of(4);
    let ids_len = n * 4;
    let by_id_at = ids_at + ids_len;
    let end = by_id_at + n * 2;
    let tmp = dir.join("lists.tmp");
    let out = File::create(&tmp)?;
    out.set_len(end)?;

    // The counting pass also writes each id's list (in id order).
    let mut offsets = vec![0u64; nlist + 1];
    let mut by_id_crc = crc32fast::Hasher::new();
    let mut by_id = Vec::with_capacity(1 << 20);
    let mut by_id_written = 0u64;
    let mut flush = |buf: &mut Vec<u8>, at: &mut u64| -> Result<()> {
        out.write_all_at(buf, by_id_at + *at)?;
        by_id_crc.update(buf);
        *at += buf.len() as u64;
        buf.clear();
        Ok(())
    };
    pass(&mut |_, l, _| {
        offsets[usize::from(l) + 1] += 1;
        by_id.extend_from_slice(&l.to_le_bytes());
        if by_id.len() >= 1 << 20 {
            flush(&mut by_id, &mut by_id_written)?;
        }
        Ok(())
    })?;
    flush(&mut by_id, &mut by_id_written)?;
    for i in 0..nlist {
        offsets[i + 1] += offsets[i];
    }
    // Ranges of lists that fit the budget, in order (a list bigger than the
    // budget gets a range of its own).
    let per_row = (m + 4) as u64;
    let mut ranges: Vec<Range<usize>> = Vec::new();
    let mut start = 0;
    for l in 0..nlist {
        if l > start && (offsets[l + 1] - offsets[start]) * per_row > budget as u64 {
            ranges.push(start..l);
            start = l;
        }
    }
    ranges.push(start..nlist);
    let (mut codes_crc, mut ids_crc) = (crc32fast::Hasher::new(), crc32fast::Hasher::new());
    for range in &ranges {
        let (a, b) = (offsets[range.start], offsets[range.end]);
        let rows = (b - a) as usize;
        let mut codes = vec![0u8; rows * m];
        let mut ids = vec![0u32; rows];
        let mut at: Vec<u64> = offsets[range.clone()].iter().map(|&o| o - a).collect();
        pass(&mut |id, l, code| {
            let l = usize::from(l);
            if range.contains(&l) {
                let j = &mut at[l - range.start];
                let r = *j as usize;
                *j += 1;
                codes[r * m..(r + 1) * m].copy_from_slice(code);
                ids[r] = u32::try_from(id).map_err(|_| Error::Invalid("more than 2^32 rows"))?;
            }
            Ok(())
        })?;
        let id_bytes: &[u8] = bytemuck::cast_slice(&ids);
        codes_crc.update(&codes);
        ids_crc.update(id_bytes);
        out.write_all_at(&codes, codes_at + a * m as u64)?;
        out.write_all_at(id_bytes, ids_at + a * 4)?;
    }
    let off: &[u8] = bytemuck::cast_slice(&offsets);
    out.write_all_at(off, 0)?;
    // Zero padding between codes and ids is never written, so it's zero.
    let mut foot = Vec::with_capacity(LISTS_FOOTER);
    foot.extend_from_slice(&LISTS_MAGIC);
    foot.extend_from_slice(&LISTS_VERSION.to_le_bytes());
    foot.extend_from_slice(&(nlist as u32).to_le_bytes());
    foot.extend_from_slice(&p.m.to_le_bytes());
    foot.extend_from_slice(&by_id_crc.finalize().to_le_bytes());
    foot.extend_from_slice(&n.to_le_bytes());
    for c in [
        crc32fast::hash(off),
        codes_crc.finalize(),
        ids_crc.finalize(),
    ] {
        foot.extend_from_slice(&c.to_le_bytes());
    }
    let c = crc32fast::hash(&foot);
    foot.extend_from_slice(&c.to_le_bytes());
    out.write_all_at(&foot, end)?;
    out.sync_data()?;
    fs::rename(&tmp, dir.join("lists"))?;
    sync_dir(dir)?;
    Ok(ranges.len())
}

/// The finished files of a build, mapped read-only.
pub struct Billion {
    dir: PathBuf,
    lists: Mmap,
    refine: Mmap,
    nlist: usize,
    m: usize,
    n: u64,
    offsets: Range<usize>,
    codes: Range<usize>,
    ids: Range<usize>,
    by_id: Range<usize>,
    crcs: [u32; 5], // offsets, codes, ids, refine, by_id
}

fn map(path: &Path) -> Result<Mmap> {
    let f = File::open(path)?;
    // SAFETY: the file is only ever written before it is renamed into
    // place and never modified after (append-only build, rename publish),
    // so the mapping's contents don't change under us.
    Ok(unsafe { Mmap::map(&f)? })
}

/// A footer's bytes, its CRC checked; the version before anything else.
fn footer<'a>(
    path: &Path,
    b: &'a [u8],
    len: usize,
    magic: [u8; 8],
    version: u32,
) -> Result<&'a [u8]> {
    let f = b
        .len()
        .checked_sub(len)
        .map(|at| &b[at..])
        .ok_or_else(|| Error::Corrupt(path.to_owned(), "shorter than its footer"))?;
    if f[..8] != magic {
        return Err(Error::Corrupt(path.to_owned(), "bad magic"));
    }
    if le_u32(f, 8) != version {
        return Err(Error::Corrupt(path.to_owned(), "unsupported version"));
    }
    if crc32fast::hash(&f[..len - 4]) != le_u32(f, len - 4) {
        return Err(Error::Checksum(path.to_owned(), "footer"));
    }
    Ok(f)
}

impl Billion {
    /// Maps `lists` and `refine` and checks their footers and sizes (not
    /// the section CRCs: [`verify`](Billion::verify)).
    pub fn open(dir: &Path) -> Result<Billion> {
        let lp = dir.join("lists");
        let rp = dir.join("refine");
        let lists = map(&lp)?;
        let refine = map(&rp)?;
        let lf = footer(&lp, &lists, LISTS_FOOTER, LISTS_MAGIC, LISTS_VERSION)?;
        let (nlist, m, n) = (
            le_u32(lf, 12) as usize,
            le_u32(lf, 16) as usize,
            le_u64(lf, 24),
        );
        let mut crcs = [
            le_u32(lf, 32),
            le_u32(lf, 36),
            le_u32(lf, 40),
            0,
            le_u32(lf, 20),
        ];
        let rf = footer(&rp, &refine, REFINE_FOOTER, REFINE_MAGIC, REFINE_VERSION)?;
        let refine_crc = le_u32(rf, 24);
        if le_u32(rf, 12) as usize != m || le_u64(rf, 16) != n {
            return Err(Error::Corrupt(rp, "m or row count differs from lists"));
        }
        if nlist == 0 || nlist > MAX_NLIST || m == 0 || n > u64::from(u32::MAX) + 1 {
            return Err(Error::Corrupt(lp, "bad footer fields"));
        }
        let size = |x: u64| usize::try_from(x).map_err(|_| Error::Invalid("file too big"));
        let codes_at = 8 * (nlist + 1);
        let codes_end = codes_at + size(n * m as u64)?;
        let ids_at = codes_end.next_multiple_of(4);
        let ids_end = ids_at + size(n * 4)?;
        let by_id_end = ids_end + size(n * 2)?;
        if lists.len() != by_id_end + LISTS_FOOTER {
            return Err(Error::Corrupt(lp, "length differs from its footer"));
        }
        if refine.len() != size(n * m as u64)? + REFINE_FOOTER {
            return Err(Error::Corrupt(rp, "length differs from its footer"));
        }
        crcs[3] = refine_crc;
        let b = Billion {
            dir: dir.to_owned(),
            lists,
            refine,
            nlist,
            m,
            n,
            offsets: 0..codes_at,
            codes: codes_at..codes_end,
            ids: ids_at..ids_end,
            by_id: ids_end..by_id_end,
            crcs,
        };
        // Offsets are read on every probe: check them now (small).
        let off = b.offsets();
        if crc32fast::hash(bytemuck::cast_slice(off)) != b.crcs[0] {
            return Err(Error::Checksum(dir.join("lists"), "offsets"));
        }
        if off.first() != Some(&0) || off.last() != Some(&n) || off.windows(2).any(|w| w[0] > w[1])
        {
            return Err(Error::Corrupt(
                dir.join("lists"),
                "offsets not ascending to n",
            ));
        }
        Ok(b)
    }

    pub fn nlist(&self) -> usize {
        self.nlist
    }

    pub fn m(&self) -> usize {
        self.m
    }

    pub fn len(&self) -> u64 {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    pub fn offsets(&self) -> &[u64] {
        bytemuck::cast_slice(&self.lists[self.offsets.clone()])
    }

    pub fn codes(&self) -> &[u8] {
        &self.lists[self.codes.clone()]
    }

    pub fn ids(&self) -> &[u32] {
        bytemuck::cast_slice(&self.lists[self.ids.clone()])
    }

    /// Each id's list.
    pub fn list_of(&self) -> &[u16] {
        bytemuck::cast_slice(&self.lists[self.by_id.clone()])
    }

    /// Refinement codes by id.
    pub fn refine(&self) -> &[u8] {
        &self.refine[..self.refine.len() - REFINE_FOOTER]
    }

    /// Prefetch hint for a list's code bytes and id rows (ranges within
    /// `codes()` and `ids()`).
    pub fn will_need_list(&self, codes: Range<usize>, ids: Range<usize>) {
        for (base, r) in [
            (self.codes.start, codes),
            (self.ids.start, ids.start * 4..ids.end * 4),
        ] {
            if r.start < r.end {
                let _ = self
                    .lists
                    .advise_range(memmap2::Advice::WillNeed, base + r.start, r.len());
            }
        }
    }

    /// Prefetch hint for refinement bytes `r` (see `Segment::will_need_blob`).
    pub fn will_need_refine(&self, r: Range<usize>) {
        if r.start <= r.end && r.end <= self.refine.len() - REFINE_FOOTER {
            let _ = self
                .refine
                .advise_range(memmap2::Advice::WillNeed, r.start, r.len());
        }
    }

    /// Checks every section's CRC (reads every byte).
    pub fn verify(&self) -> Result<()> {
        let lp = self.dir.join("lists");
        for (name, bytes, crc) in [
            ("codes", self.codes(), self.crcs[1]),
            (
                "ids",
                bytemuck::cast_slice::<u32, u8>(self.ids()),
                self.crcs[2],
            ),
            ("lists by id", &self.lists[self.by_id.clone()], self.crcs[4]),
        ] {
            if crc32fast::hash(bytes) != crc {
                return Err(Error::Checksum(lp, name));
            }
        }
        if crc32fast::hash(self.refine()) != self.crcs[3] {
            return Err(Error::Checksum(self.dir.join("refine"), "codes"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(name: &str) -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("nearpoint-billion-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    /// Row `i`: list `i * 7 % nlist`, code bytes from `i`.
    fn rows(range: Range<u64>, nlist: u64, m: usize) -> (Vec<u16>, Vec<u8>, Vec<u8>) {
        let lists = range.clone().map(|i| (i * 7 % nlist) as u16).collect();
        let codes = range
            .clone()
            .flat_map(|i| (0..m).map(move |j| (i as usize * 31 + j) as u8))
            .collect();
        let refine = range
            .flat_map(|i| (0..m).map(move |j| (i as usize * 17 + j * 3) as u8))
            .collect();
        (lists, codes, refine)
    }

    fn stream(dir: &Path, m: usize, nlist: u64, chunks: &[Range<u64>]) {
        for c in chunks {
            let (mut w, done) = StreamWriter::open(dir, m).unwrap();
            assert_eq!(done, c.start);
            let (l, co, r) = rows(c.clone(), nlist, m);
            w.append(&l, &co, &r).unwrap();
            w.commit().unwrap();
        }
    }

    /// Resume after a crash mid-batch, multi-pass sort under a small
    /// budget, and the sorted layout is exactly the counting sort's.
    #[test]
    fn stream_resume_sort_open() {
        let d = tmp_dir("rt");
        let (m, nlist, n) = (4usize, 10u64, 5000u64);
        stream(&d, m, nlist, &[0..1000, 1000..3000]);
        // A batch appended but never committed (crash): cut off on open.
        {
            let (mut w, _) = StreamWriter::open(&d, m).unwrap();
            let (l, c, r) = rows(3000..3500, nlist, m);
            w.append(&l, &c, &r).unwrap();
            w.stream.flush().unwrap();
            w.refine.flush().unwrap();
        }
        let (mut w, done) = StreamWriter::open(&d, m).unwrap();
        assert_eq!(done, 3000);
        let (l, c, r) = rows(3000..n, nlist, m);
        w.append(&l, &c, &r).unwrap();
        assert_eq!(w.finish().unwrap(), n);
        assert!(matches!(StreamWriter::open(&d, 8), Err(Error::Invalid(_))));

        let passes = sort(&d, nlist as usize, 4000).unwrap(); // ~500 rows per pass
        assert!(passes >= 8, "{passes}");
        let b = Billion::open(&d).unwrap();
        b.verify().unwrap();
        assert_eq!((b.nlist(), b.m(), b.len()), (nlist as usize, m, n));
        let (l, c, r) = rows(0..n, nlist, m);
        assert_eq!(b.refine(), &r[..]);
        assert_eq!(b.list_of(), &l[..]);
        let off = b.offsets();
        for list in 0..nlist as usize {
            let ids = &b.ids()[off[list] as usize..off[list + 1] as usize];
            let want: Vec<u32> = (0..n as u32)
                .filter(|&i| l[i as usize] as usize == list)
                .collect();
            assert_eq!(ids, &want[..]);
            for (j, &id) in ids.iter().enumerate() {
                let at = (off[list] as usize + j) * m;
                assert_eq!(&b.codes()[at..at + m], &c[id as usize * m..][..m]);
            }
        }
    }

    /// A flipped bit is found (by open for footers and offsets, by verify
    /// for big sections); a truncated file fails to open; a damaged stream
    /// fails the sort.
    #[test]
    fn corruption_and_truncation() {
        let d = tmp_dir("bad");
        let (m, nlist) = (4usize, 10u64);
        stream(&d, m, nlist, &[0..150, 150..300]);
        let (w, _) = StreamWriter::open(&d, m).unwrap();
        w.finish().unwrap();
        sort(&d, nlist as usize, 1 << 20).unwrap();
        Billion::open(&d).unwrap().verify().unwrap();
        for name in ["lists", "refine"] {
            let path = d.join(name);
            let good = fs::read(&path).unwrap();
            for at in (0..good.len()).step_by(7) {
                let mut bad = good.clone();
                bad[at] ^= 0x10;
                fs::write(&path, &bad).unwrap();
                // m = 4: no padding between codes and ids, so every byte
                // is under a CRC.
                let caught = Billion::open(&d).and_then(|b| b.verify()).is_err();
                assert!(caught, "{name} byte {at}");
            }
            fs::write(&path, &good[..good.len() - 1]).unwrap();
            assert!(Billion::open(&d).is_err());
            fs::write(&path, &good).unwrap();
        }
        let s = d.join("stream");
        let mut img = fs::read(&s).unwrap();
        img[5] ^= 1;
        fs::write(&s, &img).unwrap();
        assert!(matches!(
            sort(&d, nlist as usize, 1 << 20),
            Err(Error::Checksum(..))
        ));
        assert!(matches!(sort(&d, 3, 1 << 20), Err(Error::Corrupt(..))));
    }
}
