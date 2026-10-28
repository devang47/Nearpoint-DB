//! OS interfaces std doesn't expose. macOS only: the project targets one
//! Mac mini (AGENTS.md).

use std::io;
use std::mem::MaybeUninit;

/// High-water marks of this process's memory use, in bytes.
#[derive(Debug, Clone, Copy)]
pub struct PeakMemory {
    /// Peak resident set. Includes clean mmap'd file pages the OS can drop.
    pub rss: u64,
    /// Peak phys_footprint: what macOS memory pressure (and Activity
    /// Monitor's "Memory" column) counts. The 10 GB budget is against this.
    pub phys_footprint: u64,
}

#[cfg(target_os = "macos")]
pub fn peak_memory() -> io::Result<PeakMemory> {
    let mut ru = MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: `ru` is a valid out-pointer to a `rusage`; on success the
    // kernel has written all of it.
    let ru = unsafe {
        if libc::getrusage(libc::RUSAGE_SELF, ru.as_mut_ptr()) != 0 {
            return Err(io::Error::last_os_error());
        }
        ru.assume_init()
    };

    let mut info = MaybeUninit::<libc::rusage_info_v4>::uninit();
    // SAFETY: as above; RUSAGE_INFO_V4 tells the kernel the buffer is a
    // `rusage_info_v4`, which is exactly what `info` holds.
    let info = unsafe {
        if libc::proc_pid_rusage(
            libc::getpid(),
            libc::RUSAGE_INFO_V4,
            info.as_mut_ptr().cast(),
        ) != 0
        {
            return Err(io::Error::last_os_error());
        }
        info.assume_init()
    };

    Ok(PeakMemory {
        rss: ru.ru_maxrss as u64, // bytes on macOS (KiB on Linux)
        phys_footprint: info.ri_lifetime_max_phys_footprint,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peak_grows_with_touched_allocation() {
        const MB: u64 = 1 << 20;
        let before = peak_memory().unwrap();
        // Nonzero fill so every page is written, not lazily zero-mapped.
        let buf = vec![1u8; 256 * MB as usize];
        std::hint::black_box(&buf);
        let after = peak_memory().unwrap();
        assert!(after.phys_footprint >= before.phys_footprint.max(256 * MB));
        assert!(after.rss >= 256 * MB);
    }
}
