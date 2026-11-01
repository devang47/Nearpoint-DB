//! Writer for the SIGKILL crash test (`tests/crash.rs`): opens
//! the collection in `<dir>` and runs random upserts, deletes and seals
//! until killed. Before each op it prints the op (`U <id> <ver>`, `D <id>`,
//! `S` or `M` for a background step), and `ok` once the op is acknowledged.
//!
//! Usage: `crash-child <dir> <seed>`. Vectors and payloads are derived
//! from (id, ver) in `common.rs`, shared with `tests/crash.rs`.

mod common;

use common::{IDS, config, payload, splitmix64, vector};
use engine::collection::Collection;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let (Some(dir), Some(seed)) = (args.next(), args.next()) else {
        return Err("usage: crash-child <dir> <seed>".into());
    };
    let mut state: u64 = seed.parse()?;
    let mut rand = || {
        state = splitmix64(state);
        state
    };
    let (c, _) = Collection::open(&dir, config())?;
    loop {
        let r = rand();
        let id = r % IDS;
        // stdout is line-buffered: each line is written before the op runs
        // (or right after its ack), so a SIGKILL can't lose a printed ack.
        match (r >> 32) % 100 {
            0..3 => {
                println!("S");
                c.seal()?;
            }
            3..10 => {
                println!("M");
                c.maintain()?;
            }
            10..35 => {
                println!("D {id}");
                c.delete(id)?;
            }
            _ => {
                let ver = rand() >> 12; // fits a JSON integer exactly
                println!("U {id} {ver}");
                c.upsert(id, &vector(id, ver), &payload(id, ver))?;
            }
        }
        println!("ok");
    }
}
