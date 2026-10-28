//! Writer for the SIGKILL crash test (task 3.4b, `tests/crash.rs`): opens
//! the collection in `<dir>` and runs random upserts, deletes and seals
//! until killed. Before each op it prints the op (`U <id> <ver>`, `D <id>`,
//! `S` or `M` for a background step), and `ok` once the op is acknowledged.
//!
//! Usage: `crash-child <dir> <seed>`. Vectors and payloads are derived
//! from (id, ver); `tests/crash.rs` derives the same ones.

use engine::collection::{Collection, Config};
use index::{Metric, hnsw};
use serde_json::{Value, json};
use storage::wal::SyncPolicy;

const IDS: u64 = 64;
const DIM: u64 = 8;

fn config() -> Config {
    Config {
        dim: DIM as usize,
        metric: Metric::L2,
        hnsw: hnsw::Config {
            m: 8,
            ef_construction: 32,
            seed: 7,
        },
        schema: Default::default(),
        // Small, so background seals and merges run in most cycles.
        seal_rows: 16,
        ivf_rows: 1024,
        max_merge_rows: 1_000_000,
        sync: SyncPolicy::Batch,
    }
}

fn vector(id: u64, ver: u64) -> Vec<f32> {
    (0..DIM)
        .map(|i| (splitmix64(id ^ ver.rotate_left(17) ^ i) >> 40) as f32 / 1024.0)
        .collect()
}

fn payload(id: u64, ver: u64) -> Value {
    if ver.is_multiple_of(5) {
        return Value::Null;
    }
    json!({"ver": ver, "tag": if id.is_multiple_of(2) { "even" } else { "odd" }})
}

fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

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
