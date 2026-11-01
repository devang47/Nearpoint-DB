//! What the crash writer and `tests/crash.rs` must agree on: the config
//! and the vector and payload derived from each (id, ver). The test
//! includes this file with `#[path]`.

use engine::collection::Config;
use index::{Metric, hnsw};
use serde_json::{Value, json};
use storage::wal::SyncPolicy;

pub const IDS: u64 = 64;
pub const DIM: u64 = 8;

pub fn config() -> Config {
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
        exact_vectors: true,
        sync: SyncPolicy::Batch,
    }
}

pub fn vector(id: u64, ver: u64) -> Vec<f32> {
    (0..DIM)
        .map(|i| (splitmix64(id ^ ver.rotate_left(17) ^ i) >> 40) as f32 / 1024.0)
        .collect()
}

pub fn payload(id: u64, ver: u64) -> Value {
    if ver.is_multiple_of(5) {
        return Value::Null;
    }
    json!({"ver": ver, "tag": if id.is_multiple_of(2) { "even" } else { "odd" }})
}

pub fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
