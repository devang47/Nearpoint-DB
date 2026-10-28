//! Task 3.4b: SIGKILL a writer (`src/bin/crash-child.rs`) at random
//! moments, reopen the collection, and check that every acknowledged
//! upsert is there with its vector and payload and every acknowledged
//! delete is gone. The one op in flight at the kill may go either way.
//!
//! SIGKILL keeps the OS page cache, so this tests the process, not power
//! loss (that rests on the WAL's F_FULLFSYNC, task 3.1).
//!
//! Seeded: `NEARPOINT_CRASH_SEED=<seed>` reruns a failure.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use engine::collection::{Collection, Config, SearchOptions};
use index::{Metric, hnsw};
use serde_json::{Value, json};
use storage::wal::SyncPolicy;

// Must match src/bin/crash-child.rs.
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

#[derive(Debug, Clone, Copy)]
enum Op {
    Upsert(u64, u64),
    Delete(u64),
    Seal,
}

fn parse(line: &str) -> Op {
    let f: Vec<&str> = line.split(' ').collect();
    let n = |i: usize| f[i].parse::<u64>().expect("number in child line");
    match f[0] {
        "U" => Op::Upsert(n(1), n(2)),
        "D" => Op::Delete(n(1)),
        "S" => Op::Seal,
        _ => panic!("bad child line {line:?}"),
    }
}

/// Live ids → version, as acknowledged so far.
type Model = HashMap<u64, u64>;

fn apply(model: &mut Model, op: Op) {
    match op {
        Op::Upsert(id, ver) => {
            model.insert(id, ver);
        }
        Op::Delete(id) => {
            model.remove(&id);
        }
        Op::Seal => {}
    }
}

/// Runs one child until a random kill; returns the op in flight, if any.
fn run_child(dir: &Path, seed: u64, rng: &mut u64, model: &mut Model, ctx: &str) -> Option<Op> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_crash-child"))
        .arg(dir)
        .arg(seed.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn crash-child");
    *rng = splitmix64(*rng);
    // 1 in 8 runs dies on a timer, likely during open/replay; the rest
    // right after a random number of acks, i.e. somewhere in the next op.
    let kill_after = if rng.is_multiple_of(8) {
        0
    } else {
        1 + *rng % 64
    };
    if kill_after == 0 {
        std::thread::sleep(Duration::from_millis((*rng >> 8) % 20));
        child.kill().expect("kill");
    }
    let (mut acks, mut in_flight) = (0, None);
    for line in BufReader::new(child.stdout.take().expect("piped")).lines() {
        let line = line.expect("read child stdout");
        if line == "ok" {
            apply(model, in_flight.take().expect("ack without an op"));
            acks += 1;
            if acks == kill_after {
                child.kill().expect("kill"); // lines already in the pipe still count
            }
        } else {
            in_flight = Some(parse(&line));
        }
    }
    let status = child.wait().expect("wait");
    if acks < kill_after || status.success() {
        let mut err = String::new();
        let mut stderr = child.stderr.take().expect("piped");
        stderr.read_to_string(&mut err).expect("read child stderr");
        panic!("{ctx}: child exited on its own ({status}) after {acks} acks: {err}");
    }
    in_flight
}

/// Reopens and checks every id against the model; the in-flight op may
/// or may not have happened, and the model follows what's on disk.
fn verify(dir: &Path, model: &mut Model, in_flight: Option<Op>, ctx: &str) {
    let (c, _) =
        Collection::open(dir, config()).unwrap_or_else(|e| panic!("{ctx}: reopen failed: {e}"));
    let expect = |m: &Model, id: u64| m.get(&id).map(|&v| (vector(id, v), payload(id, v)));
    let mut alt = model.clone();
    if let Some(op) = in_flight {
        apply(&mut alt, op);
    }
    let mut took_in_flight = false;
    for id in 0..IDS {
        let got = c.get(id);
        if got != expect(model, id) {
            assert_eq!(
                got,
                expect(&alt, id),
                "{ctx}: id {id}, in flight {in_flight:?}"
            );
            took_in_flight = true;
        }
    }
    if took_in_flight {
        *model = alt;
    }
    assert_eq!(c.len(), model.len(), "{ctx}: live count");
    let hits = c
        .search(&vector(0, 0), IDS as usize, &SearchOptions::default())
        .expect("search");
    for m in &hits {
        assert!(
            model.contains_key(&m.id),
            "{ctx}: search returned dead id {}",
            m.id
        );
    }
}

fn crash_cycles(cycles: usize) {
    let seed = match std::env::var("NEARPOINT_CRASH_SEED") {
        Ok(s) => s.parse().expect("NEARPOINT_CRASH_SEED is a u64"),
        Err(_) => SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after 1970")
            .as_nanos() as u64,
    };
    println!("crash test: {cycles} cycles, NEARPOINT_CRASH_SEED={seed}");
    let dir = std::env::temp_dir().join(format!("nearpoint-crash-{}", std::process::id()));
    let mut rng = seed;
    let mut model = Model::new();
    for cycle in 0..cycles {
        // A fresh collection every 25 cycles keeps the segment count
        // bounded (no compaction yet) while still crashing over many
        // generations of seals and replays.
        if cycle.is_multiple_of(25) {
            let _ = std::fs::remove_dir_all(&dir);
            model.clear();
        }
        rng = splitmix64(rng);
        let child_seed = rng;
        let ctx = format!("NEARPOINT_CRASH_SEED={seed} cycle {cycle} child {child_seed}");
        let in_flight = run_child(&dir, child_seed, &mut rng, &mut model, &ctx);
        verify(&dir, &mut model, in_flight, &ctx);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn sigkill_20_cycles() {
    crash_cycles(20);
}

/// The 3.4 acceptance run:
/// `cargo test --release -p engine --test crash -- --ignored --nocapture`
#[test]
#[ignore]
fn sigkill_1000_cycles() {
    let t = std::time::Instant::now();
    crash_cycles(1000);
    println!("1000 cycles passed in {:.0} s", t.elapsed().as_secs_f64());
}
