//! Phase 4 acceptance load test: concurrent upserts (JSON and bulk),
//! replacements, deletes, seals and searches against a running server.
//!
//! Checks, for the whole run:
//! - every request succeeds (any non-2xx or I/O error is an error);
//! - no search returns an id whose delete was acknowledged before the
//!   search was sent (deleted ids are never re-inserted, so this is exact);
//! - no search returns an id twice, or a hit failing its filter;
//! - a get of an acknowledged-deleted id is 404, of a live id 200.
//!
//! Each writer owns the ids `≡ w (mod writers)`, so no two threads race on
//! one id. Exits non-zero if any check failed. Latency is printed for
//! eyeballing only; the machine may be shared.
//!
//! ```text
//! cargo run --release -p server --bin loadtest -- --addr 127.0.0.1:7700 --secs 1800
//! ```

use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::SeqCst};
use std::sync::{Mutex, RwLock};
use std::time::{Duration, Instant};

use anyhow::{Context, bail};
use clap::Parser;
use serde_json::{Value, json};

#[derive(Parser, Clone)]
struct Args {
    #[arg(long, default_value = "127.0.0.1:7700")]
    addr: String,
    /// How long to run.
    #[arg(long, default_value_t = 120)]
    secs: u64,
    /// Collection to (re)create; dropped at the end unless `--keep`.
    #[arg(long, default_value = "loadtest")]
    collection: String,
    #[arg(long, default_value_t = 2)]
    writers: u64,
    #[arg(long, default_value_t = 4)]
    readers: u64,
    #[arg(long, default_value_t = 32)]
    dim: usize,
    /// New points per insert request.
    #[arg(long, default_value_t = 50)]
    batch: usize,
    /// Write rounds per second per writer (each: one insert, one replace,
    /// one delete request). Bounds RAM: rows are never compacted yet.
    #[arg(long, default_value_t = 10)]
    rounds_per_sec: u64,
    /// Seal every this many seconds (0: never).
    #[arg(long, default_value_t = 60)]
    seal_secs: u64,
    #[arg(long)]
    keep: bool,
}

/// One keep-alive HTTP/1.1 connection; enough for this server's replies
/// (always `content-length`, never chunked).
struct Http {
    addr: String,
    conn: Option<BufReader<TcpStream>>,
}

impl Http {
    fn new(addr: &str) -> Http {
        Http {
            addr: addr.to_owned(),
            conn: None,
        }
    }

    fn call(
        &mut self,
        method: &str,
        path: &str,
        ctype: &str,
        body: &[u8],
    ) -> anyhow::Result<(u16, Vec<u8>)> {
        let res = self.try_call(method, path, ctype, body);
        if res.is_err() {
            self.conn = None; // reconnect next time
        }
        res
    }

    fn try_call(
        &mut self,
        method: &str,
        path: &str,
        ctype: &str,
        body: &[u8],
    ) -> anyhow::Result<(u16, Vec<u8>)> {
        if self.conn.is_none() {
            let s =
                TcpStream::connect(&self.addr).with_context(|| format!("connect {}", self.addr))?;
            s.set_nodelay(true)?;
            s.set_read_timeout(Some(Duration::from_secs(120)))?;
            self.conn = Some(BufReader::new(s));
        }
        let Some(conn) = self.conn.as_mut() else {
            bail!("no connection")
        };
        let head = format!(
            "{method} {path} HTTP/1.1\r\nHost: {}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\n\r\n",
            self.addr,
            body.len()
        );
        let s = conn.get_mut();
        s.write_all(head.as_bytes())?;
        s.write_all(body)?;
        let mut line = String::new();
        conn.read_line(&mut line)?;
        let status: u16 = line
            .split(' ')
            .nth(1)
            .and_then(|s| s.parse().ok())
            .with_context(|| format!("bad status line {line:?}"))?;
        let (mut len, mut close) = (None, false);
        loop {
            line.clear();
            conn.read_line(&mut line)?;
            let h = line.trim_end();
            if h.is_empty() {
                break;
            }
            let (k, v) = h.split_once(':').unwrap_or((h, ""));
            match k.to_ascii_lowercase().as_str() {
                "content-length" => len = v.trim().parse().ok(),
                "connection" => close = v.trim().eq_ignore_ascii_case("close"),
                "transfer-encoding" => bail!("chunked reply not supported"),
                _ => {}
            }
        }
        let mut buf = vec![0; len.context("reply without content-length")?];
        conn.read_exact(&mut buf)?;
        if close {
            self.conn = None;
        }
        Ok((status, buf))
    }

    /// A JSON call that must succeed.
    fn json(&mut self, method: &str, path: &str, body: &Value) -> anyhow::Result<Value> {
        let (status, reply) = self.call(
            method,
            path,
            "application/json",
            body.to_string().as_bytes(),
        )?;
        if !(200..300).contains(&status) {
            bail!(
                "{method} {path}: {status} {}",
                String::from_utf8_lossy(&reply)
            );
        }
        Ok(serde_json::from_slice(&reply)?)
    }
}

/// Shared between threads.
struct State {
    args: Args,
    stop: AtomicBool,
    /// Incremented after each acknowledged delete request.
    acks: AtomicU64,
    /// Deleted id → the `acks` value its delete was acknowledged at.
    deleted: RwLock<HashMap<u64, u64>>,
    errors: AtomicU64,
    violations: AtomicU64,
    stats: Mutex<Stats>,
}

#[derive(Default)]
struct Stats {
    ops: HashMap<&'static str, u64>,
    /// Search latencies, µs.
    lat: Vec<u32>,
}

impl State {
    fn path(&self, rest: &str) -> String {
        format!("/collections/{}{rest}", self.args.collection)
    }

    fn count(&self, op: &'static str) {
        *self
            .stats
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .ops
            .entry(op)
            .or_default() += 1;
    }

    fn error(&self, what: impl std::fmt::Display) {
        if self.errors.fetch_add(1, SeqCst) < 20 {
            eprintln!("error: {what}");
        }
    }

    fn violation(&self, what: impl std::fmt::Display) {
        if self.violations.fetch_add(1, SeqCst) < 20 {
            eprintln!("VIOLATION: {what}");
        }
    }
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn vector(rng: &mut u64, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|_| (splitmix64(rng) >> 40) as f32 / (1u64 << 24) as f32)
        .collect()
}

/// The tag never changes across versions of an id, so filter checks hold.
fn tag(id: u64) -> &'static str {
    if id.is_multiple_of(2) { "a" } else { "b" }
}

fn bulk_body(ids: &[u64], vectors: &[f32], dim: usize) -> Vec<u8> {
    let lines: String = ids
        .iter()
        .map(|&id| format!("{{\"tag\":\"{}\"}}\n", tag(id)))
        .collect();
    let mut b = Vec::new();
    b.extend_from_slice(&(ids.len() as u32).to_le_bytes());
    b.extend_from_slice(&(dim as u32).to_le_bytes());
    b.extend_from_slice(&(lines.len() as u32).to_le_bytes());
    ids.iter()
        .for_each(|i| b.extend_from_slice(&i.to_le_bytes()));
    vectors
        .iter()
        .for_each(|x| b.extend_from_slice(&x.to_le_bytes()));
    b.extend_from_slice(lines.as_bytes());
    b
}

fn writer(st: &State, w: u64) {
    let a = &st.args;
    let mut http = Http::new(&a.addr);
    let mut rng = 0x5EED ^ w;
    let mut next = 0u64; // the writer's ids are next * writers + w
    let mut live: Vec<u64> = Vec::new();
    let period = Duration::from_secs_f64(1.0 / a.rounds_per_sec as f64);
    let mut round = 0u64;
    while !st.stop.load(SeqCst) {
        let started = Instant::now();
        // Insert new ids, alternating JSON and binary bulk.
        let ids: Vec<u64> = (0..a.batch as u64)
            .map(|i| (next + i) * a.writers + w)
            .collect();
        next += a.batch as u64;
        let vecs: Vec<f32> = ids.iter().flat_map(|_| vector(&mut rng, a.dim)).collect();
        let res = if round.is_multiple_of(2) {
            let points: Vec<Value> = ids
                .iter()
                .zip(vecs.chunks(a.dim))
                .map(|(&id, v)| json!({"id": id, "vector": v, "payload": {"tag": tag(id)}}))
                .collect();
            http.json("POST", &st.path("/points"), &json!({"points": points}))
                .map(drop)
        } else {
            match http.call(
                "POST",
                &st.path("/points/bulk"),
                "application/octet-stream",
                &bulk_body(&ids, &vecs, a.dim),
            ) {
                Ok((200, _)) => Ok(()),
                Ok((s, b)) => Err(anyhow::anyhow!("bulk: {s} {}", String::from_utf8_lossy(&b))),
                Err(e) => Err(e),
            }
        };
        match res {
            Ok(()) => {
                live.extend(&ids);
                st.count("insert");
            }
            Err(e) => st.error(e),
        }
        // Replace a few live ids with new vectors.
        if !live.is_empty() {
            let points: Vec<Value> = (0..5)
                .map(|_| {
                    let id = live[(splitmix64(&mut rng) % live.len() as u64) as usize];
                    json!({"id": id, "vector": vector(&mut rng, a.dim), "payload": {"tag": tag(id)}})
                })
                .collect();
            match http.json("POST", &st.path("/points"), &json!({"points": points})) {
                Ok(_) => st.count("replace"),
                Err(e) => st.error(e),
            }
        }
        // Delete some: about as many as inserted once the live set is big,
        // so it stays bounded.
        let n = if live.len() > 20_000 {
            a.batch
        } else {
            a.batch / 5
        };
        let mut doomed = Vec::new();
        for _ in 0..n.min(live.len()) {
            let i = (splitmix64(&mut rng) % live.len() as u64) as usize;
            doomed.push(live.swap_remove(i));
        }
        if !doomed.is_empty() {
            match http.json("POST", &st.path("/points/delete"), &json!({"ids": doomed})) {
                Ok(_) => {
                    let at = st.acks.fetch_add(1, SeqCst) + 1;
                    let mut d = st.deleted.write().unwrap_or_else(|e| e.into_inner());
                    d.extend(doomed.iter().map(|&id| (id, at)));
                    st.count("delete");
                }
                // Unknown outcome: don't count these ids as deleted.
                Err(e) => st.error(e),
            }
        }
        // Spot-check gets.
        if round.is_multiple_of(10) {
            if let Some(&id) = doomed.first() {
                match http.call(
                    "GET",
                    &st.path(&format!("/points/{id}")),
                    "application/json",
                    b"",
                ) {
                    Ok((404, _)) => st.count("get"),
                    Ok((s, _)) => st.violation(format!("get of deleted id {id}: {s}")),
                    Err(e) => st.error(e),
                }
            }
            if let Some(&id) = live.first() {
                match http.call(
                    "GET",
                    &st.path(&format!("/points/{id}")),
                    "application/json",
                    b"",
                ) {
                    Ok((200, _)) => st.count("get"),
                    Ok((s, b)) => st.error(format!(
                        "get of live id {id}: {s} {}",
                        String::from_utf8_lossy(&b)
                    )),
                    Err(e) => st.error(e),
                }
            }
        }
        round += 1;
        if let Some(rest) = period.checked_sub(started.elapsed()) {
            std::thread::sleep(rest);
        }
    }
}

fn reader(st: &State, r: u64) {
    let a = &st.args;
    let mut http = Http::new(&a.addr);
    let mut rng = 0xFEED ^ r;
    while !st.stop.load(SeqCst) {
        let filtered = splitmix64(&mut rng).is_multiple_of(3);
        let mut req = json!({"vector": vector(&mut rng, a.dim), "k": 10, "with_payload": filtered});
        if filtered {
            req["filter"] = json!({"eq": {"tag": "a"}});
        }
        let seen = st.acks.load(SeqCst);
        let t = Instant::now();
        let reply = match http.json("POST", &st.path("/search"), &req) {
            Ok(v) => v,
            Err(e) => {
                st.error(e);
                continue;
            }
        };
        let us = t.elapsed().as_micros().min(u32::MAX as u128) as u32;
        let hits = reply["hits"].as_array().cloned().unwrap_or_default();
        let deleted = st.deleted.read().unwrap_or_else(|e| e.into_inner());
        let mut ids = HashSet::new();
        for h in &hits {
            let Some(id) = h["id"].as_u64() else {
                st.violation(format!("hit without an id: {h}"));
                continue;
            };
            if !ids.insert(id) {
                st.violation(format!("id {id} twice in one result"));
            }
            if let Some(&at) = deleted.get(&id)
                && at <= seen
            {
                st.violation(format!(
                    "id {id} deleted at ack {at}, returned by a search sent at ack {seen}"
                ));
            }
            if filtered && h["payload"]["tag"] != "a" {
                st.violation(format!("hit {h} fails filter tag = a"));
            }
        }
        drop(deleted);
        if hits.len() > 10 {
            st.violation(format!("{} hits for k = 10", hits.len()));
        }
        let mut s = st.stats.lock().unwrap_or_else(|e| e.into_inner());
        *s.ops
            .entry(if filtered {
                "search_filtered"
            } else {
                "search"
            })
            .or_default() += 1;
        s.lat.push(us);
    }
}

fn sealer(st: &State) {
    let mut http = Http::new(&st.args.addr);
    let mut last = Instant::now();
    while !st.stop.load(SeqCst) {
        std::thread::sleep(Duration::from_millis(200));
        if last.elapsed().as_secs() >= st.args.seal_secs {
            last = Instant::now();
            match http.json("POST", &st.path("/seal"), &json!(null)) {
                Ok(_) => st.count("seal"),
                Err(e) => st.error(e),
            }
        }
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let st = State {
        args: args.clone(),
        stop: AtomicBool::new(false),
        acks: AtomicU64::new(0),
        deleted: RwLock::default(),
        errors: AtomicU64::new(0),
        violations: AtomicU64::new(0),
        stats: Mutex::default(),
    };
    let mut http = Http::new(&args.addr);
    http.json("GET", "/health", &json!(null))
        .context("server not reachable")?;
    let _ = http.call("DELETE", &st.path(""), "application/json", b"")?;
    let cfg = json!({"dim": args.dim, "metric": "l2", "hnsw": {"m": 16, "ef_construction": 100},
                     "schema": {"tag": "keyword"}});
    http.json("PUT", &st.path(""), &cfg)?;
    println!(
        "loadtest: {} s, {} writers, {} readers, dim {}, collection {}",
        args.secs, args.writers, args.readers, args.dim, args.collection
    );
    let start = Instant::now();
    std::thread::scope(|s| {
        for w in 0..args.writers {
            let st = &st;
            s.spawn(move || writer(st, w));
        }
        for r in 0..args.readers {
            let st = &st;
            s.spawn(move || reader(st, r));
        }
        if args.seal_secs > 0 {
            s.spawn(|| sealer(&st));
        }
        let mut next_report = 30;
        while start.elapsed().as_secs() < args.secs {
            std::thread::sleep(Duration::from_millis(250));
            if start.elapsed().as_secs() >= next_report {
                next_report += 30;
                println!(
                    "{:>5} s: errors {}, violations {}",
                    start.elapsed().as_secs(),
                    st.errors.load(SeqCst),
                    st.violations.load(SeqCst)
                );
            }
        }
        st.stop.store(true, SeqCst);
    });
    let info = http.json("GET", &st.path(""), &json!(null))?;
    if !args.keep {
        http.json("DELETE", &st.path(""), &json!(null))?;
    }
    let mut stats = st.stats.into_inner().unwrap_or_else(|e| e.into_inner());
    let mut ops: Vec<_> = stats.ops.iter().collect();
    ops.sort();
    println!("ops: {ops:?}");
    println!("live points at the end: {}", info["points"]);
    stats.lat.sort_unstable();
    if !stats.lat.is_empty() {
        let p = |q: f64| stats.lat[((stats.lat.len() - 1) as f64 * q) as usize] as f64 / 1000.0;
        println!(
            "search latency ms (informational): p50 {:.2}, p99 {:.2}, max {:.2}",
            p(0.5),
            p(0.99),
            p(1.0)
        );
    }
    let (errors, violations) = (st.errors.load(SeqCst), st.violations.load(SeqCst));
    println!("errors {errors}, violations {violations}");
    if errors + violations > 0 {
        bail!("FAIL");
    }
    println!("PASS");
    Ok(())
}
