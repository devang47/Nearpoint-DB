//! Load test: concurrent upserts (JSON and bulk), replacements, deletes,
//! seals and searches against a running server. Phase 4 acceptance (default
//! mode) and the Phase 6 soak (`--live N`).
//!
//! Checks, for the whole run:
//! - every request succeeds (any non-2xx or I/O error is an error);
//! - no search returns an id whose delete was acknowledged before the
//!   search was sent (deleted ids are never re-inserted, so this is exact);
//! - no search returns an id twice, or a hit failing its filter;
//! - a get of an acknowledged-deleted id is 404, of a live id 200;
//! - (soak) at each probe the server's live count equals the client's.
//!
//! Each writer owns the ids `≡ w (mod writers)`, so no two threads race on
//! one id, and keeps every live vector it wrote (the client model). Exits
//! non-zero if any check failed. Latency is informational; the machine may
//! be shared.
//!
//! Soak mode (`--live N`): bulk-preload N points, then each write round
//! inserts `--batch` new ids and deletes `--batch` live ones, so the live
//! count stays at N. Every `--probe-secs` a probe pauses the writers (they
//! hold `pause` shared per round, the probe takes it exclusively, so the
//! model equals the server's state), sends `--probe-queries` k = 10
//! searches at default params and scores them against the exact top 10,
//! by brute force over the model. One line per window, also appended to
//! `--csv`: reader search p50/p99 and searches/s, write requests/s, live
//! count, recall@10, server RSS (`--server-pid`), writer pause. The run
//! ends at the last full window, then compares the medians of the first
//! and last `--drift-windows` windows (at most half each): PASS if recall
//! fell and p99 rose by at most `--max-drift` (relative).
//! A drift FAIL sets the exit code only with `--fail-on-drift`.
//!
//! ```text
//! cargo run --release -p server --bin loadtest -- --addr 127.0.0.1:7700 --secs 1800
//! cargo run --release -p server --bin loadtest -- --addr 127.0.0.1:7700 --secs 14400 \
//!     --live 1000000 --probe-secs 60 --server-pid "$(pgrep -nx server)" --csv soak.csv \
//!     --seal-secs 0 --collection-config '{"seal_rows": 200000}'
//! ```

use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
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
    /// Soak mode: preload this many points and keep the live count there
    /// (0: the default Phase 4 mode).
    #[arg(long, default_value_t = 0)]
    live: usize,
    /// Soak: probe (pause writers, measure recall) and report every this
    /// many seconds.
    #[arg(long, default_value_t = 60)]
    probe_secs: u64,
    /// Soak: k = 10 queries per probe.
    #[arg(long, default_value_t = 200)]
    probe_queries: usize,
    /// Soak: append one CSV row per window to this file.
    #[arg(long)]
    csv: Option<PathBuf>,
    /// Soak: the server's process id, for its RSS (via `ps`).
    #[arg(long)]
    server_pid: Option<u32>,
    /// Soak: allowed relative recall drop and p99 rise, first vs last window.
    #[arg(long, default_value_t = 0.10)]
    max_drift: f64,
    /// Soak: compare the medians of this many first and last windows.
    #[arg(long, default_value_t = 10)]
    drift_windows: u64,
    /// Soak: exit non-zero when the drift check fails.
    #[arg(long)]
    fail_on_drift: bool,
    /// JSON object merged into the create-collection body (top-level keys
    /// replace the defaults), e.g. '{"seal_rows": 200000}'.
    #[arg(long)]
    collection_config: Option<String>,
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
    /// Writers hold it shared for each round; a probe takes it exclusively.
    pause: RwLock<()>,
    /// Per writer: its live points, as acknowledged by the server.
    models: Vec<Mutex<Live>>,
}

#[derive(Default)]
struct Stats {
    ops: HashMap<&'static str, u64>,
    /// Search latencies, µs (drained every window in soak mode).
    lat: Vec<u32>,
}

/// Live ids and their vectors (row-major), in the same order.
#[derive(Default)]
struct Live {
    ids: Vec<u64>,
    vecs: Vec<f32>,
}

impl Live {
    fn push(&mut self, id: u64, v: &[f32]) {
        self.ids.push(id);
        self.vecs.extend_from_slice(v);
    }

    fn swap_remove(&mut self, i: usize, dim: usize) -> u64 {
        let last = self.ids.len() - 1;
        self.vecs.copy_within(last * dim..(last + 1) * dim, i * dim);
        self.vecs.truncate(last * dim);
        self.ids.swap_remove(i)
    }
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

/// Ids per writer loaded by `preload`.
fn preloaded(a: &Args) -> u64 {
    (a.live as u64).div_ceil(a.writers)
}

/// Soak mode: bulk-loads ids `0..preloaded * writers` before the writers
/// start (id `i` belongs to writer `i % writers`).
fn preload(st: &State, http: &mut Http) -> anyhow::Result<()> {
    let a = &st.args;
    let total = preloaded(a) * a.writers;
    let mut models: Vec<_> = st
        .models
        .iter()
        .map(|m| m.lock().unwrap_or_else(|e| e.into_inner()))
        .collect();
    let mut rng = 0xB01D;
    let t = Instant::now();
    for from in (0..total).step_by(50_000) {
        let ids: Vec<u64> = (from..(from + 50_000).min(total)).collect();
        let vecs: Vec<f32> = ids.iter().flat_map(|_| vector(&mut rng, a.dim)).collect();
        let body = bulk_body(&ids, &vecs, a.dim);
        let (s, b) = http.call(
            "POST",
            &st.path("/points/bulk"),
            "application/octet-stream",
            &body,
        )?;
        if s != 200 {
            bail!("preload: {s} {}", String::from_utf8_lossy(&b));
        }
        for (&id, v) in ids.iter().zip(vecs.chunks(a.dim)) {
            models[(id % a.writers) as usize].push(id, v);
        }
    }
    println!(
        "preloaded {total} points in {:.1} s",
        t.elapsed().as_secs_f64()
    );
    Ok(())
}

fn writer(st: &State, w: u64) {
    let a = &st.args;
    let mut http = Http::new(&a.addr);
    let mut rng = 0x5EED ^ w;
    let mut next = preloaded(a); // the writer's ids are next * writers + w
    let period = Duration::from_secs_f64(1.0 / a.rounds_per_sec as f64);
    let mut round = 0u64;
    while !st.stop.load(SeqCst) {
        let started = Instant::now();
        let pause = st.pause.read().unwrap_or_else(|e| e.into_inner());
        let mut live = st.models[w as usize]
            .lock()
            .unwrap_or_else(|e| e.into_inner());
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
                for (&id, v) in ids.iter().zip(vecs.chunks(a.dim)) {
                    live.push(id, v);
                }
                st.count("insert");
            }
            Err(e) => st.error(e),
        }
        // Replace a few live ids with new vectors.
        if !live.ids.is_empty() {
            let picks: Vec<(usize, Vec<f32>)> = (0..5)
                .map(|_| {
                    let i = (splitmix64(&mut rng) % live.ids.len() as u64) as usize;
                    (i, vector(&mut rng, a.dim))
                })
                .collect();
            let points: Vec<Value> = picks
                .iter()
                .map(|(i, v)| {
                    let id = live.ids[*i];
                    json!({"id": id, "vector": v, "payload": {"tag": tag(id)}})
                })
                .collect();
            match http.json("POST", &st.path("/points"), &json!({"points": points})) {
                Ok(_) => {
                    // In order, so a repeated id's last vector wins, as on the server.
                    for (i, v) in picks {
                        live.vecs[i * a.dim..(i + 1) * a.dim].copy_from_slice(&v);
                    }
                    st.count("replace")
                }
                Err(e) => st.error(e),
            }
        }
        // Delete some: as many as inserted in soak mode or once the live
        // set is big, so it stays bounded.
        let n = if a.live > 0 || live.ids.len() > 20_000 {
            a.batch
        } else {
            a.batch / 5
        };
        let mut doomed = Vec::new();
        for _ in 0..n.min(live.ids.len()) {
            let i = (splitmix64(&mut rng) % live.ids.len() as u64) as usize;
            doomed.push(live.swap_remove(i, a.dim));
        }
        if !doomed.is_empty() {
            match http.json("POST", &st.path("/points/delete"), &json!({"ids": doomed})) {
                Ok(_) => {
                    // Bump `acks` under the map's lock, so a reader that saw
                    // the new count also sees these ids when it checks.
                    let mut d = st.deleted.write().unwrap_or_else(|e| e.into_inner());
                    let at = st.acks.fetch_add(1, SeqCst) + 1;
                    d.extend(doomed.iter().map(|&id| (id, at)));
                    drop(d);
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
            if let Some(&id) = live.ids.first() {
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
        drop(live);
        drop(pause);
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

/// Exact top-10 ids per query over the client model. Squared L2 like the
/// server's `l2` metric (recall only needs the order). Queries are split
/// across threads; each thread streams the model once.
fn exact_top10(queries: &[Vec<f32>], models: &[&Live], dim: usize) -> Vec<Vec<u64>> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let per = queries.len().div_ceil(threads).max(1);
    std::thread::scope(|s| {
        let handles: Vec<_> = queries
            .chunks(per)
            .map(|qs| {
                s.spawn(move || {
                    let mut best: Vec<Vec<(f32, u64)>> = vec![Vec::new(); qs.len()];
                    for m in models {
                        for (&id, v) in m.ids.iter().zip(m.vecs.chunks_exact(dim)) {
                            for (q, b) in qs.iter().zip(&mut best) {
                                let d: f32 = q.iter().zip(v).map(|(x, y)| (x - y) * (x - y)).sum();
                                if b.len() < 10 || d < b[9].0 {
                                    b.insert(b.partition_point(|e| e.0 <= d), (d, id));
                                    b.truncate(10);
                                }
                            }
                        }
                    }
                    best.into_iter()
                        .map(|b| b.into_iter().map(|e| e.1).collect())
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
            .collect()
    })
}

/// Soak probe: pauses the writers, so the client model is exactly the
/// server's live set, checks the live count and measures recall@10 at
/// default search params. Returns (live points, recall@10, pause ms).
fn probe(st: &State, http: &mut Http, rng: &mut u64) -> (usize, f64, f64) {
    let a = &st.args;
    let t = Instant::now();
    let pause = st.pause.write().unwrap_or_else(|e| e.into_inner());
    let guards: Vec<_> = st
        .models
        .iter()
        .map(|m| m.lock().unwrap_or_else(|e| e.into_inner()))
        .collect();
    let models: Vec<&Live> = guards.iter().map(|g| &**g).collect();
    let n: usize = models.iter().map(|m| m.ids.len()).sum();
    match http.json("GET", &st.path(""), &json!(null)) {
        Ok(info) if info["points"].as_u64() == Some(n as u64) => {}
        Ok(info) => st.violation(format!(
            "server has {} live points, the client {n}",
            info["points"]
        )),
        Err(e) => st.error(e),
    }
    let (mut queries, mut got) = (Vec::new(), Vec::new());
    for _ in 0..a.probe_queries {
        let q = vector(rng, a.dim);
        match http.json("POST", &st.path("/search"), &json!({"vector": q, "k": 10})) {
            Ok(v) => {
                let hits = v["hits"].as_array().cloned().unwrap_or_default();
                got.push(
                    hits.iter()
                        .filter_map(|h| h["id"].as_u64())
                        .collect::<HashSet<_>>(),
                );
                queries.push(q);
            }
            Err(e) => st.error(e),
        }
    }
    let truth = exact_top10(&queries, &models, a.dim);
    drop(guards);
    drop(pause);
    let (mut found, mut wanted) = (0, 0);
    for (g, t) in got.iter().zip(&truth) {
        found += t.iter().filter(|id| g.contains(id)).count();
        wanted += t.len();
    }
    let recall = found as f64 / wanted.max(1) as f64;
    (n, recall, t.elapsed().as_secs_f64() * 1000.0)
}

/// Resident set size of `pid` in MiB, via `ps` (KiB).
fn rss_mib(pid: u32) -> Option<f64> {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .ok()?;
    let kib: f64 = String::from_utf8_lossy(&out.stdout).trim().parse().ok()?;
    Some(kib / 1024.0)
}

/// `q`-quantile of sorted µs latencies, in ms.
fn pct(sorted: &[u32], q: f64) -> f64 {
    sorted
        .get(((sorted.len().max(1) - 1) as f64 * q) as usize)
        .map_or(f64::NAN, |&us| us as f64 / 1000.0)
}

/// One soak report window.
struct Window {
    p99: f64,
    recall: f64,
}

/// Soak mode's main loop: a probe and a report line per window until the
/// last full window. Returns the windows.
fn soak(st: &State, start: Instant) -> anyhow::Result<Vec<Window>> {
    let a = &st.args;
    let mut csv = match &a.csv {
        Some(p) => {
            let f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(p)
                .with_context(|| format!("open {}", p.display()))?;
            Some(f)
        }
        None => None,
    };
    if let Some(f) = &mut csv
        && f.metadata()?.len() == 0
    {
        writeln!(
            f,
            "t_s,live,searches_per_s,writes_per_s,p50_ms,p99_ms,recall10,rss_mib,pause_ms,errors,violations"
        )?;
    }
    let sum = |ops: &HashMap<&str, u64>, keys: &[&str]| -> u64 {
        keys.iter().map(|k| ops.get(k).copied().unwrap_or(0)).sum()
    };
    const SEARCHES: &[&str] = &["search", "search_filtered"];
    const WRITES: &[&str] = &["insert", "replace", "delete"];
    let probe_secs = a.probe_secs.max(1);
    let mut rng = 0x9806E;
    let mut windows = Vec::new();
    let (mut last_t, mut last_ops) = (start, HashMap::new());
    for i in 1..=a.secs / probe_secs {
        std::thread::sleep(
            (start + Duration::from_secs(i * probe_secs)).saturating_duration_since(Instant::now()),
        );
        // A fresh connection per probe: the server (hyper) closes
        // keep-alive connections idle for 30 s, and probes are further apart.
        let (live, recall, pause_ms) = probe(st, &mut Http::new(&a.addr), &mut rng);
        let (ops, mut lat) = {
            let mut s = st.stats.lock().unwrap_or_else(|e| e.into_inner());
            (s.ops.clone(), std::mem::take(&mut s.lat))
        };
        let secs = last_t.elapsed().as_secs_f64();
        last_t = Instant::now();
        let sps = (sum(&ops, SEARCHES) - sum(&last_ops, SEARCHES)) as f64 / secs;
        let wps = (sum(&ops, WRITES) - sum(&last_ops, WRITES)) as f64 / secs;
        last_ops = ops;
        lat.sort_unstable();
        let (p50, p99) = (pct(&lat, 0.5), pct(&lat, 0.99));
        let rss = a.server_pid.and_then(rss_mib);
        let (errors, violations) = (st.errors.load(SeqCst), st.violations.load(SeqCst));
        let t = start.elapsed().as_secs();
        println!(
            "{t:>6} s: live {live}, search/s {sps:.0}, writes/s {wps:.0}, p50 {p50:.2} ms, \
             p99 {p99:.2} ms, recall@10 {recall:.4}, rss {} MiB, pause {pause_ms:.0} ms, \
             errors {errors}, violations {violations}",
            rss.map_or("-".into(), |r| format!("{r:.0}"))
        );
        if let Some(f) = &mut csv {
            writeln!(
                f,
                "{t},{live},{sps:.1},{wps:.1},{p50:.3},{p99:.3},{recall:.4},{},{pause_ms:.0},{errors},{violations}",
                rss.map_or(String::new(), |r| format!("{r:.1}"))
            )?;
        }
        windows.push(Window { p99, recall });
    }
    Ok(windows)
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
        pause: RwLock::default(),
        models: (0..args.writers).map(|_| Mutex::default()).collect(),
    };
    let mut http = Http::new(&args.addr);
    http.json("GET", "/health", &json!(null))
        .context("server not reachable")?;
    let _ = http.call("DELETE", &st.path(""), "application/json", b"")?;
    let mut cfg = json!({"dim": args.dim, "metric": "l2", "hnsw": {"m": 16, "ef_construction": 100},
                         "schema": {"tag": "keyword"}});
    if let Some(extra) = &args.collection_config {
        let Value::Object(extra) = serde_json::from_str(extra).context("--collection-config")?
        else {
            bail!("--collection-config must be a JSON object");
        };
        if let Value::Object(cfg) = &mut cfg {
            cfg.extend(extra);
        }
    }
    http.json("PUT", &st.path(""), &cfg)?;
    println!(
        "loadtest: {} s, {} writers, {} readers, dim {}, collection {}, config {cfg}",
        args.secs, args.writers, args.readers, args.dim, args.collection
    );
    if args.live > 0 {
        preload(&st, &mut http)?;
    }
    let start = Instant::now();
    let windows = std::thread::scope(|s| {
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
        if args.live > 0 {
            let windows = soak(&st, start);
            st.stop.store(true, SeqCst);
            return windows;
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
        Ok(Vec::new())
    })?;
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
    if args.live == 0 && !stats.lat.is_empty() {
        println!(
            "search latency ms (informational): p50 {:.2}, p99 {:.2}, max {:.2}",
            pct(&stats.lat, 0.5),
            pct(&stats.lat, 0.99),
            pct(&stats.lat, 1.0)
        );
    }
    let mut drift_ok = true;
    if args.live > 0 {
        // Medians of the first and last `--drift-windows`: one window's p99
        // swings ±40% with whatever background build overlaps it.
        let k = (args.drift_windows.max(1) as usize).min(windows.len() / 2);
        let median = |ws: &[Window], f: fn(&Window) -> f64| {
            let mut v: Vec<f64> = ws.iter().map(f).collect();
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        match k {
            1.. => {
                let (head, tail) = (&windows[..k], &windows[windows.len() - k..]);
                let f = Window {
                    recall: median(head, |w| w.recall),
                    p99: median(head, |w| w.p99),
                };
                let l = Window {
                    recall: median(tail, |w| w.recall),
                    p99: median(tail, |w| w.p99),
                };
                let recall = l.recall / f.recall - 1.0;
                let p99 = l.p99 / f.p99 - 1.0;
                // NaN (no samples) fails too.
                drift_ok = recall >= -args.max_drift && p99 <= args.max_drift;
                println!(
                    "drift, median of first vs last {k} windows: recall@10 {:.4} -> {:.4} ({:+.1}%), \
                     p99 {:.2} -> {:.2} ms ({:+.1}%), max {:.0}%: {}",
                    f.recall,
                    l.recall,
                    recall * 100.0,
                    f.p99,
                    l.p99,
                    p99 * 100.0,
                    args.max_drift * 100.0,
                    if drift_ok { "PASS" } else { "FAIL" }
                );
            }
            0 => {
                drift_ok = false;
                println!("drift: FAIL, fewer than 2 windows (need secs >= 2 * probe-secs)");
            }
        }
    }
    let (errors, violations) = (st.errors.load(SeqCst), st.violations.load(SeqCst));
    println!("errors {errors}, violations {violations}");
    if errors + violations > 0 {
        bail!("FAIL");
    }
    if !drift_ok && args.fail_on_drift {
        bail!("FAIL: drift");
    }
    println!(
        "PASS{}",
        if drift_ok {
            ""
        } else {
            " (checks; drift FAIL)"
        }
    );
    Ok(())
}
