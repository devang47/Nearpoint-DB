//! A minimal blocking HTTP/1.1 client for the load tools (no client crate,
//! AGENTS.md): one keep-alive connection, `content-length` replies only.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicU64, Ordering::SeqCst};
use std::time::{Duration, Instant};

use anyhow::{Context, bail};
use serde_json::Value;

/// The connection was closed before any reply byte (see `Http::call`).
#[derive(Debug)]
struct Stale;

impl std::fmt::Display for Stale {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("connection closed before a reply")
    }
}

impl std::error::Error for Stale {}

/// Requests retried after a `Stale` reused connection.
pub static RECONNECTS: AtomicU64 = AtomicU64::new(0);

/// One keep-alive HTTP/1.1 connection; enough for this server's replies
/// (always `content-length`, never chunked).
pub struct Http {
    addr: String,
    conn: Option<BufReader<TcpStream>>,
    /// When the last reply on `conn` finished, for error reports.
    last: Option<Instant>,
}

impl Http {
    pub fn new(addr: &str) -> Http {
        Http {
            addr: addr.to_owned(),
            conn: None,
            last: None,
        }
    }

    pub fn call(
        &mut self,
        method: &str,
        path: &str,
        ctype: &str,
        body: &[u8],
    ) -> anyhow::Result<(u16, Vec<u8>)> {
        let idle = match (&self.conn, self.last) {
            (Some(_), Some(t)) => format!("reused connection idle {} ms", t.elapsed().as_millis()),
            _ => "new connection".to_owned(),
        };
        let reused = self.conn.is_some();
        let mut res = self.try_call(method, path, ctype, body);
        // RFC 9112 §9.3.1: a server may close an idle keep-alive connection;
        // a request that fails on a reused one before any reply byte is
        // retried once on a new one. Every request here is idempotent
        // (searches, gets, upserts and deletes by id).
        if reused && matches!(&res, Err(e) if e.is::<Stale>()) {
            RECONNECTS.fetch_add(1, SeqCst);
            self.conn = None;
            res = self.try_call(method, path, ctype, body);
        }
        self.last = Some(Instant::now());
        if res.is_err() {
            self.conn = None; // reconnect next time
        }
        res.with_context(|| format!("{method} {path} ({idle})"))
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
        let mut line = String::new();
        if s.write_all(head.as_bytes()).is_err()
            || s.write_all(body).is_err()
            || conn.read_line(&mut line)? == 0
        {
            return Err(Stale.into());
        }
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
    pub fn json(&mut self, method: &str, path: &str, body: &Value) -> anyhow::Result<Value> {
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
