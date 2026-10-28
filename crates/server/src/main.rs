//! Nearpoint HTTP server. Configured by flags or environment variables
//! only; collections and their settings live in the data directory.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use clap::Parser;
use engine::Db;
use storage::wal::SyncPolicy;
use tokio::signal::unix::{SignalKind, signal};

#[derive(Parser)]
#[command(about = "Nearpoint vector database server")]
struct Args {
    /// Data directory (created if missing). Keep it outside the repo.
    #[arg(long, env = "NEARPOINT_DATA_DIR")]
    data_dir: PathBuf,
    /// Address to listen on.
    #[arg(long, env = "NEARPOINT_BIND", default_value = "127.0.0.1:7700")]
    bind: SocketAddr,
    /// WAL durability: `always` (fsync per write), `batch` (group commit;
    /// durable before the reply) or `<n>ms` (fsync every n ms; a power
    /// loss can lose that window).
    #[arg(long, env = "NEARPOINT_SYNC", default_value = "batch", value_parser = parse_sync)]
    sync: SyncPolicy,
}

fn parse_sync(s: &str) -> Result<SyncPolicy, String> {
    match s {
        "always" => Ok(SyncPolicy::Always),
        "batch" => Ok(SyncPolicy::Batch),
        _ => s
            .strip_suffix("ms")
            .and_then(|n| n.parse().ok())
            .filter(|&n| n > 0)
            .map(|n| SyncPolicy::Interval(Duration::from_millis(n)))
            .ok_or_else(|| format!("`{s}`: expected always, batch or <n>ms")),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    let args = Args::parse();
    let db = Db::open(&args.data_dir, args.sync)
        .with_context(|| format!("opening {}", args.data_dir.display()))?;
    for (name, c) in db.list() {
        tracing::info!(collection = %name, points = c.len(), "opened");
    }
    let listener = tokio::net::TcpListener::bind(args.bind)
        .await
        .with_context(|| format!("binding {}", args.bind))?;
    tracing::info!(addr = %args.bind, data_dir = %args.data_dir.display(), "listening");
    let mut term = signal(SignalKind::terminate())?;
    axum::serve(listener, server::router(Arc::new(db)))
        .with_graceful_shutdown(async move {
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {}
                _ = term.recv() => {}
            }
            tracing::info!("shutting down");
        })
        .await?;
    Ok(())
}
