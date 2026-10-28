#!/usr/bin/env bash
# The gate every commit must pass (AGENTS.md conventions).
set -euo pipefail
cd "$(dirname "$0")/.."
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
