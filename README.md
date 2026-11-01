# Nearpoint

A single-node vector database in Rust, built from scratch for one Apple
Silicon Mac mini (M4, 16 GB RAM, 256 GB SSD). It's a database, not just
an index: durable writes (WAL), CRUD, metadata filtering, an HTTP API,
concurrent reads and writes, background compaction. A compressed
read-only "billion mode" serves 1B vectors from the same machine.

> **Status:** all ten phases of [`docs/PLAN.md`](docs/PLAN.md) done.
> Every number below is measured on the Mac mini; the misses and the
> details are in [`docs/RESULTS.md`](docs/RESULTS.md).

## Results

| Scale | Dataset, mode | recall@10 | p50 | p99 | RAM | Disk | Build / load |
|------:|---------------|----------:|----:|----:|----:|-----:|-------------:|
| 1M | SIFT1M, HNSW (in-process) | 0.963 | 0.11 ms | 0.14 ms | 1.3 GB | in RAM | 44 s |
| 10M | BIGANN 10M via API, all features | 0.970 | 0.98 ms | 7.7 ms | 4.9 GB peak, 1.5 GB steady | 1.6 GB | 10 min |
| 100M | BIGANN 100M via API, all features | **0.977** | **50 ms** | **93 ms** | **9.0 GB peak**, 6.6 GB serving | 16 GB | 64 min |
| 1B | BIGANN 1B, billion mode via API | **0.666** | **39 ms** | **87 ms** | **0.16 GB** | **38 GB** (56 GB peak) | 7.7 h (the download) |

- **Targets:** 1M–100M with every feature at p99 < 100 ms, recall@10 ≥
  0.9, ≤ 10 GB of memory: met at 100M (p99 margin 7%; cold cache the
  same). 1B at p99 < 100 ms in ≤ 60 GB of disk, recall published
  against the official ground truth: met; 0.666 is what 32 bytes per
  vector buys at 1B.
- One client, queries one at a time; RAM is macOS phys_footprint. "All
  features": payloads, a 10% filter, deletes, WAL, background compaction.
- Other phases: NEON f32 L2 4.4× scalar; WAL + segments survive 1,000
  SIGKILL cycles with 0 lost writes; a 4-hour mixed insert/delete/search
  soak holds recall and p99 within 3%; filtered HNSW search on SIFT1M ≥
  0.98 recall at 1/10/50% selectivity.

## Architecture

```
                HTTP (axum): JSON, raw-binary bulk upsert, /metrics
                     │
  write ──► WAL (group commit, CRC per record) ──► mutable segment (in-RAM HNSW)
                     │                                   │ seal at 100k rows
                     │                                   ▼
                     │     sealed segments, mapped files:
                     │       HNSW (small) │ IVF-PQ (≥ 1M rows; PQ scan + int8 re-rank)
                     │       billion (read-only, 1B: lists + refine codes on disk)
                     │                                   ▲
                     │     per-collection maintenance thread: seal, absorb,
                     │     merge, drop dead rows, rebuild as IVF-PQ (one at a time)
                     │
  query ──► fan out across segments in parallel (rayon)
            ├─ skip tombstones, filter bitmaps (roaring, typed payload columns)
            ├─ prefetch re-rank rows (madvise WILLNEED)
            └─ merge top-k
```

The ANN algorithms (HNSW, k-means, PQ, IVF, fast-scan) and the SIMD
distance kernels are written by hand; no vector-search crates. Design
decisions are in [`docs/adr/`](docs/adr/) (14 ADRs), every on-disk format
in [`docs/formats/`](docs/formats/) (versioned, magic + CRC32).

## Quickstart

Requires the Rust toolchain pinned in `rust-toolchain.toml` (rustup
installs it). Builds use `target-cpu=native`.

```sh
scripts/check.sh                       # fmt + clippy + tests: the commit gate
cargo build --release
./target/release/server --data-dir ~/data/Nearpoint/db   # 127.0.0.1:7700
```

```sh
B=http://127.0.0.1:7700
curl -XPUT $B/collections/demo -H 'content-type: application/json' \
     -d '{"dim": 3, "metric": "cosine", "schema": {"color": "keyword"}}'
curl -XPOST $B/collections/demo/points -H 'content-type: application/json' \
     -d '{"points": [{"id": 1, "vector": [1, 0, 0], "payload": {"color": "red"}}]}'
curl -XPOST $B/collections/demo/search -H 'content-type: application/json' \
     -d '{"vector": [1, 0.1, 0], "k": 5, "filter": {"eq": {"color": "red"}}, "with_payload": true}'
curl $B/metrics
```

Every endpoint, flag and error is in [`docs/api.md`](docs/api.md);
filters in [`docs/filters.md`](docs/filters.md). A stdlib-only Python
client is in `clients/python/`.

### Benchmarks

Datasets live in `~/data/Nearpoint/`, never in the repo.

```sh
tools/fetch-dataset.sh sift1m          # or bigann10m, bigann100m (12.8 GB: range-fetched prefix)
cargo run --release -p bench -- --dataset sift1m --index hnsw --ef 16,32,64,128
cargo run --release -p server --bin bigload -- --n 10000000 \
    --base ~/data/Nearpoint/bigann10m/base.10M.u8bin \
    --query-file ~/data/Nearpoint/bigann10m/query.public.10K.u8bin \
    --gt ~/data/Nearpoint/bigann10m/gt.10M.ibin \
    --collection-config '{"exact_vectors": false}'   # load through the API, measure
```

### Billion mode

```sh
B=./target/release/billion D=~/data/Nearpoint/billion/b1b
$B train --sample ~/data/Nearpoint/bigann10m/base.10M.u8bin --dir $D   # ~4 min
tools/billion-build.sh $D 1000000000   # streams 128 GB, keeps only codes; resumable
$B sort --dir $D --verify && rm $D/stream
curl -XPUT $B_URL/collections/bigann -d '{"dim": 128, "metric": "l2"}' -H 'content-type: application/json'
curl -XPOST $B_URL/collections/bigann/attach_billion -d "{\"dir\": \"$D\"}" -H 'content-type: application/json'
```

(`B_URL` is the server address.) The build needs ~56 GB free at its peak
and the download is the bottleneck (~4.5 MB/s here); ADR 0014 has the
design.

## Layout

| Path | Contents |
|------|----------|
| `crates/kernels` | distance functions: scalar + NEON, f32 / int8 / 4-bit fast-scan |
| `crates/quant` | k-means, product and scalar quantization |
| `crates/index` | `Index` trait; flat, HNSW, IVF-PQ, billion mode |
| `crates/storage` | WAL, segment and billion file formats, mmap, checksums |
| `crates/engine` | collections, segments, compaction, deletes, filters |
| `crates/server` | HTTP API binary; `loadtest` and `bigload` |
| `crates/bench` | recall/latency harness, dataset loaders, `billion` tool |
| `clients/python` | stdlib-only client |
| `tools/` | dataset downloader, streaming 1B build |
| `docs/` | plan, results, ADRs, on-disk formats, API, changelog |

## Contributing

Read [`AGENTS.md`](AGENTS.md) first. It covers the hardware limits, the
allowed dependencies, the `unsafe` and error-handling rules, and the
per-task workflow.
