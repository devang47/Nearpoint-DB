# HTTP API

JSON over HTTP/1.1 (task 4.4). Code: `crates/server/src/lib.rs`; design:
`docs/adr/0011-server-and-concurrency.md`. Python client:
`clients/python/nearpoint.py`.

## Running the server

```
cargo run --release -p server -- --data-dir ~/data/Nearpoint/db
```

| Flag | Env var | Default | Meaning |
|------|---------|---------|---------|
| `--data-dir` | `NEARPOINT_DATA_DIR` | (required) | data directory; created if missing; one server per directory (`LOCK`) |
| `--bind` | `NEARPOINT_BIND` | `127.0.0.1:7700` | listen address |
| `--sync` | `NEARPOINT_SYNC` | `batch` | WAL durability: `always`, `batch` (group commit) or `<n>ms` (fsync every n ms; a power loss can lose that window) |
| `--build-threads` | `NEARPOINT_BUILD_THREADS` | `0` | threads for background index builds and compaction; 0 = half the cores, so builds don't starve searches |

A background thread seals, rebuilds and compacts segments (ADR 0012):
searches never wait for it, writes wait only for its short publish step.
Failures are logged at `error` and retried.

Logs go to stderr; `RUST_LOG` sets the level (default `info`). SIGINT or
SIGTERM stops accepting connections, finishes requests in flight, syncs
the WAL and exits. Collections and their settings are in the data
directory, so a restart needs no other flags.

## Conventions

- Request bodies are JSON with `Content-Type: application/json`, except
  the bulk upsert. Unknown fields are rejected.
- Ids are unsigned 64-bit integers (exact in JSON up to 2^64 − 1, but
  JavaScript clients lose precision above 2^53).
- Vectors are arrays of `dim` numbers with |x| ≤ 1e15 (larger ones would
  overflow f32 distances); `cosine` vectors and queries must be nonzero.
  Stored vectors are returned
  in stored form: unit length for `cosine`.
- A write is acknowledged (2xx) only after it is in the WAL and durable
  per `--sync`, and is visible to every search that starts after the
  acknowledgement. A deleted id never appears in a search that started
  after the delete was acknowledged.
- Request bodies are limited to 256 MiB.

### Errors

Errors from Nearpoint are JSON: `{"error": "<message>"}`.

| Status | When |
|--------|------|
| 400 | invalid input: wrong vector length, NaN/infinite or > 1e15 component, zero `cosine` vector, payload type conflict, bad filter type, invalid config or name, `k`/`ef`/`nprobe`/`rerank` over 10,000, malformed bulk body |
| 404 | no such collection, point, or the collection was dropped during the request |
| 409 | collection already exists |
| 413 | body over 256 MiB (plain text) |
| 415, 422, 400 | body isn't JSON, has the wrong shape or an unknown field (plain text from the framework, not JSON) |
| 500 | I/O error or corrupt data (the server logs it) |
| 503 | **write outcome unknown**: the WAL failed with this write in flight (ADR 0007). It may or may not be durable and was not applied. Every later write to that collection gets 503 until the server restarts; after the restart, read the points back to learn the outcome. Never retried as a success. |

## Endpoints

### `GET /health`

`200 {"status": "ok"}`.

### `PUT /collections/{name}`

Creates a collection. `name`: 1–64 characters of `A-Z a-z 0-9 _ -`.

```json
{
  "dim": 128,
  "metric": "l2",
  "hnsw": {"m": 16, "ef_construction": 200, "seed": 42},
  "schema": {"price": "float", "color": "keyword", "tags": "keyword_list"},
  "seal_rows": 100000,
  "ivf_rows": 1000000,
  "max_merge_rows": 4000000
}
```

| Field | Required | Meaning |
|-------|----------|---------|
| `dim` | yes | vector dimension, > 0 |
| `metric` | yes | `l2` (squared Euclidean), `inner_product` (negated dot) or `cosine` (1 − cos) |
| `hnsw` | no | any of `m` (≥ 2, default 16), `ef_construction` (≥ 1, default 200), `seed` (default 42) |
| `schema` | no | declared payload field types: `int`, `float`, `bool`, `keyword`, `keyword_list`. Undeclared fields take the type of their first value in each segment (`docs/filters.md`); segments whose undeclared fields differ in type are never merged, so declare fields you filter on |
| `seal_rows` | no | the in-memory segment is sealed to disk once it has this many rows (default 100,000, ≥ 1, ≤ half of `max_merge_rows`, and of `ivf_rows` for `inner_product`, or fresh segments could never merge) |
| `ivf_rows` | no | sealed segments with at least this many rows are rebuilt as IVF-PQ, for `l2` and `cosine` (default 1,000,000, ≥ 1,024). `inner_product` collections stay HNSW and merge only below this size |
| `max_merge_rows` | no | compaction never merges segments into one bigger than this (default 4,000,000, ≥ 1); also capped at 2 GiB of f32 vectors (4M rows at dim 128, 512k at dim 1024) |

`201` with the collection description (below). `409` if it exists, `400`
if invalid.

### `GET /collections`

`200 {"collections": [<description>, ...]}`, sorted by name. A
description:

```json
{"name": "docs", "dim": 128, "metric": "l2",
 "hnsw": {"m": 16, "ef_construction": 200, "seed": 42},
 "schema": {"price": "float"}, "seal_rows": 100000, "ivf_rows": 1000000,
 "max_merge_rows": 4000000, "points": 1000000,
 "segments": [{"index": "ivf_pq", "rows": 1000000, "dead": 1200},
              {"index": "mutable", "rows": 1200, "dead": 0}]}
```

`points` is the live (not deleted) point count. `segments` lists the
current segments (`hnsw`, `ivf_pq`, then the in-memory `mutable` one) with
their row counts, deleted or replaced rows included in `rows` and counted
in `dead`.

### `GET /collections/{name}`

`200` with the description, `404` if missing.

### `DELETE /collections/{name}`

Drops the collection and deletes its files. Writes already running on it
finish first; later ones get `404`. `200 {"dropped": true}`, `404` if
missing.

### `POST /collections/{name}/points`

Upserts a batch: inserts each point, or replaces the point with that id.

```json
{"points": [
  {"id": 1, "vector": [0.1, 0.2, 0.3], "payload": {"color": "red", "price": 10}},
  {"id": 2, "vector": [0.4, 0.5, 0.6]}
]}
```

`payload` is a flat JSON object or `null` (the default); types in
`docs/filters.md`. The whole batch is validated before anything is
written, then logged with one WAL sync; if an id repeats, its last point
wins. `200 {"upserted": 2}`. On `503` any prefix of the batch may be
durable.

One request is one WAL sync (~4 ms with `batch` on the Mac mini's SSD),
so send points in batches, not one per request.

### `POST /collections/{name}/points/bulk`

The same upsert with a binary body (`Content-Type:
application/octet-stream`), so large loads skip JSON parsing. The body
is one or more blocks, back to back; all integers and floats
little-endian:

| Size | Field |
|-----:|-------|
| 4 | `n` u32: points in this block |
| 4 | `dim` u32: must equal the collection's dim |
| 4 | `payload_len` u32: bytes of payload text (0: no payloads) |
| 8·n | ids, u64 each |
| 4·n·dim | vectors, f32 each, row-major |
| payload_len | payloads: UTF-8 JSON, one line per point (`\n`-separated, optional trailing `\n`); an empty line or `null` is no payload |

All blocks of a request form one batch (validated whole, one WAL sync).
Blocks only let a client append without knowing the total up front.
`200 {"upserted": n_total}`; `400` names the block and the problem. A
256 MiB body holds ~500k 128-d vectors without payloads; send big loads
as several requests (the Python client defaults to 50,000 points each).

### `POST /collections/{name}/points/delete`

```json
{"ids": [1, 2, 3]}
```

Deletes the ids that exist (one WAL sync). `200 {"deleted": 2}`: how
many existed.

### `GET /collections/{name}/points/{id}`

`200 {"id": 1, "vector": [...], "payload": {...}}` (`payload` is `null`
if none), `404` if missing or deleted.

### `POST /collections/{name}/search`

```json
{
  "vector": [0.1, 0.2, 0.3],
  "k": 10,
  "ef": 128,
  "nprobe": 32,
  "rerank": 100,
  "filter": {"and": [{"eq": {"color": "red"}}, {"range": {"price": {"lt": 20}}}]},
  "with_payload": true,
  "with_vector": false
}
```

| Field | Default | Meaning |
|-------|---------|---------|
| `vector` | required | query, `dim` components |
| `k` | 10 | results wanted, ≤ 10,000 |
| `ef` | 64 | HNSW beam width (raised to `k`), ≤ 10,000; higher is slower and more accurate |
| `nprobe` | 32 | IVF-PQ lists scanned in a segment of `ivf_rows` rows, ≤ 10,000; bigger segments scan the same share of their lists (capped at the list count); higher is slower and more accurate |
| `rerank` | 100 | IVF-PQ candidates re-scored with int8 vectors (at least `k`), ≤ 10,000 |
| `filter` | none | filter expression, shape and semantics in `docs/filters.md`; applied during the search |
| `with_payload` | false | include each hit's payload (`null` if none) |
| `with_vector` | false | include each hit's stored vector |

```json
{"hits": [
  {"id": 7, "dist": 0.013, "payload": {"color": "red", "price": 12}},
  {"id": 3, "dist": 0.250, "payload": {"color": "red", "price": 5}}
]}
```

Nearest first; `dist` is the metric's distance (smaller is nearer).
Fewer than `k` hits if fewer points match. `400` for a filter that
compares a field with the wrong type (`docs/filters.md`).

### `POST /collections/{name}/seal`

Seals everything written so far: freezes the in-memory segment, writes
it to disk, publishes it and truncates the WAL. Collections also seal on
their own at `seal_rows`; this forces one (tests, before a backup).
Writes go on while the files are written; searches never wait. It waits
for a background build or compaction step in flight.
`200 {"sealed": true}`.

## Examples

```
B=http://127.0.0.1:7700
curl -XPUT $B/collections/demo -H 'content-type: application/json' \
     -d '{"dim": 3, "metric": "cosine"}'
curl -XPOST $B/collections/demo/points -H 'content-type: application/json' \
     -d '{"points": [{"id": 1, "vector": [1, 0, 0], "payload": {"color": "red"}}]}'
curl -XPOST $B/collections/demo/search -H 'content-type: application/json' \
     -d '{"vector": [1, 0.1, 0], "k": 5, "with_payload": true}'
```

Bulk-load an `.fvecs` file (SIFT format) with the Python client:

```
python3 clients/python/nearpoint.py load-fvecs --url $B --collection sift \
    --metric l2 ~/data/Nearpoint/sift1m/sift_base.fvecs
```
