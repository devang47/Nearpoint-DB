//! The HTTP/JSON API (task 4.4), documented endpoint by endpoint in
//! docs/api.md. Every engine call runs on tokio's blocking pool, so a
//! WAL fsync or a long search never stalls the async workers.

use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use engine::Db;
use engine::Filter;
use engine::collection::{Collection, Config, Error, SearchOptions};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use storage::wal::WalError;

/// Largest request body. A bulk upsert is decoded into vectors and WAL
/// records next to the body, so peak memory is a few times this.
pub const MAX_BODY: usize = 256 << 20;
/// Largest `k` or `ef` a search accepts.
pub const MAX_K: usize = 10_000;

pub fn router(db: Arc<Db>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/collections", get(list))
        .route(
            "/collections/{name}",
            put(create).get(info).delete(drop_collection),
        )
        .route("/collections/{name}/seal", post(seal))
        .route("/collections/{name}/search", post(search))
        .route("/collections/{name}/points", post(upsert))
        .route("/collections/{name}/points/bulk", post(bulk))
        .route("/collections/{name}/points/delete", post(delete))
        .route("/collections/{name}/points/{id}", get(get_point))
        .layer(DefaultBodyLimit::max(MAX_BODY))
        .with_state(db)
}

/// An error response: `{"error": "..."}` with the status.
#[derive(Debug)]
pub struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        if self.0.is_server_error() {
            tracing::error!(status = %self.0, "{}", self.1);
        }
        (self.0, Json(json!({"error": self.1}))).into_response()
    }
}

fn bad(msg: impl Into<String>) -> ApiError {
    ApiError(StatusCode::BAD_REQUEST, msg.into())
}

impl From<Error> for ApiError {
    fn from(e: Error) -> ApiError {
        let status = match &e {
            // ADR 0007: the write may or may not be durable. Never a 2xx.
            Error::Unknown(_) | Error::Wal(WalError::Poisoned) => {
                return ApiError(
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!(
                        "{e}. The write may or may not be durable and was not applied; \
                         every write to this collection fails until the server restarts \
                         (restart recovers from the WAL; then read back to find out)"
                    ),
                );
            }
            Error::Invalid(_)
            | Error::Payload(_)
            | Error::Wal(WalError::RecordTooLarge(_))
            | Error::Index(index::Error::DimMismatch { .. } | index::Error::Full) => {
                StatusCode::BAD_REQUEST
            }
            Error::NotFound(_) | Error::Closed => StatusCode::NOT_FOUND,
            Error::Exists(_) => StatusCode::CONFLICT,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        ApiError(status, e.to_string())
    }
}

type ApiResult<T> = Result<T, ApiError>;

/// Runs `f` on the blocking pool. A panic there is a 500, never a 2xx.
async fn blocking<T: Send + 'static>(
    f: impl FnOnce() -> ApiResult<T> + Send + 'static,
) -> ApiResult<T> {
    tokio::task::spawn_blocking(f).await.map_err(|e| {
        ApiError(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("request task failed: {e}"),
        )
    })?
}

fn collection(db: &Db, name: &str) -> ApiResult<Arc<Collection>> {
    db.get(name)
        .ok_or_else(|| Error::NotFound(name.to_owned()).into())
}

/// The collection's config (without the sync policy), name and live count.
fn describe(name: &str, c: &Collection) -> Value {
    let mut v = serde_json::to_value(c.config()).unwrap_or_default();
    if let Value::Object(m) = &mut v {
        m.insert("name".into(), name.into());
        m.insert("points".into(), c.len().into());
    }
    v
}

async fn health() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

async fn list(State(db): State<Arc<Db>>) -> Json<Value> {
    let all: Vec<Value> = db.list().iter().map(|(n, c)| describe(n, c)).collect();
    Json(json!({"collections": all}))
}

async fn create(
    State(db): State<Arc<Db>>,
    Path(name): Path<String>,
    Json(cfg): Json<Config>,
) -> ApiResult<(StatusCode, Json<Value>)> {
    blocking(move || {
        let c = db.create(&name, cfg)?;
        Ok((StatusCode::CREATED, Json(describe(&name, &c))))
    })
    .await
}

async fn info(State(db): State<Arc<Db>>, Path(name): Path<String>) -> ApiResult<Json<Value>> {
    let c = collection(&db, &name)?;
    Ok(Json(describe(&name, &c)))
}

async fn drop_collection(
    State(db): State<Arc<Db>>,
    Path(name): Path<String>,
) -> ApiResult<Json<Value>> {
    blocking(move || {
        db.drop_collection(&name)?;
        Ok(Json(json!({"dropped": true})))
    })
    .await
}

async fn seal(State(db): State<Arc<Db>>, Path(name): Path<String>) -> ApiResult<Json<Value>> {
    let c = collection(&db, &name)?;
    blocking(move || {
        c.seal()?;
        Ok(Json(json!({"sealed": true})))
    })
    .await
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Point {
    id: u64,
    vector: Vec<f32>,
    #[serde(default)]
    payload: Value,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Points {
    points: Vec<Point>,
}

async fn upsert(
    State(db): State<Arc<Db>>,
    Path(name): Path<String>,
    Json(req): Json<Points>,
) -> ApiResult<Json<Value>> {
    let c = collection(&db, &name)?;
    blocking(move || {
        let dim = c.config().dim;
        let n = req.points.len();
        let (mut ids, mut vectors, mut payloads) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n * dim),
            Vec::with_capacity(n),
        );
        for (i, p) in req.points.into_iter().enumerate() {
            if p.vector.len() != dim {
                return Err(bad(format!(
                    "points[{i}]: vector has {} components, collection dim is {dim}",
                    p.vector.len()
                )));
            }
            ids.push(p.id);
            vectors.extend_from_slice(&p.vector);
            payloads.push(p.payload);
        }
        c.upsert_batch(&ids, &vectors, &payloads)?;
        Ok(Json(json!({"upserted": n})))
    })
    .await
}

async fn bulk(
    State(db): State<Arc<Db>>,
    Path(name): Path<String>,
    body: Bytes,
) -> ApiResult<Json<Value>> {
    let c = collection(&db, &name)?;
    blocking(move || {
        let (ids, vectors, payloads) = decode_bulk(&body, c.config().dim).map_err(bad)?;
        c.upsert_batch(&ids, &vectors, &payloads)?;
        Ok(Json(json!({"upserted": ids.len()})))
    })
    .await
}

/// Ids, row-major vectors and payloads (empty: none), as `upsert_batch`
/// takes them.
pub type Batch = (Vec<u64>, Vec<f32>, Vec<Value>);

/// Decodes a bulk body (docs/api.md): blocks of
/// `n u32 | dim u32 | payload_len u32 | ids u64×n | vectors f32×n·dim |
/// payload_len bytes of JSON lines`, little-endian. Payloads come back
/// empty if no block has any.
pub fn decode_bulk(mut b: &[u8], dim: usize) -> Result<Batch, String> {
    let (mut ids, mut vectors, mut payloads) = (Vec::new(), Vec::new(), Vec::new());
    let mut any_payload = false;
    let mut block = 0;
    while !b.is_empty() {
        let err = |what: &str| format!("block {block}: {what}");
        let (n, d, plen) = match take(&mut b, 12) {
            Some(h) => (le32(&h[0..4]), le32(&h[4..8]), le32(&h[8..12])),
            None => return Err(err("truncated header")),
        };
        if d != dim {
            return Err(err(&format!("dim {d}, collection dim is {dim}")));
        }
        let idb = take(&mut b, n * 8).ok_or_else(|| err("truncated ids"))?;
        let vb = n
            .checked_mul(dim * 4)
            .and_then(|len| take(&mut b, len))
            .ok_or_else(|| err("truncated vectors"))?;
        let pb = take(&mut b, plen).ok_or_else(|| err("truncated payloads"))?;
        ids.extend(idb.as_chunks().0.iter().map(|&x| u64::from_le_bytes(x)));
        vectors.extend(vb.as_chunks().0.iter().map(|&x| f32::from_le_bytes(x)));
        if pb.is_empty() {
            payloads.resize(ids.len(), Value::Null);
        } else {
            any_payload = true;
            let text = std::str::from_utf8(pb).map_err(|_| err("payloads are not UTF-8"))?;
            let lines: Vec<&str> = text
                .strip_suffix('\n')
                .unwrap_or(text)
                .split('\n')
                .collect();
            if lines.len() != n {
                return Err(err(&format!("{} payload lines for {n} ids", lines.len())));
            }
            for (i, line) in lines.into_iter().enumerate() {
                let v = match line {
                    "" => Value::Null,
                    l => serde_json::from_str(l)
                        .map_err(|e| err(&format!("payload line {i}: {e}")))?,
                };
                payloads.push(v);
            }
        }
        block += 1;
    }
    if !any_payload {
        payloads.clear();
    }
    Ok((ids, vectors, payloads))
}

fn take<'a>(b: &mut &'a [u8], n: usize) -> Option<&'a [u8]> {
    let (head, rest) = b.split_at_checked(n)?;
    *b = rest;
    Some(head)
}

fn le32(b: &[u8]) -> usize {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Ids {
    ids: Vec<u64>,
}

async fn delete(
    State(db): State<Arc<Db>>,
    Path(name): Path<String>,
    Json(req): Json<Ids>,
) -> ApiResult<Json<Value>> {
    let c = collection(&db, &name)?;
    blocking(move || Ok(Json(json!({"deleted": c.delete_batch(&req.ids)?})))).await
}

async fn get_point(
    State(db): State<Arc<Db>>,
    Path((name, id)): Path<(String, u64)>,
) -> ApiResult<Json<Value>> {
    let c = collection(&db, &name)?;
    blocking(move || match c.get(id) {
        Some((vector, payload)) => Ok(Json(
            json!({"id": id, "vector": vector, "payload": payload}),
        )),
        None => Err(ApiError(
            StatusCode::NOT_FOUND,
            format!("point {id} not found"),
        )),
    })
    .await
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Search {
    vector: Vec<f32>,
    #[serde(default = "default_k")]
    k: usize,
    ef: Option<usize>,
    filter: Option<Filter>,
    #[serde(default)]
    with_payload: bool,
    #[serde(default)]
    with_vector: bool,
}

fn default_k() -> usize {
    10
}

async fn search(
    State(db): State<Arc<Db>>,
    Path(name): Path<String>,
    Json(req): Json<Search>,
) -> ApiResult<Json<Value>> {
    if req.k > MAX_K || req.ef.is_some_and(|ef| ef > MAX_K) {
        return Err(bad(format!("k and ef must be at most {MAX_K}")));
    }
    let c = collection(&db, &name)?;
    blocking(move || {
        let opts = SearchOptions {
            ef: req.ef,
            filter: req.filter.as_ref(),
            with_payload: req.with_payload,
            with_vector: req.with_vector,
        };
        let hits: Vec<Value> = c
            .search(&req.vector, req.k, &opts)?
            .into_iter()
            .map(|m| {
                let mut h = Map::new();
                h.insert("id".into(), m.id.into());
                h.insert("dist".into(), m.dist.into());
                if let Some(p) = m.payload {
                    h.insert("payload".into(), p);
                }
                if let Some(v) = m.vector {
                    h.insert("vector".into(), v.into());
                }
                Value::Object(h)
            })
            .collect();
        Ok(Json(json!({"hits": hits})))
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(ids: &[u64], dim: u32, vectors: &[f32], payloads: &str) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&(ids.len() as u32).to_le_bytes());
        b.extend_from_slice(&dim.to_le_bytes());
        b.extend_from_slice(&(payloads.len() as u32).to_le_bytes());
        ids.iter()
            .for_each(|i| b.extend_from_slice(&i.to_le_bytes()));
        vectors
            .iter()
            .for_each(|x| b.extend_from_slice(&x.to_le_bytes()));
        b.extend_from_slice(payloads.as_bytes());
        b
    }

    #[test]
    fn bulk_round_trip_and_errors() {
        let mut body = block(&[1, 2], 2, &[1.0, 2.0, 3.0, 4.0], "");
        let (ids, v, p) = decode_bulk(&body, 2).unwrap();
        assert_eq!((ids, v, p.len()), (vec![1, 2], vec![1.0, 2.0, 3.0, 4.0], 0));
        body.extend(block(&[3, 4], 2, &[5.0; 4], "{\"a\":1}\n\n"));
        let (ids, v, p) = decode_bulk(&body, 2).unwrap();
        assert_eq!((ids.len(), v.len()), (4, 8));
        assert_eq!(p, [Value::Null, Value::Null, json!({"a": 1}), Value::Null]);
        assert_eq!(decode_bulk(&[], 2).unwrap().0.len(), 0);

        assert!(decode_bulk(&body, 3).unwrap_err().contains("dim 2"));
        assert!(decode_bulk(&body[..body.len() - 1], 2).is_err());
        assert!(decode_bulk(&body[..5], 2).unwrap_err().contains("header"));
        let lines = block(&[1, 2], 2, &[0.0; 4], "{}\n{}\n{}");
        assert!(
            decode_bulk(&lines, 2)
                .unwrap_err()
                .contains("3 payload lines")
        );
        let mut huge = block(&[], 2, &[], "");
        huge[..4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(decode_bulk(&huge, 2).is_err());
    }
}
