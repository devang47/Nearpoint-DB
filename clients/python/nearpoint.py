"""Minimal Nearpoint client (task 4.5): stdlib only, one method per
endpoint in docs/api.md.

    from nearpoint import Client
    c = Client("http://127.0.0.1:7700")
    c.create_collection("demo", dim=3, metric="cosine")
    c.upsert("demo", [{"id": 1, "vector": [1, 0, 0], "payload": {"color": "red"}}])
    c.search("demo", [1, 0.1, 0], k=5, with_payload=True)

As a script, bulk-loads an .fvecs file (SIFT format):

    python3 nearpoint.py load-fvecs --url http://127.0.0.1:7700 \\
        --collection sift --metric l2 ~/data/Nearpoint/sift1m/sift_base.fvecs
"""

import argparse
import array
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


class NearpointError(Exception):
    """A non-2xx response. `status` 503 means the write outcome is unknown
    (the server's WAL failed): it may or may not be durable."""

    def __init__(self, status, message):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message


class Client:
    def __init__(self, url="http://127.0.0.1:7700", timeout=600):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def _call(self, method, path, body=None, raw=None):
        headers = {}
        if body is not None:
            raw = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        elif raw is not None:
            headers["Content-Type"] = "application/octet-stream"
        req = urllib.request.Request(self.url + path, data=raw, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            text = e.read().decode(errors="replace")
            try:
                text = json.loads(text)["error"]
            except (ValueError, KeyError, TypeError):
                pass
            raise NearpointError(e.code, text) from None

    @staticmethod
    def _name(name):
        return urllib.parse.quote(name, safe="")

    def health(self):
        return self._call("GET", "/health")

    def create_collection(self, name, dim, metric="l2", hnsw=None, schema=None):
        body = {"dim": dim, "metric": metric}
        if hnsw:
            body["hnsw"] = hnsw
        if schema:
            body["schema"] = schema
        return self._call("PUT", f"/collections/{self._name(name)}", body)

    def list_collections(self):
        return self._call("GET", "/collections")["collections"]

    def get_collection(self, name):
        return self._call("GET", f"/collections/{self._name(name)}")

    def drop_collection(self, name):
        return self._call("DELETE", f"/collections/{self._name(name)}")

    def upsert(self, name, points):
        """`points`: dicts with `id`, `vector` and optionally `payload`.
        Returns the number upserted."""
        body = {"points": list(points)}
        return self._call("POST", f"/collections/{self._name(name)}/points", body)["upserted"]

    def upsert_bulk(self, name, ids, vectors, payloads=None):
        """Binary bulk upsert: `vectors` is a flat sequence of
        len(ids) * dim floats (an `array('f')` is used as is). Returns the
        number upserted."""
        return self._call(
            "POST",
            f"/collections/{self._name(name)}/points/bulk",
            raw=encode_bulk(ids, vectors, payloads),
        )["upserted"]

    def delete(self, name, ids):
        """Returns how many of `ids` existed."""
        body = {"ids": list(ids)}
        return self._call("POST", f"/collections/{self._name(name)}/points/delete", body)["deleted"]

    def get(self, name, id):
        """The point as {"id", "vector", "payload"}, or None if missing."""
        try:
            return self._call("GET", f"/collections/{self._name(name)}/points/{int(id)}")
        except NearpointError as e:
            if e.status == 404:
                return None
            raise

    def search(self, name, vector, k=10, ef=None, filter=None, with_payload=False, with_vector=False):
        """A list of hits {"id", "dist", ["payload"], ["vector"]}, nearest first."""
        body = {"vector": list(vector), "k": k, "with_payload": with_payload, "with_vector": with_vector}
        if ef is not None:
            body["ef"] = ef
        if filter is not None:
            body["filter"] = filter
        return self._call("POST", f"/collections/{self._name(name)}/search", body)["hits"]

    def seal(self, name):
        return self._call("POST", f"/collections/{self._name(name)}/seal")


def encode_bulk(ids, vectors, payloads=None):
    """One bulk block (docs/api.md): n, dim, payload_len, ids, vectors,
    payload JSON lines."""
    n = len(ids)
    if n == 0:
        return b""  # no blocks: upserts nothing
    vecs = vectors if isinstance(vectors, array.array) and vectors.typecode == "f" else array.array("f", vectors)
    if len(vecs) % n:
        raise ValueError("vectors must hold len(ids) * dim floats")
    dim = len(vecs) // n
    text = b""
    if payloads is not None:
        if len(payloads) != n:
            raise ValueError("need one payload per id")
        text = "".join(("" if p is None else json.dumps(p)) + "\n" for p in payloads).encode()
    id_arr = array.array("Q", ids)
    if sys.byteorder != "little":
        id_arr.byteswap()
        vecs = array.array("f", vecs)
        vecs.byteswap()
    return struct.pack("<III", n, dim, len(text)) + id_arr.tobytes() + vecs.tobytes() + text


def read_fvecs(path, batch):
    """Yields (first_row, dim, array('f')) chunks of up to `batch` vectors."""
    with open(path, "rb") as f:
        head = f.read(4)
        if not head:
            return
        dim = struct.unpack("<i", head)[0]
        f.seek(0)
        row_bytes = 4 + 4 * dim
        row = 0
        while True:
            buf = f.read(row_bytes * batch)
            if not buf:
                return
            if len(buf) % row_bytes:
                raise ValueError(f"{path}: truncated row")
            rows = array.array("f")
            rows.frombytes(buf)  # each row: the dim (as float bits), then dim floats
            if sys.byteorder != "little":
                rows.byteswap()
            vecs = array.array("f")
            for r in range(len(buf) // row_bytes):
                vecs.extend(rows[r * (dim + 1) + 1 : (r + 1) * (dim + 1)])
            yield row, dim, vecs
            row += len(buf) // row_bytes


def _load_fvecs(args):
    c = Client(args.url)
    names = {x["name"] for x in c.list_collections()}
    start = time.time()
    total = 0
    for first, dim, vecs in read_fvecs(os.path.expanduser(args.path), args.batch):
        if args.collection not in names:
            c.create_collection(args.collection, dim, args.metric,
                                hnsw={"m": args.m, "ef_construction": args.ef_construction})
            names.add(args.collection)
        n = len(vecs) // dim
        total += c.upsert_bulk(args.collection, range(first, first + n), vecs)
        print(f"{total} points, {time.time() - start:.1f} s", flush=True)
        if args.limit and total >= args.limit:
            break


def main():
    p = argparse.ArgumentParser(description="Nearpoint client tools")
    sub = p.add_subparsers(dest="cmd", required=True)
    lf = sub.add_parser("load-fvecs", help="bulk-load an .fvecs file; ids are row numbers")
    lf.add_argument("path")
    lf.add_argument("--url", default=os.environ.get("NEARPOINT_URL", "http://127.0.0.1:7700"))
    lf.add_argument("--collection", required=True)
    lf.add_argument("--metric", default="l2")
    lf.add_argument("--m", type=int, default=16)
    lf.add_argument("--ef-construction", type=int, default=200)
    lf.add_argument("--batch", type=int, default=50_000, help="points per request")
    lf.add_argument("--limit", type=int, default=0, help="stop after about this many points")
    args = p.parse_args()
    _load_fvecs(args)


if __name__ == "__main__":
    main()
