"""End-to-end tests for nearpoint.py against a live server (stdlib unittest).

    cargo run --release -p server -- --data-dir /tmp/nearpoint-test &
    NEARPOINT_URL=http://127.0.0.1:7700 python3 clients/python/test_nearpoint.py

Each test works in its own collection and drops it. Skipped if no server
answers at NEARPOINT_URL.
"""

import array
import os
import unittest
import uuid

from nearpoint import Client, NearpointError

URL = os.environ.get("NEARPOINT_URL", "http://127.0.0.1:7700")


def setUpModule():
    try:
        Client(URL, timeout=2).health()
    except OSError as e:
        raise unittest.SkipTest(f"no Nearpoint server at {URL}: {e}")


class ClientTest(unittest.TestCase):
    def setUp(self):
        self.c = Client(URL)
        self.name = "pytest-" + uuid.uuid4().hex[:12]
        self.c.create_collection(self.name, dim=3, metric="l2",
                                 hnsw={"m": 8, "ef_construction": 32},
                                 schema={"price": "float", "color": "keyword"})

    def tearDown(self):
        try:
            self.c.drop_collection(self.name)
        except NearpointError:
            pass

    def test_collections(self):
        self.assertEqual(self.c.health(), {"status": "ok"})
        info = self.c.get_collection(self.name)
        self.assertEqual((info["dim"], info["metric"], info["points"]), (3, "l2", 0))
        self.assertEqual(info["hnsw"]["m"], 8)
        self.assertEqual(info["schema"], {"price": "float", "color": "keyword"})
        self.assertIn(self.name, [x["name"] for x in self.c.list_collections()])
        with self.assertRaises(NearpointError) as e:
            self.c.create_collection(self.name, dim=3)
        self.assertEqual(e.exception.status, 409)
        with self.assertRaises(NearpointError) as e:
            self.c.create_collection("bad/name", dim=3)
        self.assertIn(e.exception.status, (400, 404))
        self.c.drop_collection(self.name)
        with self.assertRaises(NearpointError) as e:
            self.c.get_collection(self.name)
        self.assertEqual(e.exception.status, 404)

    def test_upsert_get_search_delete(self):
        pts = [
            {"id": 1, "vector": [0, 0, 0], "payload": {"color": "red", "price": 10}},
            {"id": 2, "vector": [1, 0, 0], "payload": {"color": "blue", "price": 10.5}},
            {"id": 3, "vector": [2, 0, 0]},
        ]
        self.assertEqual(self.c.upsert(self.name, pts), 3)
        self.assertEqual(self.c.get(self.name, 1),
                         {"id": 1, "vector": [0.0, 0.0, 0.0], "payload": {"color": "red", "price": 10.0}})
        self.assertIsNone(self.c.get(self.name, 99))

        hits = self.c.search(self.name, [0.1, 0, 0], k=2)
        self.assertEqual([h["id"] for h in hits], [1, 2])
        self.assertNotIn("payload", hits[0])
        hits = self.c.search(self.name, [0, 0, 0], k=10, ef=32, with_payload=True, with_vector=True,
                             filter={"range": {"price": {"gt": 10}}})
        self.assertEqual([(h["id"], h["payload"]["color"], h["vector"]) for h in hits],
                         [(2, "blue", [1.0, 0.0, 0.0])])

        # Replace, then delete: the old version and the deleted id are gone.
        self.c.upsert(self.name, [{"id": 1, "vector": [5, 5, 5]}])
        self.assertEqual(self.c.get(self.name, 1)["payload"], None)
        self.assertEqual(self.c.delete(self.name, [2, 2, 42]), 1)
        ids = [h["id"] for h in self.c.search(self.name, [1, 0, 0], k=10)]
        self.assertEqual(ids, [3, 1])
        self.assertEqual(self.c.get_collection(self.name)["points"], 2)
        self.assertEqual(self.c.seal(self.name), {"sealed": True})
        self.assertEqual(self.c.get(self.name, 3)["vector"], [2.0, 0.0, 0.0])

    def test_errors(self):
        self.c.upsert(self.name, [{"id": 1, "vector": [0, 0, 0]}])  # filters run on non-empty segments
        cases = [
            (lambda: self.c.upsert(self.name, [{"id": 1, "vector": [0, 0]}]), 400),
            (lambda: self.c.upsert(self.name, [{"id": 1, "vector": [0, 0, 0], "payload": {"price": "x"}}]), 400),
            (lambda: self.c.search(self.name, [0, 0, 0], filter={"range": {"color": {"gt": 1}}}), 400),
            (lambda: self.c.search(self.name, [0, 0, 0], k=1_000_000), 400),
            (lambda: self.c.search("no-such-collection", [0, 0, 0]), 404),
            (lambda: self.c.upsert_bulk(self.name, [1], [0.0] * 4), 400),
        ]
        for call, status in cases:
            with self.assertRaises(NearpointError) as e:
                call()
            self.assertEqual(e.exception.status, status, e.exception)
        # A rejected batch writes nothing, even its valid points.
        with self.assertRaises(NearpointError):
            self.c.upsert(self.name, [{"id": 5, "vector": [0, 0, 0]}, {"id": 6, "vector": [0]}])
        self.assertIsNone(self.c.get(self.name, 5))

    def test_bulk(self):
        n, dim = 2000, 3
        vecs = array.array("f", (float(i % 97) for i in range(n * dim)))
        self.assertEqual(self.c.upsert_bulk(self.name, range(n), vecs), n)
        payloads = [{"color": "red"} if i % 2 else None for i in range(10)]
        self.assertEqual(self.c.upsert_bulk(self.name, range(n, n + 10), [1.0] * 30, payloads), 10)
        self.assertEqual(self.c.upsert_bulk(self.name, [], []), 0)
        self.assertEqual(self.c.get_collection(self.name)["points"], n + 10)
        p = self.c.get(self.name, 7)
        self.assertEqual(p["vector"], list(vecs[7 * dim:8 * dim]))
        self.assertEqual(self.c.get(self.name, n + 1)["payload"], {"color": "red"})
        self.assertIsNone(self.c.get(self.name, n)["payload"])
        hits = self.c.search(self.name, [1, 1, 1], k=20, filter={"eq": {"color": "red"}})
        self.assertEqual(sorted(h["id"] for h in hits), [n + i for i in range(1, 10, 2)])


if __name__ == "__main__":
    unittest.main()
