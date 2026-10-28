# Payloads and filters

The payload and filter half of the HTTP API (task 7.2; the endpoints arrive
with 4.4 and 7.3). Code: `crates/engine/src/{payload,filter}.rs`. Design
notes: `docs/adr/0005-payloads-and-filters.md`. Every `json` block on this
page is parsed by a test (`crates/engine/tests/filter.rs`).

## Payloads

A payload is a flat JSON object (or `null`) attached to a vector:

| JSON value               | Column type   | Notes                                  |
|--------------------------|---------------|----------------------------------------|
| integer (`12`)           | int (i64)     |                                        |
| other number (`12.5`)    | float (f64)   | a float column also accepts integers   |
| `true` / `false`         | bool          |                                        |
| string                   | keyword       | exact match, dictionary-encoded        |
| array of strings         | keyword list  | tags; `[]` counts as missing           |
| `null`                   | (missing)     |                                        |
| nested object, other arrays | rejected   | `Unsupported` error                    |

Each field's type is fixed by the first value it gets in a segment. A
later value of another type rejects the whole payload with a
`TypeConflict` error; nothing is coerced. The one exception is that a
float column accepts integers, because JavaScript sends `10.0` as `10`.
The reverse doesn't work: if a field's first value is `10`, a later `10.5`
is rejected. Keyword and keyword-list fields are different types too.

## Filter expressions

A filter is a JSON object with exactly one operator key:

| Operator | Shape                                        | Matches rows where             |
|----------|----------------------------------------------|--------------------------------|
| `eq`     | `{"eq": {"field": value}}`                   | field equals value             |
| `in`     | `{"in": {"field": [v1, v2]}}`                | field equals any value         |
| `range`  | `{"range": {"field": {"gte": a, "lt": b}}}`  | every given bound holds        |
| `exists` | `{"exists": "field"}`                        | field has a value              |
| `and`    | `{"and": [f1, f2]}`                          | all match (`[]` matches every row) |
| `or`     | `{"or": [f1, f2]}`                           | any matches (`[]` matches none) |
| `not`    | `{"not": f}`                                 | `f` doesn't match              |

`eq`, `in` and `range` name exactly one field. Range bounds are `gt`,
`gte`, `lt` and `lte`, in any combination. Values are JSON scalars:
strings, numbers and booleans.

### Semantics

- **Missing fields never match a leaf.** A row without `color` matches no
  `eq`, `in`, `range` or `exists` on `color`.
- **`not` is the plain complement** over all rows in the segment. So
  `{"not": {"eq": {"color": "red"}}}` *does* match rows with no `color`
  (two-valued logic, as in Elasticsearch `must_not` and Qdrant
  `must_not`, not SQL's `NULL`). For SQL-style `!=`, add `exists`:

```json
{"and": [{"exists": "color"}, {"not": {"eq": {"color": "red"}}}]}
```

- **Unknown fields** are missing on every row, so they match nothing and
  aren't an error. A typo in a field name gives an empty result, not a 400.
  The same applies to fields that exist in other segments but not in this
  one.
- **Keyword lists:** `eq` means "contains" and `in` means "contains any".
  To require all tags, use `and` of `eq`s.
- **Numbers:** int and float compare by value, so `{"eq": {"qty": 3.0}}`
  matches `3`, and a fractional value never equals an int. Float bounds on
  int fields are exact (`gt 2.5` is `>= 3`).
- **Type mismatches are errors** (`TypeMismatch`): `range` on a keyword,
  list or bool field, a string compared with a number field, a number
  compared with a keyword field, and so on. These are reported even inside
  a branch that would otherwise match nothing.
- Deleted rows are removed by the engine after evaluation, not by the
  filter.

### Examples

Keyword equality:

```json
{"eq": {"color": "red"}}
```

Any of several values, or a tag in a list:

```json
{"in": {"color": ["red", "blue"]}}
```

```json
{"eq": {"tags": "sale"}}
```

A half-open numeric range:

```json
{"range": {"price": {"gte": 10, "lt": 20}}}
```

Combined:

```json
{"and": [
  {"eq": {"color": "red"}},
  {"range": {"price": {"gte": 10, "lt": 20}}},
  {"or": [{"eq": {"in_stock": true}}, {"not": {"exists": "discontinued"}}]}
]}
```

Everything without the `archived` tag (including rows with no tags):

```json
{"not": {"eq": {"tags": "archived"}}}
```

## Filtered search

A filter is applied during the search, not to its results, so a search
returns `k` matches whenever the segment has `k` matching rows (task 7.3,
`docs/adr/0010-filtered-search.md`):

- **HNSW** walks the graph as usual, including through rows that don't
  match, but only matching rows enter the result set; it keeps going until
  it holds `ef` matches nearer than anything left to explore.
- **When few rows match**, HNSW computes the exact distance to every
  matching row instead of walking (task 7.4). It does this when
  `matches² <= 8 · max(ef, k) · rows`: on 1M vectors at `ef` 64 that is
  under ~22,600 matches (~2.3%), and a segment of a few hundred rows always
  scans. Those results are exact.
- **Flat** visits only the matching rows.
