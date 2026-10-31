//! Per-segment payload store: one typed column per field, keyed by the
//! dense row id (`u32`) the segment's index hands out.
//!
//! A field's type is fixed by the collection's schema if it declares the
//! field, else by its first value; a conflicting later value is an error,
//! with one widening: a float column accepts JSON integers (JS clients
//! can't send `10.0`). JSON `null` and `[]` mean "missing".
//!
//! Rows are append-only: a mutable segment inserts them in increasing order
//! and an upsert is a tombstone plus a new row, so no column is ever
//! rewritten in place. Everything is plain `Vec`/map/bitmap data, ready to
//! serialize into the segment file later.

use std::collections::{BTreeMap, HashMap};
use std::io;

use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::collection::{take, u32_at};
use crate::{Error, Result};

/// A column's type, fixed by the schema or the first value the field gets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Int,
    Float,
    Bool,
    /// A string, matched exactly.
    Keyword,
    /// A non-empty array of strings (tags). Scalars and lists don't mix.
    KeywordList,
}

/// Declared field types (`{"price": "float"}`): every segment of the
/// collection gets these columns up front, so the first value can't pick
/// another type. Undeclared fields keep the first-value rule.
pub type Schema = BTreeMap<String, Kind>;

#[derive(Debug, Default)]
pub struct PayloadStore {
    columns: BTreeMap<String, Column>,
    /// One past the highest row inserted.
    len: u32,
}

#[derive(Debug)]
pub(crate) struct Column {
    /// Rows that have a value for this field.
    pub(crate) present: RoaringBitmap,
    pub(crate) data: Data,
}

#[derive(Debug)]
pub(crate) enum Data {
    /// Indexed by row; missing rows hold 0.
    // ponytail: dense per field, so a field set on few rows still costs
    // 8 bytes per row; go sparse (rows + values) if wide schemas show up.
    Int(Vec<i64>),
    Float(Vec<f64>),
    /// Rows whose value is `true`; `false` is `present - trues`.
    Bool(RoaringBitmap),
    Keyword(Keywords),
}

/// Dictionary-encoded strings with an inverted index, since eq/in on
/// keywords is the dominant filter.
#[derive(Debug, Default)]
pub(crate) struct Keywords {
    pub(crate) list: bool,
    pub(crate) dict: HashMap<String, u32>,
    pub(crate) terms: Vec<String>,
    /// Per code, the rows holding that term.
    pub(crate) postings: Vec<RoaringBitmap>,
    /// Row `r`'s codes are `codes[offsets[r]..offsets[r + 1]]`.
    offsets: Vec<u32>,
    codes: Vec<u32>,
}

/// A parsed payload value, borrowed from the JSON it came from.
enum Parsed<'a> {
    Int(i64),
    Float(f64),
    Bool(bool),
    Keyword(&'a str),
    List(Vec<&'a str>),
}

impl Parsed<'_> {
    fn kind(&self) -> Kind {
        match self {
            Parsed::Int(_) => Kind::Int,
            Parsed::Float(_) => Kind::Float,
            Parsed::Bool(_) => Kind::Bool,
            Parsed::Keyword(_) => Kind::Keyword,
            Parsed::List(_) => Kind::KeywordList,
        }
    }
}

fn parse<'a>(field: &str, v: &'a Value) -> Result<Option<Parsed<'a>>> {
    let unsupported = |what| Error::Unsupported {
        field: field.to_owned(),
        what,
    };
    Ok(Some(match v {
        Value::Null => return Ok(None),
        Value::Bool(b) => Parsed::Bool(*b),
        Value::Number(n) => match n.as_i64() {
            Some(i) => Parsed::Int(i),
            None => Parsed::Float(n.as_f64().ok_or(unsupported("non-f64 numbers"))?),
        },
        Value::String(s) => Parsed::Keyword(s),
        Value::Array(a) if a.is_empty() => return Ok(None),
        Value::Array(a) => Parsed::List(
            a.iter()
                .map(|x| x.as_str().ok_or(unsupported("arrays of non-strings")))
                .collect::<Result<_>>()?,
        ),
        Value::Object(_) => return Err(unsupported("nested objects")),
    }))
}

impl Column {
    fn new(kind: Kind) -> Column {
        let data = match kind {
            Kind::Int => Data::Int(Vec::new()),
            Kind::Float => Data::Float(Vec::new()),
            Kind::Bool => Data::Bool(RoaringBitmap::new()),
            Kind::Keyword | Kind::KeywordList => Data::Keyword(Keywords {
                list: kind == Kind::KeywordList,
                offsets: vec![0],
                ..Keywords::default()
            }),
        };
        Column {
            present: RoaringBitmap::new(),
            data,
        }
    }

    pub(crate) fn kind(&self) -> Kind {
        match &self.data {
            Data::Int(_) => Kind::Int,
            Data::Float(_) => Kind::Float,
            Data::Bool(_) => Kind::Bool,
            Data::Keyword(k) if k.list => Kind::KeywordList,
            Data::Keyword(_) => Kind::Keyword,
        }
    }

    /// Call only after `accepts(self.kind(), v.kind())`.
    fn push(&mut self, row: u32, v: Parsed) {
        let r = row as usize;
        match (&mut self.data, v) {
            (Data::Int(vals), Parsed::Int(i)) => put(vals, r, i),
            (Data::Float(vals), Parsed::Float(f)) => put(vals, r, f),
            (Data::Float(vals), Parsed::Int(i)) => put(vals, r, i as f64),
            (Data::Bool(trues), Parsed::Bool(b)) => {
                if b {
                    trues.insert(row);
                }
            }
            (Data::Keyword(k), Parsed::Keyword(s)) => k.push(row, &[s]),
            (Data::Keyword(k), Parsed::List(l)) => k.push(row, &l),
            _ => return, // ruled out by `accepts`
        }
        self.present.insert(row);
    }

    fn value(&self, row: u32) -> Value {
        let r = row as usize;
        match &self.data {
            Data::Int(vals) => vals[r].into(),
            Data::Float(vals) => vals[r].into(),
            Data::Bool(trues) => trues.contains(row).into(),
            Data::Keyword(k) => {
                let codes = &k.codes[k.offsets[r] as usize..k.offsets[r + 1] as usize];
                let mut terms = codes
                    .iter()
                    .map(|&c| Value::from(k.terms[c as usize].as_str()));
                if k.list {
                    terms.collect()
                } else {
                    terms.next().unwrap_or_default()
                }
            }
        }
    }
}

fn accepts(column: Kind, got: Kind) -> bool {
    column == got || (column == Kind::Float && got == Kind::Int)
}

/// Sets `vals[r] = v`, filling skipped rows with the default.
fn put<T: Default + Clone>(vals: &mut Vec<T>, r: usize, v: T) {
    vals.resize(r, T::default());
    vals.push(v);
}

impl Keywords {
    fn push(&mut self, row: u32, terms: &[&str]) {
        let end = self.codes.len() as u32;
        self.offsets.resize(row as usize + 1, end);
        for &t in terms {
            let code = match self.dict.get(t) {
                Some(&c) => c,
                None => {
                    let c = self.terms.len() as u32;
                    self.dict.insert(t.to_owned(), c);
                    self.terms.push(t.to_owned());
                    self.postings.push(RoaringBitmap::new());
                    c
                }
            };
            self.postings[code as usize].insert(row);
            self.codes.push(code);
        }
        self.offsets.push(self.codes.len() as u32);
    }
}

impl PayloadStore {
    pub fn new() -> PayloadStore {
        PayloadStore::default()
    }

    pub fn with_schema(schema: &Schema) -> PayloadStore {
        let columns = schema
            .iter()
            .map(|(name, &kind)| (name.clone(), Column::new(kind)))
            .collect();
        PayloadStore { columns, len: 0 }
    }

    /// One past the highest row inserted (rows skipped have no payload).
    pub fn len(&self) -> u32 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Stores `payload` (a JSON object, or `null` for none) for `row`.
    /// Rows must arrive in increasing order; gaps are rows with no payload.
    /// A payload with any bad field is rejected whole, leaving no trace.
    pub fn insert(&mut self, row: u32, payload: &Value) -> Result<()> {
        let bad_row = Error::RowOrder {
            row,
            next: self.len,
        };
        if row < self.len {
            return Err(bad_row);
        }
        let next = row.checked_add(1).ok_or(bad_row)?;
        for (name, v) in self.parse_all(payload)? {
            self.columns
                .entry(name.clone())
                .or_insert_with(|| Column::new(v.kind()))
                .push(row, v);
        }
        self.len = next;
        Ok(())
    }

    /// Whether `insert` would accept `payloads` one after another (row
    /// order aside), without storing them. Lets a writer validate a whole
    /// batch before logging it: a field new to the store gets its type
    /// from its first value in the batch.
    pub fn check(&self, payloads: &[Value]) -> Result<()> {
        let mut new: HashMap<&str, Kind> = HashMap::new();
        for p in payloads {
            for (name, v) in self.parse_all(p)? {
                if self.columns.contains_key(name) {
                    continue; // checked by `parse_all`
                }
                let column = *new.entry(name).or_insert(v.kind());
                if !accepts(column, v.kind()) {
                    return Err(Error::TypeConflict {
                        field: name.clone(),
                        column,
                        got: v.kind(),
                    });
                }
            }
        }
        Ok(())
    }

    fn parse_all<'a>(&self, payload: &'a Value) -> Result<Vec<(&'a String, Parsed<'a>)>> {
        let fields = match payload {
            Value::Null => return Ok(Vec::new()),
            Value::Object(m) => m,
            _ => return Err(Error::NotAnObject),
        };
        let mut parsed = Vec::with_capacity(fields.len());
        for (name, v) in fields {
            let Some(v) = parse(name, v)? else { continue };
            if let Some(col) = self.columns.get(name)
                && !accepts(col.kind(), v.kind())
            {
                return Err(Error::TypeConflict {
                    field: name.clone(),
                    column: col.kind(),
                    got: v.kind(),
                });
            }
            parsed.push((name, v));
        }
        Ok(parsed)
    }

    /// The payload of `row` as a JSON object, or `None` if it has no fields.
    pub fn get(&self, row: u32) -> Option<Map<String, Value>> {
        let out: Map<_, _> = self
            .columns
            .iter()
            .filter(|(_, c)| c.present.contains(row))
            .map(|(name, c)| (name.clone(), c.value(row)))
            .collect();
        (!out.is_empty()).then_some(out)
    }

    /// The type of `field`, if any row has it.
    pub fn kind(&self, field: &str) -> Option<Kind> {
        self.columns.get(field).map(Column::kind)
    }

    /// Every column's type, so a rewritten segment can start with the same
    /// columns (an inferred type must not change when rows are dropped).
    pub(crate) fn kinds(&self) -> impl Iterator<Item = (&String, Kind)> {
        self.columns.iter().map(|(name, c)| (name, c.kind()))
    }

    pub(crate) fn column(&self, field: &str) -> Option<&Column> {
        self.columns.get(field)
    }
}

/// On-disk kind codes are the `Kind` discriminants (docs/formats/payloads.md).
const KINDS: [Kind; 5] = [
    Kind::Int,
    Kind::Float,
    Kind::Bool,
    Kind::Keyword,
    Kind::KeywordList,
];

impl PayloadStore {
    /// The columnar payload body (v2, docs/formats/payloads.md) for a
    /// segment of `rows` rows.
    pub(crate) fn encode(&self, rows: u32) -> io::Result<Vec<u8>> {
        let mut out = Vec::new();
        out.extend_from_slice(&rows.to_le_bytes());
        out.extend_from_slice(&(self.columns.len() as u32).to_le_bytes());
        for (name, c) in &self.columns {
            put_bytes(&mut out, name.as_bytes());
            out.push(c.kind() as u8);
            put_bitmap(&mut out, &c.present)?;
            match &c.data {
                Data::Int(v) => put_words(&mut out, v.iter().map(|x| x.to_le_bytes())),
                Data::Float(v) => put_words(&mut out, v.iter().map(|x| x.to_le_bytes())),
                Data::Bool(trues) => put_bitmap(&mut out, trues)?,
                Data::Keyword(k) => {
                    out.extend_from_slice(&(k.terms.len() as u32).to_le_bytes());
                    for t in &k.terms {
                        put_bytes(&mut out, t.as_bytes());
                    }
                    put_words(&mut out, k.offsets.iter().map(|x| x.to_le_bytes()));
                    put_words(&mut out, k.codes.iter().map(|x| x.to_le_bytes()));
                }
            }
        }
        Ok(out)
    }

    /// Reads [`encode`](Self::encode)'s output back; `None` if it is
    /// malformed, has another row count, or conflicts with `schema`.
    pub(crate) fn decode(mut b: &[u8], rows: u32, schema: &Schema) -> Option<PayloadStore> {
        if u32_at(&mut b)? != rows {
            return None;
        }
        let mut store = PayloadStore::with_schema(schema);
        let mut prev: Option<String> = None;
        for _ in 0..u32_at(&mut b)? {
            let name = String::from_utf8(get_bytes(&mut b)?.to_vec()).ok()?;
            if prev.as_ref().is_some_and(|p| *p >= name) {
                return None; // names are strictly ascending
            }
            let kind = *KINDS.get(usize::from(*take(&mut b, 1)?.first()?))?;
            let present = get_bitmap(&mut b)?;
            if present.max().is_some_and(|m| m >= rows) {
                return None;
            }
            // A row past the values would make `get` index out of bounds.
            let covers =
                |n: usize| n <= rows as usize && present.max().is_none_or(|m| (m as usize) < n);
            let data = match kind {
                Kind::Int => {
                    Data::Int(get_words(&mut b, i64::from_le_bytes).filter(|v| covers(v.len()))?)
                }
                Kind::Float => {
                    Data::Float(get_words(&mut b, f64::from_le_bytes).filter(|v| covers(v.len()))?)
                }
                Kind::Bool => Data::Bool(get_bitmap(&mut b).filter(|t| t.is_subset(&present))?),
                Kind::Keyword | Kind::KeywordList => {
                    let mut k = Keywords {
                        list: kind == Kind::KeywordList,
                        ..Keywords::default()
                    };
                    for code in 0..u32_at(&mut b)? {
                        let t = String::from_utf8(get_bytes(&mut b)?.to_vec()).ok()?;
                        if k.dict.insert(t.clone(), code).is_some() {
                            return None;
                        }
                        k.terms.push(t);
                        k.postings.push(RoaringBitmap::new());
                    }
                    k.offsets = get_words(&mut b, u32::from_le_bytes)?;
                    k.codes = get_words(&mut b, u32::from_le_bytes)?;
                    if k.offsets.first() != Some(&0)
                        || k.offsets.last() != Some(&(k.codes.len() as u32))
                        || !covers(k.offsets.len() - 1)
                    {
                        return None;
                    }
                    // Rebuild the postings; a row has codes iff it is present
                    // (exactly one for a scalar keyword).
                    let mut seen = RoaringBitmap::new();
                    for (row, w) in (0u32..).zip(k.offsets.windows(2)) {
                        let codes = k.codes.get(w[0] as usize..w[1] as usize)?;
                        if codes.len() > 1 && !k.list {
                            return None;
                        }
                        for &c in codes {
                            // `insert`, not `push`: push rescans a dense
                            // container for its max (10x slower here).
                            k.postings.get_mut(c as usize)?.insert(row);
                        }
                        if !codes.is_empty() {
                            seen.insert(row);
                        }
                    }
                    if seen != present {
                        return None;
                    }
                    Data::Keyword(k)
                }
            };
            if store.kind(&name).is_some_and(|k| k != kind) {
                return None; // the schema declares another type
            }
            store.columns.insert(name.clone(), Column { present, data });
            prev = Some(name);
        }
        store.len = store
            .columns
            .values()
            .filter_map(|c| c.present.max())
            .max()
            .map_or(0, |m| m + 1);
        b.is_empty().then_some(store)
    }
}

fn put_bytes(out: &mut Vec<u8>, b: &[u8]) {
    out.extend_from_slice(&(b.len() as u32).to_le_bytes());
    out.extend_from_slice(b);
}

fn get_bytes<'a>(b: &mut &'a [u8]) -> Option<&'a [u8]> {
    let n = u32_at(b)? as usize;
    take(b, n)
}

fn put_bitmap(out: &mut Vec<u8>, bits: &RoaringBitmap) -> io::Result<()> {
    out.extend_from_slice(&(bits.serialized_size() as u32).to_le_bytes());
    bits.serialize_into(out)
}

fn get_bitmap(b: &mut &[u8]) -> Option<RoaringBitmap> {
    let bytes = get_bytes(b)?;
    let bits = RoaringBitmap::deserialize_from(bytes).ok()?;
    (bits.serialized_size() == bytes.len()).then_some(bits)
}

/// `n` fixed-width little-endian words after a `u32` count.
fn put_words<const W: usize>(out: &mut Vec<u8>, words: impl ExactSizeIterator<Item = [u8; W]>) {
    out.extend_from_slice(&(words.len() as u32).to_le_bytes());
    out.reserve(words.len() * W);
    for w in words {
        out.extend_from_slice(&w);
    }
}

fn get_words<T, const W: usize>(b: &mut &[u8], from: fn([u8; W]) -> T) -> Option<Vec<T>> {
    let n = u32_at(b)? as usize;
    let bytes = take(b, n.checked_mul(W)?)?;
    Some(bytes.as_chunks::<W>().0.iter().map(|&c| from(c)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn round_trips_every_type_with_gaps() {
        let mut s = PayloadStore::new();
        let a = json!({"i": -3, "f": 2.5, "b": false, "k": "red", "t": ["x", "y", "x"]});
        s.insert(0, &a).unwrap();
        s.insert(3, &json!({"f": 7, "k": "blue"})).unwrap(); // int widens into float
        s.insert(4, &json!(null)).unwrap();
        assert_eq!(s.len(), 5);
        assert_eq!(s.get(0), a.as_object().cloned());
        assert_eq!(
            s.get(3),
            json!({"f": 7.0, "k": "blue"}).as_object().cloned()
        );
        assert_eq!(s.get(1), None);
        assert_eq!(s.get(4), None);
        assert_eq!(s.get(99), None);
    }

    #[test]
    fn columnar_encoding_round_trips_and_rejects_damage() {
        let schema = Schema::from([("empty".into(), Kind::Bool), ("price".into(), Kind::Float)]);
        let mut s = PayloadStore::with_schema(&schema);
        let rows = [
            json!({"i": -3, "price": 2, "b": true, "k": "red", "t": ["x", "y", "x"], "n": null}),
            json!(null),
            json!({"b": false, "k": "réd", "t": [], "i": i64::MIN}),
            json!({}),
            json!({"price": 1e300, "t": ["y"], "i": i64::MAX, "k": "red"}),
        ];
        for (r, v) in rows.iter().enumerate() {
            s.insert(r as u32, v).unwrap();
        }
        let body = s.encode(7).unwrap();
        let back = PayloadStore::decode(&body, 7, &schema).unwrap();
        for r in 0..8 {
            assert_eq!(back.get(r), s.get(r), "row {r}");
        }
        assert_eq!(
            back.kinds().collect::<Vec<_>>(),
            s.kinds().collect::<Vec<_>>()
        );
        assert_eq!(back.len(), s.len());
        let k = |st: &PayloadStore| match &st.column("t").unwrap().data {
            Data::Keyword(k) => (k.dict.clone(), k.postings.clone()),
            _ => unreachable!(),
        };
        assert_eq!(k(&back), k(&s));
        assert!(PayloadStore::decode(&body, 6, &schema).is_none());
        let other = Schema::from([("i".into(), Kind::Float)]);
        assert!(PayloadStore::decode(&body, 7, &other).is_none());
        // Malformed bodies (CRC aside) are rejected or decode, never panic.
        for n in 0..body.len() {
            assert!(PayloadStore::decode(&body[..n], 7, &schema).is_none());
        }
        for i in 0..body.len() {
            let mut bad = body.clone();
            bad[i] ^= 0x41;
            let _ =
                PayloadStore::decode(&bad, 7, &schema).map(|b| (0..8).for_each(|r| drop(b.get(r))));
        }
    }

    #[test]
    fn type_conflicts_are_errors_and_leave_no_trace() {
        let mut s = PayloadStore::new();
        s.insert(0, &json!({"i": 1, "k": "a", "t": ["a"]})).unwrap();
        let conflict = |field: &str, column, got| Error::TypeConflict {
            field: field.into(),
            column,
            got,
        };
        let e = s.insert(1, &json!({"new": 1, "i": 1.5}));
        assert_eq!(e, Err(conflict("i", Kind::Int, Kind::Float)));
        assert_eq!(
            s.kind("new"),
            None,
            "rejected payload must not create columns"
        );
        assert_eq!(s.len(), 1);
        let e = s.insert(1, &json!({"k": ["a"]}));
        assert_eq!(e, Err(conflict("k", Kind::Keyword, Kind::KeywordList)));
        let e = s.insert(1, &json!({"t": "a"}));
        assert_eq!(e, Err(conflict("t", Kind::KeywordList, Kind::Keyword)));
        let e = s.insert(1, &json!({"k": true}));
        assert_eq!(e, Err(conflict("k", Kind::Keyword, Kind::Bool)));
    }

    #[test]
    fn schema_fixes_types_and_check_sees_the_whole_batch() {
        let schema = Schema::from([("price".into(), Kind::Float), ("tag".into(), Kind::Keyword)]);
        let mut s = PayloadStore::with_schema(&schema);
        assert_eq!(s.kind("price"), Some(Kind::Float));
        assert_eq!(s.get(0), None);
        s.check(&[json!({"price": 10}), json!({"price": 10.5})])
            .unwrap();
        s.insert(0, &json!({"price": 10})).unwrap();
        s.insert(1, &json!({"price": 10.5})).unwrap();
        assert!(s.check(&[json!({"tag": 1})]).is_err());
        // Undeclared: the batch's first value fixes the type.
        let e = s.check(&[json!({"n": 10}), json!({"n": 10.5})]);
        assert_eq!(
            e,
            Err(Error::TypeConflict {
                field: "n".into(),
                column: Kind::Int,
                got: Kind::Float
            })
        );
        s.check(&[json!({"n": 1.5}), json!({"n": 2})]).unwrap();
        assert_eq!(s.kind("n"), None, "check stores nothing");
    }

    #[test]
    fn rejects_bad_shapes_and_order() {
        let mut s = PayloadStore::new();
        assert_eq!(s.insert(0, &json!([1])), Err(Error::NotAnObject));
        assert!(matches!(
            s.insert(0, &json!({"o": {"x": 1}})),
            Err(Error::Unsupported { .. })
        ));
        assert!(matches!(
            s.insert(0, &json!({"a": [1, 2]})),
            Err(Error::Unsupported { .. })
        ));
        s.insert(5, &json!({})).unwrap();
        assert_eq!(
            s.insert(5, &json!({})),
            Err(Error::RowOrder { row: 5, next: 6 })
        );
        assert!(s.insert(u32::MAX, &json!({})).is_err());
    }
}
