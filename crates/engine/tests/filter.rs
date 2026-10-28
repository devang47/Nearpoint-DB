//! `evaluate` against a row-by-row reference over the same payloads.

use engine::{Bounds, Error, Filter, Kind, PayloadStore, Scalar, evaluate, selectivity};
use proptest::collection::vec;
use proptest::option;
use proptest::prelude::*;
use proptest::sample::select;
use roaring::RoaringBitmap;
use serde_json::{Map, Value, json};

// ---- reference evaluator: the documented semantics, one row at a time ----

/// The field's value, treating `null` and `[]` as missing.
fn field<'a>(p: Option<&'a Map<String, Value>>, name: &str) -> Option<&'a Value> {
    p?.get(name)
        .filter(|v| !v.is_null() && v.as_array().is_none_or(|a| !a.is_empty()))
}

fn eq(x: &Value, v: &Scalar) -> bool {
    match (x, v) {
        (Value::Array(a), _) => a.iter().any(|e| eq(e, v)),
        (Value::String(s), Scalar::Str(t)) => s == t,
        (Value::Bool(a), Scalar::Bool(b)) => a == b,
        (Value::Number(n), Scalar::Int(i)) => n.as_f64() == Some(*i as f64),
        (Value::Number(n), Scalar::Float(f)) => n.as_f64() == Some(*f),
        _ => false,
    }
}

fn num(s: &Option<Scalar>) -> Option<f64> {
    match *s {
        Some(Scalar::Int(i)) => Some(i as f64),
        Some(Scalar::Float(f)) => Some(f),
        _ => None,
    }
}

fn matches(f: &Filter, p: Option<&Map<String, Value>>) -> bool {
    match f {
        Filter::Eq(name, v) => field(p, name).is_some_and(|x| eq(x, v)),
        Filter::In(name, vs) => field(p, name).is_some_and(|x| vs.iter().any(|v| eq(x, v))),
        Filter::Range(name, b) => field(p, name).and_then(Value::as_f64).is_some_and(|x| {
            num(&b.gt).is_none_or(|g| x > g)
                && num(&b.gte).is_none_or(|g| x >= g)
                && num(&b.lt).is_none_or(|l| x < l)
                && num(&b.lte).is_none_or(|l| x <= l)
        }),
        Filter::Exists(name) => field(p, name).is_some(),
        Filter::And(fs) => fs.iter().all(|f| matches(f, p)),
        Filter::Or(fs) => fs.iter().any(|f| matches(f, p)),
        Filter::Not(f) => !matches(f, p),
    }
}

// ---- generators: fixed types per field so payloads never conflict ----
// i: int, f: float, b: bool, k: keyword, t: keyword list, zz: never set.

const KW: [&str; 5] = ["a", "b", "c", "d", "e"];

fn payload() -> impl Strategy<Value = Value> {
    (
        option::of(-4i64..4),
        option::of(select(vec![-1.5, 0.0, 0.5, 2.0, 3.25])),
        option::of(any::<bool>()),
        option::of(select(&KW[..4])),
        option::of(vec(select(&KW[..4]), 0..3)),
    )
        .prop_map(|(i, f, b, k, t)| {
            let mut m = Map::new();
            i.map(|v| m.insert("i".into(), v.into()));
            f.map(|v| m.insert("f".into(), v.into()));
            b.map(|v| m.insert("b".into(), v.into()));
            k.map(|v| m.insert("k".into(), v.into()));
            t.map(|v| m.insert("t".into(), v.into()));
            Value::Object(m)
        })
}

fn number() -> impl Strategy<Value = Scalar> {
    prop_oneof![
        (-5i64..5).prop_map(Scalar::Int),
        select(vec![-1.5, 0.0, 0.5, 2.0, 3.0, 3.25]).prop_map(Scalar::Float),
    ]
}

fn keyword() -> impl Strategy<Value = Scalar> {
    select(&KW[..]).prop_map(|s| Scalar::Str(s.into()))
}

fn bounds() -> impl Strategy<Value = Bounds> {
    (
        option::of(number()),
        option::of(number()),
        option::of(number()),
        option::of(number()),
    )
        .prop_map(|(gt, gte, lt, lte)| Bounds { gt, gte, lt, lte })
}

fn name(names: &'static [&'static str]) -> impl Strategy<Value = String> {
    select(names).prop_map(String::from)
}

fn filter() -> impl Strategy<Value = Filter> {
    let leaf = prop_oneof![
        (name(&["i", "f", "zz"]), number()).prop_map(|(n, v)| Filter::Eq(n, v)),
        (name(&["i", "f"]), vec(number(), 0..3)).prop_map(|(n, v)| Filter::In(n, v)),
        (name(&["i", "f", "zz"]), bounds()).prop_map(|(n, b)| Filter::Range(n, b)),
        (name(&["k", "t", "zz"]), keyword()).prop_map(|(n, v)| Filter::Eq(n, v)),
        (name(&["k", "t"]), vec(keyword(), 0..3)).prop_map(|(n, v)| Filter::In(n, v)),
        any::<bool>().prop_map(|b| Filter::Eq("b".into(), Scalar::Bool(b))),
        vec(any::<bool>().prop_map(Scalar::Bool), 0..3).prop_map(|v| Filter::In("b".into(), v)),
        name(&["i", "f", "b", "k", "t", "zz"]).prop_map(Filter::Exists),
    ];
    leaf.prop_recursive(4, 32, 4, |inner| {
        prop_oneof![
            vec(inner.clone(), 0..4).prop_map(Filter::And),
            vec(inner.clone(), 0..4).prop_map(Filter::Or),
            inner.prop_map(|f| Filter::Not(Box::new(f))),
        ]
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn evaluate_equals_row_by_row_reference(
        rows in vec(option::of(payload()), 0..40),
        extra in 0u32..3,
        shrink in 0u32..3,
        f in filter(),
    ) {
        let mut store = PayloadStore::new();
        for (r, p) in rows.iter().enumerate() {
            if let Some(p) = p {
                store.insert(r as u32, p).unwrap();
            }
        }
        let row = |r: u32| rows.get(r as usize).and_then(Option::as_ref).and_then(Value::as_object);
        for r in 0..rows.len() as u32 {
            let stored: Option<Map<_, _>> = row(r).map(|p| {
                p.iter().filter(|(k, _)| field(Some(p), k).is_some())
                    .map(|(k, v)| (k.clone(), v.clone())).collect()
            }).filter(|m: &Map<_, _>| !m.is_empty());
            prop_assert_eq!(store.get(r), stored);
        }
        // The universe may cover more rows than have payloads, or fewer.
        let universe = (rows.len() as u32 + extra).saturating_sub(shrink);
        let got = evaluate(&f, &store, universe).unwrap();
        let want: RoaringBitmap = (0..universe).filter(|&r| matches(&f, row(r))).collect();
        prop_assert_eq!(got, want);
    }

    #[test]
    fn filters_round_trip_through_json(f in filter()) {
        let s = serde_json::to_string(&f).unwrap();
        prop_assert_eq!(serde_json::from_str::<Filter>(&s).unwrap(), f);
    }
}

fn parse(v: Value) -> Filter {
    serde_json::from_value(v).expect("valid filter")
}

fn store() -> PayloadStore {
    let mut s = PayloadStore::new();
    let rows = [
        json!({"color": "red", "price": 12.0, "qty": 3, "tags": ["new", "sale"], "stock": true}),
        json!({"color": "blue", "price": 25.5, "qty": 10, "tags": ["sale"]}),
        json!({"price": 8}), // an int widens into the float column
        json!(null),
    ];
    for (r, p) in rows.iter().enumerate() {
        s.insert(r as u32, p).expect("fixture fits");
    }
    s
}

fn rows(s: &PayloadStore, f: Value) -> Vec<u32> {
    evaluate(&parse(f), s, 5)
        .expect("evaluates")
        .iter()
        .collect()
}

#[test]
fn documented_semantics() {
    let s = store();
    assert_eq!(s.kind("price"), Some(Kind::Float));
    assert_eq!(s.kind("qty"), Some(Kind::Int));
    // Missing never matches a leaf; `not` is the complement (row 4 has no payload).
    assert_eq!(rows(&s, json!({"eq": {"color": "red"}})), [0]);
    assert_eq!(
        rows(&s, json!({"not": {"eq": {"color": "red"}}})),
        [1, 2, 3, 4]
    );
    let sql_ne = json!({"and": [{"exists": "color"}, {"not": {"eq": {"color": "red"}}}]});
    assert_eq!(rows(&s, sql_ne), [1]);
    // Lists: eq is "contains", in is "contains any".
    assert_eq!(rows(&s, json!({"eq": {"tags": "sale"}})), [0, 1]);
    assert_eq!(rows(&s, json!({"in": {"tags": ["new", "x"]}})), [0]);
    // Unknown fields match nothing, even with odd operand types.
    assert_eq!(
        rows(&s, json!({"range": {"nope": {"gt": "x"}}})),
        [] as [u32; 0]
    );
    assert_eq!(
        rows(&s, json!({"not": {"eq": {"nope": 1}}})),
        [0, 1, 2, 3, 4]
    );
    // Empty and / or.
    assert_eq!(rows(&s, json!({"and": []})), [0, 1, 2, 3, 4]);
    assert_eq!(rows(&s, json!({"or": []})), [] as [u32; 0]);
    // Float bounds on an int column are exact.
    assert_eq!(
        rows(&s, json!({"range": {"qty": {"gt": 3.5, "lte": 10.0}}})),
        [1]
    );
    assert_eq!(
        rows(&s, json!({"range": {"price": {"gte": 8, "lt": 20}}})),
        [0, 2]
    );
    assert_eq!(rows(&s, json!({"eq": {"qty": 3.0}})), [0]);
    assert_eq!(rows(&s, json!({"eq": {"qty": 3.5}})), [] as [u32; 0]);
    assert_eq!(rows(&s, json!({"eq": {"stock": false}})), [] as [u32; 0]);

    let bits = evaluate(&parse(json!({"exists": "color"})), &s, 5).unwrap();
    assert_eq!(selectivity(&bits, 5), 0.4);
    assert_eq!(selectivity(&bits, 0), 0.0);
}

#[test]
fn type_mismatches_are_errors_even_when_nested() {
    let s = store();
    let mismatch = |field: &str, column, op| Error::TypeMismatch {
        field: field.into(),
        column,
        op,
    };
    let err = |f| evaluate(&parse(f), &s, 5).unwrap_err();
    assert_eq!(
        err(json!({"range": {"color": {"gt": 1}}})),
        mismatch("color", Kind::Keyword, "range")
    );
    assert_eq!(
        err(json!({"range": {"price": {"gt": "a"}}})),
        mismatch("price", Kind::Float, "range")
    );
    assert_eq!(
        err(json!({"eq": {"qty": "a"}})),
        mismatch("qty", Kind::Int, "eq/in")
    );
    assert_eq!(
        err(json!({"in": {"tags": [1]}})),
        mismatch("tags", Kind::KeywordList, "eq/in")
    );
    assert_eq!(
        err(json!({"eq": {"stock": "yes"}})),
        mismatch("stock", Kind::Bool, "eq/in")
    );
    // No short-circuit: the empty first branch doesn't hide the error.
    let nested = json!({"and": [{"eq": {"nope": 1}}, {"not": {"range": {"tags": {}}}}]});
    assert_eq!(err(nested), mismatch("tags", Kind::KeywordList, "range"));
}

#[test]
fn malformed_json_is_rejected() {
    for bad in [
        json!({"eq": {"a": 1, "b": 2}}),
        json!({"eq": {}}),
        json!({"eq": {"a": 1}, "in": {"a": [1]}}),
        json!({"range": {"a": {"gt": 1, "between": 2}}}),
        json!({"xor": []}),
        json!({"eq": {"a": [1]}}),
    ] {
        assert!(
            serde_json::from_value::<Filter>(bad.clone()).is_err(),
            "{bad}"
        );
    }
}

#[test]
fn every_example_in_the_docs_parses() {
    let doc = include_str!("../../../docs/filters.md");
    let blocks: Vec<&str> = doc
        .split("```json")
        .skip(1)
        .filter_map(|b| b.split("```").next())
        .collect();
    assert!(blocks.len() >= 5);
    for b in blocks {
        serde_json::from_str::<Filter>(b).unwrap_or_else(|e| panic!("{e}: {b}"));
    }
}
