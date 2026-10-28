//! Filter expressions over a [`PayloadStore`], evaluated to a bitmap of
//! matching rows. JSON shape and semantics: `docs/filters.md`.
//!
//! Two-valued logic: a leaf never matches a row missing the field (an
//! unknown field is missing everywhere), and `not` is the plain complement
//! against `0..universe`, so `not eq` does match rows without the field.

use std::collections::BTreeMap;
use std::ops::RangeInclusive;

use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};

use crate::payload::{Column, Data, PayloadStore};
use crate::{Error, Result};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "Wire", into = "Wire")]
pub enum Filter {
    /// Equal; on a keyword list, "contains".
    Eq(String, Scalar),
    /// Equal to any of the values; on a keyword list, "contains any".
    In(String, Vec<Scalar>),
    /// Numeric range (int and float columns only).
    Range(String, Bounds),
    Exists(String),
    /// All of; `and: []` matches every row.
    And(Vec<Filter>),
    /// Any of; `or: []` matches none.
    Or(Vec<Filter>),
    Not(Box<Filter>),
}

/// A filter operand. JSON integers parse as `Int`, other numbers as `Float`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Scalar {
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
}

/// Range bounds; all given bounds must hold.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bounds {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gt: Option<Scalar>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gte: Option<Scalar>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lt: Option<Scalar>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lte: Option<Scalar>,
}

/// The JSON form: `{"eq": {"color": "red"}}`, `{"and": [...]}`, ...
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Wire {
    Eq(BTreeMap<String, Scalar>),
    In(BTreeMap<String, Vec<Scalar>>),
    Range(BTreeMap<String, Bounds>),
    Exists(String),
    And(Vec<Filter>),
    Or(Vec<Filter>),
    Not(Box<Filter>),
}

fn one<T>(op: &str, m: BTreeMap<String, T>) -> std::result::Result<(String, T), String> {
    let mut it = m.into_iter();
    match (it.next(), it.next()) {
        (Some(kv), None) => Ok(kv),
        _ => Err(format!("`{op}` takes exactly one field")),
    }
}

impl TryFrom<Wire> for Filter {
    type Error = String;
    fn try_from(w: Wire) -> std::result::Result<Filter, String> {
        Ok(match w {
            Wire::Eq(m) => one("eq", m).map(|(f, v)| Filter::Eq(f, v))?,
            Wire::In(m) => one("in", m).map(|(f, v)| Filter::In(f, v))?,
            Wire::Range(m) => one("range", m).map(|(f, b)| Filter::Range(f, b))?,
            Wire::Exists(f) => Filter::Exists(f),
            Wire::And(fs) => Filter::And(fs),
            Wire::Or(fs) => Filter::Or(fs),
            Wire::Not(f) => Filter::Not(f),
        })
    }
}

impl From<Filter> for Wire {
    fn from(f: Filter) -> Wire {
        match f {
            Filter::Eq(f, v) => Wire::Eq([(f, v)].into()),
            Filter::In(f, v) => Wire::In([(f, v)].into()),
            Filter::Range(f, b) => Wire::Range([(f, b)].into()),
            Filter::Exists(f) => Wire::Exists(f),
            Filter::And(fs) => Wire::And(fs),
            Filter::Or(fs) => Wire::Or(fs),
            Filter::Not(f) => Wire::Not(f),
        }
    }
}

/// The rows in `0..universe` matching `filter`. `universe` is the
/// segment's row count, which may exceed `store.len()` (trailing rows with
/// no payload). Tombstones are the caller's to subtract.
pub fn evaluate(filter: &Filter, store: &PayloadStore, universe: u32) -> Result<RoaringBitmap> {
    let mut bits = eval(filter, store, universe)?;
    bits.remove_range(universe..);
    Ok(bits)
}

/// Fraction of the universe a bitmap keeps, for the brute-force planner.
pub fn selectivity(bits: &RoaringBitmap, universe: u32) -> f64 {
    if universe == 0 {
        return 0.0;
    }
    bits.len() as f64 / f64::from(universe)
}

fn all(universe: u32) -> RoaringBitmap {
    let mut b = RoaringBitmap::new();
    b.insert_range(0..universe);
    b
}

// No short-circuiting: a type error is reported whatever the data holds.
fn eval(f: &Filter, store: &PayloadStore, universe: u32) -> Result<RoaringBitmap> {
    let leaf = |field: &str| store.column(field);
    Ok(match f {
        Filter::Eq(field, v) => leaf(field).map_or(Ok(RoaringBitmap::new()), |c| {
            any_of(field, c, std::slice::from_ref(v))
        })?,
        Filter::In(field, vs) => {
            leaf(field).map_or(Ok(RoaringBitmap::new()), |c| any_of(field, c, vs))?
        }
        Filter::Range(field, b) => {
            leaf(field).map_or(Ok(RoaringBitmap::new()), |c| range(field, c, b))?
        }
        Filter::Exists(field) => leaf(field).map(|c| c.present.clone()).unwrap_or_default(),
        Filter::And(fs) => {
            let mut acc = all(universe);
            for f in fs {
                acc &= eval(f, store, universe)?;
            }
            acc
        }
        Filter::Or(fs) => {
            let mut acc = RoaringBitmap::new();
            for f in fs {
                acc |= eval(f, store, universe)?;
            }
            acc
        }
        Filter::Not(f) => all(universe) - eval(f, store, universe)?,
    })
}

fn mismatch(field: &str, c: &Column, op: &'static str) -> Error {
    Error::TypeMismatch {
        field: field.to_owned(),
        column: c.kind(),
        op,
    }
}

/// The present rows whose value satisfies `pred`.
// ponytail: numeric eq/in/range scan the column, O(present rows) per leaf
// (unmeasured yet). Add a sorted (value, row) index per sealed
// segment if numeric filters dominate latency.
fn scan(c: &Column, pred: impl Fn(usize) -> bool) -> RoaringBitmap {
    c.present.iter().filter(|&r| pred(r as usize)).collect()
}

/// -2^63, exact as an f64; i64 holds `-2^63..2^63`.
const I64_MIN: f64 = i64::MIN as f64;

fn any_of(field: &str, c: &Column, vs: &[Scalar]) -> Result<RoaringBitmap> {
    let bad = || mismatch(field, c, "eq/in");
    let mut out = RoaringBitmap::new();
    match &c.data {
        Data::Keyword(k) => {
            for v in vs {
                let Scalar::Str(s) = v else { return Err(bad()) };
                if let Some(&code) = k.dict.get(s) {
                    out |= &k.postings[code as usize];
                }
            }
        }
        Data::Bool(trues) => {
            for v in vs {
                match v {
                    Scalar::Bool(true) => out |= trues,
                    Scalar::Bool(false) => out |= &c.present - trues,
                    _ => return Err(bad()),
                }
            }
        }
        Data::Int(vals) => {
            let mut want = Vec::with_capacity(vs.len());
            for v in vs {
                match *v {
                    Scalar::Int(i) => want.push(i),
                    // Only an integral float in range can equal an int.
                    Scalar::Float(f) if f.fract() == 0.0 && (I64_MIN..-I64_MIN).contains(&f) => {
                        want.push(f as i64)
                    }
                    Scalar::Float(_) => {}
                    _ => return Err(bad()),
                }
            }
            out = scan(c, |r| want.contains(&vals[r]));
        }
        Data::Float(vals) => {
            let mut want = Vec::with_capacity(vs.len());
            for v in vs {
                want.push(match *v {
                    Scalar::Int(i) => i as f64,
                    Scalar::Float(f) => f,
                    _ => return Err(bad()),
                });
            }
            out = scan(c, |r| want.contains(&vals[r]));
        }
    }
    Ok(out)
}

fn range(field: &str, c: &Column, b: &Bounds) -> Result<RoaringBitmap> {
    match &c.data {
        Data::Int(vals) => {
            let ok = int_range(b).ok_or_else(|| mismatch(field, c, "range"))?;
            Ok(scan(c, |r| ok.contains(&i128::from(vals[r]))))
        }
        Data::Float(vals) => {
            let f = |s: &Option<Scalar>| match *s {
                None => Some(None),
                Some(Scalar::Int(i)) => Some(Some(i as f64)),
                Some(Scalar::Float(x)) => Some(Some(x)),
                Some(_) => None,
            };
            let bad = || mismatch(field, c, "range");
            let (gt, gte) = (f(&b.gt).ok_or_else(bad)?, f(&b.gte).ok_or_else(bad)?);
            let (lt, lte) = (f(&b.lt).ok_or_else(bad)?, f(&b.lte).ok_or_else(bad)?);
            Ok(scan(c, |r| {
                let x = vals[r];
                gt.is_none_or(|b| x > b)
                    && gte.is_none_or(|b| x >= b)
                    && lt.is_none_or(|b| x < b)
                    && lte.is_none_or(|b| x <= b)
            }))
        }
        Data::Bool(_) | Data::Keyword(_) => Err(mismatch(field, c, "range")),
    }
}

/// `b` as an exact inclusive integer range (`None` if a bound isn't a
/// number). Float bounds round inward: `x > 2.5` is `x >= 3`. i128 holds
/// every i64 ± 1 and saturated huge floats without overflow.
fn int_range(b: &Bounds) -> Option<RangeInclusive<i128>> {
    let t = |s: &Option<Scalar>, round: fn(f64) -> f64| match *s {
        None => Some(None),
        Some(Scalar::Int(i)) => Some(Some(i128::from(i))),
        Some(Scalar::Float(f)) => Some(Some(round(f) as i128)),
        Some(_) => None,
    };
    let mut lo = i128::MIN;
    let mut hi = i128::MAX;
    if let Some(g) = t(&b.gt, f64::floor)? {
        lo = lo.max(g.saturating_add(1));
    }
    if let Some(g) = t(&b.gte, f64::ceil)? {
        lo = lo.max(g);
    }
    if let Some(l) = t(&b.lt, f64::ceil)? {
        hi = hi.min(l.saturating_sub(1));
    }
    if let Some(l) = t(&b.lte, f64::floor)? {
        hi = hi.min(l);
    }
    Some(lo..=hi)
}
