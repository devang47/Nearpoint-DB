//! Collections, segments, compaction, deletes and filters.

pub mod filter;
pub mod payload;

pub use filter::{Bounds, Filter, Scalar, evaluate, selectivity};
pub use payload::{Kind, PayloadStore};

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum Error {
    #[error("payload must be a JSON object or null")]
    NotAnObject,
    #[error("field `{field}`: {what} are not supported")]
    Unsupported { field: String, what: &'static str },
    #[error("field `{field}` is a {column:?} column, got a {got:?} value")]
    TypeConflict {
        field: String,
        column: Kind,
        got: Kind,
    },
    #[error("row {row} is out of order: rows must increase (next is {next})")]
    RowOrder { row: u32, next: u32 },
    #[error("`{op}` does not apply to field `{field}` ({column:?} column) with that value")]
    TypeMismatch {
        field: String,
        column: Kind,
        op: &'static str,
    },
}

pub type Result<T> = std::result::Result<T, Error>;
