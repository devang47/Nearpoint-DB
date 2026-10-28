//! Distance kernels: scalar reference + NEON, f32 and int8, and the 4-bit
//! PQ fast-scan.
//!
//! The crate-root functions are the fastest version this build supports,
//! chosen at compile time (`.cargo/config.toml` builds with
//! `target-cpu=native`). `scalar` and `neon` stay public so benches and
//! tests can compare them directly.

#![allow(unsafe_code)]

#[cfg(target_arch = "aarch64")]
pub mod neon;
pub mod scalar;

pub use scalar::{PQ4_BLOCK, pq4_set};

#[cfg(target_arch = "aarch64")]
pub use neon::{cosine_distance, dot, l2_sq, pq4_scan};
#[cfg(not(target_arch = "aarch64"))]
pub use scalar::{cosine_distance, dot, l2_sq, pq4_scan};

#[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
pub use neon::{dot_i8, l2_sq_i8};
#[cfg(not(all(target_arch = "aarch64", target_feature = "dotprod")))]
pub use scalar::{dot_i8, l2_sq_i8};
