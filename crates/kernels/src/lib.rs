//! Distance kernels: scalar reference + NEON, f32 and int8.
//!
//! The crate-root functions are the fastest version this build supports,
//! chosen at compile time (`.cargo/config.toml` builds with
//! `target-cpu=native`). `scalar` and `neon` stay public so benches and
//! tests can compare them directly.

#![allow(unsafe_code)]

#[cfg(target_arch = "aarch64")]
pub mod neon;
pub mod scalar;

#[cfg(target_arch = "aarch64")]
pub use neon::{cosine_distance, dot, l2_sq};
#[cfg(not(target_arch = "aarch64"))]
pub use scalar::{cosine_distance, dot, l2_sq};

#[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
pub use neon::{dot_i8, l2_sq_i8};
#[cfg(not(all(target_arch = "aarch64", target_feature = "dotprod")))]
pub use scalar::{dot_i8, l2_sq_i8};
