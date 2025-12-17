pub mod metrics;
pub mod scalar;

#[cfg(target_arch = "x86_64")]
pub mod simd;

pub use metrics::{compute_distance, cosine_distance, euclidean_distance, dot_product};