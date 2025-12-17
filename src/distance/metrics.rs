use crate::config::DistanceMetric;

/// Compute distance with automatic SIMD dispatch
pub fn compute_distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::Cosine => cosine_distance(a, b),
        DistanceMetric::Euclidean => euclidean_distance(a, b),
        DistanceMetric::DotProduct => dot_product(a, b),
    }
}

/// Cosine distance with runtime dispatch
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && a.len() % 8 == 0 {
            unsafe { super::simd::cosine_distance_avx2(a, b) }
        } else {
            super::scalar::cosine_distance_scalar(a, b)
        }
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    {
        super::scalar::cosine_distance_scalar(a, b)
    }
}

pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && a.len() % 8 == 0 {
            unsafe { super::simd::euclidean_distance_avx2(a, b) }
        } else {
            super::scalar::euclidean_distance_scalar(a, b)
        }
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    {
        super::scalar::euclidean_distance_scalar(a, b)
    }
}

pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && a.len() % 8 == 0 {
            unsafe { super::simd::dot_product_avx2(a, b) }
        } else {
            super::scalar::dot_product_scalar(a, b)
        }
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    {
        super::scalar::dot_product_scalar(a, b)
    }
}