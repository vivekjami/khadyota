#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Cosine similarity using AVX2 (8 floats at once)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn cosine_similarity_avx2(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len() % 8, 0, "Length must be multiple of 8 for AVX2");
    
    let mut dot_sum = _mm256_setzero_ps();
    let mut norm_a_sum = _mm256_setzero_ps();
    let mut norm_b_sum = _mm256_setzero_ps();
    
    let chunks = a.len() / 8;
    
    for i in 0..chunks {
        let offset = i * 8;
        
        // Load 8 floats from a and b
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        
        // Dot product: sum += a * b
        dot_sum = _mm256_fmadd_ps(va, vb, dot_sum);
        
        // Norms: sum += a * a, b * b
        norm_a_sum = _mm256_fmadd_ps(va, va, norm_a_sum);
        norm_b_sum = _mm256_fmadd_ps(vb, vb, norm_b_sum);
    }
    
    // Horizontal sum: reduce 8 values to 1
    let dot = horizontal_sum_avx2(dot_sum);
    let norm_a = horizontal_sum_avx2(norm_a_sum).sqrt();
    let norm_b = horizontal_sum_avx2(norm_b_sum).sqrt();
    
    dot / (norm_a * norm_b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn cosine_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_similarity_avx2(a, b)
}

/// Euclidean distance squared using AVX2
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn euclidean_distance_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len() % 8, 0);
    
    let mut sum = _mm256_setzero_ps();
    let chunks = a.len() / 8;
    
    for i in 0..chunks {
        let offset = i * 8;
        
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        
        // (a - b)^2
        let diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }
    
    horizontal_sum_avx2(sum)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn euclidean_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    euclidean_distance_squared_avx2(a, b).sqrt()
}

/// Dot product using AVX2
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len() % 8, 0);
    
    let mut sum = _mm256_setzero_ps();
    let chunks = a.len() / 8;
    
    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    
    horizontal_sum_avx2(sum)
}

/// Horizontal sum: reduce __m256 (8 floats) to single float
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_avx2(v: __m256) -> f32 {
    // Extract high and low 128-bit lanes
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    
    // Add them
    let sum128 = _mm_add_ps(lo, hi);
    
    // Horizontal add within 128 bits
    let sum64 = _mm_hadd_ps(sum128, sum128);
    let sum32 = _mm_hadd_ps(sum64, sum64);
    
    // Extract final value
    _mm_cvtss_f32(sum32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::scalar::*;
    use approx::assert_relative_eq;
    
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            println!("AVX2 not available, skipping test");
            return;
        }
        
        let a: Vec<f32> = (0..512).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..512).map(|i| (i as f32).cos()).collect();
        
        unsafe {
            let simd_result = cosine_similarity_avx2(&a, &b);
            let scalar_result = cosine_similarity_scalar(&a, &b);
            
            assert_relative_eq!(simd_result, scalar_result, epsilon = 1e-5);
        }
    }
}