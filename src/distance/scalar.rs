/// Cosine similarity (scalar implementation)
pub fn cosine_similarity_scalar(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

/// Cosine distance (1 - similarity)
pub fn cosine_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_similarity_scalar(a, b)
}

/// Euclidean distance (L2)
pub fn euclidean_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    
    sum.sqrt()
}

/// Squared Euclidean distance (faster, skip sqrt)
pub fn euclidean_distance_squared_scalar(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    
    sum
}

/// Dot product
pub fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        
        let sim = cosine_similarity_scalar(&a, &b);
        assert_relative_eq!(sim, 1.0, epsilon = 1e-6);
        
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        
        let sim = cosine_similarity_scalar(&a, &b);
        assert_relative_eq!(sim, 0.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        
        let dist = euclidean_distance_scalar(&a, &b);
        assert_relative_eq!(dist, 5.0, epsilon = 1e-6);
    }
}