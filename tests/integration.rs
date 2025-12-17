use khadyota::*;
use tempfile::NamedTempFile;

#[test]
fn test_pq_end_to_end() {
    // Generate training data
    let mut training = Vec::new();
    for i in 0..1000 {
        let vec: Vec<f32> = (0..512).map(|j| ((i + j) as f32).sin()).collect();
        training.push(vec);
    }
    
    // Train PQ codec
    let pq = quantization::PQCodec::train(&training, 8).unwrap();
    
    // Encode and measure compression
    let test_vec: Vec<f32> = (0..512).map(|i| (i as f32).cos()).collect();
    let codes = pq.encode(&test_vec);
    
    println!("Original: {} bytes", test_vec.len() * 4);
    println!("Compressed: {} bytes", codes.len());
    println!("Compression ratio: {}x", (test_vec.len() * 4) as f32 / codes.len() as f32);
    
    assert!(codes.len() < test_vec.len() * 4);
}

#[test]
fn test_simd_speedup() {
    let a: Vec<f32> = (0..512).map(|i| (i as f32).sin()).collect();
    let b: Vec<f32> = (0..512).map(|i| (i as f32).cos()).collect();
    
    // Both should give same result
    let scalar_result = distance::scalar::cosine_distance_scalar(&a, &b);
    let auto_result = distance::cosine_distance(&a, &b);
    
    assert!((scalar_result - auto_result).abs() < 1e-5);
}