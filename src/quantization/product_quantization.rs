use super::codebook::Codebook;
use crate::error::Result;
use serde::{Deserialize, Serialize};

/// Product Quantization codec for vector compression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQCodec {
    pub num_subvectors: usize,
    pub subvector_size: usize,
    pub codebooks: Vec<Codebook>,
}

impl PQCodec {
    /// Create a new PQ codec and train it
    pub fn train(
        training_vectors: &[Vec<f32>],
        num_subvectors: usize,
    ) -> Result<Self> {
        assert!(!training_vectors.is_empty());
        
        let dimensions = training_vectors[0].len();
        assert_eq!(dimensions % num_subvectors, 0, "Dimensions must be divisible by num_subvectors");
        
        let subvector_size = dimensions / num_subvectors;
        let num_centroids = 256; // 8-bit quantization
        
        println!("Training PQ codec:");
        println!("  Dimensions: {}", dimensions);
        println!("  Subvectors: {}", num_subvectors);
        println!("  Subvector size: {}", subvector_size);
        println!("  Training vectors: {}", training_vectors.len());
        
        let mut codebooks = Vec::with_capacity(num_subvectors);
        
        // Train one codebook per subvector
        for subvec_idx in 0..num_subvectors {
            println!("Training codebook {}/{}", subvec_idx + 1, num_subvectors);
            
            // Extract subvectors
            let subvectors: Vec<Vec<f32>> = training_vectors
                .iter()
                .map(|v| extract_subvector(v, subvec_idx, subvector_size))
                .collect();
            
            // Train codebook
            let codebook = Codebook::train(&subvectors, num_centroids);
            codebooks.push(codebook);
        }
        
        println!("PQ training complete!");
        
        Ok(Self {
            num_subvectors,
            subvector_size,
            codebooks,
        })
    }
    
    /// Encode a vector into PQ codes
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let mut codes = Vec::with_capacity(self.num_subvectors);
        
        for (subvec_idx, codebook) in self.codebooks.iter().enumerate() {
            let subvec = extract_subvector(vector, subvec_idx, self.subvector_size);
            let code = codebook.encode(&subvec);
            codes.push(code);
        }
        
        codes
    }
    
    /// Decode PQ codes back to approximate vector
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        let mut vector = Vec::with_capacity(self.num_subvectors * self.subvector_size);
        
        for (code, codebook) in codes.iter().zip(self.codebooks.iter()) {
            let subvec = codebook.decode(*code);
            vector.extend_from_slice(subvec);
        }
        
        vector
    }
    
    /// Asymmetric distance: query is NOT quantized (more accurate)
    pub fn asymmetric_distance(&self, query: &[f32], codes: &[u8]) -> f32 {
        let mut distance_squared = 0.0;
        
        for (subvec_idx, (code, codebook)) in codes.iter().zip(self.codebooks.iter()).enumerate() {
            let query_subvec = extract_subvector(query, subvec_idx, self.subvector_size);
            distance_squared += codebook.distance_to_centroid(&query_subvec, *code);
        }
        
        distance_squared.sqrt()
    }
    
    /// Precompute distance table for faster batch queries
    pub fn precompute_distance_table(&self, query: &[f32]) -> Vec<Vec<f32>> {
        let mut tables = Vec::with_capacity(self.num_subvectors);
        
        for (subvec_idx, codebook) in self.codebooks.iter().enumerate() {
            let query_subvec = extract_subvector(query, subvec_idx, self.subvector_size);
            
            let mut table = Vec::with_capacity(256);
            for code in 0..256 {
                let dist = codebook.distance_to_centroid(&query_subvec, code as u8);
                table.push(dist);
            }
            tables.push(table);
        }
        
        tables
    }
    
    /// Fast distance lookup using precomputed table
    pub fn table_lookup_distance(&self, dist_table: &[Vec<f32>], codes: &[u8]) -> f32 {
        codes
            .iter()
            .enumerate()
            .map(|(i, &code)| dist_table[i][code as usize])
            .sum::<f32>()
            .sqrt()
    }
}

fn extract_subvector(vector: &[f32], subvec_idx: usize, subvec_size: usize) -> Vec<f32> {
    let start = subvec_idx * subvec_size;
    let end = start + subvec_size;
    vector[start..end].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pq_encode_decode() {
        // Create synthetic training data
        let mut training = Vec::new();
        for i in 0..1000 {
            let vec: Vec<f32> = (0..128)
                .map(|j| ((i + j) as f32).sin())
                .collect();
            training.push(vec);
        }
        
        // Train PQ
        let pq = PQCodec::train(&training, 8).unwrap();
        
        // Test encoding/decoding
        let test_vec: Vec<f32> = (0..128).map(|i| (i as f32).cos()).collect();
        let codes = pq.encode(&test_vec);
        let decoded = pq.decode(&codes);
        
        assert_eq!(codes.len(), 8);
        assert_eq!(decoded.len(), 128);
        
        // Measure quantization error
        let error: f32 = test_vec
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>() / 128.0;
        
        println!("Average quantization error: {:.4}", error);
        assert!(error < 1.0); // Should have reasonable accuracy
    }
}