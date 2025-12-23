use crate::quantization::PQCodec;
use serde::{Deserialize, Serialize};

/// Storage for quantized vectors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedVectors {
    /// PQ codes for each vector
    codes: Vec<Vec<u8>>,
    
    /// Original vectors (kept for reranking if needed)
    original_vectors: Option<Vec<Vec<f32>>>,
    
    /// PQ codec
    codec: PQCodec,
}

impl QuantizedVectors {
    /// Create new quantized storage
    pub fn new(codec: PQCodec) -> Self {
        Self {
            codes: Vec::new(),
            original_vectors: None,
            codec,
        }
    }
    
    /// Add a vector (will be quantized)
    pub fn add(&mut self, vector: Vec<f32>) -> u32 {
        let code = self.codec.encode(&vector);
        let id = self.codes.len() as u32;
        self.codes.push(code);
        id
    }
    
    /// Add multiple vectors in batch
    pub fn add_batch(&mut self, vectors: Vec<Vec<f32>>) {
        for vector in vectors {
            self.add(vector);
        }
    }
    
    /// Get quantized codes for a vector
    pub fn get_codes(&self, id: u32) -> &[u8] {
        &self.codes[id as usize]
    }
    
    /// Compute distance using PQ
    pub fn asymmetric_distance(&self, query: &[f32], id: u32) -> f32 {
        let codes = self.get_codes(id);
        self.codec.asymmetric_distance(query, codes)
    }
    
    /// Precompute distance table for batch queries
    pub fn precompute_distance_table(&self, query: &[f32]) -> Vec<Vec<f32>> {
        self.codec.precompute_distance_table(query)
    }
    
    /// Fast distance lookup using precomputed table
    pub fn table_lookup_distance(&self, dist_table: &[Vec<f32>], id: u32) -> f32 {
        let codes = self.get_codes(id);
        self.codec.table_lookup_distance(dist_table, codes)
    }
    
    pub fn len(&self) -> usize {
        self.codes.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }
}