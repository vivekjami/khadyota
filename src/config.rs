use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    DotProduct,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Vector dimensionality (e.g., 512 for typical embeddings)
    pub dimensions: usize,
    
    /// Distance metric to use
    pub metric: DistanceMetric,
    
    /// Enable Product Quantization compression
    pub use_pq: bool,
    
    /// Number of subvectors for PQ (typically 8)
    pub pq_subvectors: usize,
    
    /// Number of IVF clusters
    pub num_clusters: usize,
    
    /// Number of clusters to probe during search
    pub num_probe: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            dimensions: 512,
            metric: DistanceMetric::Cosine,
            use_pq: true,
            pq_subvectors: 8,
            num_clusters: 100,
            num_probe: 10,
        }
    }
}

impl Config {
    pub fn validate(&self) -> crate::error::Result<()> {
        if self.dimensions == 0 {
            return Err(crate::error::KhadyotaError::InvalidConfig(
                "Dimensions must be > 0".to_string()
            ));
        }
        
        if self.use_pq && self.dimensions % self.pq_subvectors != 0 {
            return Err(crate::error::KhadyotaError::InvalidConfig(
                format!(
                    "Dimensions ({}) must be divisible by pq_subvectors ({})",
                    self.dimensions, self.pq_subvectors
                )
            ));
        }
        
        Ok(())
    }
    
    pub fn subvector_size(&self) -> usize {
        self.dimensions / self.pq_subvectors
    }
}