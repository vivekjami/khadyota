use serde::{Deserialize, Serialize};

/// Magic bytes to identify Khadyota files
pub const MAGIC: &[u8; 4] = b"KHDY";
pub const VERSION: u32 = 1;

#[derive(Debug, Serialize, Deserialize)]
pub struct FileHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub dimensions: u32,
    pub vector_count: u64,
    pub metric: crate::config::DistanceMetric,
}

impl FileHeader {
    pub fn new(dimensions: usize, vector_count: usize, metric: crate::config::DistanceMetric) -> Self {
        Self {
            magic: *MAGIC,
            version: VERSION,
            dimensions: dimensions as u32,
            vector_count: vector_count as u64,
            metric,
        }
    }
    
    pub fn validate(&self) -> crate::error::Result<()> {
        if &self.magic != MAGIC {
            return Err(crate::error::KhadyotaError::SerializationError(
                "Invalid magic bytes".to_string()
            ));
        }
        
        if self.version != VERSION {
            return Err(crate::error::KhadyotaError::SerializationError(
                format!("Unsupported version: {}", self.version)
            ));
        }
        
        Ok(())
    }
}