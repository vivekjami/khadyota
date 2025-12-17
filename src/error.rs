use thiserror::Error;

#[derive(Error, Debug)]
pub enum KhadyotaError {
    #[error("Invalid vector dimension: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    
    #[error("Vector not found: {0}")]
    VectorNotFound(u32),
    
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Bincode error: {0}")]
    BincodeError(#[from] bincode::Error),
    
    #[error("Index not built. Call build_index() first.")]
    IndexNotBuilt,
}

pub type Result<T> = std::result::Result<T, KhadyotaError>;