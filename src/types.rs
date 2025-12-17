use serde::{Deserialize, Serialize};

/// Search result with distance and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: u32,
    pub distance: f32,
    pub metadata: Option<serde_json::Value>,
}

/// Vector with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorEntry {
    pub id: u32,
    pub vector: Vec<f32>,
    pub metadata: Option<serde_json::Value>,
}