use crate::config::Config;
use crate::error::Result;
use crate::indexing::IVFIndex;
use crate::quantization::PQCodec;
use crate::storage::QuantizedVectors;
use crate::types::SearchResult;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;

/// Main Vector Database structure
pub struct VectorDB {
    config: Config,
    vectors: Vec<Vec<f32>>,
    quantized: Option<QuantizedVectors>,
    ivf_index: Option<IVFIndex>,
    metadata: HashMap<u32, serde_json::Value>,
    next_id: u32,
    index_built: bool,
}

impl VectorDB {
    /// Create a new vector database
    pub fn new(config: Config) -> Result<Self> {
        config.validate()?;
        
        Ok(Self {
            config,
            vectors: Vec::new(),
            quantized: None,
            ivf_index: None,
            metadata: HashMap::new(),
            next_id: 0,
            index_built: false,
        })
    }
    
    /// Insert a vector with optional metadata
    pub fn insert(&mut self, vector: Vec<f32>, metadata: Option<serde_json::Value>) -> Result<u32> {
        if vector.len() != self.config.dimensions {
            return Err(crate::error::KhadyotaError::DimensionMismatch {
                expected: self.config.dimensions,
                got: vector.len(),
            });
        }
        
        let id = self.next_id;
        self.vectors.push(vector);
        
        if let Some(meta) = metadata {
            self.metadata.insert(id, meta);
        }
        
        self.next_id += 1;
        self.index_built = false; // Need to rebuild index
        
        Ok(id)
    }
    
    /// Build the search index (PQ + IVF)
    pub fn build_index(&mut self) -> Result<()> {
        if self.vectors.is_empty() {
            return Err(crate::error::KhadyotaError::InvalidConfig(
                "Cannot build index with no vectors".to_string()
            ));
        }
        
        println!("\n=== Building Search Index ===");
        println!("Vectors: {}", self.vectors.len());
        println!("Dimensions: {}", self.config.dimensions);
        
        // Step 1: Train and apply Product Quantization
        if self.config.use_pq {
            println!("\n[1/2] Training Product Quantization...");
            let pq_codec = PQCodec::train(&self.vectors, self.config.pq_subvectors)?;
            
            let mut quantized = QuantizedVectors::new(pq_codec);
            for vector in &self.vectors {
                quantized.add(vector.clone());
            }
            
            self.quantized = Some(quantized);
            println!("✓ PQ training complete");
        }
        
        // Step 2: Build IVF index
        println!("\n[2/2] Building IVF Index...");
        let mut ivf = IVFIndex::new(
            self.config.dimensions,
            self.config.num_clusters,
            self.config.num_probe,
        );
        
        ivf.build(&self.vectors, self.config.num_clusters);
        
        let stats = ivf.stats();
        println!("\n{}", stats);
        
        self.ivf_index = Some(ivf);
        self.index_built = true;
        
        println!("\n✓ Index built successfully!\n");
        
        Ok(())
    }
    
    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.config.dimensions {
            return Err(crate::error::KhadyotaError::DimensionMismatch {
                expected: self.config.dimensions,
                got: query.len(),
            });
        }
        
        if !self.index_built {
            return Err(crate::error::KhadyotaError::IndexNotBuilt);
        }
        
        // Use IVF + PQ search if available
        if let (Some(ivf), Some(quantized)) = (&self.ivf_index, &self.quantized) {
            self.search_with_index(query, k, ivf, quantized)
        } else {
            // Fallback to linear scan
            self.search_linear(query, k)
        }
    }
    
    /// Search using IVF + PQ
    fn search_with_index(
        &self,
        query: &[f32],
        k: usize,
        ivf: &IVFIndex,
        quantized: &QuantizedVectors,
    ) -> Result<Vec<SearchResult>> {
        // Step 1: Probe IVF to get candidate clusters
        let clusters = ivf.probe(query);
        let candidates = ivf.get_candidates(&clusters);
        
        // Step 2: Precompute PQ distance table
        let dist_table = quantized.precompute_distance_table(query);
        
        // Step 3: Compute distances to candidates
        let mut scored: Vec<(u32, f32)> = candidates
            .iter()
            .map(|&vec_id| {
                let distance = quantized.table_lookup_distance(&dist_table, vec_id);
                (vec_id, distance)
            })
            .collect();
        
        // Step 4: Sort and take top-k
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scored.truncate(k);
        
        // Step 5: Build results
        Ok(scored
            .into_iter()
            .map(|(id, distance)| SearchResult {
                id,
                distance,
                metadata: self.metadata.get(&id).cloned(),
            })
            .collect())
    }
    
    /// Fallback linear scan (for small datasets or when index not built)
    fn search_linear(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        use crate::distance::compute_distance;
        
        let mut scored: Vec<(u32, f32)> = self.vectors
            .iter()
            .enumerate()
            .map(|(i, vector)| {
                let distance = compute_distance(query, vector, self.config.metric);
                (i as u32, distance)
            })
            .collect();
        
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scored.truncate(k);
        
        Ok(scored
            .into_iter()
            .map(|(id, distance)| SearchResult {
                id,
                distance,
                metadata: self.metadata.get(&id).cloned(),
            })
            .collect())
    }
    
    /// Save database to disk
    pub fn save(&self, path: &Path) -> Result<()> {
        use std::fs::File;
        
        println!("Saving database to {:?}...", path);
        
        let file = File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        
        // Serialize everything
        rmp_serde::encode::write(&mut writer, &(
            &self.config,
            &self.vectors,
            &self.quantized,
            &self.ivf_index,
            &self.metadata,
            self.next_id,
            self.index_built,
        ))?;
        
        let bytes_written = writer.get_ref().metadata()?.len();
        println!("✓ Database saved ({} bytes)", bytes_written);
        
        Ok(())
    }
    
    /// Load database from disk
    pub fn load(path: &Path) -> Result<Self> {
        use std::fs::File;
        
        println!("Loading database from {:?}...", path);
        
        let file = File::open(path)?;
        let reader = std::io::BufReader::new(file);
        
        let (config, vectors, quantized, ivf_index, metadata, next_id, index_built): (Config, Vec<Vec<f32>>, Option<QuantizedVectors>, Option<IVFIndex>, HashMap<u32, serde_json::Value>, u32, bool) =
            rmp_serde::from_read(reader)?;
        
        println!("✓ Database loaded ({} vectors)", vectors.len());
        
        Ok(Self {
            config,
            vectors,
            quantized,
            ivf_index,
            metadata,
            next_id,
            index_built,
        })
    }
    
    pub fn len(&self) -> usize {
        self.vectors.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Batch search multiple queries in parallel
    pub fn batch_search(&self, queries: &[Vec<f32>], k: usize) -> Result<Vec<Vec<SearchResult>>> {
        if !self.index_built {
            return Err(crate::error::KhadyotaError::IndexNotBuilt);
        }
        
        queries
            .par_iter()
            .map(|query| self.search(query, k))
            .collect()
    }
    
    /// Parallel candidate scoring for large result sets
    fn search_with_index_parallel(
        &self,
        query: &[f32],
        k: usize,
        ivf: &IVFIndex,
        quantized: &QuantizedVectors,
    ) -> Result<Vec<SearchResult>> {
        // Probe IVF
        let clusters = ivf.probe(query);
        let candidates = ivf.get_candidates(&clusters);
        
        // Precompute distance table
        let dist_table = quantized.precompute_distance_table(query);
        
        // Parallel distance computation
        let mut scored: Vec<(u32, f32)> = candidates
            .par_iter()
            .map(|&vec_id| {
                let distance = quantized.table_lookup_distance(&dist_table, vec_id);
                (vec_id, distance)
            })
            .collect();
        
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scored.truncate(k);
        
        Ok(scored
            .into_iter()
            .map(|(id, distance)| SearchResult {
                id,
                distance,
                metadata: self.metadata.get(&id).cloned(),
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_vector_db_end_to_end() {
        let config = Config {
            dimensions: 128,
            use_pq: true,
            pq_subvectors: 8,
            num_clusters: 10,
            num_probe: 3,
            ..Default::default()
        };
        
        let mut db = VectorDB::new(config).unwrap();
        
        // Insert vectors
        for i in 0..1000 {
            let vector: Vec<f32> = (0..128)
                .map(|j| ((i + j) as f32).sin())
                .collect();
            
            db.insert(vector, Some(serde_json::json!({"id": i}))).unwrap();
        }
        
        // Build index
        db.build_index().unwrap();
        
        // Search
        let query: Vec<f32> = (0..128).map(|i| (i as f32).cos()).collect();
        let results = db.search(&query, 10).unwrap();
        
        assert_eq!(results.len(), 10);
        
        // Test save/load
        let temp = NamedTempFile::new().unwrap();
        db.save(temp.path()).unwrap();
        
        let loaded = VectorDB::load(temp.path()).unwrap();
        assert_eq!(loaded.len(), 1000);
        
        let results2 = loaded.search(&query, 10).unwrap();
        assert_eq!(results2.len(), 10);
    }
}