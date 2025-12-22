use crate::distance::metrics::euclidean_distance;
use crate::quantization::kmeans::{kmeans, KMeansResult};
use serde::{Deserialize, Serialize};

/// Inverted File Index for fast approximate search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVFIndex {
    /// Cluster centroids
    centroids: Vec<Vec<f32>>,
    
    /// Inverted lists: cluster_id -> vector_ids in that cluster
    inverted_lists: Vec<Vec<u32>>,
    
    /// Number of clusters to probe during search
    num_probe: usize,
    
    /// Dimensionality
    dimensions: usize,
}

impl IVFIndex {
    /// Create a new empty IVF index
    pub fn new(dimensions: usize, num_clusters: usize, num_probe: usize) -> Self {
        Self {
            centroids: Vec::new(),
            inverted_lists: vec![Vec::new(); num_clusters],
            num_probe,
            dimensions,
        }
    }
    
    /// Build the IVF index from training vectors
    pub fn build(&mut self, vectors: &[Vec<f32>], num_clusters: usize) {
        assert!(!vectors.is_empty(), "Cannot build index from empty vectors");
        
        println!("Building IVF index with {} clusters...", num_clusters);
        
        // Step 1: Learn cluster centroids using K-means
        println!("  Running K-means clustering...");
        let result = kmeans(vectors, num_clusters, 100, 0.001);
        self.centroids = result.centroids;
        
        println!("  K-means complete. Inertia: {:.2}", result.inertia);
        
        // Step 2: Assign each vector to its nearest cluster
        println!("  Assigning vectors to clusters...");
        self.inverted_lists = vec![Vec::new(); num_clusters];
        
        for (vec_id, vector) in vectors.iter().enumerate() {
            let cluster_id = self.find_nearest_cluster(vector);
            self.inverted_lists[cluster_id].push(vec_id as u32);
        }
        
        // Print cluster statistics
        let mut cluster_sizes: Vec<_> = self.inverted_lists
            .iter()
            .map(|list| list.len())
            .collect();
        cluster_sizes.sort();
        
        println!("  Cluster size stats:");
        println!("    Min: {}", cluster_sizes.first().unwrap_or(&0));
        println!("    Median: {}", cluster_sizes[cluster_sizes.len() / 2]);
        println!("    Max: {}", cluster_sizes.last().unwrap_or(&0));
        println!("IVF index built successfully!");
    }
    
    /// Find the nearest cluster centroid for a vector
    fn find_nearest_cluster(&self, vector: &[f32]) -> usize {
        self.centroids
            .iter()
            .enumerate()
            .map(|(i, centroid)| {
                let dist = euclidean_distance(vector, centroid);
                (i, dist)
            })
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    }
    
    /// Find the k nearest clusters to probe for a query
    pub fn probe(&self, query: &[f32]) -> Vec<usize> {
        let mut distances: Vec<(usize, f32)> = self.centroids
            .iter()
            .enumerate()
            .map(|(i, centroid)| {
                let dist = euclidean_distance(query, centroid);
                (i, dist)
            })
            .collect();
        
        // Sort by distance and take top num_probe
        distances.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
        
        distances
            .iter()
            .take(self.num_probe)
            .map(|(i, _)| *i)
            .collect()
    }
    
    /// Get candidate vector IDs from probed clusters
    pub fn get_candidates(&self, cluster_ids: &[usize]) -> Vec<u32> {
        let mut candidates = Vec::new();
        
        for &cluster_id in cluster_ids {
            candidates.extend_from_slice(&self.inverted_lists[cluster_id]);
        }
        
        candidates
    }
    
    /// Get statistics about the index
    pub fn stats(&self) -> IVFStats {
        let total_vectors: usize = self.inverted_lists.iter().map(|l| l.len()).sum();
        let non_empty_clusters = self.inverted_lists.iter().filter(|l| !l.is_empty()).count();
        
        let mut sizes: Vec<_> = self.inverted_lists.iter().map(|l| l.len()).collect();
        sizes.sort();
        
        IVFStats {
            num_clusters: self.centroids.len(),
            total_vectors,
            non_empty_clusters,
            min_cluster_size: *sizes.first().unwrap_or(&0),
            median_cluster_size: sizes[sizes.len() / 2],
            max_cluster_size: *sizes.last().unwrap_or(&0),
            num_probe: self.num_probe,
        }
    }
    
    /// Set number of clusters to probe
    pub fn set_num_probe(&mut self, num_probe: usize) {
        self.num_probe = num_probe.min(self.centroids.len());
    }
}

#[derive(Debug, Clone)]
pub struct IVFStats {
    pub num_clusters: usize,
    pub total_vectors: usize,
    pub non_empty_clusters: usize,
    pub min_cluster_size: usize,
    pub median_cluster_size: usize,
    pub max_cluster_size: usize,
    pub num_probe: usize,
}

impl std::fmt::Display for IVFStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "IVF Index Stats:\n\
             - Clusters: {} ({} non-empty)\n\
             - Total vectors: {}\n\
             - Cluster sizes: min={}, median={}, max={}\n\
             - Probe: {} clusters per query",
            self.num_clusters,
            self.non_empty_clusters,
            self.total_vectors,
            self.min_cluster_size,
            self.median_cluster_size,
            self.max_cluster_size,
            self.num_probe
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ivf_build() {
        // Create synthetic vectors in 3 clear clusters
        let mut vectors = Vec::new();
        
        // Cluster 1: around [0, 0]
        for _ in 0..100 {
            vectors.push(vec![
                rand::random::<f32>() * 0.5,
                rand::random::<f32>() * 0.5,
            ]);
        }
        
        // Cluster 2: around [10, 0]
        for _ in 0..100 {
            vectors.push(vec![
                10.0 + rand::random::<f32>() * 0.5,
                rand::random::<f32>() * 0.5,
            ]);
        }
        
        // Cluster 3: around [0, 10]
        for _ in 0..100 {
            vectors.push(vec![
                rand::random::<f32>() * 0.5,
                10.0 + rand::random::<f32>() * 0.5,
            ]);
        }
        
        let mut index = IVFIndex::new(2, 3, 1);
        index.build(&vectors, 3);
        
        let stats = index.stats();
        println!("{}", stats);
        
        assert_eq!(stats.num_clusters, 3);
        assert_eq!(stats.total_vectors, 300);
        assert!(stats.non_empty_clusters >= 2); // At least 2 should have vectors
    }
    
    #[test]
    fn test_ivf_probe() {
        let mut vectors = Vec::new();
        for i in 0..1000 {
            let vec: Vec<f32> = (0..128)
                .map(|j| ((i + j) as f32).sin())
                .collect();
            vectors.push(vec);
        }
        
        let mut index = IVFIndex::new(128, 10, 3);
        index.build(&vectors, 10);
        
        // Test probing
        let query: Vec<f32> = (0..128).map(|i| (i as f32).cos()).collect();
        let clusters = index.probe(&query);
        
        assert_eq!(clusters.len(), 3); // Should return num_probe clusters
        
        let candidates = index.get_candidates(&clusters);
        println!("Got {} candidates from 3 clusters", candidates.len());
        
        // Should get candidates from probed clusters
        assert!(!candidates.is_empty());
    }
}