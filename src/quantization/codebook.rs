use super::kmeans::{kmeans, KMeansResult};

/// A codebook is a set of learned centroids for quantization
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Codebook {
    pub centroids: Vec<Vec<f32>>,
    pub dimensions: usize,
}

impl Codebook {
    /// Train a codebook from training vectors
    pub fn train(training_vectors: &[Vec<f32>], num_centroids: usize) -> Self {
        assert!(!training_vectors.is_empty());
        let dimensions = training_vectors[0].len();
        
        println!(
            "Training codebook: {} centroids, {} dims, {} training vectors",
            num_centroids,
            dimensions,
            training_vectors.len()
        );
        
        let result = kmeans(training_vectors, num_centroids, 100, 0.001);
        
        println!("Codebook training complete. Inertia: {:.4}", result.inertia);
        
        Self {
            centroids: result.centroids,
            dimensions,
        }
    }
    
    /// Encode a vector to its nearest centroid index
    pub fn encode(&self, vector: &[f32]) -> u8 {
        assert_eq!(vector.len(), self.dimensions);
        
        self.centroids
            .iter()
            .enumerate()
            .map(|(i, centroid)| {
                let dist = euclidean_distance_squared(vector, centroid);
                (i, dist)
            })
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as u8
    }
    
    /// Decode a centroid index back to a vector
    pub fn decode(&self, code: u8) -> &[f32] {
        &self.centroids[code as usize]
    }
    
    /// Compute distance from query vector to a centroid
    pub fn distance_to_centroid(&self, query: &[f32], code: u8) -> f32 {
        euclidean_distance_squared(query, &self.centroids[code as usize])
    }
}

fn euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_codebook() {
        let vectors = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];
        
        let codebook = Codebook::train(&vectors, 2);
        
        assert_eq!(codebook.centroids.len(), 2);
        
        // Vectors close to [0,0] should map to same code
        let code1 = codebook.encode(&vec![0.0, 0.0]);
        let code2 = codebook.encode(&vec![0.1, 0.1]);
        assert_eq!(code1, code2);
        
        // Vectors close to [10,10] should map to different code
        let code3 = codebook.encode(&vec![10.0, 10.0]);
        assert_ne!(code1, code3);
    }
}