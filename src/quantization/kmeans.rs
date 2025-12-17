use rand::seq::SliceRandom;
use rand::thread_rng;

/// K-means clustering result
#[derive(Debug, Clone)]
pub struct KMeansResult {
    pub centroids: Vec<Vec<f32>>,
    pub assignments: Vec<usize>,
    pub inertia: f32,
}

/// Run K-means clustering
pub fn kmeans(
    vectors: &[Vec<f32>],
    k: usize,
    max_iterations: usize,
    tolerance: f32,
) -> KMeansResult {
    assert!(!vectors.is_empty(), "Cannot cluster empty vectors");
    assert!(k <= vectors.len(), "K must be <= number of vectors");
    
    let dimensions = vectors[0].len();
    
    // Initialize centroids using k-means++
    let mut centroids = kmeans_plus_plus_init(vectors, k);
    let mut assignments = vec![0; vectors.len()];
    let mut prev_inertia = f32::INFINITY;
    
    for iteration in 0..max_iterations {
        // Assignment step: assign each vector to nearest centroid
        let mut inertia = 0.0;
        for (i, vector) in vectors.iter().enumerate() {
            let (nearest_idx, distance) = find_nearest_centroid(vector, &centroids);
            assignments[i] = nearest_idx;
            inertia += distance * distance;
        }
        
        // Check convergence
        if (prev_inertia - inertia).abs() < tolerance {
            println!("K-means converged at iteration {}", iteration);
            break;
        }
        prev_inertia = inertia;
        
        // Update step: recompute centroids
        let mut new_centroids = vec![vec![0.0; dimensions]; k];
        let mut counts = vec![0usize; k];
        
        for (vector, &cluster) in vectors.iter().zip(assignments.iter()) {
            counts[cluster] += 1;
            for (j, &val) in vector.iter().enumerate() {
                new_centroids[cluster][j] += val;
            }
        }
        
        // Average to get new centroids
        for (centroid, count) in new_centroids.iter_mut().zip(counts.iter()) {
            if *count > 0 {
                for val in centroid.iter_mut() {
                    *val /= *count as f32;
                }
            }
        }
        
        // Handle empty clusters by reinitializing from random point
        for (i, count) in counts.iter().enumerate() {
            if *count == 0 {
                let random_vec = vectors.choose(&mut thread_rng()).unwrap();
                new_centroids[i] = random_vec.clone();
            }
        }
        
        centroids = new_centroids;
    }
    
    let inertia = compute_inertia(vectors, &centroids, &assignments);
    
    KMeansResult {
        centroids,
        assignments,
        inertia,
    }
}

/// K-means++ initialization for better starting centroids
fn kmeans_plus_plus_init(vectors: &[Vec<f32>], k: usize) -> Vec<Vec<f32>> {
    let mut rng = thread_rng();
    let mut centroids = Vec::with_capacity(k);
    
    // Choose first centroid randomly
    let first = vectors.choose(&mut rng).unwrap().clone();
    centroids.push(first);
    
    // Choose remaining centroids with probability proportional to distance²
    for _ in 1..k {
        let mut distances: Vec<f32> = vectors
            .iter()
            .map(|v| {
                let (_, dist) = find_nearest_centroid(v, &centroids);
                dist * dist
            })
            .collect();
        
        // Weighted random selection
        let total: f32 = distances.iter().sum();
        let mut threshold = rand::random::<f32>() * total;
        
        for (i, &dist) in distances.iter().enumerate() {
            threshold -= dist;
            if threshold <= 0.0 {
                centroids.push(vectors[i].clone());
                break;
            }
        }
    }
    
    centroids
}

/// Find nearest centroid and its distance
fn find_nearest_centroid(vector: &[f32], centroids: &[Vec<f32>]) -> (usize, f32) {
    centroids
        .iter()
        .enumerate()
        .map(|(i, centroid)| {
            let dist = euclidean_distance(vector, centroid);
            (i, dist)
        })
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
}

/// Compute total inertia (sum of squared distances to centroids)
fn compute_inertia(
    vectors: &[Vec<f32>],
    centroids: &[Vec<f32>],
    assignments: &[usize],
) -> f32 {
    vectors
        .iter()
        .zip(assignments.iter())
        .map(|(vec, &cluster)| {
            let dist = euclidean_distance(vec, &centroids[cluster]);
            dist * dist
        })
        .sum()
}

/// Euclidean distance between two vectors
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_kmeans_simple() {
        // Create two clear clusters
        let mut vectors = Vec::new();
        
        // Cluster 1: around [0, 0]
        for _ in 0..50 {
            vectors.push(vec![
                rand::random::<f32>() * 0.1,
                rand::random::<f32>() * 0.1,
            ]);
        }
        
        // Cluster 2: around [10, 10]
        for _ in 0..50 {
            vectors.push(vec![
                10.0 + rand::random::<f32>() * 0.1,
                10.0 + rand::random::<f32>() * 0.1,
            ]);
        }
        
        let result = kmeans(&vectors, 2, 100, 0.001);
        
        assert_eq!(result.centroids.len(), 2);
        // Centroids should be roughly at [0,0] and [10,10]
        // (order may vary)
    }
}