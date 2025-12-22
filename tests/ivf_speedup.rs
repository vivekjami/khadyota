use khadyota::indexing::IVFIndex;
use khadyota::distance::cosine_distance;
use std::time::Instant;

#[test]
fn test_ivf_speedup() {
    // Generate test data
    println!("Generating 10,000 test vectors...");
    let mut vectors = Vec::new();
    for i in 0..10_000 {
        let vec: Vec<f32> = (0..512)
            .map(|j| ((i * 512 + j) as f32).sin())
            .collect();
        vectors.push(vec);
    }
    
    let query: Vec<f32> = (0..512).map(|i| (i as f32).cos()).collect();
    
    // Naive linear scan baseline
    println!("\n--- Naive Linear Scan ---");
    let start = Instant::now();
    
    let mut distances: Vec<(u32, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let dist = cosine_distance(&query, v);
            (i as u32, dist)
        })
        .collect();
    
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let naive_top10: Vec<u32> = distances.iter().take(10).map(|(i, _)| *i).collect();
    
    let naive_time = start.elapsed();
    println!("Time: {:?}", naive_time);
    
    // IVF search with different probe values
    for num_probe in [1, 3, 5, 10] {
        println!("\n--- IVF Search (probe={}) ---", num_probe);
        
        let mut index = IVFIndex::new(512, 100, num_probe);
        
        let build_start = Instant::now();
        index.build(&vectors, 100);
        println!("Build time: {:?}", build_start.elapsed());
        
        let search_start = Instant::now();
        let clusters = index.probe(&query);
        let candidates = index.get_candidates(&clusters);
        
        let mut distances: Vec<(u32, f32)> = candidates
            .iter()
            .map(|&vec_id| {
                let dist = cosine_distance(&query, &vectors[vec_id as usize]);
                (vec_id, dist)
            })
            .collect();
        
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let ivf_top10: Vec<u32> = distances.iter().take(10).map(|(i, _)| *i).collect();
        
        let ivf_time = search_start.elapsed();
        println!("Search time: {:?}", ivf_time);
        println!("Speedup: {:.2}x", naive_time.as_secs_f64() / ivf_time.as_secs_f64());
        
        // Calculate recall
        let recall = ivf_top10.iter()
            .filter(|id| naive_top10.contains(id))
            .count() as f32 / 10.0;
        println!("Recall@10: {:.1}%", recall * 100.0);
        println!("Candidates searched: {} / {}", candidates.len(), vectors.len());
    }
}